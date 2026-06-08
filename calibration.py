"""
Model and utilities to calibrate a trained waveform generator model to the target waveform.
We will use the training data to calibrate the generated outputs, by predicting the residuals
between the generated [amp,freq] and the target [amp,freq], for example.
"""

import os
import json
import h5py
import argparse
import numpy as np
import pandas as pd
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from datacvae import CustomDataset, CustomDataLoader
from optimize import load_flex_model

from maincvae import plot_polarization_mismatch
from plotutils import putils

from utils.generic import init_logging, init_verbosity_args
global logger


PROJECT_DIR = 'v0p1'

TODAY = datetime.date.today().strftime("%Y%m%d")
TIME = datetime.datetime.now().strftime("%H%M%S")
NOW = TODAY + '-' + TIME

# -- define some constants for waveform generation
SAMPLE_RATE = 8192  # Hz
DURATION = 1.0  # seconds
FMIN = 20.0  # Hz
FREF = 50.0  # Hz


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    PRECISION = 'float32'  # Can also use float64 for CUDA, but float32 is usually sufficient and faster on GPU
    torch.set_float32_matmul_precision("high")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    PRECISION = 'float32'  # Use float32 for MPS since it does not support float64 well
else:
    DEVICE = torch.device("cpu")
    PRECISION = 'float32'  # Use double precision for CPU



class ConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(num_groups=8, num_channels=channels),
        )

        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class ResidualCalibrationCNN(nn.Module):
    """
    A ResNet-style 1D CNN model that takes in some ML generated [amp,freq] and parameters,
    and predicts the residuals to be added to the ML generated outputs to better match the target waveforms.
    """
    def __init__(
        self,
        input_channels,
        output_channels=2,
        hidden_channels=128,
        num_blocks=5,
        kernel_size=7,
        dropout=0.0,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.input_proj = nn.Conv1d(
            input_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.blocks = nn.Sequential(
            *[
                ConvBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_proj = nn.Conv1d(
            hidden_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        # Important: initialize final layer near zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        """
        x shape: (batch, input_channels, n)
        output shape: (batch, output_channels, n)

        Output is the predicted residual, preferably in standardized/scaled units.
        """
        z = self.input_proj(x)
        z = self.blocks(z)
        residual = self.output_proj(z)
        return residual
    


def get_calibrator_input(wfmodel, originals, labels, 
                         data_hdf=None, indices=None,
                         params_mean=None, params_std=None):
    """
    Function to obtain the input and target for the calibrator model, 
    given the originals waveforms, the parameters, and the trained waveform model.

    Arguments:
    ----------
    wfmodel : torch.nn.Module
        The trained waveform generator model that takes in the parameters and generates [amp,freq].
    originals : torch.Tensor
        The originals waveforms in the form of [amp,freq], shape: (batch, 2, n)
    labels : torch.Tensor
        The parameters corresponding to the waveforms, shape: (batch, num_params)
    data_hdf : str, optional
        The name of the HDF file to save the calibrator input and target residuals, by default None. 
        If None, the data will not be saved or read from the HDF file.
    indices : torch.Tensor, optional
        The original sample indices from the dataset for this batch, shape: (batch,), used for saving 
        the data to HDF file with unique group names, by default None.
        NOTE: These are the only groups that will be saved to the HDF file.
    params_mean : torch.Tensor, optional
        The mean values for normalizing the parameters, by default None.
    params_std : torch.Tensor, optional
        The standard deviation values for normalizing the parameters, by default None.

    Returns:
    --------
    calibrator_input : torch.Tensor
        The input to the calibrator model, which includes the ML generated [amp,freq] and the parameters, 
        shape: (batch, input_channels, n)
    calibrator_target : tuple of torch.Tensor
        The target residuals for amplitude and frequency, each of shape: (batch, 2, n)
    """
    orig_amp, orig_freq = originals[:, 0, :], originals[:, 1, :]

    # -- get the ml predictions for this batch
    with torch.no_grad():
        ml_outputs = wfmodel.generate(labels, convert_to_hphc=False)  # shape: (batch, 2, n)
    ml_amp, ml_freq = ml_outputs[:, 0, :], ml_outputs[:, 1, :]

    # -- repeat first element in ML generated outputs which have shape (batch, 2, 8190),
    # -- while original [amp,freq] are of shape (batch, 2, 8191)!
    ml_amp = torch.cat([ml_amp[:, 0:1], ml_amp], dim=1)  # shape: (batch, n+1)
    ml_freq = torch.cat([ml_freq[:, 0:1], ml_freq], dim=1)  # shape: (batch, n+1)

    assert ml_amp.shape == orig_amp.shape, f"ML generated amplitude shape {ml_amp.shape} does not match original amplitude shape {orig_amp.shape}"
    assert ml_freq.shape == orig_freq.shape, f"ML generated frequency shape {ml_freq.shape} does not match original frequency shape {orig_freq.shape}"

    target_amp_residual = orig_amp - ml_amp
    target_freq_residual = orig_freq - ml_freq
    logger.debug(f"Target amplitude residual shape: {target_amp_residual.shape}, Target frequency residual shape: {target_freq_residual.shape}")

    # -- calibrator takes in ml generated [amp,freq] and the parameters, and predicts the residuals
    calibrator_input = torch.stack([ml_amp, ml_freq], dim=1)  # shape: (batch, 2, n)
    param_m1, param_m2, param_s1z, param_s2z = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]

    # -- repeat the parameters across the time dimension to match the shape of ml_amp/ml_freq
    param_m1 = param_m1.unsqueeze(-1).expand(-1, ml_amp.shape[-1])
    param_m2 = param_m2.unsqueeze(-1).expand(-1, ml_amp.shape[-1])
    param_s1z = param_s1z.unsqueeze(-1).expand(-1, ml_amp.shape[-1])
    param_s2z = param_s2z.unsqueeze(-1).expand(-1, ml_amp.shape[-1])

    # fig, ax = plt.subplots(4, 1, figsize=(12, 12))
    # ax[0].plot(ml_amp[0].cpu().numpy(), label='ML Amp')
    # ax[0].plot(orig_amp[0].cpu().numpy(), label='Original Amp')
    # ax[0].set_title('ML Generated Amplitude vs Original Amplitude')
    # ax[0].legend()
    # ax[1].plot(ml_freq[0].cpu().numpy(), label='ML Freq')
    # ax[1].plot(orig_freq[0].cpu().numpy(), label='Original Freq')
    # ax[1].set_title('ML Generated Frequency vs Original Frequency')
    # ax[1].legend()
    # ax[2].plot(target_amp_residual[0].cpu().numpy(), label='Target Amp Residual')
    # ax[2].set_title('Target Amplitude Residual')
    # ax[2].legend()
    # ax[3].plot(target_freq_residual[0].cpu().numpy(), label='Target Freq Residual')
    # ax[3].set_title('Target Frequency Residual')
    # ax[3].legend()
    # plt.tight_layout()
    # plt.savefig(f'calibrator_input_example_{NOW}.png')
    # plt.close()

    if params_mean is not None and params_std is not None:
        # -- normalize the parameters using the mean and std from the model config
        param_m1 = (param_m1 - params_mean[0]) / params_std[0]
        param_m2 = (param_m2 - params_mean[1]) / params_std[1]
        param_s1z = (param_s1z - params_mean[2]) / params_std[2]
        param_s2z = (param_s2z - params_mean[3]) / params_std[3]

    calibrator_input = torch.cat([calibrator_input, param_m1.unsqueeze(1), param_m2.unsqueeze(1),
                                param_s1z.unsqueeze(1), param_s2z.unsqueeze(1)], dim=1)  # shape: (batch, 6, n)
    logger.debug(f"Calibrator input shape: {calibrator_input.shape}")

    if data_hdf is not None and indices is not None:
        # # logger.debug(f"Saving calibrator input and target residuals to {data_hdf} for this batch...")
        # savename = f'../data/{data_hdf.strip(".hdf")}_{NOW}.hdf'
        # os.makedirs(os.path.dirname(savename), exist_ok=True)

        # Save calibaration input, target residuals, and parameters the first time, so that we can reuse them next time!
        # Save this data to a HDF file repeatedly appending to it every time we call this function, 
        # and then we can load this data directly in the calibrator training loop, instead of having to 
        # generate it on the fly every time, which is computationally expensive since it requires running the ML model inference every time.
        with h5py.File('../data/' + data_hdf, 'a') as f:
            # -- create a new group for each data waveform in the batch, with datasets for:
            # -- [ml_amp, ml_freq, target_amp_residual, target_freq_residual, param_m1, param_m2, param_s1z, param_s2z]
            for i in range(len(indices)):
                group_name = f'sample{int(indices[i])}'  # use the original sample index from the dataset as the group name
                if group_name in f:
                    del f[group_name]  # delete existing group if it exists, to avoid appending to old data
                grp = f.create_group(group_name)
                grp.create_dataset('ml_amp', data=calibrator_input[i, 0, :].cpu().numpy())
                grp.create_dataset('ml_freq', data=calibrator_input[i, 1, :].cpu().numpy())
                grp.create_dataset('target_amp_residual', data=target_amp_residual[i].cpu().numpy())
                grp.create_dataset('target_freq_residual', data=target_freq_residual[i].cpu().numpy())
                grp.create_dataset('param_m1', data=param_m1[i, 0].cpu().numpy())
                grp.create_dataset('param_m2', data=param_m2[i, 0].cpu().numpy())
                grp.create_dataset('param_s1z', data=param_s1z[i, 0].cpu().numpy())
                grp.create_dataset('param_s2z', data=param_s2z[i, 0].cpu().numpy())
        # logger.debug(f"Saved calibrator input and target residuals to {savename}")
    return calibrator_input, (target_amp_residual, target_freq_residual)


def read_calibrator_input(data_hdf, indices, params_mean=None, params_std=None):
    """
    Read Calibrator input and target residuals data from HDF file.

    Arguments:
    ----------
    data_hdf : str
        The name of the HDF file to read the calibrator input and target residuals from.
    indices : torch.Tensor
        The original sample indices from the dataset for this batch, shape: (batch,), used for reading 
        the data from HDF file with unique group names.
    params_mean : torch.Tensor, optional
        The mean values for normalizing the parameters, by default None.
    params_std : torch.Tensor, optional
        The standard deviation values for normalizing the parameters, by default None.

    Returns:
    --------
    calibrator_input : torch.Tensor
        The input to the calibrator model, which includes the ML generated [amp,freq] and the parameters, 
        shape: (batch, input_channels, n)
    calibrator_target : tuple of torch.Tensor
        The target residuals for amplitude and frequency, each of shape: (batch, 2, n)
    """
    data_hdf = data_hdf + '.hdf' if not data_hdf.endswith('.hdf') else data_hdf
    data_path = f'../data/{data_hdf}'
    calibrator_inputs = []
    target_amp_residuals = []
    target_freq_residuals = []
    with h5py.File(data_path, 'r') as f:
        for i in range(len(indices)):
            group_name = f'sample{int(indices[i])}'
            ml_amp = torch.tensor(f[group_name]['ml_amp'][:], dtype=getattr(torch, PRECISION), device=DEVICE)
            ml_freq = torch.tensor(f[group_name]['ml_freq'][:], dtype=getattr(torch, PRECISION), device=DEVICE)
            target_amp_residual = torch.tensor(f[group_name]['target_amp_residual'][:], dtype=getattr(torch, PRECISION), device=DEVICE)
            target_freq_residual = torch.tensor(f[group_name]['target_freq_residual'][:], dtype=getattr(torch, PRECISION), device=DEVICE)

            # -- parameter values are scalars for each data, so we don't convert them to tensors until we read them, and then we repeat them across the time dimension to match the shape of ml_amp/ml_freq, which is (n,)
            # -- Scalar values in HDF datasets are available via ellipsis indexing `[...]`
            param_m1 = f[group_name]['param_m1'][...].astype(getattr(np, PRECISION)).item()
            param_m2 = f[group_name]['param_m2'][...].astype(getattr(np, PRECISION)).item()
            param_s1z = f[group_name]['param_s1z'][...].astype(getattr(np, PRECISION)).item()
            param_s2z = f[group_name]['param_s2z'][...].astype(getattr(np, PRECISION)).item()

            if params_mean is not None and params_std is not None:
                # -- normalize the parameters using the mean and std from the model config
                param_m1 = (param_m1 - params_mean[0].item()) / params_std[0].item()
                param_m2 = (param_m2 - params_mean[1].item()) / params_std[1].item()
                param_s1z = (param_s1z - params_mean[2].item()) / params_std[2].item()
                param_s2z = (param_s2z - params_mean[3].item()) / params_std[3].item()
            logger.debug(f"Read calibrator input from HDF for group {group_name}: param_m1={param_m1}, param_m2={param_m2}, param_s1z={param_s1z}, param_s2z={param_s2z}")

            params = torch.tensor([param_m1, param_m2, param_s1z, param_s2z], dtype=getattr(torch, PRECISION), device=DEVICE)
            params = params.unsqueeze(-1).expand(-1, ml_amp.shape[-1])  # shape: (4, n)

            calibrator_input = torch.stack([ml_amp, ml_freq, params[0], params[1], params[2], params[3]], dim=0)  # shape: (6, n)
            calibrator_inputs.append(calibrator_input)
            target_amp_residuals.append(target_amp_residual)
            target_freq_residuals.append(target_freq_residual)

    calibrator_inputs = torch.stack(calibrator_inputs, dim=0)  # shape: (batch, input_channels, n)
    target_amp_residuals = torch.stack(target_amp_residuals, dim=0)  # shape: (batch, n)
    target_freq_residuals = torch.stack(target_freq_residuals, dim=0)  # shape: (batch, n)

    return calibrator_inputs, (target_amp_residuals, target_freq_residuals)



def match_data_between_wf_and_calibrator_hdfs(wf_hdf, calibrator_hdf, wfmodel, 
                                              wf_dataset_obj=None,
                                              params_mean=None, params_std=None):
    """
    Check if all the groups / waveforms present in the `wf_hdf` are also present in the
    `calibrator_hdf`! If not, then create the missing groups in the `calibrator_hdf` by 
    reading data from the `wf_hdf` and using the the `get_calibrator_input` function!
    """
    logger.info(f"Checking for missing groups in {calibrator_hdf} that are present in {wf_hdf}...")
    wf_hdf = wf_hdf + '.hdf' if not wf_hdf.endswith('.hdf') else wf_hdf
    calibrator_hdf = calibrator_hdf + '.hdf' if not calibrator_hdf.endswith('.hdf') else calibrator_hdf

    wfhf = h5py.File('../data/' + wf_hdf, 'r')
    chf = h5py.File('../data/' + calibrator_hdf, 'a')  # open in append mode to create missing groups if needed
    wf_groups = set(wfhf.keys())
    calibrator_groups = set(chf.keys())
    missing_groups = wf_groups - calibrator_groups

    if len(missing_groups) > 0:
        logger.info(f"Found {len(missing_groups)} missing groups in {calibrator_hdf} that are present in {wf_hdf}. Will create these groups in {calibrator_hdf} by reading data from {wf_hdf} and using the `get_calibrator_input` function.")
        
        for group_name in tqdm(missing_groups, desc="Creating missing groups in calibrator HDF"):
            # -- read the original waveform and parameters from the `wf_hdf` for this group
            data = wf_dataset_obj.__getitem__(int(group_name.strip('sample')))
            originals, _, labels, keys, _, indices = data

            originals = torch.from_numpy(originals).to(getattr(torch, PRECISION)).to(DEVICE)  # shape: (2, n)
            labels = torch.from_numpy(labels).to(getattr(torch, PRECISION)).to(DEVICE)  # shape: (num_params,)
            originals = originals.unsqueeze(0)  # add batch dimension, shape: (1, 2, n)
            labels = labels.unsqueeze(0)  # add batch dimension, shape: (1, num_params)
            
            # -- get the calibrator input and target residuals for this group using the `get_calibrator_input` function
            _, _ = get_calibrator_input(
                wfmodel=wfmodel,
                originals=originals,  # shape: (1, 2, n)
                labels=labels,  # shape: (1, num_params)
                data_hdf=calibrator_hdf,  # save this data group to the calibrator HDF file!
                indices=torch.tensor([int(group_name.strip('sample'))]),  # use the original sample index from the dataset!
                params_mean=params_mean,
                params_std=params_std,
            )
            logger.info(f"Created missing group {group_name} in {calibrator_hdf} with calibrator input and target residuals for the corresponding waveform in {wf_hdf}.")
        logger.info(f"Finished creating missing groups in {calibrator_hdf}. All groups in {wf_hdf} are now present in {calibrator_hdf}.")
    else:
        logger.info(f"All groups in {wf_hdf} are already present in {calibrator_hdf}. No missing groups found.")

    assert len(wf_groups - set(chf.keys())) == 0, f"After attempting to create missing groups, there are still {len(wf_groups - set(chf.keys()))} missing groups in {calibrator_hdf} that are present in {wf_hdf}. Please check the logs for details."
    # close the HDF files
    wfhf.close()
    chf.close()



class CalibratorDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for the calibrator model, which reads the calibrator input and target residuals from a HDF file.
    """
    def __init__(self, data_hdf, params_mean=None, params_std=None):
        self.data_hdf = data_hdf
        self.params_mean = params_mean
        self.params_std = params_std
        assert self.params_mean is None or self.params_mean.shape == (4,), f"Expected params_mean to be of shape (4,), but got {self.params_mean.shape}"
        assert self.params_std is None or self.params_std.shape == (4,), f"Expected params_std to be of shape (4,), but got {self.params_std.shape}"
        self.init_hdf()  # initialize the HDF file for reading the data in the `__getitem__` method

    def __len__(self):
        return len(self.data_file.keys())  # number of groups in the HDF file, which corresponds to the number of data samples
    
    def init_hdf(self):
        # -- check if the HDF file exists, if not, create an empty HDF file with the same name, so that we can write to it later on in the `get_calibrator_input` function without having to worry about file not found errors.
        data_hdf = self.data_hdf + '.hdf' if not self.data_hdf.endswith('.hdf') else self.data_hdf
        data_path = f'../data/{data_hdf}'

        if not os.path.exists(data_path):
            with h5py.File(data_path, 'w') as f:
                pass  # just create an empty HDF file
            logger.info(f"Created empty HDF file at {data_path} for storing calibrator input and target residuals.")
        else:
            logger.info(f"HDF file {data_path} already exists. Will read from it or append to it when generating calibrator input and target residuals.")

        # open the HDF file for reading in the dataset initialization, so that we can read from it in the `__getitem__` method without having to open and close the file every time, which is inefficient. We will keep this file open for the lifetime of the dataset, and close it when the dataset is deleted.
        self.data_file = h5py.File(data_path, 'r')
        logger.info(f"Opened HDF file {data_path} for reading calibrator input and target residuals in the CalibratorDataset.")

    def close_hdf(self):
        # close the HDF file when the dataset is deleted, to free up resources
        if hasattr(self, 'data_file') and self.data_file is not None:
            self.data_file.close()
            logger.info(f"Closed HDF file {self.data_hdf} after reading calibrator input and target residuals.")

    def __del__(self):
        self.close_hdf()

    def read_input_data(self, index):
        # -- read the calibrator input and target residuals from the HDF file for the given indices
        group_name = f'sample{int(index)}'
        ml_amp = torch.tensor(self.data_file[group_name]['ml_amp'][:], dtype=getattr(torch, PRECISION), device=DEVICE)
        ml_freq = torch.tensor(self.data_file[group_name]['ml_freq'][:], dtype=getattr(torch, PRECISION), device=DEVICE)
        target_amp_residual = torch.tensor(self.data_file[group_name]['target_amp_residual'][:], dtype=getattr(torch, PRECISION), device=DEVICE)
        target_freq_residual = torch.tensor(self.data_file[group_name]['target_freq_residual'][:], dtype=getattr(torch, PRECISION), device=DEVICE)

        # -- parameter values are scalars for each data, so we don't convert them to tensors until we read them, and then we repeat them across the time dimension to match the shape of ml_amp/ml_freq, which is (n,)
        # -- Scalar values in HDF datasets are available via ellipsis indexing `[...]`
        param_m1 = self.data_file[group_name]['param_m1'][...].astype(getattr(np, PRECISION)).item()
        param_m2 = self.data_file[group_name]['param_m2'][...].astype(getattr(np, PRECISION)).item()
        param_s1z = self.data_file[group_name]['param_s1z'][...].astype(getattr(np, PRECISION)).item()
        param_s2z = self.data_file[group_name]['param_s2z'][...].astype(getattr(np, PRECISION)).item()

        if self.params_mean is not None and self.params_std is not None:
            # -- normalize the parameters using the mean and std from the model config
            param_m1 = (param_m1 - self.params_mean[0].item()) / self.params_std[0].item()
            param_m2 = (param_m2 - self.params_mean[1].item()) / self.params_std[1].item()
            param_s1z = (param_s1z - self.params_mean[2].item()) / self.params_std[2].item()
            param_s2z = (param_s2z - self.params_mean[3].item()) / self.params_std[3].item()
            logger.debug(f"Read calibrator input from HDF for group {group_name}: param_m1={param_m1}, param_m2={param_m2}, param_s1z={param_s1z}, param_s2z={param_s2z}")
        
        params = torch.tensor([param_m1, param_m2, param_s1z, param_s2z], dtype=getattr(torch, PRECISION), device=DEVICE)
        params = params.unsqueeze(-1).expand(-1, ml_amp.shape[-1])  # shape: (4, n)
        calibrator_input = torch.stack([ml_amp, ml_freq, params[0], params[1], params[2], params[3]], dim=0)  # shape: (6, n)
        return (calibrator_input, target_amp_residual, target_freq_residual)

    def __getitem__(self, idx):
        calibrator_input, target_amp_residual, target_freq_residual = self.read_input_data(
            index=idx
        )
        return (calibrator_input.squeeze(0), target_amp_residual.squeeze(0), target_freq_residual.squeeze(0))
    

class CalibratorDataLoader(torch.utils.data.DataLoader):
    """
    A custom data loader class for the calibrator dataset, which simply wraps the CalibratorDataset and allows for batching and shuffling.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
            
        
def merger_weighted_mse_loss_func(true, predicted, amp_ml):
    """
    A custom loss function that computes a weighted MSE loss, where the weights are based on the amplitude of the ML generated waveform.
    The idea is to give more weight to the parts of the waveform where the amplitude is higher, since those parts are more important for the overall waveform shape and the mismatch calculation. The weights are computed as a function of the normalized amplitude of the ML generated waveform, with a minimum weight to ensure that we don't completely ignore the low amplitude parts of the waveform. The gamma parameter can be used to control how much more weight we give to the high amplitude parts compared to the low amplitude parts. This loss function can help the calibrator model focus on learning the residuals in the parts of the waveform that matter the most for improving the overall waveform accuracy, while still allowing it to learn from the entire waveform.
    """
    eps = 1e-12
    w_min = 0.05
    gamma = 1.0  # try 1.0 first, then 2.0 if needed

    # amp_ml shape: (batch, n)
    amp_norm = torch.abs(amp_ml) / (torch.amax(torch.abs(amp_ml), dim=-1, keepdim=True) + eps)

    weights = w_min + (1.0 - w_min) * amp_norm**gamma
    # shape: (batch, n)

    loss = torch.sum(weights * (predicted - true)**2) / torch.sum(weights)
    return loss
    
    

def train_calibrator(wfmodel_modelpath=f'../trained-models/model-20251004_072338-10', 
                     wfmodel_configpath='modelconfig-cvae-paper-I.json',
                     approximant='SEOBNRv4', batch_size=64, num_epochs=25,
                     timestamp=NOW, use_calibrator_dataloaders=True,
                     dummyrun=False):
    """
    Train the residual calibrator model.

    NOTE: ML generated waveforms have shape (batch, 2, 8190), for the Paper-I model,
    likely due to the way data was preprocessed for it. Whereas, the saved waveforms
    in the `regen` HDF files have shape (batch, 2, 8191). This is because the `regen` 
    HDF files were generated such that one element from the amplitude array is removed
    to match the shape of the frequency array, which is one element shorter by definition.
    Something else happended in the ML model training that caused the generated waveforms
    to have one less element in the time dimension, which is not ideal but we can work with it for now.

    I bypassed this issue when caculating the mismatch for the ML waveforms earlier, by removing
    one element from the frequency array before calculating the phase, and then adding a start
    phase to the calculated phase, so that it matches the shape of the amplitude array. Still,
    I think that time I didn't use the `regen` version of the HDF files, so that the original
    waveforms and [amp,freq] were also of 8190 length.

    For now, we just repeat the first element twice for the ML generated [amp,freq] and train
    the calibrator to predict the residuals for a waveform of length 8191 instead of 8190. This
    allows us to directly compare the calibrated waveforms with the original waveforms in the `regen`
    HDF files, without having to worry about the shape mismatch issue. We can always retrain the ML 
    model later with the correct shape of the waveforms, and then retrain the calibrator on top of that.
    """
    logger.info(f"Training residual calibrator model with ML waveform model from {wfmodel_modelpath} and config from {wfmodel_configpath}")
    # -- init waveform model
    try:
        wfmodel = load_flex_model(model_path=wfmodel_modelpath, 
                                configpath=wfmodel_configpath, 
                                device=DEVICE, precision=PRECISION,)
    except FileNotFoundError:
        wfmodel_modelpath = f'../{PROJECT_DIR}/trained-models/model-20251004_072338-10'
        wfmodel = load_flex_model(model_path=wfmodel_modelpath, 
                                configpath=wfmodel_configpath, 
                                device=DEVICE, precision=PRECISION,)
    wfmodel.eval()  # set to eval mode since we are only using it for inference to generate the calibrator inputs
    logger.info(f"Loaded waveform model for calibrator input generation: {wfmodel}")

    # -- Load `labels_mean` and `labels_std` from model config file, since original CVAE model `state_dict` doesn't have them!
    with open(wfmodel_configpath, 'r') as f:
        model_config = json.load(f)
    params_mean = torch.tensor(model_config['labels_mean'], dtype=getattr(torch, PRECISION), device=DEVICE)
    params_std = torch.tensor(model_config['labels_std'], dtype=getattr(torch, PRECISION), device=DEVICE)

    savename = f'../{PROJECT_DIR}/results/{TODAY}'
    os.makedirs(savename, exist_ok=True)
    os.makedirs(f'../{PROJECT_DIR}/trained-models', exist_ok=True)
    
    trainhdf = '../data/SEOBNRv4-train-100000-fcutoff-uniform-aligned-regen.hdf'
    valhdf = '../data/SEOBNRv4-val-100000-fcutoff-uniform-aligned-regen.hdf'

    calmodel = ResidualCalibrationCNN(
        input_channels=6,  # [ml_amp, ml_freq, param_m1, param_m2, param_s1z, param_s2z]
        output_channels=2,
        hidden_channels=128,
        num_blocks=5,
        kernel_size=7,
        dropout=0.0,
    )
    calmodel.to(device=DEVICE, dtype=getattr(torch, PRECISION))
    if DEVICE==torch.device("cuda"):
        # calmodel = torch.nn.DataParallel(calmodel)
        calmodel.compile() # compile the model for faster training on CUDA
        logger.info("Compiled the calibrator model for faster training on CUDA.")
    logger.info(f"Calibrator model architecture: {calmodel}")

    
    wf_train_set = CustomDataset(forwhat='train', approximant=approximant, hdf_fname=trainhdf, 
                                train_device=DEVICE, precision=PRECISION, return_sample_indices=True)
    wf_valid_set = CustomDataset(forwhat='valid', approximant=approximant, hdf_fname=valhdf, 
                                train_device=DEVICE, precision=PRECISION, return_sample_indices=True)

    if use_calibrator_dataloaders:
        # -- make sure that calibarator HDF datagroups match those in the waveform HDF!
        match_data_between_wf_and_calibrator_hdfs(
            wf_hdf=trainhdf, 
            calibrator_hdf=f'calibrator_training_data_{timestamp}.hdf', 
            wfmodel=wfmodel,
            wf_dataset_obj=wf_train_set,
            params_mean=params_mean, params_std=params_std
        )
        match_data_between_wf_and_calibrator_hdfs(
            wf_hdf=valhdf, 
            calibrator_hdf=f'calibrator_validation_data_{timestamp}.hdf', 
            wfmodel=wfmodel,
            wf_dataset_obj=wf_valid_set,
            params_mean=params_mean, params_std=params_std
        )
        train_set = CalibratorDataset(data_hdf=f'calibrator_training_data_{timestamp}', 
                                        params_mean=params_mean, params_std=params_std)
        valid_set = CalibratorDataset(data_hdf=f'calibrator_validation_data_{timestamp}', 
                                        params_mean=params_mean, params_std=params_std)
        training_loader = CalibratorDataLoader(train_set, batch_size=batch_size, shuffle=True)
        validation_loader = CalibratorDataLoader(valid_set, batch_size=batch_size, shuffle=True)
        logger.info("Using CalibratorDataset and CalibratorDataLoader for training the calibrator model, which read the calibrator input and target residuals from HDF files.")
    else:
        train_set = wf_train_set
        valid_set = wf_valid_set
        training_loader = CustomDataLoader(train_set, batch_size=batch_size, shuffle=True)
        validation_loader = CustomDataLoader(valid_set, batch_size=batch_size, shuffle=True)
        logger.info("Using CustomDataset and CustomDataLoader for training the calibrator model, which generate the calibrator input and target residuals on the fly by running the ML model inference every time. This is computationally expensive, so it's recommended to use the CalibratorDataset and CalibratorDataLoader instead, which read the pre-generated data from HDF files.")
    logger.info(training_loader.__dict__)
    ntbatches = len(training_loader)
    nvbatches = len(validation_loader)
    logger.info(f'Number of Training batches: {ntbatches}')  # this doesn't return the batchsize!
    logger.info(f'Number of Validationg batches: {nvbatches}')

    optimizer = torch.optim.AdamW(calmodel.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
    )
    loss_fn = torch.nn.MSELoss()
    logger.info(f"Starting training loop for {num_epochs} epochs...")

    epoch_losses = pd.DataFrame(columns=['epoch', 'train_loss_amp', 'train_loss_freq', 'val_loss_amp', 'val_loss_freq'],
                                index=range(num_epochs))
    running_train_loss_file = open(f'{savename}/calibrator_running_train_losses_{NOW}.csv', 'w')
    running_train_loss_file.write(','.join(['epoch', 'train_loss_amp', 'train_loss_freq']) + '\n')
    running_val_loss_file = open(f'{savename}/calibrator_running_val_losses_{NOW}.csv', 'w')
    running_val_loss_file.write(','.join(['epoch', 'val_loss_amp', 'val_loss_freq']) + '\n')

    logger.info(f"Training calibrator model for {num_epochs} epochs with batch size {batch_size}...")

    calibrator_training_data_hdf = f'calibrator_training_data_{timestamp}.hdf'
    calibrator_val_data_hdf = f'calibrator_validation_data_{timestamp}.hdf'
    logger.info(f"Calibrator training data file is set to {calibrator_training_data_hdf}, \
                and validation data is set to {calibrator_val_data_hdf}")

    if dummyrun:
        logger.info("Running in dummy mode for quick testing...")
        num_epochs = 1

    best_val_loss = float('inf')
    for epoch in tqdm(range(num_epochs), desc='Epoch'):
        calmodel.train(True)

        # -- lower learning rate after some epochs, so that model doesn't diverge after reaching a good loss value!
        if epoch >= 8:
            for group in optimizer.param_groups:
                group["lr"] = 3e-4

        train_loss_amp = 0.0
        train_loss_freq = 0.0
        counter = 0
        
        for databatch in tqdm(training_loader, total=len(training_loader), desc='Train Steps'):
            counter += 1
            if dummyrun and counter > 2:
                break


            # originals, _, labels, keys, _, indices = databatch
            # originals = originals.to(DEVICE)
            # labels = labels.to(DEVICE)

            # if epoch == 0:
            #     try:
            #         calibrator_input, calibrator_target = read_calibrator_input(
            #             data_hdf=calibrator_training_data_hdf,          
            #             indices=indices, params_mean=params_mean, params_std=params_std)
            #     except FileNotFoundError:
            #         logger.warning("Calibrator input data not found, generating new data...")
            #         calibrator_input, calibrator_target = get_calibrator_input(
            #             wfmodel, originals, labels, 
            #             data_hdf=calibrator_training_data_hdf, 
            #             indices=indices, params_mean=params_mean, params_std=params_std)
            # else:
            #     calibrator_input, calibrator_target = read_calibrator_input(
            #         data_hdf=calibrator_training_data_hdf,          
            #         indices=indices, params_mean=params_mean, params_std=params_std)
            # target_amp_residual, target_freq_residual = calibrator_target

            calibrator_input, target_amp_residual, target_freq_residual = databatch

            out = calmodel(calibrator_input)  # shape: (batch, 2, n)
            pred_amp_residual, pred_freq_residual = out[:, 0, :], out[:, 1, :]

            # -- compute loss and backprop
            # amp_loss = loss_fn(pred_amp_residual, target_amp_residual)
            # freq_loss = loss_fn(pred_freq_residual, target_freq_residual)
            amp_loss = merger_weighted_mse_loss_func(target_amp_residual, pred_amp_residual, calibrator_input[:, 0, :])
            freq_loss = merger_weighted_mse_loss_func(target_freq_residual, pred_freq_residual, calibrator_input[:, 0, :])

            loss = amp_loss + freq_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # -- log training loss for this batch
            running_train_loss_file.write(f"{epoch+1},{amp_loss.item()},{freq_loss.item()}\n")
            running_train_loss_file.flush()
            train_loss_amp += amp_loss.item()
            train_loss_freq += freq_loss.item()
            logger.debug(f"Epoch {epoch+1}, Batch {counter}, Amp Loss: {amp_loss.item()}, Freq Loss: {freq_loss.item()}")
        logger.info(f"Epoch {epoch+1}, Batch {counter}, Amp Loss: {amp_loss.item()}, Freq Loss: {freq_loss.item()}")

        # -- validation loop
        calmodel.eval()
        with torch.no_grad():
            val_loss_amp = 0.0
            val_loss_freq = 0.0
            counter = 0

            for databatch in tqdm(validation_loader, total=len(validation_loader), desc='Val Steps'):
                counter += 1
                if dummyrun and counter > 2:
                    break

                calibrator_input, target_amp_residual, target_freq_residual = databatch

                # originals, _, labels, keys, _, indices = databatch
                # originals = originals.to(DEVICE)
                # labels = labels.to(DEVICE)

                # if epoch == 0:
                #     try:
                #         calibrator_input, calibrator_target = read_calibrator_input(
                #             data_hdf=calibrator_val_data_hdf,          
                #             indices=indices, params_mean=params_mean, params_std=params_std)
                #     except FileNotFoundError:
                #         logger.warning("Calibrator input data not found, generating new data...")
                #         calibrator_input, calibrator_target = get_calibrator_input(
                #             wfmodel, originals, labels, 
                #             data_hdf=calibrator_val_data_hdf, 
                #             indices=indices, params_mean=params_mean, params_std=params_std)
                # else:
                #     calibrator_input, calibrator_target = read_calibrator_input(
                #         data_hdf=calibrator_val_data_hdf,          
                #         indices=indices, params_mean=params_mean, params_std=params_std)
                # target_amp_residual, target_freq_residual = calibrator_target

                out = calmodel(calibrator_input)  # shape: (batch, 2, n)
                pred_amp_residual, pred_freq_residual = out[:, 0, :], out[:, 1, :]

                # val_amp_loss = loss_fn(pred_amp_residual, target_amp_residual)
                # val_freq_loss = loss_fn(pred_freq_residual, target_freq_residual)
                val_amp_loss = merger_weighted_mse_loss_func(target_amp_residual, pred_amp_residual, calibrator_input[:, 0, :])
                val_freq_loss = merger_weighted_mse_loss_func(target_freq_residual, pred_freq_residual, calibrator_input[:, 0, :])
                val_loss = val_loss_amp + val_loss_freq

                # -- log validation loss for this batch
                running_val_loss_file.write(f"{epoch+1},{val_amp_loss.item()},{val_freq_loss.item()}\n")
                running_val_loss_file.flush()
                val_loss_amp += val_amp_loss.item()
                val_loss_freq += val_freq_loss.item()
                logger.debug(f"Epoch {epoch+1}, Batch {counter}, Val Amp Loss: {val_amp_loss.item()}, Val Freq Loss: {val_freq_loss.item()}")
            logger.info(f"Epoch {epoch+1}, Batch {counter}, Val Amp Loss: {val_amp_loss.item()}, Val Freq Loss: {val_freq_loss.item()}")

        # -- take a step in the learning rate scheduler based on the validation loss
        scheduler.step(val_loss)

        # -- log epoch losses
        epoch_losses.iloc[epoch] = {
            'epoch': epoch+1,
            'train_loss_amp': train_loss_amp/len(training_loader),
            'train_loss_freq': train_loss_freq/len(training_loader),
            'val_loss_amp': val_loss_amp/len(validation_loader),
            'val_loss_freq': val_loss_freq/len(validation_loader),
        }
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(validation_loader)}")

        # # -- every 5 epochs, save the model checkpoint
        # if (epoch + 1) % 5 == 0:
        #     outpath = f'../{PROJECT_DIR}/trained-models/calibrator_model_{NOW}_epoch{epoch+1}.pt'
        #     torch.save(calmodel.state_dict(), outpath)

        # -- save the best model checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            outpath = f'../{PROJECT_DIR}/trained-models/calibrator_model_{NOW}_epoch{epoch+1}.pt'
            torch.save(calmodel.state_dict(), outpath)

    # -- save the trained calibrator model
    outpath = f'../{PROJECT_DIR}/trained-models/calibrator_model_{NOW}.pt'
    torch.save(calmodel.state_dict(), outpath)
    # -- save the epoch losses to a CSV file
    epoch_losses.to_csv(f'{savename}/calibrator_epoch_losses_{NOW}.csv', index=False)
    # -- close the running loss files
    running_train_loss_file.close()
    running_val_loss_file.close()
    print("Training complete!")



def test_calibrator(wfmodel_modelpath=f'trained-models/model-20251004_072338-10', 
                    wfmodel_configpath='modelconfig-cvae-paper-I.json',
                    calibrator_modelpath=f'trained-models/calibrator_model_20260605-011742_epoch20.pt',
                    dataset_path='../data/SEOBNRv4-test-100000-fcutoff-uniform-aligned-regen.hdf',
                    batch_size=128, dummyrun=False,
                    timestamp=NOW):
    """
    Read parameters from some test dataset, generate the output waveform from the ML model,
    then use the calibrator model to the predict the residuals based on the output ML waveform.
    Finally, add the calibrator predicted residuals to the ML generated waveform to obtain the
    calibrated output waveform. The plot the original waveform the dataset HDF and the calibrated
    output waveform on the same plot. Return the calibrated output waveform that we can use
    for mismatch calculation and comparison with the original waveform using other functions.
    """
    logger.info(f"Testing residual calibrator model with ML waveform model from {wfmodel_modelpath} and config from {wfmodel_configpath}, and calibrator model from {calibrator_modelpath} on dataset {dataset_path}")

    wfmodel_modelpath = wfmodel_modelpath if os.path.exists(wfmodel_modelpath) else f'../{PROJECT_DIR}/{wfmodel_modelpath}'
    calibrator_modelpath = calibrator_modelpath if os.path.exists(calibrator_modelpath) else f'../{PROJECT_DIR}/{calibrator_modelpath}'

    wfmodel = load_flex_model(model_path=wfmodel_modelpath, 
                            configpath=wfmodel_configpath, 
                            device=DEVICE, precision=PRECISION,)
    wfmodel.eval()  # set to eval mode since we are only using it for inference to generate the calibrator inputs
    logger.info(f"Loaded waveform model for calibrator input generation: {wfmodel}")

    # -- Load `labels_mean` and `labels_std` from model config file, since original CVAE model `state_dict` doesn't have them!
    with open(wfmodel_configpath, 'r') as f:
        model_config = json.load(f)
    params_mean = torch.tensor(model_config['labels_mean'], dtype=getattr(torch, PRECISION), device=DEVICE)
    params_std = torch.tensor(model_config['labels_std'], dtype=getattr(torch, PRECISION), device=DEVICE)

    # -- init calibrator model
    calmodel = ResidualCalibrationCNN(
        input_channels=6,  # [ml_amp, ml_freq, param_m1, param_m2, param_s1z, param_s2z]
        output_channels=2,
        hidden_channels=128,
        num_blocks=5,
        kernel_size=7,
        dropout=0.0,
    )
    calmodel.to(device=DEVICE, dtype=getattr(torch, PRECISION))
    calmodel.load_state_dict(torch.load(calibrator_modelpath, map_location=DEVICE))
    calmodel.eval()
    logger.info(f"Loaded calibrator model from {calibrator_modelpath} for testing: {calmodel}")

    test_dataset = CustomDataset(approximant='SEOBNRv4', 
                                 hdf_fname=dataset_path, 
                                precision=PRECISION, 
                                return_phases=True, return_sample_indices=True, return_attributes=True)
    testloader = CustomDataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    calibrator_data_hdf = f'calibrator_test_data_{timestamp}.hdf'

    for i, databatch in enumerate(testloader):
        if dummyrun and i >= 5:  # just test on the first num_samples samples for now
            break
        originals, _, labels, keys, strains, phases, indices, attr = databatch
        originals.to(DEVICE)
        labels.to(DEVICE)
        assert type(indices) == torch.Tensor and indices.shape == (batch_size,), f"Expected indices to be a tensor of shape ({batch_size},), but got {type(indices)} with shape {indices.shape}"

        calibrator_input, _ = get_calibrator_input(
            wfmodel=wfmodel,
            originals=originals.to(DEVICE),  # shape: (1, 2, n)
            labels=labels.to(DEVICE),  # shape: (1, num_params)
            data_hdf=calibrator_data_hdf,  # save this data group to the calibrator HDF file!
            indices=indices,  # use the original sample index from the dataset!
            params_mean=params_mean,
            params_std=params_std,
        )
        orig_amp, orig_freq = originals[:, 0, :], originals[:, 1, :]
        ml_amp, ml_freq = calibrator_input[:, 0, :], calibrator_input[:, 1, :]

        with torch.no_grad():
            out = calmodel(calibrator_input)  # shape: (1, 2, n)
            pred_amp_residual, pred_freq_residual = out[:, 0, :], out[:, 1, :]
            calibrated_amp = calibrator_input[:, 0, :] + pred_amp_residual
            calibrated_freq = calibrator_input[:, 1, :] + pred_freq_residual

        if i == 0:  # just plot the first batch for now, which is of shape (batch_size, 2, n)
            plot_calibration_results(
                original=originals,
                calibrated=torch.stack([calibrated_amp, calibrated_freq], dim=1),  # shape: (batch, 2, n)
                target_residual=torch.stack([pred_amp_residual, pred_freq_residual], dim=1),
                output_residual=torch.stack([pred_amp_residual, pred_freq_residual], dim=1),
                title=f'$m_1 = {labels[0, 0].item():.2f}, m_2 = {labels[0, 1].item():.2f}, s_1^z = {labels[0, 2].item():.2f}, s_2^z = {labels[0, 3].item():.2f}$',
                savename=f'calibrator_test_results_wf{int(indices[0].item())}_{NOW}.png'
            )

        plot_polarization_mismatch(
            original=originals,
            reconst=torch.stack([calibrated_amp, calibrated_freq], dim=1),  # shape: (batch, 2, n)
            labels=labels,
            keys=keys,
            phases=phases,
            strains=strains,
            attr=attr
        )


def calc_time_array(n, sample_rate=SAMPLE_RATE):
    """
    Calculate the time array for a given number of samples and sample rate.
    """
    return torch.linspace(0, n / sample_rate, steps=n)


def plot_calibration_results(original: torch.Tensor, 
                             calibrated: torch.Tensor, 
                             target_residual: torch.Tensor, 
                             output_residual: torch.Tensor,
                             title: str = '', savename: str = None,
                             fontsize=12, labelsize=10):
    """
    Plots the calibration results in 4 subplots, one each for original and calibrated 
    amplitude and frequency, and then two for the target and output residuals.
    The residual subplot height is scaled for better visualization.

    It is assumed that a batch of data is passed! The first data in the batch is plotted.
    """
    orig_amp, orig_freq = original[0, 0, :], original[0, 1, :]
    calibrated_amp, calibrated_freq = calibrated[0, 0, :], calibrated[0, 1, :]
    target_amp_residual, target_freq_residual = target_residual[0, 0, :], target_residual[0, 1, :]
    output_amp_residual, output_freq_residual = output_residual[0, 0, :], output_residual[0, 1, :]
    time_arr = calc_time_array(orig_amp.shape[-1])
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True,
                             height_ratios=[1, 1, 0.5, 0.5, 0.5])
    # adjust the height of the residual subplot for better visualization
    axes[0].plot(time_arr.cpu(), orig_amp.cpu(), label='Orig Amp', color='blue')
    axes[0].plot(time_arr.cpu(), calibrated_amp.cpu(), label='Cal Amp', color='orange')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[1].plot(time_arr.cpu(), orig_freq.cpu(), label='Orig Freq', color='blue')
    axes[1].plot(time_arr.cpu(), calibrated_freq.cpu(), label='Cal Freq', color='orange')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].legend()
    axes[2].plot(time_arr.cpu(), target_amp_residual.cpu(), label='Target Amp Residual', color='blue')
    axes[2].plot(time_arr.cpu(), output_amp_residual.cpu(), label='Output Amp Residual', color='orange')
    axes[2].plot(time_arr.cpu(), target_freq_residual.cpu(), label='Target Freq Residual', color='green')
    axes[2].plot(time_arr.cpu(), output_freq_residual.cpu(), label='Output Freq Residual', color='red')
    axes[2].set_ylabel('Residuals')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[3].plot(time_arr.cpu(), target_amp_residual.cpu(), label='Target Amp Residual', color='blue')
    axes[3].plot(time_arr.cpu(), output_amp_residual.cpu(), label='Output Amp Residual', color='orange')
    axes[3].set_ylabel('Amplitude Residuals')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()
    axes[4].plot(time_arr.cpu(), target_freq_residual.cpu(), label='Target Freq Residual', color='green')
    axes[4].plot(time_arr.cpu(), output_freq_residual.cpu(), label='Output Freq Residual', color='red')
    axes[4].set_ylabel('Frequency Residuals')
    axes[4].set_xlabel('Time (s)')
    axes[4].legend()
    fig.suptitle(title, fontsize=fontsize)
    putils.beautifyPlot(axes, labelsize=labelsize, tickDirection='in', minor=True)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved calibration results plot to {savename}")
    plt.show()
    plt.close()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the residual calibrator model for waveform generation.")

    parser.add_argument('--timestamp', type=str, default=NOW, 
                        help='Timestamp string to use for saving outputs, default is current date and time.')

    trainparser = parser.add_argument_group('Training Hyperparameters')
    trainparser.add_argument('--batch-size', type=int, default=64, 
                        help='Batch size for training the calibrator model.')
    trainparser.add_argument('--num-epochs', type=int, default=25, 
                        help='Number of epochs to train the calibrator model.')
    trainparser.add_argument('--dummy-run', action='store_true', 
                        help='If set, runs a quick dummy training loop for testing purposes.')
    
    testparser = parser.add_argument_group('Testing Hyperparameters')

    parser = init_verbosity_args(parser)
    args = parser.parse_args()
    logger = init_logging(args)

    logger.info(f"Using device: {DEVICE}, with precision: {PRECISION}")

    # train_calibrator(
    #     batch_size=args.batch_size,
    #     num_epochs=args.num_epochs,
    #     dummyrun=args.dummy_run,
    #     timestamp=args.timestamp,
    # )

    test_calibrator(batch_size=args.batch_size, dummyrun=args.dummy_run)