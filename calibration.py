"""
Model and utilities to calibrate a trained waveform generator model to the target waveform.
We will use the training data to calibrate the generated outputs, by predicting the residuals
between the generated [amp,freq] and the target [amp,freq], for example.
"""

import os
import h5py
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

from utils.generic import init_logging, init_verbosity_args

parser = init_verbosity_args()
args = parser.parse_args()
logger = init_logging(args)

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
logger.info(f"Using device: {DEVICE}, with precision: {PRECISION}")




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
    


def get_calibrator_input(wfmodel, originals, labels, save_data_to_disk=False):
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

    calibrator_input = torch.cat([calibrator_input, param_m1.unsqueeze(1), param_m2.unsqueeze(1),
                                param_s1z.unsqueeze(1), param_s2z.unsqueeze(1)], dim=1)  # shape: (batch, 6, n)
    logger.debug(f"Calibrator input shape: {calibrator_input.shape}")
    # TODO: Can save data separately later!
    # if save_data_to_disk:
    #     # Save calibaration input, target residuals, and parameters the first time, so that we can reuse them next time!
    #     # Save this data to a HDF file repeatedly appending to it every time we call this function, 
    #     # and then we can load this data directly in the calibrator training loop, instead of having to 
    #     # generate it on the fly every time, which is computationally expensive since it requires running the ML model inference every time.
    #     outpath = f'../data/calibrator_input_data_{NOW}.hdf'
    #     with h5py.File(outpath, 'a') as f:
    #         # -- create a new group for each data waveform in the batch, with datasets for:
    #         # -- [ml_amp, ml_freq, target_amp_residual, target_freq_residual, param_m1, param_m2, param_s1z, param_s2z]
    #         for i in range(calibrator_input.shape[0]):
    #             group_name = f"data_{i}"
    #             if group_name in f:
    #                 del f[group_name]  # delete existing group if it exists, to avoid appending to old data
    #             grp = f.create_group(group_name)
    #             grp.create_dataset('ml_amp', data=calibrator_input[i, 0, :].cpu().numpy())
    #             grp.create_dataset('ml_freq', data=calibrator_input[i, 1, :].cpu().numpy())
    #             grp.create_dataset('target_amp_residual', data=target_amp_residual[i].cpu().numpy())
    #             grp.create_dataset('target_freq_residual', data=target_freq_residual[i].cpu().numpy())
    #             grp.create_dataset('param_m1', data=param_m1[i, 0].cpu().numpy())
    #             grp.create_dataset('param_m2', data=param_m2[i, 0].cpu().numpy())
    #             grp.create_dataset('param_s1z', data=param_s1z[i, 0].cpu().numpy())
    #             grp.create_dataset('param_s2z', data=param_s2z[i, 0].cpu().numpy())
    #     logger.info(f"Saved calibrator input and target residuals to {outpath}")
    return calibrator_input, (target_amp_residual, target_freq_residual)
            
        
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
    logger.info(f"Calibrator model architecture: {calmodel}")

    train_set = CustomDataset(forwhat='train', approximant=approximant, returnattr=True,
                            convert=False, hdf_fname=trainhdf, 
                            train_device=DEVICE, precision=PRECISION)
    logger.info(f'Reading validation data from {valhdf}.hdf')
    valid_set = CustomDataset(forwhat='valid', approximant=approximant, returnattr=True,
                            convert=False, hdf_fname=valhdf, 
                            train_device=DEVICE, precision=PRECISION)
    
    training_loader = CustomDataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = CustomDataLoader(valid_set, batch_size=batch_size, shuffle=True)
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
    fname = f'../{PROJECT_DIR}/results/{TODAY}'
    os.makedirs(fname, exist_ok=True)
    running_train_loss_file = open(f'{fname}/calibrator_running_train_losses_{NOW}.csv', 'w')
    running_train_loss_file.write(','.join(['epoch', 'train_loss_amp', 'train_loss_freq']) + '\n')
    running_val_loss_file = open(f'{fname}/calibrator_running_val_losses_{NOW}.csv', 'w')
    running_val_loss_file.write(','.join(['epoch', 'val_loss_amp', 'val_loss_freq']) + '\n')

    if dummyrun:
        logger.info("Running in dummy mode for quick testing...")
        num_epochs = 1
        training_loader = CustomDataLoader(train_set, batch_size=16, shuffle=True)
        validation_loader = CustomDataLoader(valid_set, batch_size=16, shuffle=True)

    for epoch in tqdm(range(num_epochs), desc='Epoch'):
        calmodel.train(True)

        train_loss_amp = 0.0
        train_loss_freq = 0.0
        counter = 0
        for originals, _, labels, keys, _, attr in tqdm(training_loader, total=len(training_loader), desc='Steps/Batchs'):
            counter += 1
            if dummyrun and counter > 2:
                break

            originals = originals.to(DEVICE)
            labels = labels.to(DEVICE)

            calibrator_input, calibrator_target = get_calibrator_input(wfmodel, originals, labels)
            target_amp_residual, target_freq_residual = calibrator_target

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

        # validation loop
        calmodel.eval()
        with torch.no_grad():
            val_loss_amp = 0.0
            val_loss_freq = 0.0
            counter = 0
            for originals, _, labels, keys, _, attr in tqdm(validation_loader, total=len(validation_loader), desc='Validation Steps'):
                counter += 1
                if dummyrun and counter > 2:
                    break

                originals = originals.to(DEVICE)
                labels = labels.to(DEVICE)

                calibrator_input, calibrator_target = get_calibrator_input(wfmodel, originals, labels)
                target_amp_residual, target_freq_residual = calibrator_target

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

        # -- every 10 epochs, save the model checkpoint
        if (epoch + 1) % 10 == 0:
            outpath = f'../{PROJECT_DIR}/trained-models/calibrator_model_{NOW}_epoch_{epoch+1}.pt'
            torch.save(calmodel.state_dict(), outpath)

    # -- save the trained calibrator model
    outpath = f'../{PROJECT_DIR}/trained-models/calibrator_model_{NOW}.pt'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    torch.save(calmodel.state_dict(), outpath)
    # -- save the epoch losses to a CSV file
    epoch_losses.to_csv(f'{fname}/calibrator_epoch_losses_{NOW}.csv', index=False)
    # -- close the running loss files
    running_train_loss_file.close()
    running_val_loss_file.close()
    print("Training complete!")





if __name__ == "__main__":
    train_calibrator()