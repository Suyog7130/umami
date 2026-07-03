"""
Finetune the output [hp,hc] polarizations to match the target [hp,hc],
by predicting the residual errors. This makes the third stage of the full model.
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
import matplotlib.ticker as tck

import pycbc

from tqdm import tqdm

from datacvae import CustomDataset, CustomDataLoader
from optimize import (
    load_flex_model, 
    set_waveform_dataloaders, 
    WaveformDataset,
    plot_reconstructions
)

from calibration import (
    ResidualCalibrationCNN, 
    CalibrationModel,
    merger_weighted_mse_loss_func
)

from maincvae import plot_mismatch, plot_polarization_mismatch
from plotutils import putils

from utils.io import ensure_dirs_and_files, ensure_dir
from utils.gwutils import (
    calculate_cosine_distance,
    polarizations_from_ampfreq,
    polarizations_from_amp_phase,
    calc_polarization_mismatch,
    calc_chirp_mass,
    calc_chieff,
    calc_time_array
)
from utils.plotting import plot_twopanel

from plotutils import putils

import logging
from utils.generic import init_logging, init_verbosity_args
logger = logging.getLogger(__name__)


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


# -- get mean and std of labels for normalization
params_fname = '../data/params-SEOBNRv4-train-100000-fcutoff-uniform-aligned-regen-4vals.csv'
params_df = pd.read_csv(params_fname, index_col=0, sep=',')
params_mean = params_df.mean().values
params_std = params_df.std().values
logger.info(f"Labels mean: {params_mean}")
logger.info(f"Labels std: {params_std}")
params_mean = torch.tensor(params_mean, dtype=getattr(torch, PRECISION))
params_std = torch.tensor(params_std, dtype=getattr(torch, PRECISION))


def calc_residual_via_best_match(orig, ml, delta_t=None, use_psd=True):
    """
    DEPRECATED: We will directly use the residuals and use mismatch in the loss function!

    Calculate the residual between the original and ML-generated waveforms
    by finding the best match (minimum mismatch) between them, by time shifting
    and phase shifting the ML waveform to align with the original waveform.

    Arguments
    ---------
    orig : torch.Tensor
        The original waveform of shape (num_samples,).
    ml : torch.Tensor
        The ML-generated waveform of shape (num_samples,).

    Returns
    -------
    residual : torch.Tensor
        The residual waveform of shape (num_samples,).
    """
    dt = delta_t or 1 / SAMPLE_RATE
    orig = orig.detach().cpu().numpy()
    ml = ml.detach().cpu().numpy()
    orig = np.array(orig, dtype=np.float64)
    ml = np.array(ml, dtype=np.float64)
    orig_ts = pycbc.types.TimeSeries(orig, delta_t=dt)
    ml_ts = pycbc.types.TimeSeries(ml, delta_t=dt)
    if use_psd:
        psd = pycbc.psd.aLIGOZeroDetHighPower(len(orig_ts), delta_f=1/(len(orig_ts) * dt), low_freq_cutoff=FMIN)
        orig_fs = orig_ts.to_frequencyseries()
        ml_fs = ml_ts.to_frequencyseries()
        psd.astype(np.float64)
        psd_interp = np.interp(orig_fs.sample_frequencies, psd.sample_frequencies, psd.data)
        psd_resampled = pycbc.types.FrequencySeries(psd_interp, delta_f=orig_ts.delta_f, dtype=psd.dtype)
        m, t = pycbc.filter.match(ml_fs, orig_fs, psd=psd_resampled, low_frequency_cutoff=FMIN,)
    else:
        m, t = pycbc.filter.match(ml_ts, orig_ts, low_frequency_cutoff=FMIN)

    ml_shifted_ts = ml_ts.cyclic_time_shift(t)
    ml_shifted = np.array(ml_shifted_ts, dtype=np.float64)
    residual = orig - ml_shifted
    return residual


def get_finetuner_input(wfmodel, calmodel, originals, labels, hf_file, indices, attr=None,
                        labels_mean=None, labels_std=None,
                        inputnames=['ml_hp', 'ml_hc'], targetnames=['target_hp_residual', 'target_hc_residual'],
    ):
    """
    Generate the fine tuner input and target data for a batch of original waveforms and labels.

    Arguments
    ---------
    wfmodel : nn.Module
        The waveform generation model (FlexC-VAE) used to generate the predicted waveforms.
    calmodel : CalibrationModel
        The calibration model used to generate the residuals.
    originals : torch.Tensor
        The original waveforms (targets) of shape (batch_size, 2, num_samples).
    labels : torch.Tensor
        The labels corresponding to the original waveforms of shape (batch_size, num_labels).
    indices : torch.Tensor
        The indices of the original waveforms in the dataset.
    inputnames : list of str
        The names of the input data to be saved in the HDF file.
    targetnames : list of str
        The names of the target data to be saved in the HDF file.
    correct_length : bool
        Whether to correct the length of the generated waveforms to match the original waveforms.
    savedir : str
        The directory to save the HDF file.
    """
    logger.debug(f"Generating fine tuner input and target data for batch with indices: {indices}")

    orig_hp, orig_hc = originals[:, 0, :], originals[:, 1, :]  # shape (batch_size, num_samples)
    logger.debug(f"Original waveforms shape: {originals.shape}, hp shape: {orig_hp.shape}, hc shape: {orig_hc.shape}")

    outwaves = wfmodel.generate(labels, convert_to_hphc=False)  # has shape (1, 2=[amp,phase], seq_len)!
    logger.debug(f"Generated waveform from ML model with shape: {outwaves.shape}")

    ml_hp, ml_hc = calmodel.calibrate_waveform(outwaves, labels, convert_to_hphc=True)
    logger.debug(f"Calibrated waveform shape: {ml_hp.shape}, {ml_hc.shape}")

    assert ml_hp.shape == orig_hp.shape, f"Shape mismatch: ml_hp {ml_hp.shape} vs orig_hp {orig_hp.shape}"
    assert ml_hc.shape == orig_hc.shape, f"Shape mismatch: ml_hc {ml_hc.shape} vs orig_hc {orig_hc.shape}"

    # # -- compute the residuals between the original and ML-generated waveforms
    # target_hp_residual = torch.tensor([calc_residual_via_best_match(orig_hp[i].cpu(), ml_hp[i].cpu(), 
    #                                    delta_t=attr['delta_t'][i]) 
    #                                    for i in range(orig_hp.shape[0])], dtype=getattr(torch, PRECISION), device=DEVICE)
    # target_hc_residual = torch.tensor([calc_residual_via_best_match(orig_hc[i].cpu(), ml_hc[i].cpu(), 
    #                                    delta_t=attr['delta_t'][i]) 
    #                                    for i in range(orig_hc.shape[0])], dtype=getattr(torch, PRECISION), device=DEVICE)

    # -- Compute residual and normalize them!
    target_hp_residual = orig_hp - ml_hp
    target_hc_residual = orig_hc - ml_hc

    logger.debug(f"Computed target residuals with shapes: {target_hp_residual.shape}, {target_hc_residual.shape}")

    finetuner_input = torch.stack([ml_hp, ml_hc], dim=1)  # shape (batch_size, 2, num_samples)
    finetuner_target = torch.stack([target_hp_residual, target_hc_residual], dim=1)  # shape (batch_size, 2, num_samples)
    logger.debug(f"Stacked finetuner input shape: {finetuner_input.shape}, target shape: {finetuner_target.shape}")

    param_m1, param_m2, param_s1z, param_s2z = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]

    # -- repeat the parameters across the time dimension to match the shape of ml_amp/ml_freq
    param_m1 = param_m1.unsqueeze(-1).expand(-1, ml_hp.shape[-1])
    param_m2 = param_m2.unsqueeze(-1).expand(-1, ml_hp.shape[-1])
    param_s1z = param_s1z.unsqueeze(-1).expand(-1, ml_hp.shape[-1])
    param_s2z = param_s2z.unsqueeze(-1).expand(-1, ml_hp.shape[-1])

    # fig, ax = plt.subplots(4, 1, figsize=(12, 12))
    # ax[0].plot(ml_hp[0].cpu().numpy(), label=inputnames[0])
    # ax[0].plot(orig_hp[0].cpu().numpy(), label='original_hp')
    # ax[0].set_title('ML Generated HP vs Original HP')
    # ax[0].legend()
    # ax[1].plot(target_hp_residual[0].cpu().numpy(), label=targetnames[0])
    # ax[1].set_title('Target HP Residual')
    # ax[1].legend()
    # ax[2].plot(ml_hc[0].cpu().numpy(), label=inputnames[1])
    # ax[2].plot(orig_hc[0].cpu().numpy(), label='original_hc')
    # ax[2].set_title('ML Generated HC vs Original HC')
    # ax[2].legend()
    # ax[3].plot(target_hc_residual[0].cpu().numpy(), label=targetnames[1])
    # ax[3].set_title(targetnames[1])
    # ax[3].legend()
    # putils.beautifyPlot(ax, top=True, right=True)
    # plt.tight_layout()
    # plt.savefig(f'finetuner_input_example_{NOW}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    if labels_mean is None or labels_std is None:
        labels_mean = wfmodel.MODEL_CONFIG['labels_mean']
        labels_std = wfmodel.MODEL_CONFIG['labels_std']
        logger.debug(f"Obtained labels_mean and labels_std from the model config: {labels_mean}, {labels_std}")
        if labels_mean is None or labels_std is None:
            logger.debug("labels_mean and labels_std are not provided and not found in the model config. So, we will use the global labels_mean and labels_std calculated from the training data CSV file.")
            labels_mean = params_mean
            labels_std = params_std
        else:
            logger.debug(f"Using labels_mean and labels_std from the model config: {labels_mean}, {labels_std}")

    param_m1 = (param_m1 - labels_mean[0]) / labels_std[0]
    param_m2 = (param_m2 - labels_mean[1]) / labels_std[1]
    param_s1z = (param_s1z - labels_mean[2]) / labels_std[2]
    param_s2z = (param_s2z - labels_mean[3]) / labels_std[3]
    logger.debug(f"Normalized parameters: param_m1={param_m1}, param_m2={param_m2}, param_s1z={param_s1z}, param_s2z={param_s2z}")

    finetuner_input = torch.cat([finetuner_input, param_m1.unsqueeze(1), param_m2.unsqueeze(1),
                                param_s1z.unsqueeze(1), param_s2z.unsqueeze(1)], dim=1)  # shape: (batch, 6, n)
    logger.debug(f"Finetuner input shape: {finetuner_input.shape}")

    # -- Save the finetuner input and target data to HDF file
    for i in range(len(indices)):
        grp_name = f'sample{int(indices[i])}'
        if grp_name in hf_file:
            logger.warning(f"Group {grp_name} already exists in HDF file. Overwriting...")
            del hf_file[grp_name]
        grp = hf_file.create_group(grp_name)
        grp.create_dataset(inputnames[0], data=finetuner_input[i, 0, :].cpu().numpy())
        grp.create_dataset(inputnames[1], data=finetuner_input[i, 1, :].cpu().numpy())
        grp.create_dataset(targetnames[0], data=finetuner_target[i, 0, :].cpu().numpy())
        grp.create_dataset(targetnames[1], data=finetuner_target[i, 1, :].cpu().numpy())
        grp.create_dataset('param_m1', data=finetuner_input[i, 2, :].cpu().numpy())
        grp.create_dataset('param_m2', data=finetuner_input[i, 3, :].cpu().numpy())
        grp.create_dataset('param_s1z', data=finetuner_input[i, 4, :].cpu().numpy())
        grp.create_dataset('param_s2z', data=finetuner_input[i, 5, :].cpu().numpy())
    return



def save_finetuner_data(wfmodel_modelname, wfmodel_configname, calibrator_modelname,
                        timestamp = NOW):
    """
    Generate and save the fine tuner input and target data to HDF files.
    This is useful for pre-generating the data for faster training later.
    """
    logger.info(f"Loading waveform generation model from {wfmodel_modelname} with config {wfmodel_configname}...")
    savedir = '../data'
    project_dir = os.path.join('../', PROJECT_DIR)
    inputnames = ['ml_hp', 'ml_hc']
    targetnames = ['target_hp_residual', 'target_hc_residual']

    model_path = os.path.join(project_dir, 'trained-models', wfmodel_modelname)
    config_path = os.path.join(project_dir, 'trained-models', wfmodel_configname)
    calmodel_path = os.path.join(project_dir, 'trained-models', calibrator_modelname)

    logger.info(f"Searching for waveform generation model at: {model_path}")
    if not os.path.isfile(model_path):
        model_path = os.path.join('../', 'trained-models', args.wfmodel_modelname)
        if not os.path.isfile(model_path):
            logger.error(f"Provided MODEL_PATH does not exist: {model_path}")
            raise FileNotFoundError(f"MODEL_PATH file not found at {model_path}")
    logger.info(f"Using MODEL_PATH: {model_path}")
    logger.info(f"Searching for waveform generation model config at: {config_path}")
    if not os.path.isfile(config_path):
        config_path = os.path.join('../', 'trained-models', args.wfmodel_configname)
        if not os.path.isfile(config_path):
            logger.error(f"Provided MODEL_CONFIG_PATH does not exist: {config_path}")
            raise FileNotFoundError(f"MODEL_CONFIG_PATH file not found at {config_path}")
    logger.info(f"Using MODEL_CONFIG_PATH: {config_path}")
    logger.info(f"Searching for calibration model at: {calmodel_path}")
    if not os.path.isfile(calmodel_path):
        calmodel_path = os.path.join('../', 'trained-models', args.calibrator_modelname)
        if not os.path.isfile(calmodel_path):
            logger.error(f"Provided CALMODEL_PATH does not exist: {calmodel_path}")
            raise FileNotFoundError(f"CALMODEL_PATH file not found at {calmodel_path}")
    logger.info(f"Using CALMODEL_PATH: {calmodel_path}")


    ml_wfmodel = load_flex_model(model_path=model_path, 
                                configpath=config_path, 
                                device=None, precision=None)
    ml_calmodel = CalibrationModel(calibrator_modelpath=calmodel_path,
                                   device=None, precision=None)
    logger.info(f"Loaded waveform generation model and calibration model successfully.")
    
    # -- Read waveform generation model input data for labels and original waveforms
    wftrainloader, wfvalidloader = set_waveform_dataloaders(target_type='amp_phase', 
                                                            num_workers=0, return_indices=True)
    wftestloader = set_waveform_dataloaders(target_type='amp_phase', 
                                            return_test_loader=True, 
                                            num_workers=0, return_indices=True)
    logger.info(f"Loaded waveform generation model input data for train, valid, and test sets successfully.")
    dataloaders = [wftrainloader, wfvalidloader, wftestloader]
    datasets = ['train', 'valid', 'test']

    for i in range(len(dataloaders)):
        logger.info(f"Generating and saving fine tuner input and target data for {datasets[i]} set...")
        ensure_dir(savedir)
        savename = f'finetuner_data_{datasets[i]}_{timestamp}.hdf'
        hf_file = h5py.File(os.path.join(savedir, savename), 'a')
        for batch in tqdm(dataloaders[i], desc="batches"):
            input, target, labels, keys, strains, indices, attr = batch
            get_finetuner_input(
                wfmodel=ml_wfmodel,
                calmodel=ml_calmodel,
                originals=strains,   # targets are original [hp,hc] strains waveforms
                labels=labels,
                indices=indices,
                hf_file=hf_file,
                attr=attr,
                inputnames=inputnames,
                targetnames=targetnames,
            )
        hf_file.close()
    logger.info(f"Finished generating and saving finetuner input and target data for {datasets[i]} set to HDF file: {savename}")
    
    # -- Save finetuner data config to JSON file
    config = {
        'wfmodel_modelname': wfmodel_modelname,
        'wfmodel_configname': wfmodel_configname,
        'calibrator_modelname': calibrator_modelname,
        'inputnames': inputnames,
        'targetnames': targetnames,
        'timestamp': timestamp
    }
    config_fname = f'finetuner_data_{timestamp}_config.json'
    with open(os.path.join(savedir, config_fname), 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Finished generating and saving finetuner input and target data to HDF files for all sets (train, valid, test).")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune the output [hp,hc] polarizations to match the target [hp,hc], by predicting the residual errors.")

    parser.add_argument('--wfmodel_configname', type=str, default='modelconfig-flexcvae-20260619-064140.json',
                        help="Name of the waveform generation model configuration JSON file (default: %(default)s)")
    parser.add_argument('--wfmodel_modelname', type=str, default='flexcvae-model-backup-20260619-064140-epoch98.pt',
                        help="Name of the trained waveform generation model (default: %(default)s)")
    parser.add_argument('--calibrator_modelname', type=str, default='calibrator_model_20260623-010953_epoch74.pt',
                        help="Name of the trained calibration model checkpoint (default: %(default)s)")
    
    parser.add_argument('--timestamp', type=str, default=NOW,
                        help="Timestamp for saving the generated data (default: %(default)s)")


    methodargs = parser.add_mutually_exclusive_group(required=True)
    methodargs.add_argument('--save-data', action='store_true',
                        help='Generate and save the fine tuner input and target data to HDF files, without training the model. This is useful for pre-generating the data for faster training later.')
    methodargs.add_argument('--train', action='store_true',
                        help='Train the fine tuner model using the training dataset.')
    methodargs.add_argument('--test', action='store_true',
                        help='Test the fine tuner model using the test dataset.')
    methodargs.add_argument('--plot-results', type=str, default=None,
                        help='Plot the calibrated mismatch histogram from the given HDF file path.')
    
    
    parser = init_verbosity_args(parser)
    args = parser.parse_args()
    init_logging(args)

    logger.info(f"Using device: {DEVICE}, with precision: {PRECISION}")

    if args.save_data:
        logger.info("Generating and saving fine tuner input and target data to HDF files...")
        save_finetuner_data(
            wfmodel_modelname=args.wfmodel_modelname,
            wfmodel_configname=args.wfmodel_configname,
            calibrator_modelname=args.calibrator_modelname,
            timestamp=args.timestamp,
        )