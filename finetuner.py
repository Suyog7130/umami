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

from tqdm import tqdm

from datacvae import CustomDataset, CustomDataLoader
from optimize import (
    load_flex_model, 
    set_waveform_dataloaders, 
    WaveformDataset,
    plot_reconstructions
)

from calibration import (
    ResidualCalibratorCNN, 
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



def save_finetuner_data(wfmodel_modelpath, wfmodel_configpath, calibrator_modelpath,
                        timestamp = NOW):
    """
    Generate and save the fine tuner input and target data to HDF files.
    This is useful for pre-generating the data for faster training later.
    """
    logger.info(f"Loading waveform generation model from {wfmodel_modelpath} with config {wfmodel_configpath}...")
    savedir = '../data'
    inputnames = ['ml_hp', 'ml_hc']
    targetnames = ['target_hp_residual', 'target_hc_residual']

    ml_wfmodel = load_flex_model(model_path=wfmodel_modelpath, 
                                configpath=wfmodel_configpath, 
                                device=None, precision=None)
    ml_calmodel = CalibrationModel(calibrator_modelpath=calibrator_modelpath,
                                   device=None, precision=None)
    
    # -- Read waveform generation model input data for labels and original waveforms
    wftrainloader, wfvalidloader = set_waveform_dataloaders(target_type='amp_phase', 
                                                            num_workers=0, return_indices=True)
    wftestloader = set_waveform_dataloaders(target_type='amp_phase', 
                                            return_test_loader=True, 
                                            num_workers=0, return_indices=True)
    dataloaders = [wftrainloader, wfvalidloader, wftestloader]
    savenames = ['train', 'valid', 'test']

    for i in range(len(dataloaders)):
        savename = f'finetuner_data_{savenames[i]}_{timestamp}.hdf'
        for batch in tqdm(dataloaders[i], desc="batches"):
            input, target, labels, keys, strains, indices, attr = batch
            get_finetuner_input(
                wfmodel=ml_wfmodel,
                calmodel=ml_calmodel,
                originals=strains,   # targets are original [hp,hc] strains waveforms
                labels=labels,
                data_hdf=savename,
                indices=indices,
                inputnames=inputnames,
                targetnames=targetnames,
                correct_length=False,
                savedir=savedir
            )
        logger.info(f"Finished generating and saving calibrator input and target data for {savenames[i]} set to HDF file: {savename}")

    # -- Save calibration data config to JSON file
    config = {
        'wfmodel_modelpath': wfmodel_modelpath,
        'wfmodel_configpath': wfmodel_configpath,
        'calibrator_modelpath': calibrator_modelpath,
        'inputnames': inputnames,
        'targetnames': targetnames,
        'timestamp': timestamp
    }
    config_fname = f'finetuner_data_{timestamp}_config.json'
    with open(savedir + config_fname, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Finished generating and saving calibrator input and target data to HDF files for all sets (train, valid, test).")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune the output [hp,hc] polarizations to match the target [hp,hc], by predicting the residual errors.")

    parser.add_argument('--model-config', type=str, default='modelconfig-flexcvae-20260619-064140.json',
                        help="Name of the waveform generation model configuration JSON file (default: %(default)s)")
    parser.add_argument('--model-name', type=str, default='flexcvae-model-backup-20260619-064140-epoch98.pt',
                        help="Name of the trained waveform generation model (default: %(default)s)")
    parser.add_argument('--calmodel-name', type=str, default='calibrator_model_20260623-010953_epoch74.pt',
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
            wfmodel_modelpath=args.wfmodel_modelpath,
            wfmodel_configpath=args.wfmodel_configpath,
            calibrator_modelpath=args.calmodel_name,
            timestamp=args.timestamp,
        )