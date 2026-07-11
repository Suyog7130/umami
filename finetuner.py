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

import time
from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Dict, Tuple, List

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import math
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

from wfconditioner import get_conditioned_waveform
from plotutils import putils

from utils.io import ensure_dirs_and_files, ensure_dir
from utils.gwutils import (
    calculate_cosine_distance,
    polarizations_from_ampfreq,
    polarizations_from_amp_phase,
    amp_phase_from_polarizations,
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
DURATION = 8.0  # seconds


if torch.cuda.is_available():
    n_cuda = torch.cuda.device_count()
    if n_cuda > 1:
        free_memories = []
        for idx in range(n_cuda):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(idx)
            except TypeError:
                with torch.cuda.device(idx):
                    free_bytes, _ = torch.cuda.mem_get_info()
            free_memories.append(free_bytes)
        best_cuda_idx = int(np.argmax(free_memories))
        torch.cuda.set_device(best_cuda_idx)
        DEVICE = torch.device(f"cuda:{best_cuda_idx}")
    else:
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
                        inputnames=['ml_hp', 'ml_hc'], 
                        targetnames=['target_hp_residual', 'target_hc_residual'],
                        perform_conditioning=True
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

    if perform_conditioning:
        amp_mlcal, phase_mlcal = calmodel.calibrate_waveform(outwaves, labels, convert_to_hphc=False)
        hp_mlcond, hc_mlcond = get_conditioned_waveform(amp_mlcal, phase_mlcal)
        amp_orig, phase_orig = amp_phase_from_polarizations(orig_hp, orig_hc, use_pycbc=True)
        hp_origcond, hc_origcond = get_conditioned_waveform(amp_orig, phase_orig)
        hp_ml_final, hc_ml_final = hp_mlcond, hc_mlcond
        hp_orig_final, hc_orig_final = hp_origcond, hc_origcond
    else:
        ml_hp, ml_hc = calmodel.calibrate_waveform(outwaves, labels, convert_to_hphc=True)
        hp_ml_final, hc_ml_final = ml_hp, ml_hc
        hp_orig_final, hc_orig_final = orig_hp, orig_hc

    assert hp_ml_final.shape == hp_orig_final.shape, f"Shape mismatch: hp_ml_final {hp_ml_final.shape} vs hp_orig_final {hp_orig_final.shape}"
    assert hc_ml_final.shape == hc_orig_final.shape, f"Shape mismatch: hc_ml_final {hc_ml_final.shape} vs hc_orig_final {hc_orig_final.shape}"

    finetuner_input = torch.stack([hp_ml_final, hc_ml_final], dim=1)  # shape (batch_size, 2, num_samples)

    # -- Compute residual and normalize them!
    target_hp_residual = hp_orig_final - hp_ml_final
    target_hc_residual = hc_orig_final - hc_ml_final

    logger.debug(f"Computed target residuals with shapes: {target_hp_residual.shape}, {target_hc_residual.shape}")

    finetuner_target = torch.stack([target_hp_residual, target_hc_residual], dim=1)  # shape (batch_size, 2, num_samples)
    logger.debug(f"Stacked finetuner input shape: {finetuner_input.shape}, target shape: {finetuner_target.shape}")

    param_m1, param_m2, param_s1z, param_s2z = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]

    # -- repeat the parameters across the time dimension to match the shape of ml_amp/ml_freq
    param_m1 = param_m1.unsqueeze(-1).expand(-1, hp_ml_final.shape[-1])
    param_m2 = param_m2.unsqueeze(-1).expand(-1, hp_ml_final.shape[-1])
    param_s1z = param_s1z.unsqueeze(-1).expand(-1, hp_ml_final.shape[-1])
    param_s2z = param_s2z.unsqueeze(-1).expand(-1, hp_ml_final.shape[-1])

    fig, ax = plt.subplots(4, 1, figsize=(12, 12))
    ax[0].plot(hp_ml_final[0].cpu().numpy(), label=inputnames[0])
    ax[0].plot(hp_orig_final[0].cpu().numpy(), label='original_hp')
    ax[0].set_title('ML Generated HP vs Original HP')
    ax[0].legend()
    ax[1].plot(target_hp_residual[0].cpu().numpy(), label=targetnames[0])
    ax[1].set_title('Target HP Residual')
    ax[1].legend()
    ax[2].plot(hc_ml_final[0].cpu().numpy(), label=inputnames[1])
    ax[2].plot(hc_orig_final[0].cpu().numpy(), label='original_hc')
    ax[2].set_title('ML Generated HC vs Original HC')
    ax[2].legend()
    ax[3].plot(target_hc_residual[0].cpu().numpy(), label=targetnames[1])
    ax[3].set_title(targetnames[1])
    ax[3].legend()
    putils.beautifyPlot(ax, top=True, right=True)
    plt.tight_layout()
    plt.savefig(f'finetuner_input_example_{NOW}.png', dpi=300, bbox_inches='tight')
    plt.close()
    exit()

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
                                                            batch_size=64,
                                                            num_workers=0, return_indices=True)
    wftestloader = set_waveform_dataloaders(target_type='amp_phase', 
                                            batch_size=64,
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



# ============================================================
# Config
# ============================================================

@dataclass
class FinetunerConfig:
    train_path: str
    valid_path: str
    outdir: str = "stage3_polarization_calibrator_runs"

    input_names: str = "inputs"
    target_names: str = "targets"

    num_epochs: int = 100
    batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = 1.0

    coeffs = {
        "res": 1.0,
        "overlap": 0.05,
        "high": 0.0,
        "smooth": 1e-5,
    }

    use_amp: bool = True
    amp_dtype: str = "float16"

    validate_every: int = 1
    plot_every: int = 5
    checkpoint_every: int = 5
    hard_refresh_every: int = 5

    hard_start_frac: float = 0.80
    hard_top_frac: float = 0.15
    hard_sample_weight: float = 8.0

    w_min: float = 0.05
    gamma: float = 1.0
    topk_frac: float = 0.10
    smooth_kind: str = "curvature"

    eps: float = 1e-12
    num_plot_samples: int = 4

    max_train_samples: Optional[int] = None
    max_valid_samples: Optional[int] = None

    seed: int = 1234
    device: Optional[str] = None



# ============================================================
# Dataset
# ============================================================

class FinetunerDataset(Dataset):
    """
    HDF5 dataset for stage-3 hplus/hcross residual calibration.

    Expected:
        inputs:  (N, 6, n)
        targets: (N, 2, n)

    Input channels:
        0: h_ml_plus
        1: h_ml_cross
        2: m1 repeated over time
        3: m2 repeated over time
        4: chi1 repeated over time
        5: chi2 repeated over time

    Target channels:
        0: residual for plus polarization, `h_true_plus - h_ml_plus`
        1: residual for cross polarization, `h_true_cross - h_ml_cross`

    NOTE: Targets saved in the dataset are not normalized!
    """

    def __init__(
        self,
        hdf_path: str,
        input_names: list = ["ml_hp", "ml_hc"],
        target_names: list = ["target_hp_residual", "target_hc_residual"],
        max_samples: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.hdf_path = hdf_path
        self.input_names = input_names
        self.target_names = target_names
        self.max_samples = max_samples
        self.dtype = dtype

        self._file = None
        self._inputs = None
        self._targets = None

        # -- open HDF file once to read number of groups
        with h5py.File(self.hdf_path, 'r') as f:
            self.num_samples = len(f.keys())
            self.group_names = list(f.keys())  # -- we use these instead of indices!

        if max_samples is not None:
            self.num_samples = min(self.num_samples, max_samples)

    def _init_hdf(self):
        if not os.path.exists(self.hdf_path):
            raise FileNotFoundError(f"HDF file not found at {self.hdf_path}. Please generate the finetuner data first using save_finetuner_data().")
        self._file = h5py.File(self.hdf_path, 'r')
        logger.info(f"Opened HDF file {self.hdf_path} for reading calibrator input and target residuals in the CalibratorDataset.")

    def __len__(self):
        return self.num_samples

    def _read_data(self, idx):
        if self._file is None:
            self._init_hdf()

        group_name = self.group_names[idx]
        group = self._file[group_name]

        x = torch.tensor(
            np.stack([group[self.input_names[0]][:], 
                      group[self.input_names[1]][:],
                      group['param_m1'][:],
                      group['param_m2'][:],
                      group['param_s1z'][:],
                      group['param_s2z'][:]
                      ], axis=0),
            dtype=self.dtype,
        )
        y = torch.tensor(
            np.stack([group[self.target_names[0]][:], 
                      group[self.target_names[1]][:]], axis=0),
            dtype=self.dtype,
        )
        return x, y

    def __getitem__(self, idx):
        x, y = self._read_data(idx)
        return {
            "input": x,
            "target_residual": y,
            "index": torch.tensor(idx, dtype=torch.long),
        }

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._inputs = None
            self._targets = None


# ============================================================
# Custom FinetuneDataLoader
# ============================================================

class FinetunerDataLoader:
    """
    Normal training:
        shuffled uniform batches

    Hard curriculum:
        hard samples get larger sampling weight
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.drop_last = drop_last
        self.hard_indices = None
        self.hard_weight = 1.0

    def set_hard_indices(
        self,
        hard_indices: Optional[Sequence[int]],
        hard_weight: float = 8.0,
    ) -> None:
        if hard_indices is None or len(hard_indices) == 0:
            self.hard_indices = None
            self.hard_weight = 1.0
        else:
            self.hard_indices = np.asarray(hard_indices, dtype=np.int64)
            self.hard_weight = float(hard_weight)

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

    def build(self) -> DataLoader:
        kwargs = dict(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor

        if self.hard_indices is None:
            return DataLoader(
                **kwargs,
                shuffle=True,
            )

        weights = torch.ones(len(self.dataset), dtype=torch.double)
        valid_hard = self.hard_indices[
            (self.hard_indices >= 0) & (self.hard_indices < len(self.dataset))
        ]
        weights[valid_hard] = self.hard_weight

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.dataset),
            replacement=True,
        )

        return DataLoader(
            **kwargs,
            sampler=sampler,
            shuffle=False,
        )


# ============================================================
# Waveform helpers
# ============================================================

def split_stage3_input(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    h_ml = x[:, 0:2, :]
    params = x[:, 2:, :]
    # print("h_ml mean:", h_ml.mean().item())
    # print("h_ml std:", h_ml.std().item())
    return h_ml, params


def compute_peak_amplitude(h_ml: torch.Tensor) -> torch.Tensor:
    amp = torch.sqrt(h_ml[:, 0, :] ** 2 + h_ml[:, 1, :] ** 2)
    amax = amp.amax(dim=-1, keepdim=True).unsqueeze(1)
    return amax


def reconstruct_true_from_normalized_residual(
    h_ml: torch.Tensor,
    target_norm_residual: torch.Tensor,
    norm_factor: float = 1.0,
) -> torch.Tensor:
    return h_ml + target_norm_residual * norm_factor

def reconstruct_true_from_residual(
    h_ml: torch.Tensor,
    target_residual: torch.Tensor,
) -> torch.Tensor:
    return h_ml + target_residual


def apply_predicted_normalized_residual(
    h_ml: torch.Tensor,
    pred_norm_residual: torch.Tensor,
    norm_factor: float = 1.0,
) -> torch.Tensor:
    return h_ml + pred_norm_residual * norm_factor


# ============================================================
# Losses
# ============================================================

def amplitude_weights(
    h_ref: torch.Tensor,
    w_min: float = 0.05,
    gamma: float = 1.0,
) -> torch.Tensor:
    amp = torch.sqrt(h_ref[:, 0, :] ** 2 + h_ref[:, 1, :] ** 2)
    amp_norm = amp / (amp.amax(dim=-1, keepdim=True))
    w = w_min + (1.0 - w_min) * amp_norm.pow(gamma)
    return w.unsqueeze(1)


def weighted_normalized_residual_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    h_ref: torch.Tensor,
    w_min: float = 0.05,
    gamma: float = 1.0,
    eps: float = 1e-12
) -> torch.Tensor:
    w = amplitude_weights(h_ref, w_min=w_min, gamma=gamma)
    loss = w * (pred_norm - target_norm) ** 2
    denom = w.sum() * pred_norm.shape[1]
    return loss.sum() / (denom + eps)


def overlap_and_mismatch(
    h_true: torch.Tensor,
    h_pred: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h_true = h_true.float()
    h_pred = h_pred.float()

    true_c = torch.complex(h_true[:, 0, :], h_true[:, 1, :])
    pred_c = torch.complex(h_pred[:, 0, :], h_pred[:, 1, :])

    inner = torch.real(torch.sum(true_c * torch.conj(pred_c), dim=-1))
    norm_true = torch.sqrt(torch.sum(torch.abs(true_c) ** 2, dim=-1))
    norm_pred = torch.sqrt(torch.sum(torch.abs(pred_c) ** 2, dim=-1))

    overlap = inner / (norm_true * norm_pred)
    mismatch = 1.0 - overlap
    return overlap, mismatch


def overlap_loss(
    h_true: torch.Tensor,
    h_pred: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, mismatch = overlap_and_mismatch(h_true, h_pred)
    return mismatch.mean(), mismatch


def topk_mismatch_loss(
    mismatch: torch.Tensor,
    frac: float = 0.10,
) -> torch.Tensor:
    batch = mismatch.shape[0]
    k = max(1, int(math.ceil(frac * batch)))
    vals = torch.topk(mismatch, k=k, largest=True).values
    return vals.mean()


def smoothness_loss(
    residual_norm: torch.Tensor,
    kind: str = "curvature",
) -> torch.Tensor:
    if residual_norm.shape[-1] < 3:
        return residual_norm.new_tensor(0.0)

    if kind == "slope":
        diff = residual_norm[:, :, 1:] - residual_norm[:, :, :-1]
        return (diff ** 2).mean()

    if kind == "curvature":
        curv = (
            residual_norm[:, :, 2:]
            - 2.0 * residual_norm[:, :, 1:-1]
            + residual_norm[:, :, :-2]
        )
        return (curv ** 2).mean()

    raise ValueError(f"Unknown smoothness kind: {kind}")


# ============================================================
# Schedules
# ============================================================

def get_training_loss_coefficients(epoch: int, cfg: FinetunerConfig) -> Dict[str, float]:
    frac = epoch / max(1, cfg.num_epochs)
    if frac < 0.20:
        return {
            "res": 1.0,
            "overlap": 0.0,
            "high": 0.0,
            "smooth": 1e-5,
        }
    elif frac >= cfg.hard_start_frac:
        return {
            "res": 1.0,
            "overlap": 0.5,
            "high": 0.05,
            "smooth": 1e-5,
        }
    else:
        return {
        "res": 1.0,
        "overlap": 0.05,
        "high": 0.0,
        "smooth": 1e-5,
    }
    # if frac < cfg.hard_start_frac:
    # return {
    #     "res": 0.20,
    #     "overlap": 0.05,
    #     "high": 0.0,
    #     "smooth": 1e-5,
    # }


def set_lr_for_epoch(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: FinetunerConfig,
) -> float:
    frac = epoch / max(1, cfg.num_epochs)

    if frac < 0.20:
        lr = cfg.lr
    elif frac < cfg.hard_start_frac:
        lr = min(cfg.lr, 1e-4)
    else:
        lr = min(cfg.lr, 3e-5)

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


# ============================================================
# Full loss computation
# ============================================================

def compute_finetuner_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    coeffs: Dict[str, float],
    cfg: FinetunerConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    
    x = batch["input"].to(device, non_blocking=True)
    target = batch["target_residual"].to(device, non_blocking=True)

    # # -- print scales of input and target tensors
    # print("x hp mean:", x[:, 0].mean().item())
    # print("x hp std:", x[:, 0].std().item())
    # print("x hc mean:", x[:, 1].mean().item())
    # print("x hc std:", x[:, 1].std().item())
    # print("target mean:", target.mean().item())
    # print("target std:", target.std().item())

    # -- ML waveform output and true waveform will remain unnormalized!
    h_ml, _ = split_stage3_input(x)
    h_true = reconstruct_true_from_residual(h_ml, target)

    # -- Now normalize both input waveforms and target residuals
    norm_factor = compute_peak_amplitude(h_ml)
    normed_target = target / norm_factor

    # -- create a copy and then replace, to avoid modifying the original input tensor
    normed_input = x.clone()
    normed_input[:, 0:2, :] = normed_input[:, 0:2, :] / norm_factor

    pred_norm = model(normed_input)

    h_pred = apply_predicted_normalized_residual(h_ml, pred_norm, norm_factor)

    # # -- print scales of all h tensors
    # print("h_ml mean:", h_ml.mean().item())
    # print("h_ml std:", h_ml.std().item())
    # print("h_true mean:", h_true.mean().item())
    # print("h_true std:", h_true.std().item())
    # print("h_pred mean:", h_pred.mean().item())
    # print("h_pred std:", h_pred.std().item())

    # # -- print scales of all normalized tensors
    # print("normed_target mean:", normed_target.mean().item())
    # print("normed_target std:", normed_target.std().item())
    # print("pred_norm mean:", pred_norm.mean().item())
    # print("pred_norm std:", pred_norm.std().item())

    loss_res = weighted_normalized_residual_loss(
        pred_norm=pred_norm,
        target_norm=normed_target,
        h_ref=h_true,
        w_min=cfg.w_min,
        gamma=cfg.gamma,
        eps=cfg.eps,
    )

    loss_ov, mismatch = overlap_loss(
        h_true=h_true,
        h_pred=h_pred,
    )

    loss_high = topk_mismatch_loss(
        mismatch=mismatch,
        frac=cfg.topk_frac,
    )

    loss_smooth = smoothness_loss(
        residual_norm=pred_norm,
        kind=cfg.smooth_kind,
    )

    total = (
        coeffs["res"] * loss_res
        + coeffs["overlap"] * loss_ov
        + coeffs["high"] * loss_high
        + coeffs["smooth"] * loss_smooth
    )

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_res": float(loss_res.detach().cpu()),
        "loss_overlap": float(loss_ov.detach().cpu()),
        "loss_high": float(loss_high.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "mismatch_mean": float(mismatch.mean().detach().cpu()),
        "mismatch_median": float(mismatch.median().detach().cpu()),
        "mismatch_max": float(mismatch.max().detach().cpu()),
    }

    tensors = {
        "x": x.detach(),
        "h_ml": h_ml.detach(),
        "h_true": h_true.detach(),
        "h_pred": h_pred.detach(),
        "target_norm": normed_target.detach(),
        "pred_norm": pred_norm.detach(),
        "mismatch": mismatch.detach(),
    }

    return total, stats, tensors


# ============================================================
# Evaluation and hard mining
# ============================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    cfg: FinetunerConfig,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    losses = []
    res_losses = []
    ov_losses = []
    high_losses = []
    smooth_losses = []
    mismatches = []

    coeffs = {
        "res": 1.0,
        "overlap": 0.05,
        "high": 0.0,
        "smooth": 1e-5,
    }

    for bidx, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
        if max_batches is not None and bidx >= max_batches:
            break

        loss, stats, tensors = compute_finetuner_loss(
            model=model,
            batch=batch,
            coeffs=coeffs,
            cfg=cfg,
            device=device,
        )

        losses.append(stats["loss_total"])
        res_losses.append(stats["loss_res"])
        ov_losses.append(stats["loss_overlap"])
        high_losses.append(stats["loss_high"])
        smooth_losses.append(stats["loss_smooth"])
        mismatches.append(tensors["mismatch"].detach().cpu())

    mismatch_all = torch.cat(mismatches, dim=0).numpy()

    val_stats_per_epoch = {
        "loss_total": float(np.mean(losses)),
        "loss_res": float(np.mean(res_losses)),
        "loss_overlap": float(np.mean(ov_losses)),
        "loss_high": float(np.mean(high_losses)),
        "loss_smooth": float(np.mean(smooth_losses)),
        "mismatch_mean": float(np.mean(mismatch_all)),
        "mismatch_median": float(np.median(mismatch_all)),
        "mismatch_p90": float(np.quantile(mismatch_all, 0.90)),
        "mismatch_p95": float(np.quantile(mismatch_all, 0.95)),
        "mismatch_p99": float(np.quantile(mismatch_all, 0.99)),
        "mismatch_max": float(np.max(mismatch_all)),
    }
    val_stats_per_batch = {
        "loss_total": losses,
        "loss_res": res_losses,
        "loss_overlap": ov_losses,
        "loss_high": high_losses,
        "loss_smooth": smooth_losses,
        "mismatch": mismatch_all
    }
    return val_stats_per_epoch, val_stats_per_batch


@torch.no_grad()
def mine_hard_samples(
    model: nn.Module,
    dataset: Dataset,
    cfg: FinetunerConfig,
    device: torch.device,
    top_frac: float = 0.15,
) -> np.ndarray:
    kwargs = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        drop_last=False,
    )

    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor

    loader = DataLoader(**kwargs)

    model.eval()

    all_indices = []
    all_mismatch = []

    for batch in tqdm(loader, desc="Mining hard samples", leave=False):
        _, _, tensors = compute_finetuner_loss(
            model=model,
            batch=batch,
            coeffs=cfg.coeffs,
            cfg=cfg,
            device=device,
        )

        all_indices.append(batch["index"].cpu())
        all_mismatch.append(tensors["mismatch"].detach().cpu())

    indices = torch.cat(all_indices, dim=0).numpy()
    mismatch = torch.cat(all_mismatch, dim=0).numpy()

    n_hard = max(1, int(math.ceil(top_frac * len(indices))))
    hard_order = np.argsort(mismatch)[-n_hard:]
    hard_indices = indices[hard_order]

    return hard_indices


# ============================================================
# Plotting
# ============================================================

@torch.no_grad()
def plot_batch_predictions(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: FinetunerConfig,
    device: torch.device,
    outpath: str,
    title: str = "",
    num_samples: int = 4,
) -> None:
    model.eval()

    _, _, tensors = compute_finetuner_loss(
        model=model,
        batch=batch,
        coeffs=cfg.coeffs,
        cfg=cfg,
        device=device,
    )

    h_ml = tensors["h_ml"].cpu()
    h_true = tensors["h_true"].cpu()
    h_pred = tensors["h_pred"].cpu()
    pred_norm = tensors["pred_norm"].cpu()
    target_norm = tensors["target_norm"].cpu()
    mismatch = tensors["mismatch"].cpu()

    b = min(num_samples, h_ml.shape[0])
    t = np.arange(h_ml.shape[-1])

    fig, axes = plt.subplots(
        b,
        4,
        figsize=(22, 4 * b),
        squeeze=False,
    )

    for i in range(b):
        axes[i, 0].plot(t, h_true[i, 0].numpy(), label="true hp", linewidth=1)
        axes[i, 0].plot(t, h_ml[i, 0].numpy(), label="ml hp", linewidth=1)
        axes[i, 0].plot(t, h_pred[i, 0].numpy(), label="corr hp", linewidth=1)
        axes[i, 0].set_title(f"hp, mismatch={mismatch[i].item():.3e}")
        axes[i, 0].legend()

        axes[i, 1].plot(t, h_true[i, 1].numpy(), label="true hc", linewidth=1)
        axes[i, 1].plot(t, h_ml[i, 1].numpy(), label="ml hc", linewidth=1)
        axes[i, 1].plot(t, h_pred[i, 1].numpy(), label="corr hc", linewidth=1)
        axes[i, 1].set_title("hc")
        axes[i, 1].legend()

        axes[i, 2].plot(t, target_norm[i, 0].numpy(), label="target r_norm hp", linewidth=1)
        axes[i, 2].plot(t, pred_norm[i, 0].numpy(), label="pred r_norm hp", linewidth=1)
        axes[i, 2].set_title("normalized hp residual")
        axes[i, 2].legend()

        axes[i, 3].plot(t, target_norm[i, 1].numpy(), label="target r_norm hc", linewidth=1)
        axes[i, 3].plot(t, pred_norm[i, 1].numpy(), label="pred r_norm hc", linewidth=1)
        axes[i, 3].set_title("normalized hc residual")
        axes[i, 3].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_loss_curves(history: List[Dict[str, float]], outpath: str,) -> None:
    if len(history) == 0:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, [h["train_loss_total"] for h in history], label="train total")
    if "valid_loss_total" in history[-1]:
        axes[0, 0].plot(epochs, [h.get("valid_loss_total", np.nan) for h in history], label="valid total")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Total loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [h["train_loss_res"] for h in history], label="res")
    axes[0, 1].plot(epochs, [h["train_loss_overlap"] for h in history], label="overlap")
    axes[0, 1].plot(epochs, [h["train_loss_high"] for h in history], label="high")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Train loss components")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [h["train_mismatch_mean"] for h in history], label="train mean")
    axes[1, 0].plot(epochs, [h["train_mismatch_median"] for h in history], label="train median")
    if "valid_mismatch_mean" in history[-1]:
        axes[1, 0].plot(epochs, [h.get("valid_mismatch_mean", np.nan) for h in history], label="valid mean")
        axes[1, 0].plot(epochs, [h.get("valid_mismatch_median", np.nan) for h in history], label="valid median")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Mismatch mean and median")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, [h["train_mismatch_max"] for h in history], label="train max")
    if "valid_mismatch_p99" in history[-1]:
        axes[1, 1].plot(epochs, [h.get("valid_mismatch_p99", np.nan) for h in history], label="valid p99")
        axes[1, 1].plot(epochs, [h.get("valid_mismatch_max", np.nan) for h in history], label="valid max")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Worst-case mismatch")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_running_losses(history: List[Dict[str, float]], outpath: str,
                        fontsize=15, labelsize=12) -> None:
    """
    Plot running loss curves for train and validation.
    These are saved as np.array in the history dict for each epoch, where all keys are appended
    with the prefix "train_running_" or "valid_running_". The total loss is stored as "train_running_loss_total" and "valid_running_loss_total".
    """
    if len(history) == 0:
        return
        
    total_steps = 0
    run_train_loss = []
    run_valid_loss = []

    for row in history:
        steps_this_epoch = len(row.get("train_running_loss_total", []))
        if steps_this_epoch == 0:
            continue
        total_steps += steps_this_epoch
        run_train_loss.append(row["train_running_loss_total"])
        run_valid_loss.append(row.get("valid_running_loss_total", np.nan))

    run_train_loss = np.concatenate(run_train_loss)
    run_valid_loss = np.concatenate(run_valid_loss)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(len(run_train_loss)), run_train_loss, label="Train loss")
    ax.plot(np.arange(len(run_valid_loss)), run_valid_loss, label="Valid loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Steps", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)
    ax.legend()
    fig.tight_layout()
    putils.beautifyPlot(ax, fontsize=fontsize, labelsize=labelsize)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Saving
# ============================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: FinetunerConfig,
    history: List[Dict[str, float]],
    outpath: str,
    extra: Optional[Dict] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "history": history,
        "extra": extra or {},
    }
    torch.save(payload, outpath)


def save_history_csv(history: List[Dict[str, float]], outpath: str,
                     do_not_write_running_loss_cols: bool = True) -> None:
    if len(history) == 0:
        return
    keys = sorted(set().union(*[h.keys() for h in history]))
    if do_not_write_running_loss_cols:
        keys = [k for k in keys 
                if not k.startswith("train_running_") and not k.startswith("valid_running_")]
    with open(outpath, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in history:
            vals = [str(row.get(k, "")) for k in keys]
            f.write(",".join(vals) + "\n")


# ============================================================
# Main training loop
# ============================================================


def overfit_one_batch_residual_only(model, loader, device="cuda", steps=2000):
    model = model.to(device)
    model.train()

    batch = next(iter(loader))
    x = batch["input"].to(device).float()
    target = batch["target_residual"].to(device).float()
    
    # -- normalize input waveforms by peak amplitude of the ML waveform
    norm_factor = compute_peak_amplitude(x[:, 0:2, :])
    x[:, 0:2, :] = x[:, 0:2, :] / norm_factor

    # -- normalize target by peak amplitude of the ML waveform
    target = target / norm_factor

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    print("x shape:", x.shape)
    print("target shape:", target.shape)
    print("target mean:", target.mean().item())
    print("target std:", target.std().item())
    print("target abs mean:", target.abs().mean().item())
    print("target abs max:", target.abs().max().item())

    with torch.no_grad():
        pred0 = model(x).float()
        zero_loss = (target ** 2).mean()
        init_loss = ((pred0 - target) ** 2).mean()
        print("zero baseline loss:", zero_loss.item())
        print("initial model loss:", init_loss.item())
        print("initial pred abs mean:", pred0.abs().mean().item())
        print("initial pred abs max:", pred0.abs().max().item())

    for step in range(steps):
        pred = model(x).float()
        loss = ((pred - target) ** 2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if step in [0, 1, 2, 10]:
            grad_sum = 0.0
            max_grad = 0.0
            for name, p in model.named_parameters():
                if p.grad is not None:
                    g = p.grad.detach().abs().mean().item()
                    gm = p.grad.detach().abs().max().item()
                    grad_sum += g
                    max_grad = max(max_grad, gm)
                    if "output" in name or "out" in name or "proj" in name:
                        print(f"step {step} | {name} grad mean {g:.3e}, grad max {gm:.3e}")
            print(f"step {step} | grad_sum {grad_sum:.3e}, max_grad {max_grad:.3e}")

        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                pred_abs = pred.abs().mean().item()
                pred_max = pred.abs().max().item()
                corr = torch.mean(pred * target) / (
                    torch.sqrt(torch.mean(pred ** 2) * torch.mean(target ** 2)) + 1e-12
                )

            print(
                f"step {step:04d} | "
                f"loss {loss.item():.6e} | "
                f"pred_abs {pred_abs:.3e} | "
                f"pred_max {pred_max:.3e} | "
                f"corr {corr.item():.3e}"
            )
    return model


def train_finetuner(
    model: nn.Module,
    cfg: FinetunerConfig,
    do_demo_train_run: bool = False,
) -> nn.Module:
    # -- set random number seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = DEVICE if cfg.device is None else torch.device(cfg.device)
    ensure_dir(cfg.outdir)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_id = NOW
    run_dir = os.path.join(cfg.outdir, f"finetuner_run_{run_id}/")
    plot_dir = os.path.join(run_dir, "plots/")
    ckpt_dir = os.path.join(run_dir, "checkpoints/")

    ensure_dir(run_dir)
    ensure_dir(plot_dir)
    ensure_dir(ckpt_dir)

    with open(os.path.join(run_dir, "finetuner_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")

    train_dataset = FinetunerDataset(
        hdf_path=cfg.train_path,
        input_names=cfg.input_names,
        target_names=cfg.target_names,
        max_samples=cfg.max_train_samples,
    )

    valid_dataset = FinetunerDataset(
        hdf_path=cfg.valid_path,
        input_names=cfg.input_names,
        target_names=cfg.target_names,
        max_samples=cfg.max_valid_samples,
    )
    logger.info(f"Train dataset: {len(train_dataset)} samples from {cfg.train_path}")
    logger.info(f"Valid dataset: {len(valid_dataset)} samples from {cfg.valid_path}")

    train_loader_builder = FinetunerDataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        drop_last=True,
    )
    logger.info(f"Built training DataLoader with {len(train_loader_builder)} batches")


    if do_demo_train_run:
        logger.info("Running demo training run on 1 batch...")
        model = overfit_one_batch_residual_only(
            model=model,
            loader=train_loader_builder.build(),
            device=device,
            steps=2000,
        )
        logger.info("Demo training run complete. Exiting.")
        return model


    valid_kwargs = dict(
        dataset=valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        drop_last=False,
    )

    logger.info("Building validation DataLoader...")
    if cfg.num_workers > 0:
        valid_kwargs["prefetch_factor"] = cfg.prefetch_factor
    valid_loader = DataLoader(**valid_kwargs)

    model = model.to(device)
    model.train()
    logger.info(f"Model moved to device {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    use_cuda_amp = cfg.use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda" if use_cuda_amp else "cpu")

    history = []
    best_valid_score = float("inf")
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    fixed_plot_batch = None
    hard_indices = None

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.perf_counter()

        lr = set_lr_for_epoch(optimizer, epoch, cfg)
        coeffs = get_training_loss_coefficients(epoch, cfg)

        hard_phase = (epoch / cfg.num_epochs) >= cfg.hard_start_frac

        if hard_phase:
            should_refresh = (
                hard_indices is None
                or ((epoch - 1) % cfg.hard_refresh_every == 0)
            )

            if should_refresh:
                hard_indices = mine_hard_samples(
                    model=model,
                    dataset=train_dataset,
                    cfg=cfg,
                    device=device,
                    top_frac=cfg.hard_top_frac,
                )
                np.save(
                    os.path.join(run_dir, f"hard_indices_epoch_{epoch:04d}.npy"),
                    hard_indices,
                )

            train_loader_builder.set_hard_indices(
                hard_indices=hard_indices,
                hard_weight=cfg.hard_sample_weight,
            )
        else:
            train_loader_builder.set_hard_indices(None)

        train_loader = train_loader_builder.build()
        model.train()

        accum = {
            "loss_total": [],
            "loss_res": [],
            "loss_overlap": [],
            "loss_high": [],
            "loss_smooth": [],
            "mismatch_mean": [],
            "mismatch_median": [],
            "mismatch_max": [],
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}")

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            if use_cuda_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loss, stats, tensors = compute_finetuner_loss(
                        model=model,
                        batch=batch,
                        coeffs=coeffs,
                        cfg=cfg,
                        device=device,
                    )

                scaler.scale(loss).backward()

                if cfg.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                scaler.step(optimizer)
                scaler.update()

            else:
                loss, stats, tensors = compute_finetuner_loss(
                    model=model,
                    batch=batch,
                    coeffs=coeffs,
                    cfg=cfg,
                    device=device,
                )

                loss.backward()

                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                optimizer.step()

            for k in accum:
                accum[k].append(stats[k])

            pbar.set_postfix({
                "loss": f"{stats['loss_total']:.3e}",
                "mm": f"{stats['mismatch_mean']:.3e}",
                "lr": f"{lr:.1e}",
            })

            if fixed_plot_batch is None:
                fixed_plot_batch = {
                    key: val.detach().cpu() if torch.is_tensor(val) else val
                    for key, val in batch.items()
                }

        train_stats_per_epoch = {
            f"train_{k}": float(np.mean(v))
            for k, v in accum.items()
        }
        # -- Also save per-batch stats for plotting. One train step is one batch!
        train_stats_per_batch = {
            f"train_running_{k}": v
            for k, v in accum.items()
        }

        row = {
            "epoch": epoch,
            "lr": lr,
            "coef_res": coeffs["res"],
            "coef_overlap": coeffs["overlap"],
            "coef_high": coeffs["high"],
            "coef_smooth": coeffs["smooth"],
            "hard_phase": int(hard_phase),
            **train_stats_per_epoch,
            **train_stats_per_batch,
        }

        # -- Validation step

        if epoch % cfg.validate_every == 0:
            val_stats_per_epoch, val_stats_per_batch = evaluate_model(
                model=model,
                loader=valid_loader,
                cfg=cfg,
                device=device,
            )

            row.update({
                f"valid_{k}": v
                for k, v in val_stats_per_epoch.items()
            })
            row.update({
                f"valid_running_{k}": v
                for k, v in val_stats_per_batch.items()
            })

            valid_score = val_stats_per_epoch["mismatch_p99"]

            if valid_score < best_valid_score:
                best_valid_score = valid_score
                logger.info(f"New best validation score found: {best_valid_score:.3e}")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    cfg=cfg,
                    history=history + [row],
                    outpath=best_path,
                    extra={
                        "best_valid_score": best_valid_score,
                        "score_name": "valid_mismatch_p99",
                    },
                )

        epoch_time = time.perf_counter() - epoch_start
        row["epoch_time_sec"] = epoch_time

        history.append(row)

        save_history_csv(
            history,
            os.path.join(run_dir, f"history_{run_id}.csv"),
        )

        plot_loss_curves(
            history,
            os.path.join(plot_dir, f"loss_curves_{run_id}.png"),
        )
        plot_running_losses(
            history,
            os.path.join(plot_dir, f"running_loss_curves_{run_id}.png"),
        )

        if epoch % cfg.plot_every == 0 and fixed_plot_batch is not None:
            plot_batch_predictions(
                model=model,
                batch=fixed_plot_batch,
                cfg=cfg,
                device=device,
                outpath=os.path.join(plot_dir, f"predictions_epoch_{epoch:04d}.png"),
                title=f"Epoch {epoch}",
                num_samples=cfg.num_plot_samples,
            )

        if epoch % cfg.checkpoint_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                history=history,
                outpath=os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pt"),
                extra={
                    "best_valid_score": best_valid_score,
                },
            )

        latest_path = os.path.join(ckpt_dir, f"latest.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            history=history,
            outpath=latest_path,
            extra={
                "best_valid_score": best_valid_score,
            },
        )

        msg = (
            f"Epoch {epoch:04d} done in {epoch_time:.1f}s | "
            f"train mismatch mean={row['train_mismatch_mean']:.3e}, "
            f"train mismatch max={row['train_mismatch_max']:.3e}"
        )

        if "valid_mismatch_p99" in row:
            msg += (
                f" | valid median={row['valid_mismatch_median']:.3e}, "
                f"valid p99={row['valid_mismatch_p99']:.3e}, "
                f"valid max={row['valid_mismatch_max']:.3e}"
            )

        print(msg)

    plot_loss_curves(
        history,
        os.path.join(plot_dir, f"loss_curves_final_{run_id}.png"),
    )
    plot_running_losses(
        history,
        os.path.join(plot_dir, f"running_loss_curves_final_{run_id}.png"),
    )

    final_path = os.path.join(ckpt_dir, f"fineturner_final_model_{run_id}.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=cfg.num_epochs,
        cfg=cfg,
        history=history,
        outpath=final_path,
        extra={
            "best_valid_score": best_valid_score,
        },
    )

    # -- Save just the losses to a separate CSV for easier plotting
    loss_history = [
        {
            "epoch": h["epoch"],
            "train_loss_total": h["train_loss_total"],
            "train_mismatch_mean": h["train_mismatch_mean"],
            "train_mismatch_max": h["train_mismatch_max"],
            "valid_loss_total": h.get("valid_loss_total", np.nan),
            "valid_mismatch_mean": h.get("valid_mismatch_mean", np.nan),
            "valid_mismatch_p99": h.get("valid_mismatch_p99", np.nan),
            "valid_mismatch_max": h.get("valid_mismatch_max", np.nan),
        }
        for h in history
    ]
    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(os.path.join(run_dir, f"loss_history_{run_id}.csv"), index=False)

    # -- Save just the train running losses to a separate h5 file
    running_loss_history = []
    for h in history:
        steps_this_epoch = len(h.get("train_running_loss_total", []))
        if steps_this_epoch == 0:
            continue
        for step in range(steps_this_epoch):
            running_loss_history.append({
                "epoch": h["epoch"],
                "step": step + 1,
                "train_running_loss_total": h["train_running_loss_total"][step],
                "train_running_loss_res": h["train_running_loss_res"][step],
                "train_running_loss_overlap": h["train_running_loss_overlap"][step],
                "train_running_loss_high": h["train_running_loss_high"][step],
                "train_running_loss_smooth": h["train_running_loss_smooth"][step],
                "train_running_mismatch_mean": h["train_running_mismatch_mean"][step],
                "train_running_mismatch_median": h["train_running_mismatch_median"][step],
                "train_running_mismatch_max": h["train_running_mismatch_max"][step],
            })
    running_loss_df = pd.DataFrame(running_loss_history)
    running_loss_df.to_hdf(os.path.join(run_dir, f"train_running_losses_{run_id}.h5"), 
                           key="train_running_losses", index=False)

    # -- Save just the valid running losses to a separate h5 file for easier plotting
    running_valid_loss_history = []
    for h in history:
        steps_this_epoch = len(h.get("valid_running_loss_total", []))
        if steps_this_epoch == 0:
            continue
        for step in range(steps_this_epoch):
            running_valid_loss_history.append({
                "epoch": h["epoch"],
                "step": step + 1,
                "valid_running_loss_total": h["valid_running_loss_total"][step],
                "valid_running_loss_res": h["valid_running_loss_res"][step],
                "valid_running_loss_overlap": h["valid_running_loss_overlap"][step],
                "valid_running_loss_high": h["valid_running_loss_high"][step],
                "valid_running_loss_smooth": h["valid_running_loss_smooth"][step],
                "valid_running_mismatch": h["valid_running_mismatch"][step],
            })
    running_valid_loss_df = pd.DataFrame(running_valid_loss_history)
    running_valid_loss_df.to_hdf(os.path.join(run_dir, f"valid_running_losses_{run_id}.h5"), 
                                key="valid_running_losses", index=False)

    print(f"Training complete. Best model: {best_path}")
    return model




def training_main(args: argparse.Namespace) -> None:

    cfg = FinetunerConfig(
        train_path = "../data/finetuner_data_train_20260703-173318.hdf",
        valid_path = "../data/finetuner_data_valid_20260703-173318.hdf",
        outdir = f"../v0p1/results/{TODAY}/",

        input_names = ['ml_hp', 'ml_hc'],
        target_names = ['target_hp_residual', 'target_hc_residual'],

        validate_every=1,
        plot_every=5,
        checkpoint_every=5,

        w_min=0.05,
        gamma=1.0,
        topk_frac=0.10,

        use_amp=True,
        amp_dtype="float16",
    )

    # -- Override config with command-line arguments
    for key in vars(args):
        if hasattr(cfg, key):
            setattr(cfg, key, getattr(args, key))

    model = ResidualCalibrationCNN(
        input_channels=6,
        output_channels=2,
        hidden_channels=64,
        num_blocks=6,
        kernel_size=7,
        dropout=0.0,
    )

    if args.debug_training:
        cfg.num_epochs = 10
        cfg.batch_size = 32
        cfg.num_workers = 0
        cfg.max_train_samples = 256
        cfg.max_valid_samples = 64
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    elif args.demo_training:
        cfg.num_epochs = 10
        cfg.batch_size = 64
        cfg.num_workers = 0
        cfg.max_train_samples = 512
        cfg.max_valid_samples = 64
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_finetuner(model, cfg, do_demo_train_run=True)
        return

    logger.info(f"Starting fine tuner training with config: {cfg}")
    train_finetuner(model, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune the output [hp,hc] polarizations to match the target [hp,hc], by predicting the residual errors."
        )

    parser.add_argument('--wfmodel_configname', type=str, default='modelconfig-flexcvae-20260619-064140.json',
                        help="Name of the waveform generation model configuration JSON file (default: %(default)s)")
    parser.add_argument('--wfmodel_modelname', type=str, default='flexcvae-model-backup-20260619-064140-epoch98.pt',
                        help="Name of the trained waveform generation model (default: %(default)s)")
    parser.add_argument('--calibrator_modelname', type=str, default='calibrator_model_20260623-010953_epoch74.pt',
                        help="Name of the trained calibration model checkpoint (default: %(default)s)")
    
    parser.add_argument('--timestamp', type=str, default=NOW,
                        help="Timestamp for saving the generated data (default: %(default)s)")

    parser.add_argument('--debug-training', action='store_true',
                        help='Enable debug mode for training, which uses a smaller dataset and fewer epochs for quick testing.')
    parser.add_argument('--demo-training', action='store_true',
                        help='Enable demo mode for training, which uses a moderate dataset and fewer epochs for demonstration purposes.')
    
    trainargs = parser.add_argument_group('Training arguments')
    trainargs.add_argument('-ne', '--num-epochs', type=int, default=100,
                        help='Number of training epochs (default: %(default)s)')
    trainargs.add_argument('-bs', '--batch-size', type=int, default=216,
                        help='Batch size for training (default: %(default)s)')
    trainargs.add_argument('-lr', '--learning-rate', type=float, default=3e-3,
                        help='Starting learning rate for the optimizer (default: %(default)s)')
    trainargs.add_argument('-wd', '--weight-decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization) for the optimizer (default: %(default)s)')
    trainargs.add_argument('-ua', '--use-amp', action='store_true',
                        help='Use automatic mixed precision (AMP) for training (default: %(default)s)')
    trainargs.add_argument('-ad', '--amp-dtype', type=str, 
                        choices=['float16', 'bfloat16'], default='float16',
                        help='Data type for AMP (default: %(default)s)')
    trainargs.add_argument('-hs', '--hard-start-frac', type=float, default=0.80,
                        help='Fraction of training epochs after which to start hard sample mining (default: %(default)s)')
    trainargs.add_argument('-ht', '--hard-top-frac', type=float, default=0.15,
                        help='Fraction of hardest samples to mine during hard sample mining (default: %(default)s)')
    trainargs.add_argument('-hw', '--hard-sample-weight', type=float, default=8.0,
                        help='Weight for hard samples during training (default: %(default)s)')

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
    if args.train:
        logger.info("Starting fine tuner training...")
        training_main(args)