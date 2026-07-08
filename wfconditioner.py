
import os
import gc
import json
import logging
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
import datetime

import bilby
# from bilby.core.utils import logger
from bilby.core import utils
from bilby.gw import WaveformGenerator

import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp

from flexcvae import FlexTwoC2E1D, FlexCAE, FlexCAEPhase
from optimize import load_flex_model
from calibration import CalibrationModel
from cvae import CVAE

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
    PRECISION = 'float32'  # Use float32 for CUDA if available
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    PRECISION = 'float32'  # Use float32 for MPS since it does not support float64 well
else:
    DEVICE = torch.device("cpu")
    PRECISION = 'float32'  # Use float32 for CPU
logger.info(f"Using device: {DEVICE}, with precision: {PRECISION}")


CACHED_MLMODEL = {'ml_wfmodel': None, 'ml_calmodel': None}


def set_cached_mlmodel(ml_wfmodel, ml_calmodel):
    global CACHED_MLMODEL
    CACHED_MLMODEL = {'ml_wfmodel': ml_wfmodel, 'ml_calmodel': ml_calmodel}
    logger.info("ML model cached successfully for future use in waveform generation.")

def get_cached_mlmodel():
    return (CACHED_MLMODEL['ml_wfmodel'], CACHED_MLMODEL['ml_calmodel'])



def as_1d_float_array(x, name):
    x = np.asarray(x, dtype=np.float64).squeeze()
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    return x


def resample_to_length(x, n_target):
    x = as_1d_float_array(x, "x")
    if len(x) == n_target:
        return x.copy()

    old_grid = np.linspace(0.0, 1.0, len(x), endpoint=False)
    new_grid = np.linspace(0.0, 1.0, n_target, endpoint=False)
    return np.interp(new_grid, old_grid, x)


def sin2_rise_taper(n_total, n_taper):
    w = np.ones(n_total, dtype=np.float64)

    n_taper = int(max(0, min(n_taper, n_total)))
    if n_taper <= 1:
        return w

    theta = np.linspace(0.0, np.pi / 2.0, n_taper)
    w[:n_taper] = np.sin(theta) ** 2
    return w


def cos2_fall_taper(n_total, n_taper):
    w = np.ones(n_total, dtype=np.float64)

    n_taper = int(max(0, min(n_taper, n_total)))
    if n_taper <= 1:
        return w

    theta = np.linspace(0.0, np.pi / 2.0, n_taper)
    w[-n_taper:] = np.cos(theta) ** 2
    return w


def find_one_cycle_taper_length_from_phase(phase, sampling_frequency, min_taper_seconds=0.005, max_taper_seconds=0.20):
    """
    Find the first sample where the unwrapped phase has changed by 2*pi.

    This gives the start taper length. If the phase is pathological, use a safe fallback.
    """
    phase = as_1d_float_array(phase, "phase")
    unwrapped = np.unwrap(phase)

    dphi = unwrapped - unwrapped[0]

    if np.nanmax(np.abs(dphi)) < 2.0 * np.pi:
        fallback = int(round(0.02 * sampling_frequency))
        return max(2, min(fallback, len(phase)))

    progress = np.maximum.accumulate(np.abs(dphi))
    idx = int(np.argmax(progress >= 2.0 * np.pi))

    n_min = int(round(min_taper_seconds * sampling_frequency))
    n_max = int(round(max_taper_seconds * sampling_frequency))

    idx = max(idx, n_min)
    idx = min(idx, n_max)
    idx = min(idx, len(phase))

    return max(2, idx)


def reconstruct_hp_hc_from_amp_phase(amplitude, phase, cross_sign=1.0, phase_offset=0.0):
    """
    Default convention:
        h_plus  = A cos(phi + phase_offset)
        h_cross = cross_sign A sin(phi + phase_offset)

    If your training convention uses the opposite cross sign, set cross_sign=-1.
    """
    amplitude = as_1d_float_array(amplitude, "amplitude")
    phase = as_1d_float_array(phase, "phase")

    if len(amplitude) != len(phase):
        raise ValueError(f"amplitude and phase lengths differ: {len(amplitude)} vs {len(phase)}")

    phi = np.unwrap(phase) + phase_offset

    h_plus = amplitude * np.cos(phi)
    h_cross = cross_sign * amplitude * np.sin(phi)

    return h_plus, h_cross


def shift_with_zero_padding(x, shift_samples):
    """
    Shift array by shift_samples without wraparound.

    shift_samples > 0 shifts waveform to the right.
    shift_samples < 0 shifts waveform to the left.
    """
    x = as_1d_float_array(x, "x")
    y = np.zeros_like(x)

    n = len(x)

    if shift_samples == 0:
        return x.copy()

    if shift_samples > 0:
        if shift_samples < n:
            y[shift_samples:] = x[:n - shift_samples]
    else:
        s = -shift_samples
        if s < n:
            y[:n - s] = x[s:]
    return y


def shift_merger_to_fraction(h_plus, h_cross, amplitude_for_merger=None, merger_fraction=0.8,
                             extend_length=True):
    """
    Shift waveform so that the maximum amplitude occurs at merger_fraction of the 1-second waveform.
    If `extend_length` is True, the waveform will be extended with zeros if necessary to accommodate the shift,
    this ensures that no cycles are lost.
    """
    h_plus = as_1d_float_array(h_plus, "h_plus")
    h_cross = as_1d_float_array(h_cross, "h_cross")

    if len(h_plus) != len(h_cross):
        raise ValueError("h_plus and h_cross lengths differ")

    n = len(h_plus)

    if amplitude_for_merger is None:
        envelope = np.sqrt(h_plus**2 + h_cross**2)
    else:
        envelope = as_1d_float_array(amplitude_for_merger, "amplitude_for_merger")
        envelope = resample_to_length(envelope, n)

    current_merger_idx = int(np.argmax(np.abs(envelope)))
    target_merger_idx = int(round(merger_fraction * n))

    target_merger_idx = max(0, min(target_merger_idx, n - 1))

    shift_samples = target_merger_idx - current_merger_idx

    h_plus_shifted = shift_with_zero_padding(h_plus, shift_samples)
    h_cross_shifted = shift_with_zero_padding(h_cross, shift_samples)
    
    return h_plus_shifted, h_cross_shifted, current_merger_idx, target_merger_idx, shift_samples


def embed_1s_waveform_in_8s(h_plus_1s, h_cross_1s, sampling_frequency, segment_duration=8.0, insertion_start_seconds=3.5):
    h_plus_1s = as_1d_float_array(h_plus_1s, "h_plus_1s")
    h_cross_1s = as_1d_float_array(h_cross_1s, "h_cross_1s")

    if len(h_plus_1s) != len(h_cross_1s):
        raise ValueError("h_plus_1s and h_cross_1s lengths differ")

    n_segment = int(round(segment_duration * sampling_frequency))
    start_idx = int(round(insertion_start_seconds * sampling_frequency))
    end_idx = start_idx + len(h_plus_1s)

    if start_idx < 0 or end_idx > n_segment:
        raise ValueError(
            f"1s waveform does not fit in segment: start={start_idx}, end={end_idx}, n_segment={n_segment}"
        )

    h_plus_8s = np.zeros(n_segment, dtype=np.float64)
    h_cross_8s = np.zeros(n_segment, dtype=np.float64)

    h_plus_8s[start_idx:end_idx] = h_plus_1s
    h_cross_8s[start_idx:end_idx] = h_cross_1s

    return h_plus_8s, h_cross_8s, start_idx, end_idx


def fft_td_waveform(h_td, sampling_frequency, use_bilby=True):
    h_td = as_1d_float_array(h_td, "h_td")

    if use_bilby:
        h_fd, freqs = bilby.core.utils.nfft(h_td, sampling_frequency)
    else:
        freqs = np.fft.rfftfreq(len(h_td), d=1.0 / sampling_frequency)
        h_fd = np.fft.rfft(h_td) / sampling_frequency

    return freqs, h_fd


def get_amp_phase_from_mlwavegen(mass_1, mass_2, chi_1, chi_2,
                                 wfmodel_modelpath='../trained-models/flexcvae-model-backup-20260619-064140-epoch98.pt',
                                 wfmodel_configpath='../trained-models/modelconfig-flexcvae-20260619-064140.json',
                                 calibrator_modelpath='../v0p1/trained-models/calibrator_model_20260623-010953_epoch74.pt'):
    """
    Get calibrated amplitude and phase from MLWaveGen for given parameters.
    """
    ml_wfmodel, ml_calmodel = get_cached_mlmodel()
    if ml_wfmodel is None:
        logger.warning("No ML model provided to get_td_SEOBNRv4ml. We will initialize the model using the provided model_path and config_path in kwargs!")
        ml_wfmodel = load_flex_model(model_path=wfmodel_modelpath, configpath=wfmodel_configpath, device=None, precision=None)
        ml_calmodel = CalibrationModel(calibrator_modelpath=calibrator_modelpath, device=None, precision=None)
        set_cached_mlmodel(ml_wfmodel, ml_calmodel)
    
    parameters = {'mass_1': mass_1, 'mass_2': mass_2, 'spin_1z': chi_1, 'spin_2z': chi_2}
    
    labels = [parameters[key] for key in sorted(parameters.keys())]
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)  # shape: (1, 4, 1)
    logger.debug(f"Converted parameters to tensor labels for ML model: {labels}")
    
    outwaves = ml_wfmodel.generate(labels, convert_to_hphc=False)  # has shape (1, 2=[amp,phase], seq_len)!
    logger.info(f"Generated waveform from ML model with shape: {outwaves.shape}")

    amp_mlcal, phase_mlcal = ml_calmodel.calibrate_waveform(outwaves, labels, convert_to_hphc=False)
    logger.info(f"Calibrated waveform from ML model with shape: {amp_mlcal.shape}, {phase_mlcal.shape}")
    logger.info(f"Calibrated waveform shapes: amp={amp_mlcal.shape}, phase={phase_mlcal.shape}")

    # plt.plot(amp_mlcal.squeeze().cpu().numpy(), label='Calibrated Amplitude')
    # plt.plot(phase_mlcal.squeeze().cpu().numpy(), label='Calibrated Phase')
    # plt.title('Calibrated Amplitude and Phase from MLWaveGen')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Amplitude / Phase')
    # plt.legend()
    # plt.show()
    # exit()
    return (amp_mlcal.squeeze().cpu().numpy(), phase_mlcal.squeeze().cpu().numpy())


def build_8s_tapered_ml_waveform(
    mass_1,
    mass_2,
    chi_1,
    chi_2,
    sampling_frequency=8192,
    short_duration=1.0,
    segment_duration=8.0,
    insertion_start_seconds=3.5,
    merger_fraction=0.8,
    ringdown_taper_seconds=0.02,
    cross_sign=1.0,
    phase_offset=0.0,
):
    """
    Full pipeline:
        MLWaveGen -> A, phase
        start taper from first full phase cycle
        reconstruct h_plus, h_cross
        shift merger to 80 percent of 1s waveform
        taper final 0.02s
        embed into 8s segment
        FFT
    """
    n_short = int(round(short_duration * sampling_frequency))

    amplitude_raw, phase_raw = get_amp_phase_from_mlwavegen(
        mass_1=mass_1,
        mass_2=mass_2,
        chi_1=chi_1,
        chi_2=chi_2,
    )

    amplitude_raw = resample_to_length(amplitude_raw, n_short)
    phase_raw = resample_to_length(phase_raw, n_short)
    phase_unwrapped = np.unwrap(phase_raw)

    start_taper_samples = find_one_cycle_taper_length_from_phase(
        phase_unwrapped,
        sampling_frequency=sampling_frequency,
    )

    start_window = sin2_rise_taper(n_short, start_taper_samples)
    amplitude_start_tapered = amplitude_raw * start_window

    hp_start, hc_start = reconstruct_hp_hc_from_amp_phase(
        amplitude_start_tapered,
        phase_unwrapped,
        cross_sign=cross_sign,
        phase_offset=phase_offset,
    )

    hp_shifted, hc_shifted, merger_idx_old, merger_idx_target, shift_samples = shift_merger_to_fraction(
        hp_start,
        hc_start,
        amplitude_for_merger=amplitude_start_tapered,
        merger_fraction=merger_fraction,
    )

    ringdown_taper_samples = int(round(ringdown_taper_seconds * sampling_frequency))
    end_window = cos2_fall_taper(n_short, ringdown_taper_samples)

    hp_1s_final = hp_shifted * end_window
    hc_1s_final = hc_shifted * end_window

    hp_8s, hc_8s, insert_start_idx, insert_end_idx = embed_1s_waveform_in_8s(
        hp_1s_final,
        hc_1s_final,
        sampling_frequency=sampling_frequency,
        segment_duration=segment_duration,
        insertion_start_seconds=insertion_start_seconds,
    )

    freqs, hp_fd = fft_td_waveform(hp_8s, sampling_frequency)
    _, hc_fd = fft_td_waveform(hc_8s, sampling_frequency)

    t_short = np.arange(n_short) / sampling_frequency
    t_segment = np.arange(len(hp_8s)) / sampling_frequency

    merger_time_in_short = merger_idx_target / sampling_frequency
    merger_time_in_segment = insertion_start_seconds + merger_time_in_short

    return {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "chi_1": chi_1,
        "chi_2": chi_2,
        "sampling_frequency": sampling_frequency,
        "short_duration": short_duration,
        "segment_duration": segment_duration,
        "insertion_start_seconds": insertion_start_seconds,
        "insertion_end_seconds": insertion_start_seconds + short_duration,
        "merger_fraction": merger_fraction,
        "merger_time_in_short": merger_time_in_short,
        "merger_time_in_segment": merger_time_in_segment,
        "start_taper_samples": start_taper_samples,
        "start_taper_seconds": start_taper_samples / sampling_frequency,
        "ringdown_taper_samples": ringdown_taper_samples,
        "ringdown_taper_seconds": ringdown_taper_seconds,
        "merger_idx_old": merger_idx_old,
        "merger_idx_target": merger_idx_target,
        "shift_samples": shift_samples,
        "t_short": t_short,
        "t_segment": t_segment,
        "amplitude_raw": amplitude_raw,
        "phase_raw": phase_raw,
        "phase_unwrapped": phase_unwrapped,
        "start_window": start_window,
        "end_window": end_window,
        "amplitude_start_tapered": amplitude_start_tapered,
        "hp_start_tapered": hp_start,
        "hc_start_tapered": hc_start,
        "hp_1s_final": hp_1s_final,
        "hc_1s_final": hc_1s_final,
        "hp_8s": hp_8s,
        "hc_8s": hc_8s,
        "freqs": freqs,
        "hp_fd": hp_fd,
        "hc_fd": hc_fd,
    }


def plot_tapered_waveform_stages(result, outdir=".", label="ml_taper_debug"):
    os.makedirs(outdir, exist_ok=True)

    t = result["t_short"]
    ts = result["t_segment"]
    freqs = result["freqs"]

    hp_fd = result["hp_fd"]
    hc_fd = result["hc_fd"]

    eps = 1e-40

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, result["amplitude_raw"], label="raw amplitude")
    ax.plot(t, result["amplitude_start_tapered"], label="start-tapered amplitude")
    ax.axvline(result["start_taper_seconds"], linestyle="--", label="one-cycle taper end")
    ax.axvline(result["merger_idx_old"] / result["sampling_frequency"], linestyle=":", label="raw merger")
    ax.set_xlabel("Time in 1 s ML waveform [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Stage 1: raw amplitude and start taper")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{label}_stage1_amp_start_taper.png"), dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, result["phase_unwrapped"], label="unwrapped phase")
    ax.axhline(result["phase_unwrapped"][0] + 2.0 * np.pi, linestyle="--", label="phase + 2pi")
    ax.axvline(result["start_taper_seconds"], linestyle="--", label="taper length")
    ax.set_xlabel("Time in 1 s ML waveform [s]")
    ax.set_ylabel("Unwrapped phase [rad]")
    ax.set_title("Stage 2: phase-based one-cycle taper length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{label}_stage2_phase_taper_length.png"), dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, result["hp_start_tapered"], label=r"$h_+$ after start taper")
    ax.plot(t, result["hp_1s_final"], label=r"$h_+$ after shift and end taper")
    ax.axvline(result["merger_idx_target"] / result["sampling_frequency"], linestyle="--", label="target merger")
    ax.axvline(1.0 - result["ringdown_taper_seconds"], linestyle=":", label="end taper start")
    ax.set_xlabel("Time in 1 s waveform [s]")
    ax.set_ylabel("Strain")
    ax.set_title("Stage 3: reconstructed and shifted waveform")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{label}_stage3_hp_shift_taper.png"), dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, result["hp_8s"], label=r"$h_+$ embedded in 8 s")
    ax.plot(ts, result["hc_8s"], label=r"$h_\times$ embedded in 8 s")
    ax.axvline(result["insertion_start_seconds"], linestyle=":", label="insertion start")
    ax.axvline(result["merger_time_in_segment"], linestyle="--", label="merger")
    ax.axvline(result["insertion_end_seconds"], linestyle=":", label="insertion end")
    ax.set_xlabel("Time in 8 s segment [s]")
    ax.set_ylabel("Strain")
    ax.set_title("Stage 4: 1 s waveform embedded in 8 s segment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{label}_stage4_8s_embedding.png"), dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, np.maximum(np.abs(hp_fd), eps), label=r"$|\tilde h_+(f)|$")
    ax.plot(freqs, np.maximum(np.abs(hc_fd), eps), label=r"$|\tilde h_\times(f)|$")
    ax.set_yscale("log")
    ax.set_xlim(0, result["sampling_frequency"] / 2.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Frequency-domain strain amplitude")
    ax.set_title("Stage 5: Fourier transform of 8 s tapered waveform")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{label}_stage5_fft_log.png"), dpi=250)
    plt.close(fig)


def run_conditioner():
    params = dict(
        mass_1=58.66327592946544,
        mass_2=42.140402119374166,
        chi_1=0.06979998634467655,
        chi_2=0.6961158780604293,
    )

    out = build_8s_tapered_ml_waveform(
        mass_1=params["mass_1"],
        mass_2=params["mass_2"],
        chi_1=params["chi_1"],
        chi_2=params["chi_2"],
        sampling_frequency=8192,
        short_duration=1.0,
        segment_duration=8.0,
        insertion_start_seconds=3.5,
        merger_fraction=0.9,
        ringdown_taper_seconds=0.02,
        cross_sign=1.0,
    )

    savedir = f"../v0p1/results/{TODAY}/taper_debug_plots_{NOW}/"
    plot_tapered_waveform_stages(
        out,
        outdir=savedir,
        label="ml_8s_taper_test",
    )

    hp_8s = out["hp_8s"]
    hc_8s = out["hc_8s"]
    freqs = out["freqs"]
    hp_fd = out["hp_fd"]
    hc_fd = out["hc_fd"]

    # -- Save configuration to file
    config_save_path = os.path.join(savedir, "ml_8s_taper_test_config.json")
    os.makedirs(savedir, exist_ok=True)
    with open(config_save_path, "w") as f:
        json.dump(params, f, indent=4)

    print("Merger time in 8 s segment:", out["merger_time_in_segment"])
    print("For Bilby use start_time = geocent_time -", out["merger_time_in_segment"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run waveform conditioner for ML-generated waveforms.")

    parser = init_verbosity_args(parser)
    args = parser.parse_args()
    init_logging(args, log_dir=f'../{PROJECT_DIR}/logs/{TODAY}')

    run_conditioner()

    # get_amp_phase_from_mlwavegen(
    #     mass_1=58.66327592946544,
    #     mass_2=42.140402119374166,
    #     chi_1=0.06979998634467655,
    #     chi_2=0.6961158780604293,
    # )