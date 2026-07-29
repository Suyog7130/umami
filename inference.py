
"""
Use trained surrogate ML model and do Bayesian parameter estimation on this
using Bilby, and obtain a Probability-Probability plot.
"""

import os
import glob
import copy
import json
import time
import argparse
import logging
import datetime
import numpy as np
import pandas as pd

import scipy
from scipy.special import logsumexp

import matplotlib.pyplot as plt
import matplotlib.ticker as tck

plt.rcParams['text.usetex'] = True

import torch
import torch.multiprocessing as mp

import bilby
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.detector import InterferometerList
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.core.prior import Uniform
from bilby.gw.prior import PriorDict, BBHPriorDict
from bilby.core.result import make_pp_plot

import copy
from typing import Dict, List, Tuple, Optional, Any

from tqdm import tqdm

from pycbc.waveform import get_td_waveform
from mlwavegen import MLWaveformGenerator, convert_to_ml_parameters
from wfconditioner import get_conditioned_waveform
from utils.gwutils import amp_phase_from_polarizations, calc_polarization_mismatch
from utils.io import (
    save_json, 
    load_json,
    save_pickle, 
    save_txt, 
    write_DONE_file, 
    ensure_dir, 
    check_DONE_file_exists
    )

from utils.generic import init_logging, init_verbosity_args
logger = logging.getLogger(__name__)


# -- Block standard python warnings from clogging the stream
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Force PyTorch backend logs to only show CRITICAL errors, silencing warning logs
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"



PROJECT_DIR = 'v0p1'

TODAY = datetime.date.today().strftime("%Y%m%d")
TIME = datetime.datetime.now().strftime("%H%M%S")
NOW = TODAY + '-' + TIME

# -- define some constants for waveform generation
SAMPLE_RATE = 8192  # Hz
DURATION = 8.0  # seconds

# NOTE: My ML training has $f_{min}\in [11,60]$ with mean of 19.5 Hz
# FIXME: This `FMIN` value should ideally be lower than the lowest f_low value used in ML model training.
# But then the duration of the data exceeds the length of 8 second!
FMIN = 14.0  # Hz   

FREF = 50.0  # Hz
LUMINOSITY_DISTANCE = 400.0  # Mpc, should be same as for the ML waveform training data, to avoid bias in amplitudes!

# -- Constants for all waveform generators and IFOs.
MERGER_TIME = 1126259642.413
MERGER_TIME_IN_SEGMENT = 6.4   # 80% of the 8s long data segment, after wf conditioning!
START_TIME = MERGER_TIME - MERGER_TIME_IN_SEGMENT



if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    PRECISION = 'float32'  # Use float32 for CUDA if available
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    PRECISION = 'float32'  # Use float32 for MPS since it does not support float64 well
else:
    DEVICE = torch.device("cpu")
    PRECISION = 'float32'  # Use float32 for CPU


LATEX_LABELS = {
    "mass_1": "$m_1 \, [M_\odot]$",
    "mass_2": "$m_2 \, [M_\odot]$",
    "chi_1": "$\chi_1$",
    "chi_2": "$\chi_2$",
    "chirp_mass": "$\mathcal{M} \, [M_\odot]$",
    "chi_eff": "$\chi_{\mathrm{eff}}$",
}


# bilby.core.utils.setup_logger(outdir=f'../logs/{TODAY}', label='umamipe', log_level="INFO")


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    bilby.core.utils.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed} for reproducibility.")


# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])


def make_default_base_injection() -> Dict[str, float]:
    return dict(
        mass_1=36.0,
        mass_2=32.0,  # keep inside prior
        # a_1=0.4,
        # a_2=0.3,
        # tilt_1=0.0,
        # tilt_2=0.0,
        # phi_12=0.0,
        # phi_jl=0.0,
        luminosity_distance=LUMINOSITY_DISTANCE,
        theta_jn=0.4,
        psi=2.659,
        phase=1.3,
        geocent_time=MERGER_TIME,
        ra=1.375,
        dec=-1.2108,
        chi_1=0.4,   # for ML waveform generator, equivalent to spin1z
        chi_2=0.3,   # for ML waveform generator, equivalent to spin2z
    )


base_injection = make_default_base_injection()

p2m_cond_best_worst_params = {
    "best_mismatch": 0.0032256899092882874,
    "best_mismatch_params": {
        "mass_1": 52.29399871826172,
        "mass_2": 32.418025970458984,
        "chi_1": 0.31453636288642883,
        "chi_2": -0.16409091651439667
    },
    "worst_mismatch": 0.2469990280857144,
    "worst_mismatch_params": {
        "mass_1": 5.130840301513672,
        "mass_2": 9.432706832885742,
        "chi_1": 0.8170799016952515,
        "chi_2": -0.8445430397987366
    },
    "median_mismatch": 0.011362908620650036,
    "median_mismatch_params": {
        "mass_1": 65.2022476196289,
        "mass_2": 22.095273971557617,
        "chi_1": 0.35439324378967285,
        "chi_2": 0.8108929991722107
    }
}


# def signed_chi_to_bilby_spins(params):
#     p = dict(params)

#     chi1 = p.pop("spin_1z")
#     chi2 = p.pop("spin_2z")

#     p["a_1"] = abs(chi1)
#     p["tilt_1"] = 0.0 if chi1 >= 0 else np.pi

#     p["a_2"] = abs(chi2)
#     p["tilt_2"] = 0.0 if chi2 >= 0 else np.pi

#     p.setdefault("phi_12", 0.0)
#     p.setdefault("phi_jl", 0.0)

#     return p


def make_analysis_priors(
    injection_parameters: Dict[str, float],
    active_keys: Tuple[str, ...] = ("mass_1", "mass_2", "chi_1", "chi_2"),
    without_mass_ratio_constraint: bool = False,
) -> bilby.gw.prior.BBHPriorDict:
    """
    Build priors for one PE run.
    Fixed parameters become delta-function priors at injected values.
    Active parameters are sampled.

    `bilby.gw.priors.BBHPriorDict` provides the option for `aligned_spin` priors,
    [mass_1, mass_2, chi_1, chi_2], which we can directly use. My ML waveform requires 
    `spin_1z` and `spin_2z` instead of `chi_1` and `chi_2`, and these are converted using 
    the supplied convertion factor in the MLWaveformGenerator.

    Arguments
    ---------
    injection_parameters: dict
        Dictionary of all injection parameters and their values.
    active_keys: tuple of str
        Tuple of parameter names to be treated as active parameters, and sampled over 
        in PE run. All other parameters will be fixed to their injected values.
    """
    priors = bilby.gw.prior.BBHPriorDict(
        aligned_spin=True,
    )
    print(f"Default priors for BBH parameters:\n{priors.keys()}")
    print(f"Full priors for BBH parameters:\n{priors}")

    # NOTE: The following parameters are fixed, and therefore consitute delta-function priors at the injection value. These parameters are ignored during sampling, and only the active parameters are considered.
    fixed_keys = [
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "luminosity_distance",
        "theta_jn",
        "psi",
        "ra",
        "dec",
        "geocent_time",
        "phase",
    ]

    for key in fixed_keys:
        if key in injection_parameters and key not in active_keys:
            priors[key] = injection_parameters[key]

    if "mass_1" in active_keys:
        priors["mass_1"] = bilby.core.prior.Uniform(
            30, 75, name="mass_1", latex_label=LATEX_LABELS["mass_1"]
        )
    else:
        priors["mass_1"] = injection_parameters["mass_1"]

    if "mass_2" in active_keys:
        priors["mass_2"] = bilby.core.prior.Uniform(
            30, 75, name="mass_2", latex_label=LATEX_LABELS["mass_2"]
        )
    else:
        priors["mass_2"] = injection_parameters["mass_2"]

    if "chi_1" in active_keys:
        priors["chi_1"] = bilby.core.prior.Uniform(
            -0.80, 0.80, name="chi_1", latex_label=LATEX_LABELS["chi_1"]
        )
    else:
        priors["chi_1"] = injection_parameters["chi_1"]

    if "chi_2" in active_keys:
        priors["chi_2"] = bilby.core.prior.Uniform(
            -0.80, 0.80, name="chi_2", latex_label=LATEX_LABELS["chi_2"]
        )
    else:        
        priors["chi_2"] = injection_parameters["chi_2"]

    if not without_mass_ratio_constraint:
        priors["mass_ratio"] = bilby.gw.prior.Constraint(
            minimum=1.0/10.0,  # NOTE: m_1>=m_2, and m_1/m_2<10.0, but Bilby convention is m2/m1!
            maximum=1.0,
            name="mass_ratio",
        )
        # -- Add the "mass_ratio" parameter to check for constraints
        priors.conversion_function = add_mass_ratio
    else:
        priors.pop("mass_ratio", None)

    priors.pop("chirp_mass", None)
    print("Updated priors for BBH parameters:")
    for key, prior in priors.items():
        print(f"  {key}: {prior}")
    return priors


def add_mass_ratio(parameters):
    converted = parameters.copy()
    if "mass_1" in converted and "mass_2" in converted:
        converted["mass_ratio"] = converted["mass_2"] / converted["mass_1"]
    return converted


def sample_injection_from_priors(
    base_injection: Dict[str, float],
    active_priors: bilby.core.prior.PriorDict,
    active_keys: Tuple[str, ...] = ("mass_1", "mass_2", "chi_1", "chi_2"),
) -> Dict[str, float]:
    """
    Samples only active parameters, copies all other values from base_injection.
    """
    injection = copy.deepcopy(base_injection)

    # logger.debug(f"Active priors for injection sampling: {active_priors}")
    sampled = active_priors.sample()
    logger.debug(f"Sampled injection parameters from priors: {sampled}")

    for key in active_keys:
        injection[key] = sampled[key]

    return injection



def aligned_chi_to_lal_parameters(parameters, 
                                  return_with_bilby_converter=False):
    """
    Convert aligned-spin chi_1, chi_2 parameters into LAL-style
    spin magnitude and tilt parameters for non-precessing approximants.

    Input sampled parameters:
        chi_1, chi_2 in [-1, 1]

    Output LAL-style parameters:
        a_1, a_2 >= 0
        tilt_1, tilt_2 in {0, pi}
        phi_12 = 0
        phi_jl = 0
    """
    logger.debug(f"Converting aligned-spin parameters to LAL parameters: {parameters}")
    converted = parameters.copy()

    if "chi_1" in converted:
        chi_1 = float(converted["chi_1"])
        converted["a_1"] = abs(chi_1)
        converted["tilt_1"] = 0.0 if chi_1 >= 0 else np.pi

    if "chi_2" in converted:
        chi_2 = float(converted["chi_2"])
        converted["a_2"] = abs(chi_2)
        converted["tilt_2"] = 0.0 if chi_2 >= 0 else np.pi

    converted["phi_12"] = 0.0
    converted["phi_jl"] = 0.0

    # # Also keep these for the ML model if useful.
    # if "chi_1" in converted:
    #     converted["spin_1z"] = converted["chi_1"]
    # if "chi_2" in converted:
    #     converted["spin_2z"] = converted["chi_2"]
    logger.debug(f"Converted LAL parameters: {converted}")
    if return_with_bilby_converter:
        return bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(converted)
    return converted, []


def lal_binary_black_hole_aligned_chi(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    chi_1,
    chi_2,
    theta_jn,
    phase,
    **kwargs,
):
    a_1 = abs(float(chi_1))
    tilt_1 = 0.0 if chi_1 >= 0 else np.pi

    a_2 = abs(float(chi_2))
    tilt_2 = 0.0 if chi_2 >= 0 else np.pi

    return bilby.gw.source.lal_binary_black_hole(
        frequency_array=frequency_array,
        mass_1=mass_1,
        mass_2=mass_2,
        luminosity_distance=luminosity_distance,
        a_1=a_1,
        tilt_1=tilt_1,
        phi_12=0.0,
        a_2=a_2,
        tilt_2=tilt_2,
        phi_jl=0.0,
        theta_jn=theta_jn,
        phase=phase,
        **kwargs,
    )

def identity_parameter_conversion(parameters):
    return parameters.copy(), []


# priors = BBHPriorDict(aligned_spin=True)
# print("Default priors for BBH parameters:")
# for key, prior in priors.items():
#     print(f"  {key}: {prior}")
# print(f'Default priors for BBH parameters: {priors.keys()}')

# # NOTE: Other parameters should be allowed to vary freely, for injection generation!
# # # -- remove redundant or irrelevant parameters
# params_to_use = ["mass_1", "mass_2", "chi_1", "chi_2"]
# # for param in list(priors.keys()):
# #     if param not in params_to_use:
# #         priors.pop(param)
# priors.pop("mass_ratio", None)
# priors.pop("chirp_mass", None)
# priors["geocent_time"] = 0.0 

# # -- Set the priors for the parameters we want to estimate!
# priors["mass_1"] = bilby.core.prior.Uniform(30, 75, name="mass_1", latex_label="$m_1$")
# priors["mass_2"] = bilby.core.prior.Uniform(30, 75, name="mass_2", latex_label="$m_2$")
# priors["chi_1z"] = bilby.core.prior.Uniform(-0.80, 0.80, name="chi_1z", latex_label="$\\chi_{1z}$")
# priors["chi_2z"] = bilby.core.prior.Uniform(-0.80, 0.80, name="chi_2z", latex_label="$\\chi_{2z}$")

# # priors["chi_1"].a_prior.maximum = 0.80
# # priors["chi_2"].a_prior.maximum = 0.80
# # priors["a_1"] = bilby.core.prior.Uniform(0, 0.80, name="a_1", latex_label="$\\chi_{1z}$")
# # priors["a_2"] = bilby.core.prior.Uniform(0, 0.80, name="a_2", latex_label="$\\chi_{2z}$")
# # priors["a_1"] = injection_parameters["a_1"]  
# # priors["a_2"] = injection_parameters["a_2"]  

# print("Updated priors for BBH parameters:")
# for key, prior in priors.items():
#     print(f"  {key}: {prior}")
# print(f'Updated priors for BBH parameters: {priors.keys()}')

def get_network_optimal_snr(ifos):
    snrs = []
    for ifo in ifos:
        snrs.append(float(ifo.meta_data["optimal_SNR"]))
    return np.sqrt(np.sum(np.asarray(snrs) ** 2))


def run_single_injection(run_idx, seed=None, label_base='umamipe', 
                         outdir=f'../{PROJECT_DIR}/results/',
                         active_priors=None,
                         injection_generator=None, waveform_generator=None, 
                         use_set_injection_params=False,
                         sampler=None, **sampler_kwargs):
    if seed is not None:
        rng_seed = seed + run_idx
    set_random_seed(rng_seed)
    this_label = f"{label_base}_inj_{run_idx:04d}"
    logger.info(f"Running sampler for injection {run_idx} with label {this_label} using sampler {sampler}...")

    if use_set_injection_params:
        logger.warning("Using pre-defined injection parameters for best mismatch, instead of sampling from priors!")
        injection_parameters = base_injection.copy()
        set_params = p2m_cond_best_worst_params["best_mismatch_params"]
        injection_parameters.update(set_params)
    else:
        injection_parameters = sample_injection_from_priors(
            base_injection=base_injection,
            active_priors=active_priors,
        )
    logger.info(f"Sampled injection parameters for run {run_idx}: {injection_parameters}")

    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLE_RATE,
        duration=DURATION,
        start_time=START_TIME,   # NOTE: Injection signal should be within data segment!
    )

    # -- Send model to device only here! We will keep the model on CPU until we need to generate the waveform, to save GPU memory and avoid potential issues with multiprocessing in Bilby!
    for generator in [injection_generator, waveform_generator]:
        if isinstance(generator, MLWaveformGenerator):
            logger.debug(f"Sending ML model to device: {DEVICE} with dtype: {PRECISION}")
            generator.ml_wfmodel.to(DEVICE, dtype=getattr(torch, PRECISION))
            generator.ml_calmodel.model.to(DEVICE, dtype=getattr(torch, PRECISION))
            generator.check_model_weights_on_device(device=DEVICE, precision=getattr(torch, PRECISION))
            logger.debug(f"Sent injection_generator.loaded_mlmodel to device: {DEVICE} with dtype: {PRECISION}")

    ifos.inject_signal(
        waveform_generator=injection_generator,
        parameters=injection_parameters,
    )
    logger.info(f"Injection parameters for run {run_idx}: {injection_parameters}")

    # -- Compute network optimal SNR for the injected signal
    network_snr = get_network_optimal_snr(ifos)
    logger.info(f"Network optimal SNR for run {run_idx}: {network_snr}")
    save_json({"network_optimal_snr": network_snr,
               "H1_optimal_snr": float(ifos[0].meta_data["optimal_SNR"]),
               "L1_optimal_snr": float(ifos[1].meta_data["optimal_SNR"]),}, 
              os.path.join(outdir, f"{this_label}_network_optimal_snr.json"))

    # -- Save IFOs with injected signal and noise to file
    save_pickle(ifos, os.path.join(outdir, f"{this_label}_ifos.pkl"))
    save_json(injection_parameters, os.path.join(outdir, f"{this_label}_injection_parameters.json"))

    likelihood = GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
    )

    time_start = time.time()
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=active_priors,
        sampler=sampler,
        injection_parameters=injection_parameters,
        # conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
        outdir=outdir,
        label=this_label,
        resume=False,
        save=True,
        **sampler_kwargs
    )
    time_end = time.time()
    logger.info(f"Completed sampler for injection {run_idx} with label {this_label}. Time taken: {time_end - time_start:.2f} seconds.")

    # Write total sampling time to TXT file
    save_txt(f"{time_end - time_start:.2f}", os.path.join(outdir, f"{this_label}_sampling_time.txt"))

    # Make a corner plot
    true_params = {"mass_1": injection_parameters["mass_1"],
                    "mass_2": injection_parameters["mass_2"],
                    "chi_1": injection_parameters["chi_1"],
                    "chi_2": injection_parameters["chi_2"]}
    result.plot_corner(save=True, parameters=true_params,
                       filename=outdir+f'{this_label}_corner.png')
    return result


def run_injection_campaign(num_injections=50, base_seed=1234, 
                           label_base='umamipe', outdir=f'../{PROJECT_DIR}/results/',
                           injection_generator=None, waveform_generator=None, 
                           sampler=None, active_priors=None, **sampler_kwargs):
    if injection_generator is None or waveform_generator is None:
        raise NotImplementedError("Both injection and waveform generators must be provided")
    results = []
    for i in tqdm(range(num_injections), desc="Running injections"):
        logger.info(f"Injection {i+1}/{num_injections}")
        res = run_single_injection(i, seed=base_seed, outdir=outdir, label_base=label_base,
                                   injection_generator=injection_generator, waveform_generator=waveform_generator, 
                                   sampler=sampler, active_priors=active_priors, **sampler_kwargs)
        results.append(res)
    return results


TRUNCATE_EOB_WAVEFORM_TO_1S = False
FORCE_EOB_WF_TRUNCATION = False
PLOT_CONDITIONED_WAVEFORM = False


def ml_to_eob_param_conversion(parameters):
    """
    Convert ML model parameters to EOB model parameters.
    [spin_1z, spin_2z] -> [chi_1, chi_2] for EOB model.
    """
    converted = parameters.copy()
    converted["chi_1"] = converted.pop("spin_1z", converted.get("chi_1"))
    converted["chi_2"] = converted.pop("spin_2z", converted.get("chi_2"))
    return converted, []


def pycbc_seobnrv4_time_domain_source_model(time_array, 
        mass_1, mass_2, chi_1, chi_2, 
        theta_jn,
        psi,
        ra,
        dec,
        phase,
        luminosity_distance=1.0,
        **kwargs):
    """
    Bilby-compatible time-domain source model using PyCBC get_td_waveform,
    which generates a waveform in the time domain with a set low frequency cutoff,
    and adds this signal to a 8 second data segment with a fixed merger time of 6.4 second.
    The signal is also conditioned such that the start and ends are tapered smoothly to zero.

    Returns:
        {"plus": hp, "cross": hc}
    """
    hp, hc = get_td_waveform(
        approximant="SEOBNRv4",
        mass1=mass_1,
        mass2=mass_2,
        chi1z=chi_1,
        chi2z=chi_2,
        theta_jn=theta_jn,
        psi=psi,
        coa_phase=phase,
        ra=ra,
        dec=dec,
        delta_t=1/SAMPLE_RATE,   # -- this should always be such that we have 8192 samples in 1 second!
        f_lower=FMIN,
        f_ref=FREF
    )

    hp = hp.trim_zeros()
    hc = hc.trim_zeros()

    amp_arr, phase_arr = amp_phase_from_polarizations(hp, hc, use_pycbc=True)
    hplus, hcross = get_conditioned_waveform(amp_arr, phase_arr,
                                             scale_factor=1.0,  # No scaling req! 
                                             plot_result=PLOT_CONDITIONED_WAVEFORM,
                                             wf_type='eob',
                                             truncate_wf_to_short_duration=TRUNCATE_EOB_WAVEFORM_TO_1S,
                                             force=FORCE_EOB_WF_TRUNCATION,)
    
    if luminosity_distance != 1.0:
        hplus /= luminosity_distance
        hcross /= luminosity_distance

    return {"plus": hplus, "cross": hcross}


def make_wf_generator(type: {'eobbilby', 'eob', 'ml'}, 
                      wfkwargs: Dict = {},
                      wf_source_model = None,
                      parameter_converter = None) -> WaveformGenerator:
    if type=='eobbilby':
        if wf_source_model is None:
            wf_source_model = bilby.gw.source.lal_binary_black_hole
        if parameter_converter is None:
            parameter_conversion = aligned_chi_to_lal_parameters
        wfgen = WaveformGenerator(
            duration=DURATION,
            sampling_frequency=SAMPLE_RATE,
            # NOTE: The `lal_binary_black_hole` source model works basically FrequencyDomain approximants!
            frequency_domain_source_model=wf_source_model,
            parameter_conversion=parameter_converter,
            start_time=START_TIME,
            waveform_arguments=dict(
                waveform_approximant="SEOBNRv4",      #"IMRPhenomPv2",
                reference_frequency=FREF,
                minimum_frequency=FMIN,
                mode_array=[[2,2]],
                catch_waveform_errors=True,
            )
        )
        assert wfgen.start_time == START_TIME, f"Waveform generator start time {wfgen.start_time} does not match expected {START_TIME}"
        return wfgen
    
    elif type=='eob':
        wfgen = WaveformGenerator(
            duration=DURATION,
            sampling_frequency=SAMPLE_RATE,
            time_domain_source_model=pycbc_seobnrv4_time_domain_source_model,
            parameter_conversion=ml_to_eob_param_conversion,
            start_time=START_TIME,
            waveform_arguments={}
        )
        assert wfgen.start_time == START_TIME, f"Waveform generator start time {wfgen.start_time} does not match expected {START_TIME}"
        return wfgen

    elif type=='ml':
        if 'distance_scale_factor' not in wfkwargs:
            wfkwargs['distance_scale_factor'] = LUMINOSITY_DISTANCE
        logger.info(f"Waveform generator kwargs for ML model: {wfkwargs}")
        wfgen = MLWaveformGenerator(
            duration=DURATION,
            sampling_frequency=SAMPLE_RATE,
            time_domain_source_model=None,   # We will load ML model at initialization!
            parameter_conversion=convert_to_ml_parameters,
            start_time=START_TIME,
            waveform_arguments=wfkwargs,
        )
        assert wfgen.start_time == START_TIME, f"Waveform generator start time {wfgen.start_time} does not match expected {START_TIME}"
        return wfgen
    logger.error(f"Invalid waveform generator type: {type}. Must be 'eob' or 'ml'.")


def set_sampler_kwargs(args, sampler):
    if sampler=='nessai':
        logger.info("Using multiprocessing with spawn context for parallel sampling.")
        nworkers = min(3, mp.cpu_count() - 1)
        sampler_kwargs = dict(
            nlive=args.nlive,
            n_pool=args.nessai_npool,  # Set this arg for nessai's multiprocessing that can use GPU workers!
            pytorch_threads=args.pytorch_threads,   # limit PyTorch to 1 thread per worker!
            npool=args.dynesty_npool, # Set this arg for Bilby's internal multiprocessing that only works on CPU!
            flow_proposal_class='flowproposal',     # 'gwflowproposal' instead reparameterisation full 15D space!
            reparameterisations=None,  
            # max_iteration=7500,    # NOTE: This forces nessai to abruptly end, leaving results JSON file incomplete!
            stopping=args.threshold,    # Stop if log evidence `logZ` value will change by less than this amount in next iteration!
            reset_flow=16,          # Periodic reset to clear "stuck" AI states
            analytic_priors=True,  # this is a bool, to indicate directly using supplying prior samples.
        )
    elif sampler=='pocomc':
        logger.info("Using the preconditioned Monte Carlo sampler (pocoMC)!")
        sampler_kwargs = dict(
            # NOTE: Bilby by default removes the "vectorize" behaviour from any samplers that support it, since the waveforms that are supported by Bilby cannot generate data in batches!!! Thus, "pocomc" also basically the same amount of time as `nessai` or more!
            vectorize=True,   # THIS IS IGNORED!
            npool=args.npool,
            n_effective=args.nlive,  # larger the value, the better it is. `n_active = n_effective // 2` in pocomc!
            save_every=15,  # Save intermediate results every 15 iterations (default: 5)
            track_sampling_time=True,
            pytorch_threads=args.pytorch_threads,
        )
    else:
        logger.info("Not using multiprocessing. Running sampler in single-process mode.")
        sampler_kwargs = dict(
            nlive=args.nlive,
            dlogz=args.threshold,
            naccept=10,
            sample="acceptance-walk",
            npool=args.dynesty_npool,  # Set this arg for Bilby's internal multiprocessing that only works on CPU!
        )
    return sampler_kwargs


def main(args, label='umamipe', 
         pe_run_type: {'ml2ml', 'eobbilby2ml', 'eobbilby2eobbilby', 'eob2ml', 'eob2eob'} = 'ml2ml', 
         sampler: {'nessai', 'dynesty', 'pocomc'} = 'nessai',):
    label = label + f'_{pe_run_type}_{sampler}'
    project_dir = f'../{args.project_dir}/'
    if args.outdir is None:
        outdir = os.path.join(project_dir, f'results/{TODAY}/')
    else:
        outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    model_path = os.path.join(project_dir, 'trained-models', args.model_name)
    config_path = os.path.join(project_dir, 'trained-models', args.model_config)
    calmodel_path = os.path.join(project_dir, 'trained-models', args.calmodel_name)
    if not os.path.isfile(model_path):
        model_path = os.path.join('../', 'trained-models', args.model_name)
        if not os.path.isfile(model_path):
            logger.error(f"Provided MODEL_PATH does not exist: {model_path}")
            raise FileNotFoundError(f"MODEL_PATH file not found at {model_path}")
    logger.info(f"Using MODEL_PATH: {model_path}")
    if not os.path.isfile(config_path):
        config_path = os.path.join('../', 'trained-models', args.model_config)
        if not os.path.isfile(config_path):
            logger.error(f"Provided MODEL_CONFIG_PATH does not exist: {config_path}")
            raise FileNotFoundError(f"MODEL_CONFIG_PATH file not found at {config_path}")
    logger.info(f"Using MODEL_CONFIG_PATH: {config_path}")
    if not os.path.isfile(calmodel_path):
        calmodel_path = os.path.join('../', 'trained-models', args.calmodel_name)
        if not os.path.isfile(calmodel_path):
            logger.error(f"Provided CALMODEL_PATH does not exist: {calmodel_path}")
            raise FileNotFoundError(f"CALMODEL_PATH file not found at {calmodel_path}")
    logger.info(f"Using CALMODEL_PATH: {calmodel_path}")

    if args.truncate_eob_waveform_to_1s:
        global TRUNCATE_EOB_WAVEFORM_TO_1S, FORCE_EOB_WF_TRUNCATION
        TRUNCATE_EOB_WAVEFORM_TO_1S = True
        FORCE_EOB_WF_TRUNCATION = True
        logger.warning("We will be truncating EOB waveforms to 1 second duration for injection and/or recovery.")
    else:
        logger.warning("Not truncating EOB waveforms to 1 second duration. It is assumed for waveform conditioning that the actual EOB waveform length is shorter than 8 second long data segment it will be embedded into.")

    wfkwargs={'wfmodel_modelpath': model_path, 
                'wfmodel_configpath': config_path,
                'calibrator_modelpath': calmodel_path}
    if args.distance_factor is not None:
        wfkwargs['distance_scale_factor'] = args.distance_factor
        base_injection['luminosity_distance'] = args.distance_factor
        logger.info(f"Using distance factor / luminosity distance for ML waveform generator: {args.distance_factor} Mpc")

    if pe_run_type == 'eob2eob':
        # -- this uses PyCBC SEOBNRv4 time-domain waveform generator!
        injection_generator = make_wf_generator('eob')
        waveform_generator = make_wf_generator('eob')
        logger.info("Initialized EOB waveform generator for both injection and recovery.")
    elif pe_run_type == 'eobbilby2eobbilby':
        # -- this uses bilby waveform generator, without conditioning!
        injection_generator = make_wf_generator('eobbilby')
        waveform_generator = make_wf_generator('eobbilby')
        logger.info("Initialized EOB Bilby waveform generator for both injection and recovery.")
    else:
        waveform_generator = make_wf_generator('ml', wfkwargs=wfkwargs)
        if pe_run_type == 'ml2ml':
            injection_generator = make_wf_generator('ml', wfkwargs=wfkwargs)
            logger.info("Initialized ML waveform generator for both injection and recovery.")
        elif pe_run_type == 'eobbilby2ml':
            injection_generator = make_wf_generator('eobbilby')
            logger.info("Initialized EOB Bilby waveform generator for injection and ML waveform generator for recovery.")
        elif pe_run_type == 'eob2ml':
            injection_generator = make_wf_generator('eob')
            logger.info("Initialized EOB waveform generator for injection and ML waveform generator for recovery.")
    
    # TODO: For EOB waveforms, priors should be [m_1, m_2, a_1, a_2, tilt_1, tilt_2], since otherwise the "spin1z" and "spin2z" parameters will be ignored by the EOB waveform generator, since internally Bilby requires aforementioned parameter names, and then uses its the `bilby_to_lal_bbh_...` function to convert them to LAL parameters, before calling the waveform model.

    # Use the same active priors for drawing injections.
    if args.without_mass_ratio_constraint:
        logger.warning("Running without mass ratio constraint. This may lead to unphysical injections with m1 < m2. So, it is assumed that the MLWaveformGenerator can handles such cases automatically by swapping m1 and m2 internally, and returning the correct waveform.")

    active_priors = make_analysis_priors(
        injection_parameters=base_injection,
        without_mass_ratio_constraint=args.without_mass_ratio_constraint
    )

    logger.info("Active priors for injection sampling:")
    for key, prior in active_priors.items():
        logger.info(f"  {key}: {prior}")
    logger.info(f'Active priors for injection sampling: {active_priors.keys()}')

    # Perform a check that the prior does not extend to a parameter space longer than the data
    active_priors.validate_prior(DURATION, FMIN)

    sampler_kwargs = set_sampler_kwargs(args, sampler)

    if args.run_one_injection:

        # Allow injection index start to wary, so that new runs can be performed via HTCondor.
        injection_index = args.injection_index + args.injection_index_start

        if args.use_set_injection_params:
            injection_index = 0

        if check_DONE_file_exists(outdir, label=label, injection_index=injection_index, check_all_subdirs=False):
            if not args.force:
                logger.info(f"PE results for injection index {injection_index} already exist. Skipping this injection.")
                return
            else:
                logger.info(f"PE results for injection index {injection_index} already exist. Overwriting due to --force flag.")

        outdir = os.path.join(outdir, f'{label}_inj_{injection_index}_{NOW}/')
        ensure_dir(outdir)

        logger.info(f"Running a single injection and PE with fixed seed 42, for index {injection_index}...")
        results = run_single_injection(injection_index, seed=42, 
                                        label_base=label, outdir=outdir,
                                        active_priors=active_priors,
                                        injection_generator=injection_generator, 
                                        waveform_generator=waveform_generator,
                                        use_set_injection_params=args.use_set_injection_params,
                                        sampler=sampler, **sampler_kwargs)
        
    elif args.run_pe_campaign:
        logger.info(f"Running a PE campaign with {args.num_injections} injections...")

        results = run_injection_campaign(num_injections=args.num_injections, 
                                        base_seed=42,
                                        active_priors=active_priors,
                                        injection_generator=injection_generator, 
                                        waveform_generator=waveform_generator,
                                        label_base=label+f'_{NOW}', 
                                        outdir=outdir,
                                        sampler=sampler, 
                                        **sampler_kwargs)
        
    elif args.plot_corner_from_result_file:
        logger.info(f"Plotting corner plot from existing result file: {args.results_fname}...")
        injection_index = args.injection_index + args.injection_index_start
        result = bilby.gw.result.CBCResult.from_json(f"{outdir}/{args.results_fname}")
        injection_parameters = result.injection_parameters
        true_params = {"mass_1": injection_parameters["mass_1"],
                       "mass_2": injection_parameters["mass_2"],
                       "chi_1": injection_parameters["chi_1"],
                       "chi_2": injection_parameters["chi_2"]}
        logger.info(f"Loaded result from {outdir}/{args.results_fname}. Plotting corner plot with true parameters: {true_params}")
        result.plot_corner(save=True, parameters=true_params,
                           filename=outdir+f'{label}_inj_{injection_index:04d}_corner.png')
    
    # -- Save all args and config to a results config JSON file!
    config_snapshot = {
        'args': vars(args),
        'pe_run_type': pe_run_type,
        'sampler': sampler,
        'sampler_kwargs': sampler_kwargs,
        'base_seed': 42,
    }
    # -- convert any non-serializable objects in config_snapshot to strings or dicts
    for key, value in config_snapshot['sampler_kwargs'].items():
        if isinstance(value, (bilby.core.prior.PriorDict, bilby.gw.prior.BBHPriorDict)):
            config_snapshot['sampler_kwargs'][key] = {k: str(v) for k, v in value.items()}
        elif isinstance(value, torch.nn.Module):
            config_snapshot['sampler_kwargs'][key] = str(value)
        elif isinstance(value, dict):
            config_snapshot['sampler_kwargs'][key] = {k: str(v) for k, v in value.items()}
        elif isinstance(value, list):
            config_snapshot['sampler_kwargs'][key] = [str(v) for v in value]
    with open(os.path.join(outdir, f'{label}_{NOW}_config.json'), 'w') as f:
        json.dump(config_snapshot, f, indent=4)

    write_DONE_file(outdir, '')
    # analyze_results(results=results, label=label+f'_{NOW}', outdir=outdir)
    print("Completed injection campaign and sampling for all injections.")



# ------
# PE Bias Estimation
# ------


def correct_posterior_bias(posterior_samples: pd.DataFrame,
                           bias_factors: dict,) -> pd.DataFrame:
    """
    Applies bias correction to the posterior samples based on the provided bias factors.
    These bias factors are the slope and intercept of the `y=mx+c` linear fit line between
    the inferred posterior mode and the ratio of the inferred posterior mode to the true injection value, 
    for each parameter.
    A bias factor is first predicted based on the inferred posterior mode and these linear fit slope
    and intercept values. And then the corrected posterior is a shifted version of the original posterior,
    where each sample is divided by the predicted bias factor for that parameter.

    Arguments
    ---------
        posterior_samples: (pd.DataFrame)
            The posterior samples to correct.
        bias_factors: (dict)
            A dictionary containing the bias factors for each parameter.

    Returns:
        pd.DataFrame: The bias-corrected posterior samples.
    """
    logger.info(f"Correcting posterior samples for bias ...")
    corrected_samples = posterior_samples.copy()

    for param in corrected_samples.columns:
        if param in bias_factors:
            slope = bias_factors[param]['inferred-mode_mode-ratio_slope']
            intercept = bias_factors[param]['inferred-mode_mode-ratio_intercept']

            counts, bin_edges = np.histogram(corrected_samples[param], bins=50, density=False)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            posterior_mode = bin_centers[np.argmax(counts)]

            pred_bias_factor = slope * posterior_mode + intercept
            corrected_samples[param] = corrected_samples[param] / pred_bias_factor

    logger.info(f"Bias correction applied to posterior samples.")
    return corrected_samples


def extract_marginalized_posteriors(
        results_fname: str = None,
        results_dir: str = f'../{PROJECT_DIR}/results/{TODAY}/',
        outdir: str = f'../{PROJECT_DIR}/results/{TODAY}/',
        save_to_outdir: bool = False,
        plot_waveforms: bool = False,
        apply_bias_correction: bool = False,
        bias_factors_dict: dict = None,
        fontsize=15, labelsize=13,
        nolog=False, force=False, **kwargs):
    """
    Extract marginalized posterior samples for the specified parameters from a Bilby result object.
    These posteriors are then plotted as single parameter marginalized histograms, with the true
    injection value shown as a vertical line.
    Then marginalied posteriors are also calculated for the derived parameters, chirp_mass and chi_eff!
    All these plots are saved to the same directory as the result object, with the derived quantity
    posteriors also saved to a JSON file for later analysis.

    Arguments
    ---------
    results_fname : str
        Path to the Bilby result JSON file.
    results_dir : str
        Directory containing the Bilby result JSON files.
    label : str
        Label used to identify the result files.
    pe_run_type : str
        Type of parameter estimation run, e.g., 'ml2ml', 'eob2ml', 'eob2eob'.
    sampler : str
        Sampler used for the parameter estimation, e.g., 'nessai', 'dynesty', 'pocomc'.
    outdir : str
        Directory to save the marginalized posterior plots and JSON files.
    save_to_outdir : bool
        If True, save the marginalized posterior plots and JSON files to the specified outdir.
    plot_waveforms : bool
        If True, plot the injected and recovered waveforms for each parameter estimation run.
    apply_bias_correction : bool
        If True, apply bias correction to the posterior samples.
    bias_factors_dict : dict
        Dictionary containing bias correction factors for the posterior samples.
    fontsize : int
        Font size for the plots.
    labelsize : int
        Label size for the plots.
    nolog : bool
        If True, suppress logging output.
    force : bool
        If True, overwrite existing files.
    """
    if nolog:
        logger.setLevel(logging.WARNING)
    logger.info(f"Extracting marginalized posteriors from results in: {results_dir}")

    if results_fname is None:
        raise ValueError("results_fname must be provided to extract marginalized posteriors.")
    elif not results_fname.endswith('.json'):
        results_fname += '.json'
    fname = os.path.join(results_dir, results_fname)
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Results file not found: {fname}")

    savedir = outdir if save_to_outdir else results_dir
    ensure_dir(savedir)
    
    # -- Check if extracted distance and ratio files already exist, and skip if they do!
    distances_fname = os.path.join(savedir, results_fname.replace('_result.json', '_param_distances.json'))
    ratios_fname = os.path.join(savedir, results_fname.replace('_result.json', '_param_ratios.json'))
    if os.path.isfile(distances_fname) and os.path.isfile(ratios_fname) \
        and not force and not apply_bias_correction:
        logger.info(f"Parameter distances and ratios files already exist: {distances_fname}, {ratios_fname}.")
        logger.info("Skipping extraction.")
        return

    result = bilby.gw.result.CBCResult.from_json(fname)
    logger.info(f"Loaded Bilby result from: {fname}")

    posterior = result.posterior
    injection_parameters = result.injection_parameters

    if apply_bias_correction:
        if bias_factors_dict is None:
            logger.warning("Bias factors dictionary must be provided if apply_bias_correction is True.")
            logger.warning("Proceeding without bias correction.")

        # Check if bias correction was already performed
        shifted_posterior_fname = os.path.join(savedir, results_fname.replace('_result.json', '_shifted_posterior.json'))
        if os.path.isfile(shifted_posterior_fname) and not force:
            logger.info(f"Bias-corrected posterior samples already exist: {shifted_posterior_fname}.")
            logger.info("Skipping bias correction.")
            # TODO: Write a bias-correction DONE file instead!
            return

        posterior = correct_posterior_bias(result.posterior, bias_factors_dict)
        logger.info("Bias correction applied to posterior samples.")

        # Save the bias-corrected posterior samples to a new JSON file
        logger.info(f"Saving bias-corrected posterior samples to: {shifted_posterior_fname}")
        save_json(posterior, shifted_posterior_fname)

    # Extract marginalized posteriors for the specified parameters
    parameters_of_interest = ["mass_1", "mass_2", "chi_1", "chi_2"]
    marginalized_posteriors = {param: posterior[param].values 
                               for param in parameters_of_interest if param in posterior.columns}
    
    # Extract derived parameters: chirp_mass and chi_eff
    marginalized_posteriors["chirp_mass"] = calculate_chirp_mass(posterior["mass_1"], posterior["mass_2"])
    marginalized_posteriors["chi_eff"] = calculate_chi_eff(posterior["mass_1"],  posterior["mass_2"], 
                                                           posterior["chi_1"], posterior["chi_2"])
    
    # Calculate true values for derived parameters
    injection_parameters["chirp_mass"] = calculate_chirp_mass(injection_parameters["mass_1"], injection_parameters["mass_2"])
    injection_parameters["chi_eff"] = calculate_chi_eff(injection_parameters["mass_1"], injection_parameters["mass_2"], 
                                                        injection_parameters["chi_1"], injection_parameters["chi_2"])
    
    # Save marginalized posteriors to JSON file
    post_fname = os.path.join(savedir, results_fname.replace('_result.json', '_marginalized_posteriors.json'))
    if apply_bias_correction:
        post_fname = post_fname.replace('.json', '_shifted.json')
    logger.info(f"Saving marginalized posteriors to: {post_fname}")
    save_json(marginalized_posteriors, post_fname)

    # Store the distance and ratio, of posterior mode and median, from the true value, for each param!
    param_distances = {}
    param_ratios = {}
    for i, param in enumerate(marginalized_posteriors):
        if param in injection_parameters:
            true_value = injection_parameters[param]

            # mode = scipy.stats.mode(marginalized_posteriors[param], keepdims=True).mode[0]
            # median = np.median(marginalized_posteriors[param])

            # -- Calculate mode, mean and median of the data by first taking the average of all histogram bins,
            # and then reporting the statistics of this data. This ensures that the mode reported will be the
            # average value of the bin with the highest count, rather than some singleton value elsewhere in the data.
            counts, bin_edges = np.histogram(marginalized_posteriors[param], bins=50, density=False)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mode = bin_centers[np.argmax(counts)]
            mean = np.average(bin_centers, weights=counts)
            median = np.median(np.repeat(bin_centers, counts))
            
            # Calculate distance from true value
            param_distances[param] = {
                "mode": np.abs(mode - true_value),
                "median": np.abs(median - true_value)
            }
            # Calculate ratio from true value
            param_ratios[param] = {
                "mode": mode / true_value if true_value != 0 else np.inf,
                "median": median / true_value if true_value != 0 else np.inf
            }

    distances_fname = os.path.join(savedir, results_fname.replace('_result.json', '_param_distances.json'))
    ratios_fname = os.path.join(savedir, results_fname.replace('_result.json', '_param_ratios.json'))
    if apply_bias_correction:
        distances_fname = distances_fname.replace('.json', '_shifted.json')
        ratios_fname = ratios_fname.replace('.json', '_shifted.json')
    logger.info(f"Saving parameter distances to: {distances_fname}")
    logger.info(f"Saving parameter ratios to: {ratios_fname}")
    save_json(param_distances, distances_fname)
    save_json(param_ratios, ratios_fname)

    # Plot marginalized posteriors with true injection values
    for i, param in enumerate(marginalized_posteriors):
        fig, ax = plt.subplots(figsize=(8, 6))

        plt.hist(marginalized_posteriors[param], bins=50, density=True, 
                 alpha=0.7, label='Posterior', edgecolor='black')
        if param in injection_parameters:
            plt.axvline(injection_parameters[param], color='r', linestyle='--', label='True Value')
        plt.xlabel(LATEX_LABELS.get(param, param), fontsize=fontsize)
        plt.ylabel('Probability Density', fontsize=fontsize)
        plt.legend(loc='upper right', fontsize=labelsize-2)
        ax.tick_params(which="both", direction='in', top=True, right=True)
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.tick_params(labelsize=labelsize)

        # Put mode, median of posterior, and true value in the plot
        true_value = injection_parameters[param]
        mode = param_distances[param]["mode"] + true_value
        median = param_distances[param]["median"] + true_value
        textstr = f'Mode: {mode:.3f}\nMedian: {median:.3f}\nTrue: {true_value:.3f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=labelsize-2, verticalalignment='top', bbox=props)

        plot_fname = os.path.join(savedir, results_fname.replace('_result.json', f'_{param}_posterior.png'))
        if apply_bias_correction:
            plot_fname = plot_fname.replace('.png', '_shifted.png')
        logger.info(f"Saving marginalized posterior plot for {param} to: {plot_fname}")
        plt.tight_layout()
        plt.savefig(plot_fname, dpi=300, bbox_inches='tight')
        plt.close()
    logger.info("Completed extraction and plotting of marginalized posteriors.")

    # Plot original injected EOB waveform and recovery ML and EOB waveforms at posterior median/mode
    if plot_waveforms:
        logger.info("Generating waveform comparison plots for injected EOB waveform and recovered ML/EOB waveforms at posterior median/mode...")
        eob_generator = make_wf_generator("eob")
        wfkwargs = {'wfmodel_modelpath': kwargs.get('model_path'),
                    'wfmodel_configpath': kwargs.get('config_path'),
                    'calibrator_modelpath': kwargs.get('calmodel_path'),
                    'distance_scale_factor': injection_parameters.get('luminosity_distance', LUMINOSITY_DISTANCE)
                    }
        ml_generator = make_wf_generator("ml", wfkwargs=wfkwargs)

        # global PLOT_CONDITIONED_WAVEFORM
        # PLOT_CONDITIONED_WAVEFORM = True  # Enable plotting of conditioned waveforms

        post_median_params = {param: np.median(marginalized_posteriors[param]) for param in marginalized_posteriors}
        post_mode_params = {param: scipy.stats.mode(marginalized_posteriors[param], keepdims=True).mode[0] 
                            for param in marginalized_posteriors}

        logger.info(f"Injection parameters: {injection_parameters}")
        logger.info(f"Posterior median parameters: {post_median_params}")
        logger.info(f"Posterior mode parameters: {post_mode_params}")

        plot_waveforms_comparison(eob_generator, ml_generator, 
                                  injection_parameters, post_median_params, post_mode_params,
                                  outdir=savedir, 
                                  label=results_fname.replace('_result.json', '')
                                  )
        plot_waveforms_comparison(eob_generator, ml_generator, 
                                  injection_parameters, post_median_params, post_mode_params,
                                  outdir=savedir, 
                                  zoomed=True,
                                  label=results_fname.replace('_result.json', '_zoomed')
                                  )

        # Change luminosity distance to default value of 400 Mpc, and replot zoomed waveforms for comparison
        injection_parameters['luminosity_distance'] = LUMINOSITY_DISTANCE
        post_median_params['luminosity_distance'] = LUMINOSITY_DISTANCE
        post_mode_params['luminosity_distance'] = LUMINOSITY_DISTANCE
        wfkwargs['distance_scale_factor'] = LUMINOSITY_DISTANCE
        ml_generator = make_wf_generator("ml", wfkwargs=wfkwargs)

        plot_waveforms_comparison(eob_generator, ml_generator, 
                                  injection_parameters, post_median_params, post_mode_params,
                                  outdir=savedir, 
                                  zoomed=True,
                                  label=results_fname.replace('_result.json', '_zoomed_d0p4k')
                                  )
    return injection_parameters, marginalized_posteriors, param_distances, param_ratios


def plot_waveforms_comparison(eob_generator, ml_generator,
                              inj_params, post_median_params, post_mode_params,
                              outdir, label, 
                              zoomed=False,):
    """
    Plot EOB and ML waveforms at injection parameters, posterior median, and posterior mode, for comparison.
    """
    logger.debug(f"Generating waveforms for injection parameters: {inj_params}")
    time_array = np.arange(0, DURATION, 1/SAMPLE_RATE)

    if zoomed:
        # Zoom to 5-7 second window for better visualization
        start_idx = int(5/8 * len(time_array))
        end_idx = int(7/8 * len(time_array))
        time_array = time_array[start_idx:end_idx]

    # copy fixed parameters from injection params to posterior median and mode params, if they are not present
    for param in ['theta_jn', 'psi', 'ra', 'dec', 'phase', 'luminosity_distance']:
        if param in inj_params:
            if param not in post_median_params:
                post_median_params[param] = inj_params[param]
            if param not in post_mode_params:
                post_mode_params[param] = inj_params[param]

    titles = ['Inj', 'Post-Median', 'Post-Mode']

    # Plot EOB v/s ML at different param sets: injection, posterior median, posterior mode
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for i, param_arr in enumerate([inj_params, post_median_params, post_mode_params]):
        logger.debug(f"Generating waveforms for parameter set {i}: {param_arr}")

        h_eob = eob_generator.time_domain_strain(param_arr)
        logger.debug("Generated EOB injection waveform.")
        h_ml = ml_generator.time_domain_strain(param_arr)
        logger.debug("Generated ML injection waveform.")

        if zoomed:
            h_eob_wf = h_eob['plus'][start_idx:end_idx]
            h_ml_wf = h_ml['plus'][start_idx:end_idx]
        else:
            h_eob_wf = h_eob['plus']
            h_ml_wf = h_ml['plus']

        ax[i].plot(time_array, h_eob_wf, label='EOB', color='blue', alpha=0.75)
        ax[i].plot(time_array, h_ml_wf, label='ML', color='orange', alpha=0.75)
        ax[i].set_title(f'{titles[i]}:  ' \
            f'$m_1={param_arr["mass_1"]:.2f}$, $m_2={param_arr["mass_2"]:.2f}$, ' \
            f'$\chi_1={param_arr["chi_1"]:.2f}$, $\chi_2={param_arr["chi_2"]:.2f}$, ' \
            f'$d_L={param_arr["luminosity_distance"]:.2f}$ Mpc', fontsize=12)
        ax[i].set_xlabel('Time (s)', fontsize=15)
        ax[i].set_ylabel('$h_+(t)$ Strain', fontsize=15)

        # Add mismatch value between this pair of waveforms
        mismatch = calc_polarization_mismatch(h_eob_wf, h_ml_wf)
        ax[i].text(0.05, 0.9, f'Mismatch: {mismatch:.3e}', 
                   transform=ax[i].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        ax[i].legend()
        ax[i].tick_params(which="both", direction='in', top=True, right=True)

    plt.tight_layout()
    plot_fname = os.path.join(outdir, f'{label}_waveforms_comparison.png')
    logger.info(f"Saving waveform comparison plot to: {plot_fname}")
    plt.savefig(plot_fname, dpi=300, bbox_inches='tight')
    plt.close()

    # Now, plot EOB at injection param, v/s ML at diff param sets: injection, posterior median, posterior mode
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    h_eob_inj = eob_generator.time_domain_strain(inj_params)['plus']
    for i, param_arr in enumerate([inj_params, post_median_params, post_mode_params]):
        logger.debug(f"Generating ML waveform for parameter set {i}: {param_arr}")

        h_ml = ml_generator.time_domain_strain(param_arr)

        if zoomed:
            h_eob_wf = h_eob_inj[start_idx:end_idx]
            h_ml_wf = h_ml['plus'][start_idx:end_idx]
        else:
            h_eob_wf = h_eob_inj
            h_ml_wf = h_ml['plus']

        ax[i].plot(time_array, h_eob_wf, label='EOB@Inj', color='blue', alpha=0.75)
        ax[i].plot(time_array, h_ml_wf, label=f'ML@{titles[i]}', color='orange', alpha=0.75)
        ax[i].set_title(f'{titles[i]}:  ' \
            f'$m_1={param_arr["mass_1"]:.2f}$, $m_2={param_arr["mass_2"]:.2f}$, ' \
            f'$\chi_1={param_arr["chi_1"]:.2f}$, $\chi_2={param_arr["chi_2"]:.2f}$, ' \
            f'$d_L={param_arr["luminosity_distance"]:.2f}$ Mpc', fontsize=12)
        ax[i].set_xlabel('Time (s)', fontsize=15)
        ax[i].set_ylabel('$h_+(t)$ Strain', fontsize=15)

        mismatch = calc_polarization_mismatch(h_eob_wf, h_ml_wf)
        ax[i].text(0.05, 0.9, f'Mismatch: {mismatch:.3e}', 
                   transform=ax[i].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        ax[i].legend()
        ax[i].tick_params(which="both", direction='in', top=True, right=True)

    plt.tight_layout()
    plot_fname = os.path.join(outdir, f'{label}_waveforms_comparison_eob_inj.png')
    logger.info(f"Saving waveform comparison plot (EOB injection) to: {plot_fname}")
    plt.savefig(plot_fname, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Completed waveform comparison plotting.")



def analyze_results(results_dir=f'../{PROJECT_DIR}/results/{TODAY}/',
                  label: str = 'umamipe',
                  pe_run_type: {'eob2eob', 'ml2ml', 'eob2ml'} = 'ml2ml',
                  sampler: {'nessai', 'dynesty', 'pocomc'} = 'nessai',
                  apply_bias_correction: bool = False,
                  outdir=None, force=False):
    """
    Extract the marginalized 1D posteriors from Bilby result objects, for all subdirs in the `results_dir`.
    And then from the distances and ratios of the posterior mode and median from the true injection values, for each parameter,
    calculate the mean and standard deviation of these distances and ratios across all injections. Also, plot the distribution of these distances and ratios for each parameter, and save the plots to the `results_dir`.
    """
    if outdir is None:
        outdir = results_dir
    savedir = os.path.join(outdir, f'post_summary_{NOW}/')
    ensure_dir(savedir)

    logger.info(f"Analyzing results in directory: {results_dir}")
    all_param_distances = {}
    all_param_ratios = {}
    all_inj_params = {}

    dirnames = []

    for subdir in tqdm(os.listdir(results_dir), desc="Processing"):
        if not os.path.isdir(os.path.join(results_dir, subdir)):
            continue

        if label in subdir and pe_run_type in subdir and sampler in subdir:
            logger.info(f"Found result directory: {subdir} for PP plot generation...")
        else:
            logger.debug(f"Skipping directory: {subdir} as it does not match label: {label}, pe_run_type: {pe_run_type}, sampler: {sampler}")
            continue

        for fname in os.listdir(os.path.join(results_dir, subdir)):
            if not fname.endswith('_result.json'):
                continue
            else:
                dirnames.append(subdir)
            logger.debug(f"Processing result file: {fname} in subdir: {subdir}")

            extract_marginalized_posteriors(
                results_fname=fname,
                results_dir=os.path.join(results_dir, subdir),
                nolog=True,
                force=force
            )

            distances_fname = os.path.join(os.path.join(results_dir, subdir), 
                                            fname.replace('_result.json', '_param_distances.json'))
            ratios_fname = os.path.join(os.path.join(results_dir, subdir), 
                                        fname.replace('_result.json', '_param_ratios.json'))
            distances = load_json(distances_fname)
            ratios = load_json(ratios_fname)

            for param in distances:
                if param not in all_param_distances:
                    all_param_distances[param] = []
                all_param_distances[param].append(distances[param])
            for param in ratios:
                if param not in all_param_ratios:
                    all_param_ratios[param] = []
                all_param_ratios[param].append(ratios[param])

            # -- Store injection parameters for later analysis
            injection_params = load_json(os.path.join(os.path.join(results_dir, subdir), 
                                                  fname.replace('_result.json', '_injection_parameters.json')))
            for param in injection_params:
                if param not in all_inj_params:
                    all_inj_params[param] = []
                all_inj_params[param].append(injection_params[param])

    logger.setLevel(logging.INFO)

    # Save derived quantities (chirp_mass and chi_eff) to `all_inj_params` for later analysis
    all_inj_params['chirp_mass'] = calculate_chirp_mass(np.array(all_inj_params['mass_1']), 
                                              np.array(all_inj_params['mass_2']))
    all_inj_params['chi_eff'] = calculate_chi_eff(np.array(all_inj_params['mass_1']),
                                        np.array(all_inj_params['mass_2']),
                                        np.array(all_inj_params['chi_1']), 
                                        np.array(all_inj_params['chi_2']))

    # Write metadata about the analysis to a JSON file
    analysis_metadata = {
        'results_dir': results_dir,
        'outdir': outdir,
        'savedir': savedir,
        'num_injections': sum(len(all_param_distances[param]) for param in all_param_distances),
        'parameters_analyzed': list(all_param_distances.keys()),
        'result_dirs': dirnames,
        'result_files': [os.path.join(subdir, fname) for subdir in dirnames 
                         for fname in os.listdir(os.path.join(results_dir, subdir)) 
                         if fname.endswith('_result.json')],
    }
    metadata_fname = os.path.join(savedir, f'{label}_{pe_run_type}_analysis_metadata_{NOW}.json')
    logger.info(f"Saving analysis metadata to: {metadata_fname}")
    save_json(analysis_metadata, metadata_fname)
    
    # Now calculate mean and std of distances and ratios for each parameter
    summary_distances = {}
    summary_ratios = {}
    for param in all_param_distances:
        modes = [d['mode'] for d in all_param_distances[param]]
        medians = [d['median'] for d in all_param_distances[param]]
        summary_distances[param] = {
            'mode_mean': np.mean(modes),
            'mode_std': np.std(modes),
            'median_mean': np.mean(medians),
            'median_std': np.std(medians)
        }
    for param in all_param_ratios:
        modes = [r['mode'] for r in all_param_ratios[param]]
        medians = [r['median'] for r in all_param_ratios[param]]
        summary_ratios[param] = {
            'mode_mean': np.mean(modes),
            'mode_std': np.std(modes),
            'median_mean': np.mean(medians),
            'median_std': np.std(medians)
        }
    logger.info(f"Summary of parameter distances: {summary_distances}")
    logger.info(f"Summary of parameter ratios: {summary_ratios}")

    # Save summary to JSON files
    summary_distances_fname = os.path.join(savedir, f'{label}_{pe_run_type}_summary_param_distances.json')
    summary_ratios_fname = os.path.join(savedir, f'{label}_{pe_run_type}_summary_param_ratios.json')
    logger.info(f"Saving summary of parameter distances to: {summary_distances_fname}")
    logger.info(f"Saving summary of parameter ratios to: {summary_ratios_fname}")
    save_json(summary_distances, summary_distances_fname)
    save_json(summary_ratios, summary_ratios_fname)

    # -- Save `y=mx+c` linear fit slope and intercept value for the 
    # -- inferred mode v/s mode ratio quantities, as the bias factor in a dictionary 
    # -- for each parameter. These values are used to correct the bias in the marginalized
    # -- posteriors of each parameter, by shifting the posterior samples by the bias factor.
    bias_factors = {}

    # Plot distributions of distances and ratios for each parameter
    for quantity, all_param_data in zip(['distances', 'ratios'], [all_param_distances, all_param_ratios]):
        for i, param in enumerate(all_param_data):
            bias_factors[param] = {}
            for xvalname in ['True', 'Inferred Mode', 'Inferred Median']:

                injection_values = all_inj_params[param]
                if xvalname == 'True':
                    xvals = injection_values
                elif xvalname == 'Inferred Mode':
                    dist_from_mode = [d['mode'] for d in all_param_distances[param]]
                    xvals = injection_values - np.array(dist_from_mode)
                elif xvalname == 'Inferred Median':
                    dist_from_median = [d['median'] for d in all_param_distances[param]]
                    xvals = injection_values - np.array(dist_from_median)

                if quantity == 'distances':
                    modes = [d['mode'] for d in all_param_distances[param]]
                    medians = [d['median'] for d in all_param_distances[param]]
                    xlabel = f'Distance from {xvalname} Value'
                else:
                    modes = [r['mode'] for r in all_param_ratios[param]]
                    medians = [r['median'] for r in all_param_ratios[param]]
                    xlabel = f'Ratio of {xvalname} to True Value'

                bins = 30
                mode_counts, mode_edges = np.histogram(modes, bins=bins)
                mode_max_bin_idx = np.argmax(mode_counts)
                mode_peak_center = 0.5 * (mode_edges[mode_max_bin_idx] + mode_edges[mode_max_bin_idx + 1])

                median_counts, median_edges = np.histogram(medians, bins=bins)
                median_max_bin_idx = np.argmax(median_counts)
                median_peak_center = 0.5 * (median_edges[median_max_bin_idx] + median_edges[median_max_bin_idx + 1])

                fig, ax = plt.subplots(figsize=(8, 6))
                plt.hist(modes, bins=bins, alpha=0.5,  edgecolor='black',
                        label=f'with post mode (peak={mode_peak_center:.2f})')
                plt.hist(medians, bins=bins, alpha=0.5,  edgecolor='black',
                        label=f'with post median (peak={median_peak_center:.2f})')
                plt.xlabel(f'{xlabel} for {LATEX_LABELS.get(param, param)}', fontsize=15)
                plt.ylabel('Count', fontsize=15)
                plt.legend(title=f'N={len(medians)}', loc='upper right', fontsize=13, title_fontsize=15)
                ax.tick_params(which="both", direction='in', top=True, right=True)
                ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
                ax.tick_params(labelsize=13)
                # ax.text(0.05, 0.95, f'Max (with mode): {np.max(modes):.3f}\nMax (with median): {np.max(medians):.3f}',
                #         transform=ax.transAxes, fontsize=13, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                plot_fname = os.path.join(savedir, f'{param}_{quantity}_distribution_{NOW}.png')
                logger.info(f"Saving {quantity} distribution plot for {param} to: {plot_fname}")
                plt.tight_layout()
                plt.savefig(plot_fname, dpi=300, bbox_inches='tight')
                plt.close()

                # Now plot True value v/s distance/ratio from true value for different params, across all injections!
                fig, ax = plt.subplots(figsize=(8, 6))
                if quantity == 'distances':
                    plt.scatter(xvals,
                                [d['mode'] for d in all_param_distances[param]], 
                                marker='o', color='blue', alpha=0.5, label='Post Mode Distance')
                    plt.scatter(xvals,
                                [d['median'] for d in all_param_distances[param]], 
                                marker='s', color='red', alpha=0.5, label='Post Median Distance')
                    # plot horizontal line at y=0 to indicate perfect inference
                    plt.axhline(0, color='black', linestyle='--', label='Perfect Inference')
                    plt.ylabel(f'Distance from True value for {LATEX_LABELS.get(param, param)}', fontsize=15)

                else:
                    plt.scatter(xvals,
                                [r['mode'] for r in all_param_ratios[param]], 
                                marker='o', color='blue', alpha=0.5, label='Post Mode Ratio')
                    plt.scatter(xvals,
                                [r['median'] for r in all_param_ratios[param]], 
                                marker='s', color='red', alpha=0.5, label='Post Median Ratio')
                    
                    # plot horizontal line at y=1 to indicate perfect inference
                    plt.axhline(1, color='black', linestyle='--', label='Perfect Inference')
                    plt.ylabel(f'Ratio of Inferred to True value for {LATEX_LABELS.get(param, param)}', fontsize=15)

                    # -- now, plot a linear fit line, `y=mx+c`
                    for type, color in zip(['mode', 'median'], ['green', 'purple']):
                        x = np.array(xvals)
                        y = np.array([r[type] for r in all_param_ratios[param]])
                        m, c = np.polyfit(x, y, 1)
                        plt.plot(x, m*x + c, color=color, linestyle='--', label=f'$y={m:.5f}x+{c:.2f}$ ({type})')

                        # -- Store slope, intercept with the mode, median ratios!
                        bias_factors[param][f'{xvalname.lower().replace(" ", "-")}_{type}-ratio_slope'] = m
                        bias_factors[param][f'{xvalname.lower().replace(" ", "-")}_{type}-ratio_intercept'] = c

                plt.xlabel(f'{xvalname} Value of {LATEX_LABELS.get(param, param)}', fontsize=15)
                plt.legend(title=f'N={len(medians)}', loc='upper right', fontsize=10, title_fontsize=12)
                ax.tick_params(which="both", direction='in', top=True, right=True)
                ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
                ax.tick_params(labelsize=13)
                plot_fname = os.path.join(savedir, f'{param}_{xvalname.lower().replace(" ", "-")}_vs_{quantity}_{NOW}.png')
                logger.info(f"Saving {xvalname} value v/s {quantity} plot for {param} to: {plot_fname}")
                plt.tight_layout()
                plt.savefig(plot_fname, dpi=300, bbox_inches='tight')
                plt.close()

    # Save bias factors to JSON file
    bias_factors_fname = os.path.join(savedir, f'{label}_{pe_run_type}_bias_factors_{NOW}.json')
    logger.info(f"Saving bias factors to: {bias_factors_fname}")
    save_json(bias_factors, bias_factors_fname)

    # Now, apply bias correction to the marginalized posteriors, and resave them!
    if apply_bias_correction:
        logger.info("Applying bias correction to marginalized posteriors...")
        # print(f"List of subdirectories to process for bias correction: {dirnames}")
        for subdir in dirnames:
            logger.debug(f"Processing subdir: {subdir} for bias correction...")
            for fname in os.listdir(os.path.join(results_dir, subdir)):
                if not fname.endswith('_result.json'):
                    continue
                logger.debug(f"Applying bias correction to result file: {fname} in subdir: {subdir}")
                extract_marginalized_posteriors(
                    results_fname=fname,
                    results_dir=os.path.join(results_dir, subdir),
                    nolog=True,
                    force=force,
                    apply_bias_correction=True,
                    bias_factors_dict=bias_factors
                )

    logger.info("Completed analysis of results and plotting of parameter distance distributions.")


# ------
# PP plots
# ------


def calculate_chirp_mass(mass_1, mass_2):
    mass_1 = np.asarray(mass_1)
    mass_2 = np.asarray(mass_2)
    return (mass_1 * mass_2) ** (3.0 / 5.0) / (mass_1 + mass_2) ** (1.0 / 5.0)


def calculate_chi_eff(mass_1, mass_2, chi_1, chi_2):
    mass_1 = np.asarray(mass_1)
    mass_2 = np.asarray(mass_2)
    chi_1 = np.asarray(chi_1)
    chi_2 = np.asarray(chi_2)
    return (mass_1 * chi_1 + mass_2 * chi_2) / (mass_1 + mass_2)


def add_derived_parameters_to_result(result):
    """
    Return a copied Bilby result with derived posterior columns and
    derived injection parameters added.

    No likelihood object is needed for PP plots.
    """
    r = copy.deepcopy(result)
    post = r.posterior.copy()

    post["chirp_mass"] = calculate_chirp_mass(
        post["mass_1"],
        post["mass_2"],
    )
    post["chi_eff"] = calculate_chi_eff(
        post["mass_1"],
        post["mass_2"],
        post["chi_1"],
        post["chi_2"],
    )

    r.posterior = post

    inj = dict(r.injection_parameters)
    inj["chirp_mass"] = float(
        calculate_chirp_mass(
            inj["mass_1"],
            inj["mass_2"],
        )
    )
    inj["chi_eff"] = float(
        calculate_chi_eff(
            inj["mass_1"],
            inj["mass_2"],
            inj["chi_1"],
            inj["chi_2"],
        )
    )
    r.injection_parameters = inj
    return r


def order_component_samples(df):
    """
    Convert posterior samples to ordered primary-secondary convention.

    If mass_2 > mass_1 for a sample, swap:
        mass_1 <-> mass_2
        chi_1  <-> chi_2

    This preserves the physical component pairing.
    """
    out = df.copy()
    swap = out["mass_2"].to_numpy() > out["mass_1"].to_numpy()

    m1 = out["mass_1"].to_numpy().copy()
    m2 = out["mass_2"].to_numpy().copy()
    out.loc[swap, "mass_1"] = m2[swap]
    out.loc[swap, "mass_2"] = m1[swap]

    if "chi_1" in out.columns and "chi_2" in out.columns:
        chi1 = out["chi_1"].to_numpy().copy()
        chi2 = out["chi_2"].to_numpy().copy()
        out.loc[swap, "chi_1"] = chi2[swap]
        out.loc[swap, "chi_2"] = chi1[swap]
    return out


def order_injection_parameters(inj):
    """
    Convert injection dictionary to ordered primary-secondary convention.
    """
    p = dict(inj)

    if p["mass_2"] > p["mass_1"]:
        p["mass_1"], p["mass_2"] = p["mass_2"], p["mass_1"]

        if "chi_1" in p and "chi_2" in p:
            p["chi_1"], p["chi_2"] = p["chi_2"], p["chi_1"]
    return p


def make_ordered_result(result):
    """
    Return a copied Bilby result with ordered posterior samples and ordered
    injection parameters.
    """
    r = copy.deepcopy(result)
    r.posterior = order_component_samples(r.posterior)
    r.injection_parameters = order_injection_parameters(r.injection_parameters)
    return r


def swap_dataframe_component_labels(df):
    out = df.copy()

    pairs = [
        ("mass_1", "mass_2"),
        ("chi_1", "chi_2"),
        ("spin_1z", "spin_2z"),
        ("a_1", "a_2"),
        ("tilt_1", "tilt_2"),
    ]

    for a, b in pairs:
        if a in out.columns and b in out.columns:
            tmp = out[a].copy()
            out[a] = out[b].copy()
            out[b] = tmp

    if "mass_1" in out.columns and "mass_2" in out.columns:
        out["mass_ratio"] = out["mass_2"] / out["mass_1"]
        out["chirp_mass"] = calculate_chirp_mass(out["mass_1"], out["mass_2"])

    if all(k in out.columns for k in ["mass_1", "mass_2", "chi_1", "chi_2"]):
        out["chi_eff"] = calculate_chi_eff(
            out["mass_1"],
            out["mass_2"],
            out["chi_1"],
            out["chi_2"],
        )
    return out


def swap_dict_component_labels(d):
    out = dict(d)

    pairs = [
        ("mass_1", "mass_2"),
        ("chi_1", "chi_2"),
        ("spin_1z", "spin_2z"),
        ("a_1", "a_2"),
        ("tilt_1", "tilt_2"),
    ]

    for a, b in pairs:
        if a in out and b in out:
            out[a], out[b] = out[b], out[a]

    if "mass_1" in out and "mass_2" in out:
        out["mass_ratio"] = out["mass_2"] / out["mass_1"]
        out["chirp_mass"] = float(calculate_chirp_mass(out["mass_1"], out["mass_2"]))

    if all(k in out for k in ["mass_1", "mass_2", "chi_1", "chi_2"]):
        out["chi_eff"] = float(
            calculate_chi_eff(out["mass_1"], out["mass_2"], out["chi_1"], out["chi_2"])
        )
    return out


def make_swapped_label_result(result, swap_injection=False, swap_nested_samples=True):
    """
    Return a copied Bilby result with component labels swapped.

    Default behavior:
        posterior mass_1 <-> mass_2
        posterior chi_1  <-> chi_2
        injection parameters unchanged

    This is the correct diagnostic if the recovered posterior labels are suspected
    to be swapped relative to the physical injection labels.

    Set swap_injection=True only if the stored injection labels themselves are
    known to be swapped incorrectly.
    """
    r = copy.deepcopy(result)

    r.posterior = swap_dataframe_component_labels(r.posterior)

    if swap_nested_samples and hasattr(r, "nested_samples"):
        if r.nested_samples is not None:
            r.nested_samples = swap_dataframe_component_labels(r.nested_samples)

    if swap_injection:
        r.injection_parameters = swap_dict_component_labels(r.injection_parameters)
    else:
        r.injection_parameters = dict(r.injection_parameters)
    return r


def print_quantile_summary(results, keys=("mass_1", "mass_2", "chi_1", "chi_2")):
    rows = []
    for i, r in enumerate(results):
        row = {"i": i}
        for key in keys:
            if key in r.posterior.columns and key in r.injection_parameters:
                row[f"q_{key}"] = np.mean(
                    r.posterior[key].to_numpy() < r.injection_parameters[key]
                )
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.describe())
    return df


def make_pp_plots(results_dir: str = f'../{PROJECT_DIR}/results/{TODAY}/',
                  label: str = 'umamipe',
                  pe_run_type: {'eob2eob', 'ml2ml', 'eob2ml'} = 'ml2ml',
                  sampler: {'nessai', 'dynesty', 'pocomc'} = 'nessai',
                  apply_bias_correction: bool = False,
                  max_injections_to_include: int = None,
                  outdir: str = f'../{PROJECT_DIR}/results/{TODAY}/',
                  save_to_outdir: bool = False):
    """
    Make PP plots for a list of Bilby CBCResult objects.
    """
    if save_to_outdir:
        outdir = os.path.join(outdir, f'pp_plots_{NOW}/')
    else:
        outdir = os.path.join(results_dir, f'pp_plots_{NOW}/')
    ensure_dir(outdir)
    
    results = []
    for dirname in os.listdir(results_dir):
        if label in dirname and pe_run_type in dirname and sampler in dirname:
            for fname in os.listdir(f"{results_dir}/{dirname}"):
                if fname.endswith('result.json'):
                    logger.info(f"Found result directory: {dirname} for PP plot generation...")
                    result = bilby.gw.result.CBCResult.from_json(f"{results_dir}/{dirname}/{fname}")

                    if apply_bias_correction:
                        # Read bias shifted posteriors from `~shifted_posteriors.json` file, if it exists
                        shifted_post_fname = os.path.join(results_dir, dirname, 
                                                          fname.replace('_result.json', '_shifted_posteriors.json'))
                        
                        if not os.path.exists(shifted_post_fname):
                            logger.debug(f"Skipping bias correction for result: {fname}. No shifted posteriors file found.")
                            continue

                        logger.info(f"Found shifted posteriors at: {shifted_post_fname}")
                        shifted_posterior = load_json(shifted_post_fname)
                        result.posterior = shifted_posterior

                        # Make new corner plot with bias-corrected posteriors
                        corner_plot_fname = os.path.join(outdir,
                                                         f"{label}_{pe_run_type}_{sampler}_corner_bias_corrected.png")
                        logger.info(f"Saving corner plot with bias-corrected posteriors to: {corner_plot_fname}")
                        result.plot_corner(filename=corner_plot_fname, save=True)

                    results.append(result)
                    logger.debug(f"Loaded result from {results_dir}/{dirname}/{fname} for PP plot generation...")
                    if max_injections_to_include is not None and len(results) >= max_injections_to_include:
                        logger.info(f"Reached max_injections_to_include={max_injections_to_include}. Breaking out!")
                        break

    logger.info(f"Loaded {len(results)} results from {results_dir} for PP plot generation...")

    savename = os.path.join(outdir, f"{label}_{pe_run_type}_{sampler}_pp-plot_{NOW}.png")
    fig, pvals = make_pp_plot(
        results,
        filename=savename,
        save=True,
    )
    print("Combined p-value:", pvals.combined_pvalue)

    # -- Save PP plot metadata to JSON file
    pp_metadata = {
        "results_dir": results_dir,
        "label": label,
        "pe_run_type": pe_run_type,
        "sampler": sampler,
        "num_results": len(results),
        "pvalues": pvals.pvalues,
        "combined_pvalue": pvals.combined_pvalue,
        "result_dirs": [dirname for dirname in os.listdir(results_dir) if label in dirname and pe_run_type in dirname and sampler in dirname],
        "result_fnames": [fname for dirname in os.listdir(results_dir) if label in dirname and pe_run_type in dirname and sampler in dirname for fname in os.listdir(f"{results_dir}/{dirname}") if fname.endswith('result.json')],
        "timestamp": NOW,
    }
    with open(savename.replace('.png', '.json'), 'w') as f:
        json.dump(pp_metadata, f, indent=4)
    logger.info(f"Saved PP plot for {len(results)} results to {savename}")

    # -- Make ordered parameter PP plot
    ordered_results = [
        make_ordered_result(r)
        for r in results
    ]

    savename_ordered = os.path.join(
        outdir, 
        f"{label}_{pe_run_type}_{sampler}_pp-plot_ordered_{NOW}.png"
        )
    fig, pvals = make_pp_plot(
        ordered_results,
        keys=["mass_1", "mass_2", "chi_1", "chi_2"],
        filename=savename_ordered,
        save=True,
    )
    logger.info(f"Saved ordered-parameter PP plot for {len(ordered_results)} results to {savename_ordered}")

    # -- Do Posterior label swapping, for a check
    swapped_results = [
        make_swapped_label_result(r, swap_injection=False)
        for r in results
    ]

    print("Quantiles after posterior-only label swap:")
    df_swapped = print_quantile_summary(
        swapped_results,
        keys=("mass_1", "mass_2", "chi_1", "chi_2"),
    )

    savename_swapped = os.path.join(
        outdir,
        f"{label}_{pe_run_type}_{sampler}_pp-plot_swapped_labels_{NOW}.png",
    )
    fig, pvals = make_pp_plot(
        swapped_results,
        keys=["mass_1", "mass_2", "chi_1", "chi_2"],
        filename=savename_swapped,
        save=True,
    )
    logger.info(f"Saved swapped-label PP plot for {len(swapped_results)} results to {savename_swapped}")

    # -- Debug PP plot and results
    rows = []
    for i, r in enumerate(results):
        inj = r.injection_parameters
        post = r.posterior
        row = {
            "i": i,
            "label": r.label,
            "inj_m1": inj["mass_1"],
            "inj_m2": inj["mass_2"],
            "inj_chi1": inj["chi_1"],
            "inj_chi2": inj["chi_2"],
            "m1_ge_m2_inj": inj["mass_1"] >= inj["mass_2"],
            "npost": len(post),
            "post_m1_median": post["mass_1"].median(),
            "post_m2_median": post["mass_2"].median(),
            "post_chi1_median": post["chi_1"].median(),
            "post_chi2_median": post["chi_2"].median(),
            "post_m1_lt_m2_count": int((post["mass_1"] < post["mass_2"]).sum()),
        }
        for p in ["mass_1", "mass_2", "chi_1", "chi_2"]:
            row[f"q_{p}"] = np.mean(post[p].to_numpy() < inj[p])
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df)
    print(df[["q_mass_1", "q_mass_2", "q_chi_1", "q_chi_2"]].describe())
    print("unique injections:", len(df.drop_duplicates(["inj_m1", "inj_m2", "inj_chi1", "inj_chi2"])))
    print("bad injection mass order:", (~df["m1_ge_m2_inj"]).sum())
    print("posterior mass-order violations:", df["post_m1_lt_m2_count"].sum())
    df.to_csv(savename.replace('.png', '_debug.csv'), index=False)
    print(f"Saved debug CSV for PP plot to {savename.replace('.png', '_debug.csv')}")

    # -- Make PP plot for derived parameters: chirp mass and effective spin 
    derived_keys = ["chirp_mass", "chi_eff"]

    derived_results = [
        add_derived_parameters_to_result(r)
        for r in results
    ]

    savename_derived = os.path.join(
        outdir, 
        f"{label}_{pe_run_type}_{sampler}_pp-plot_derived_{NOW}.png"
        )
    fig, pvals = make_pp_plot(
        derived_results,
        keys=derived_keys,
        filename=savename_derived,
        save=True,
    )
    print("Derived-parameter PP plot saved to:", savename_derived)
    print("Derived PP p-values:", pvals)
    print("Combined p-value for derived parameters:", pvals.combined_pvalue)


# ------
# Re-weighting
# ------


def read_ifos_from_file(fname: str, outdir: str = f'../{PROJECT_DIR}/results/'):
    if not fname.endswith('.pkl'):
        fname += '.pkl'
    ifos = bilby.gw.detector.InterferometerList.from_pickle(f"{outdir}/{fname}")
    logger.info(f"Loaded interferometers from {outdir}/{fname} for analysis...")
    return ifos

def add_fixed_parameters_to_result_posterior(result, fixed_parameters):
    """
    Bilby reweight evaluates likelihoods using rows from result.posterior.
    If fixed parameters are not columns in result.posterior, the waveform
    generator can fail with KeyError, e.g. KeyError: 'phase'.
    """
    for key, value in fixed_parameters.items():
        if key not in result.posterior.columns:
            result.posterior[key] = value
    return result

def debug_reweight_samples(result, new_likelihood, nsamp=100):
    logger.info(f"Debugging reweighting of posterior samples using new likelihood: {new_likelihood}")
    df = result.posterior.iloc[:nsamp].copy()

    logl_old = df["log_likelihood"].to_numpy()
    logl_new = []

    for _, row in df.iterrows():
        params = row.to_dict()
        ll = new_likelihood.log_likelihood(parameters=params)
        logl_new.append(ll)

    logl_new = np.asarray(logl_new)
    logw = logl_new - logl_old

    print("old logL finite:", np.isfinite(logl_old).sum(), "/", len(logl_old))
    print("new logL finite:", np.isfinite(logl_new).sum(), "/", len(logl_new))
    print("logw finite:", np.isfinite(logw).sum(), "/", len(logw))

    print("old logL min/median/max:", np.nanmin(logl_old), np.nanmedian(logl_old), np.nanmax(logl_old))
    print("new logL min/median/max:", np.nanmin(logl_new), np.nanmedian(logl_new), np.nanmax(logl_new))
    print("logw min/median/max:", np.nanmin(logw), np.nanmedian(logw), np.nanmax(logw))
    print("logw std:", np.nanstd(logw))

    finite = np.isfinite(logw)
    if finite.sum() == 0:
        print("ALL LOG WEIGHTS ARE NON-FINITE. Reweighting cannot work.")
        return df, logl_new, logw, None, 0.0

    logw_finite = logw[finite]
    logw_norm = logw_finite - logsumexp(logw_finite)
    w = np.exp(logw_norm)

    n_eff = 1.0 / np.sum(w**2)
    eff_frac = n_eff / len(w)

    print("N_eff:", n_eff)
    print("N_eff fraction:", eff_frac)
    print("max normalized weight:", np.max(w))
    print("top 5 weights:", np.sort(w)[-5:])
    return df, logl_new, logw, w, n_eff


def imp_reweight_posteriors(fname: str, 
                            outdir: str = f'../{PROJECT_DIR}/results/',
                            use_old_likelihood_from_file: bool = True,
                            use_nested_samples: bool = False,
                            eob_wf_gen_type: {'eob', 'eobbilby'} = 'eob',
                            use_bibly_reweight_function: bool = False,
                            npool: int = 8):
    """
    Do importance reweighting for a posterior using built-in `bilby.gw.core.result.reweight`
    function, to correct EOB2ML case posterior samples by appling rejection sampling using the
    log likelihood ratio between EOB and ML waveforms.
    """
    logger.info(f"Starting importance reweighting for posterior samples in {outdir}/{fname}...")
    logger.info(f"Supplied arguments: use_old_likelihood_from_file={use_old_likelihood_from_file}, use_nested_samples={use_nested_samples}, npool={npool}")
    if not fname.endswith('_result.json'):
        fname = fname.replace('.json', '_result.json')
    result = bilby.core.result.read_in_result(f"{outdir}/{fname}")
    logger.info(f"Loaded result from {outdir}/{fname} for importance reweighting...")

    # if use_old_likelihood_from_file:
    #     old_likelihood = result.likelihood
    logger.info(f"Read results object is: {result}")
    logger.info(f"Result injection parameters are: {result.injection_parameters}")
    logger.info(f"Result posterior columns are: {result.posterior.columns}")

    fixed_parameters = {
        "luminosity_distance": result.injection_parameters["luminosity_distance"],
        "theta_jn": result.injection_parameters["theta_jn"],
        "psi": result.injection_parameters["psi"],
        "phase": result.injection_parameters["phase"],
        "geocent_time": result.injection_parameters["geocent_time"],
        "ra": result.injection_parameters["ra"],
        "dec": result.injection_parameters["dec"],
    }
    result = add_fixed_parameters_to_result_posterior(result, fixed_parameters)
    logger.info(f"Updated result posterior columns after adding fixed parameters: {result.posterior.columns}")

    if use_nested_samples:
        # Update parameters in result.nested_samples as well, if available
        for key, value in fixed_parameters.items():
            if key not in result.nested_samples.columns:
                result.nested_samples[key] = value
        logger.info(f"Updated result nested_samples columns after adding fixed parameters: {result.nested_samples.columns}")

    ifos = read_ifos_from_file(fname=fname.replace('_result.json', '_ifos.pkl'), outdir=outdir)
    logger.info(f"Interferometer injection parameters are: {ifos[0].__dict__}")

    if eob_wf_gen_type == 'eobbilby':
        waveform_generator = make_wf_generator('eobbilby', 
                                            wf_source_model=lal_binary_black_hole_aligned_chi,param_converter=identity_parameter_conversion,)
    else:
        waveform_generator = make_wf_generator('eob')

    eob_likelihood = GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
    )

    # Set wfgenerator start_time to that of interferometer geocent_time
    waveform_generator.start_time = ifos[0].meta_data['parameters']['geocent_time']

    # logger.info("Running debug reweighting of posterior samples ...")
    # df_dbg, logl_eob_dbg, logw_dbg, w_dbg, neff_dbg = debug_reweight_samples(
    #     result,
    #     eob_likelihood,
    #     nsamp=200,
    # )

    # sample = result.posterior.iloc[0].to_dict()
    # print("sample keys:", sorted(sample.keys()))
    # eob_likelihood.parameters.update(sample)
    # print("new EOB logL:", eob_likelihood.log_likelihood())

    if use_bibly_reweight_function:
        new_result, new_logl, new_logpriors, _, _ = bilby.core.result.reweight(
            result=result,
            label="imp-reweighted",
            new_likelihood=eob_likelihood,
            npool=npool,
            verbose_output=True,
            n_checkpoint=5000,
            use_nested_samples=use_nested_samples,
        )
        new_result.plot_corner(save=True, filename=f"{outdir}/{fname.replace('_result.json', '_corner_imp-reweighted.png')}")
        savename = f"{outdir}/{fname.replace('_result.json', '_result_imp-reweighted.json')}"
        new_result.to_json(savename)
        logger.info(f"Saved importance reweighted result to {savename}")
        return
    
    logger.info("Starting manual importance reweighting of posterior samples...")

    from peanalysis import run_manual_eob_reweighting_workflow
    
    weighted_post, summary, diag = run_manual_eob_reweighting_workflow(
        eob2ml_result=result,
        eob_likelihood=eob_likelihood,
        parameters=["mass_1", "mass_2", "chi_1", "chi_2"],  # -- ignore other fixed params
        eob2eob_result=None,     # set to None if unavailable
        interval="90",                     # or "1sigma"
        proposal_logl_column="log_likelihood",
        checkpoint_csv="eob_reweighting_checkpoint.csv",
        out_prefix=fname.replace('_result.json', '_eob-reweighted'),
        outdir=outdir,
    )
    logging.info(f"Completed manual importance reweighting of posterior samples. Summary: {summary}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a CVAE model on GW waveforms")
    
    parser.add_argument('--label', type=str, default='umamipe',
                        help="Label for the analysis (default: umamipe)")
    parser.add_argument('--project-dir', type=str, choices=['cvae@taiwan', 'v0p1', '@alvin', '@korea'], 
                        default=PROJECT_DIR, help="Base directory for the project (default: %(default)s)")
    
    parser.add_argument('--model-config', type=str, default='modelconfig-cvae-paper-I',
                        help="Name of the model configuration JSON file (default: %(default)s)")
    parser.add_argument('--model-name', type=str, default='model-20251004_072338-10',
                        help="Name of the trained model checkpoint (default: %(default)s)")
    parser.add_argument('--calmodel-name', type=str, default='calibrator_model_20260623-010953_epoch74.pt',
                        help="Name of the trained calibration model checkpoint (default: %(default)s)")

    parser.add_argument('--results-fname', type=str, default=None,
                        help="Filename of the results JSON file to analyze in analyze-only mode (default: None, required if --analyze-only is set)")
    parser.add_argument('--results-dir', type=str, default=f'../{PROJECT_DIR}/results/',
                        help="Directory where the results JSON file is located (default: %(default)s)")
    parser.add_argument('--outdir', type=str, default=None,
                        help="Directory where the results will be saved (default: %(default)s)")
    
    parser.add_argument('--num-injections', type=int, default=None,
                        help="Number of injections to run in the campaign (default: %(default)s)")
    parser.add_argument('--injection-index', type=int, default=0,
                        help="Index of the specific injection to run (default: %(default)s)")
    parser.add_argument('--injection-index-start', type=int, default=0,
                        help="Starting index of the injection runs to analyze (default: %(default)s)")
    parser.add_argument('--injection-index-end', type=int, default=None,
                        help="Ending index of the injection runs to analyze (default: %(default)s)")
    parser.add_argument('--force', action='store_true',
                        help="Force overwrite of existing results for the given injection (default: False)")
    parser.add_argument('--use-set-injection-params', action='store_true',
                        help="Whether to use a fixed set of injection parameters instead of random sampling (default: False)")
    
    parser.add_argument('--apply-bias-correction', action='store_true',
                        help="Whether to apply bias correction to the posterior samples (default: False)")
    parser.add_argument('--max-injections-to-include', type=int, default=None,
                        help="Maximum number of injections to include in the analysis (default: %(default)s)")
    
    parser.add_argument('--pe-run-type', type=str, choices=['eob2eob', 'ml2ml', 'eob2ml', 'eobbilby2ml'], default='ml2ml',
                        help="Type of PE run: 'eob2eob' for EOB injection and EOB recovery, 'ml2ml' for ML injection and ML recovery, 'eob2ml' for EOB injection and ML recovery (default: ml2ml)")
    parser.add_argument('--sampler', type=str, choices=['nessai', 'dynesty', 'pocomc'], default='nessai',
                        help="Sampler to use for parameter estimation: 'nessai' for neural density estimation sampler, 'dynesty' for nested sampling, 'pocomc' for preconditioned Monte Carlo (default: nessai)")
    
    parser.add_argument('--distance-factor', type=float, default=None,
                        help="Distance scale factor for the injection (default: %(default)s)")
    parser.add_argument('--truncate-eob-waveform-to-1s', action='store_true',
                        help="Whether to truncate the EOB waveform to 1 second (default: False)")
    
    parser.add_argument('--nlive', type=int, default=300,
                        help="Number of live points for the sampler (default: %(default)s)")
    parser.add_argument('--threshold', type=float, default=0.1,
                        help="Stopping threshold for the sampler, corresponding to the change in the log evidence `logZ` in the next iteration. If the change falls below this threshold, the sampler will stop! This value is equal to `dlogZ`! (default: %(default)s)")
    parser.add_argument('--nessai-npool', type=int, default=1,
                        help="Number of workers for nessai's multiprocessing kwarg `n_pool` (default: %(default)s)")
    parser.add_argument('--dynesty-npool', type=int, default=1,
                        help="Number of workers for dynesty's multiprocessing kwarg `npool` (default: %(default)s)")
    parser.add_argument('--npool', type=int, default=1,
                        help="Number of workers for Bilby's internal multiprocessing or for the sampler (default: %(default)s)")
    parser.add_argument('--pytorch-threads', type=int, default=1,
                        help="Number of threads for PyTorch (default: %(default)s)")

    parser.add_argument('--with-original-model', action='store_true',
                        help="Whether to use the original CVAE model instead of the FlexCVAE (default: False)")
    parser.add_argument('--without-mass-ratio-constraint', action='store_true',
                        help="Whether to disable the mass ratio constraint in the prior (default: False)")
    
    methodargs = parser.add_mutually_exclusive_group(required=True)
    methodargs.add_argument('--run-one-injection', action='store_true',
                              help="Whether to run a single injection (default: False)")
    methodargs.add_argument('--run-pe-campaign', action='store_true', 
                              help="Whether to run the full PE campaign (default: False)")
    methodargs.add_argument('--plot-corner-from-result-file', action='store_true',
                              help="Whether to plot a corner plot from a previous result file (default: False)")
    methodargs.add_argument('--imp-reweight', action='store_true',
                              help="Whether to perform importance reweighting on a previous result, using the provided JSON file (default: False)")
    methodargs.add_argument('--make-pp-plots', action='store_true',
                              help="Whether to make PP plots from a previous run, using the provided JSON file (default: False)")
    methodargs.add_argument('--extract-marginalized-posteriors', action='store_true',
                              help="Whether to extract marginalized posteriors from a previous result, using the provided JSON file (default: False)")
    methodargs.add_argument('--analyze-results', action='store_true',
                              help="Whether to analyze a previous result files without running new injections (default: False)")
    
    parser = init_verbosity_args(parser)
    args = parser.parse_args()
    init_logging(args, log_dir=f'../{PROJECT_DIR}/logs/{TODAY}')

    # Force PyTorch's spawn context globally
    mp.set_start_method('spawn', force=True)
    
    if args.imp_reweight:
        logger.info("Running in importance reweighting mode. Will reweight results from a previous run using the provided JSON file.")
        imp_reweight_posteriors(fname=args.results_fname, outdir=args.results_dir, 
                                npool=args.npool,
                                use_nested_samples=False)
        
    elif args.make_pp_plots:
        logger.info("Running in make-PP-plots mode. Will generate PP plots from previous results.")
        make_pp_plots(results_dir=args.results_dir, label=args.label, 
                      pe_run_type=args.pe_run_type, sampler=args.sampler,
                      apply_bias_correction=args.apply_bias_correction,
                      max_injections_to_include=args.max_injections_to_include,
                      outdir=f'../{PROJECT_DIR}/results/{TODAY}/')
        
    elif args.extract_marginalized_posteriors:
        logger.info("Running in extract-marginalized-posteriors mode.")
        extract_marginalized_posteriors(
            results_fname=args.results_fname,
            results_dir=args.results_dir,
            plot_waveforms=True,
            force=args.force,
            model_path='flexcvae-model-backup-20260619-064140-epoch98.pt',
            config_path='modelconfig-flexcvae-20260619-064140.json',
            calmodel_path='calibrator_model_20260623-010953_epoch74.pt',
            )
    
    elif args.analyze_results:
        logger.info("Running in analyze-results mode. Will analyze posteriors from previous results files.")
        analyze_results(
            results_dir=args.results_dir,
            label=args.label,
            pe_run_type=args.pe_run_type,
            sampler=args.sampler,
            force=args.force,
            apply_bias_correction=args.apply_bias_correction
        )

    else:
        main(args, label=args.label, 
            pe_run_type=args.pe_run_type,
            sampler=args.sampler,)