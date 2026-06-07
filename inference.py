
"""
Use trained surrogate ML model and do Bayesian parameter estimation on this
using Bilby, and obtain a Posterior Probability plot.
"""

import os
import argparse
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from mlwavegen import MLWaveformGenerator, convert_to_ml_parameters

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


# bilby.core.utils.setup_logger(outdir=f'../logs/{TODAY}', label='umamipe', log_level="INFO")


def make_default_base_injection() -> Dict[str, float]:
    return dict(
        mass_1=36.0,
        mass_2=32.0,  # keep inside prior
        a_1=0.4,
        a_2=0.3,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_12=0.0,
        phi_jl=0.0,
        luminosity_distance=400.0,  # NOTE: This should be same as for the ML waveform training data, to avoid bias in amplitudes!
        theta_jn=0.4,
        psi=2.659,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
    )


def sample_injection_from_priors(
    base_injection: Dict[str, float],
    active_priors: bilby.core.prior.PriorDict,
    active_keys: Tuple[str, ...] = ("mass_1", "mass_2", "spin_1z", "spin_2z"),
    rng_seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Samples only active parameters, copies all other values from base_injection.
    """

    if rng_seed is not None:
        np.random.seed(rng_seed)

    injection = copy.deepcopy(base_injection)

    sampled = active_priors.sample()
    for key in active_keys:
        injection[key] = sampled[key]
    return injection



def make_analysis_priors(
    injection_parameters: Dict[str, float],
    active_keys: Tuple[str, ...] = ("mass_1", "mass_2", "spin_1z", "spin_2z"),
) -> bilby.gw.prior.BBHPriorDict:
    """
    Build priors for one PE run.

    Fixed parameters become delta-function priors at injected values.
    Active parameters are sampled.

    spin_magnitudes=True means a_1, a_2 are non-negative spin magnitudes.
    If you truly want signed aligned spins, use different parameter names.
    """

    priors = bilby.gw.prior.BBHPriorDict()

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
            30, 75, name="mass_1", latex_label="$m_1$"
        )
    else:
        priors["mass_1"] = injection_parameters["mass_1"]

    if "mass_2" in active_keys:
        priors["mass_2"] = bilby.core.prior.Uniform(
            30, 75, name="mass_2", latex_label="$m_2$"
        )
    else:
        priors["mass_2"] = injection_parameters["mass_2"]

    if "spin_1z" in active_keys:
        priors["spin_1z"] = bilby.core.prior.Uniform(
            -0.80, 0.80, name="spin_1z", latex_label="$\\chi_{1z}$"
        )
    else:
        priors["spin_1z"] = injection_parameters["spin_1z"]

    if "spin_2z" in active_keys:
        priors["spin_2z"] = bilby.core.prior.Uniform(
            -0.80, 0.80, name="spin_2z", latex_label="$\\chi_{2z}$"
        )
    else:        
        priors["spin_2z"] = injection_parameters["spin_2z"]

    priors.pop("mass_ratio", None)
    priors.pop("chirp_mass", None)
    return priors


base_injection = make_default_base_injection()

# Use the same active priors for drawing injections.
active_priors = make_analysis_priors(
    injection_parameters=base_injection,
)
print("Active priors for injection sampling:")
for key, prior in active_priors.items():
    print(f"  {key}: {prior}")
print(f'Active priors for injection sampling: {active_priors.keys()}')


# priors = BBHPriorDict(aligned_spin=True)
# print("Default priors for BBH parameters:")
# for key, prior in priors.items():
#     print(f"  {key}: {prior}")
# print(f'Default priors for BBH parameters: {priors.keys()}')

# # NOTE: Other parameters should be allowed to vary freely, for injection generation!
# # # -- remove redundant or irrelevant parameters
# params_to_use = ["mass_1", "mass_2", "chi_1z", "chi_2z"]
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

# Perform a check that the prior does not extend to a parameter space longer than the data
active_priors.validate_prior(DURATION, FMIN)



# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])


def run_single_injection(run_idx, seed=None, label_base='umamipe', outdir=f'../{PROJECT_DIR}/results/',
                         injection_generator=None, waveform_generator=None, 
                         sampler=None, **sampler_kwargs):
    if seed is not None:
        np.random.seed(seed + run_idx)
    this_label = f"{label_base}_inj_{run_idx:04d}"
    logger.info(f"Running sampler for injection {run_idx} with label {this_label} using sampler {sampler}...")

    # injection_parameters = priors.sample()

    injection_parameters = sample_injection_from_priors(
        base_injection=base_injection,
        active_priors=active_priors,
        rng_seed=seed + run_idx if seed is not None else None
    )


    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLE_RATE,
        duration=DURATION,
        start_time=injection_parameters["geocent_time"]
    )

    # -- Send model to device only here! We will keep the model on CPU until we need to generate the waveform, to save GPU memory and avoid potential issues with multiprocessing in Bilby!
    for generator in [injection_generator, waveform_generator]:
        if isinstance(generator, MLWaveformGenerator):
            logger.debug(f"Before sending to device, generator.loaded_mlmodel is on device: \
                        {next(generator.loaded_mlmodel.parameters()).device}, dtype: {next(generator.loaded_mlmodel.parameters()).dtype}")
            generator.loaded_mlmodel.to(DEVICE, dtype=getattr(torch, PRECISION))
            generator.check_model_weights_on_device(device=DEVICE, 
                                                   precision=getattr(torch, PRECISION))
            logger.debug(f"Sent injection_generator.loaded_mlmodel to device: {DEVICE} with dtype: {PRECISION}")

    ifos.inject_signal(
        waveform_generator=injection_generator,
        parameters=injection_parameters,
    )
    logger.debug(f"Injection parameters for run {run_idx}: {injection_parameters}")

    likelihood = GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
    )

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
        **sampler_kwargs
    )
    logger.info(f"Completed sampler for injection {run_idx} with label {this_label}.")
    return result


def run_injection_campaign(N_injections=50, base_seed=1234, 
                           label_base='umamipe', outdir=f'../{PROJECT_DIR}/results/',
                           injection_generator=None, waveform_generator=None, 
                           sampler=None, **sampler_kwargs):
    if injection_generator is None or waveform_generator is None:
        raise NotImplementedError("Both injection and waveform generators must be provided")
    results = []
    for i in tqdm(range(N_injections), desc="Running injections"):
        logger.info(f"Injection {i+1}/{N_injections}")
        res = run_single_injection(i, seed=base_seed, outdir=outdir, label_base=label_base,
                                   injection_generator=injection_generator, waveform_generator=waveform_generator, 
                                   sampler=sampler,**sampler_kwargs)
        results.append(res)
    return results


def make_wf_generator(type: {'eob', 'ml'}, wfkwargs: Dict = None):
    if type=='eob':
        return WaveformGenerator(
            duration=DURATION,
            sampling_frequency=SAMPLE_RATE,
            # NOTE: The `lal_binary_black_hole` source model works basically FrequencyDomain approximants!
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                waveform_approximant="SEOBNRv4",      #"IMRPhenomPv2",
                reference_frequency=FREF,
                minimum_frequency=FMIN,
                mode_array=[[2,2]],
                catch_waveform_errors=True, 
            )
        )
    elif type=='ml':
        return MLWaveformGenerator(
            duration=DURATION,
            sampling_frequency=SAMPLE_RATE,
            time_domain_source_model=None,   # We will load ML model at initialization!
            parameter_conversion=convert_to_ml_parameters,
            waveform_arguments={'model_path': wfkwargs.get('model_path'), 
                                'config_path': wfkwargs.get('config_path')},
            )


def main(args, label='umamipe', 
         pe_run_type: {'eob2eob', 'ml2ml', 'eob2ml'} = 'ml2ml', 
         multiprocessing=True):

    if multiprocessing:
        logger.info("Using multiprocessing with spawn context for parallel sampling.")
        sampler = "nessai"
        nworkers = min(3, mp.cpu_count() - 1)
        sampler_kwargs = dict(
            nlive=750,
            n_pool=1,
            pytorch_threads=nworkers,
            npool=1, # Set this arg for Bilby's internal multiprocessing that only works on CPU!
            flow_proposal_class='flowproposal',     # 'gwflowproposal' instead reparameterisation full 15D space!
            reparameterisations=None,  # We only  
            # max_iteration=7500,    # NOTE: This forces nessai to abruptly end, leaving results JSON file incomplete!
            stopping=10,   # Stop if `dlogz` doesn't improve by this amt in consecutive iterations.
            reset_flow=16,          # Periodic reset to clear "stuck" AI states
            analytic_priors=active_priors,  # Pass the priors to nessai for better sampling efficiency
        )
    else:
        logger.info("Not using multiprocessing. Running sampler in single-process mode.")
        sampler = 'dynesty'
        sampler_kwargs = dict(
            nlive=100,
            dlogz=0.5,
            naccept=10,
            sample="acceptance-walk",
            npool=1,
        )

    project_dir = '../' + args.project_dir + '/'
    outdir = os.path.join(project_dir, f'results/{TODAY}')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    model_path = os.path.join(project_dir, 'trained-models', args.model_name)
    try:
        if not os.path.isfile(model_path):
            logger.error(f"Provided MODEL_PATH does not exist: {model_path}")
            raise FileNotFoundError(f"MODEL_PATH file not found at {model_path}")
    except Exception as e:
        model_path = os.path.join('../', 'trained-models', args.model_name)
        if not os.path.isfile(model_path):
            logger.error(f"Provided MODEL_PATH does not exist: {model_path}")
            raise FileNotFoundError(f"MODEL_PATH file not found at {model_path}")
    logger.info(f"Using MODEL_PATH: {model_path}")

    # configpath = args.model_config
    # if configpath is not None:
    #     if not configpath.endswith('.json'):
    #         configpath += '.json'
    #     if not os.path.isfile(configpath):
    #         logger.error(f"Provided MODEL_CONFIG path does not exist: {configpath}")
    #         raise FileNotFoundError(f"MODEL_CONFIG file not found at {configpath}")
    #     logger.info(f"Using MODEL_CONFIG: {configpath}")
    #     MODEL_CONFIG = json.load(open(configpath, 'r'))

    # if args.with_original_model:
    #     # Load the trained model
    #     preset_array_size = 8190
    #     num_classes = 4
    #     model = CVAE(input_shape=(2, preset_array_size), num_classes=num_classes, key_shape=(2,2),
    #                  MODEL_CONFIG=MODEL_CONFIG)
    #     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    #     model.to(getattr(torch, PRECISION))
    #     model.to(DEVICE)
    #     model.eval()
    #     logger.info("Model loaded and set to evaluation mode.")

    # else:
    # model = load_flex_model(model_path=model_path, 
    #                         configpath=args.model_config)

    if pe_run_type == 'eob2eob':
        injection_generator = make_wf_generator('eob')
        waveform_generator = make_wf_generator('eob')
        logger.info("Initialized EOB waveform generator for both injection and recovery.")
    elif pe_run_type == 'ml2ml':
        wfkwargs={'model_path': model_path, 'config_path': args.model_config}
        injection_generator = make_wf_generator('ml', wfkwargs=wfkwargs)
        waveform_generator = make_wf_generator('ml', wfkwargs=wfkwargs)
        logger.info("Initialized ML waveform generator for both injection and recovery.")
    elif pe_run_type == 'eob2ml':
        injection_generator = make_wf_generator('eob')
        waveform_generator = make_wf_generator('ml', wfkwargs={'model_path': model_path, 
                                                               'config_path': args.model_config})
        logger.info("Initialized EOB waveform generator for injection and ML waveform generator for recovery.")

    results = run_injection_campaign(N_injections=3, 
                                     base_seed=42,
                                     injection_generator=injection_generator, 
                                     waveform_generator=waveform_generator,
                                     label_base=label+f'_{NOW}', 
                                     outdir=outdir,
                                     sampler=sampler, 
                                     **sampler_kwargs)
    analyze_results(results=results, label=label+f'_{NOW}', outdir=outdir)
    logger.info("Completed injection campaign and sampling for all injections.")



def analyze_results(fname: str = None, results: Optional[List[bilby.gw.result.CBCResult]] = None,
                    label: str = 'umamipe',
                    outdir: str = f'../{PROJECT_DIR}/results/'):
    
    if results is None:
        if not fname.endswith('.json'):
            fname += '.json'
        results = bilby.gw.result.CBCResult.from_json(f"{outdir}/{fname}")

    # Plot the inferred waveform superposed on the actual data.
    results[0].plot_waveform_posterior(n_samples=100)

    # Make a corner plot.
    results[0].plot_corner(save=True, filename=f'{label}_corner.png')

    # Bilby built-in PP plot
    fig, pvals = make_pp_plot(
        results,
        filename=f"{outdir}/{label}_pp.png",
        save=True,
    )
    print("Combined p-value:", pvals.combined_pvalue)

    logger.info("Saved PP plot, waveform posterior plot, and corner plot for the first injection result.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CVAE model on GW waveforms")
    parser.add_argument('--label', type=str, default='umamipe',
                        help="Label for the analysis (default: umamipe)")
    
    parser.add_argument('--project-dir', type=str, choices=['cvae@taiwan', 'v0p1', '@alvin', '@korea'], 
                        default=PROJECT_DIR, help="Base directory for the project (default: v0p1)")
    parser.add_argument('--model-config', type=str, default='modelconfig-cvae-paper-I',
                        help="Name of the model configuration JSON file (default: None)")
    parser.add_argument('--model-name', type=str, default='model-20251004_072338-10',
                        help="Name of the trained model checkpoint (default: None)")
    
    parser.add_argument('--num-injections', type=int, default=50,
                        help="Number of injections to run in the campaign (default: 50)")
    parser.add_argument('--no-multiprocessing', action='store_true',
                        help="Whether to use multiprocessing for parallel sampling (default: False)")
    
    parser.add_argument('--with-original-model', action='store_true',
                        help="Whether to use the original CVAE model instead of the FlexCVAE (default: False)")
    
    parser = init_verbosity_args(parser)
    args = parser.parse_args()
    init_logging(args, log_dir=f'../{PROJECT_DIR}/logs/{TODAY}', label=args.label)

    logging.getLogger("bilby").setLevel(logging.INFO)  # Allow INFO level logs from Bilby to be printed, but suppress DEBUG logs
    logging.getLogger("nessai").setLevel(logging.INFO)  # Allow INFO level logs from nessai to be printed, but suppress DEBUG logs

    # Force PyTorch's spawn context globally
    mp.set_start_method('spawn', force=True)


    main(args, label=args.label, pe_run_type='eob2eob',
        multiprocessing = not args.no_multiprocessing,)

    # analyze_results(fname='ml2ml-4d_20260607-153120_inj_0001_result.json', 
    #                 label=args.label, 
    #                 outdir=f'../{PROJECT_DIR}/results/{TODAY}')