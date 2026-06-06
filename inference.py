
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

from tqdm import tqdm
from umamipe import MLWaveformGenerator, convert_to_ml_parameters

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


# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])


priors = BBHPriorDict(aligned_spin=True)
print("Default priors for BBH parameters:")
for key, prior in priors.items():
    print(f"  {key}: {prior}")
print(f'Default priors for BBH parameters: {priors.keys()}')

# NOTE: Other parameters should be allowed to vary freely, for injection generation!
# # -- remove redundant or irrelevant parameters
# params_to_use = ["mass_1", "mass_2", "chi_1z", "chi_2z"]
# for param in list(priors.keys()):
#     if param not in params_to_use:
#         priors.pop(param)
priors.pop("mass_ratio", None)
priors.pop("chirp_mass", None)
priors["geocent_time"] = 0.0 

# -- Set the priors for the parameters we want to estimate!
priors["mass_1"] = bilby.core.prior.Uniform(30, 75, name="mass_1", latex_label="$m_1$")
priors["mass_2"] = bilby.core.prior.Uniform(30, 75, name="mass_2", latex_label="$m_2$")
priors["chi_1z"] = bilby.core.prior.Uniform(-0.80, 0.80, name="chi_1z", latex_label="$\\chi_{1z}$")
priors["chi_2z"] = bilby.core.prior.Uniform(-0.80, 0.80, name="chi_2z", latex_label="$\\chi_{2z}$")

# priors["chi_1"].a_prior.maximum = 0.80
# priors["chi_2"].a_prior.maximum = 0.80
# priors["a_1"] = bilby.core.prior.Uniform(0, 0.80, name="a_1", latex_label="$\\chi_{1z}$")
# priors["a_2"] = bilby.core.prior.Uniform(0, 0.80, name="a_2", latex_label="$\\chi_{2z}$")
# priors["a_1"] = injection_parameters["a_1"]  
# priors["a_2"] = injection_parameters["a_2"]  

print("Updated priors for BBH parameters:")
for key, prior in priors.items():
    print(f"  {key}: {prior}")
print(f'Updated priors for BBH parameters: {priors.keys()}')

# Perform a check that the prior does not extend to a parameter space longer than the data
priors.validate_prior(DURATION, FMIN)


def run_single_injection(run_idx, seed=None, label_base='umamipe', outdir=f'../{PROJECT_DIR}/results/',
                         injection_generator=None, waveform_generator=None, 
                         sampler=None, **sampler_kwargs):
    if seed is not None:
        np.random.seed(seed + run_idx)
    this_label = f"{label_base}_inj_{run_idx:04d}"
    logger.info(f"Running sampler for injection {run_idx} with label {this_label} using sampler {sampler}...")

    injection_parameters = priors.sample()

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
        priors=priors,
        sampler=sampler,
        injection_parameters=injection_parameters,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
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


def main(args, label='umamipe', multiprocessing=True):

    if multiprocessing:
        logger.info("Using multiprocessing with spawn context for parallel sampling.")
        sampler = "nessai"
        nworkers = min(4, mp.cpu_count() - 1)
        sampler_kwargs = dict(
            nlive=1000,
            n_pool=1,
            pytorch_threads=nworkers,

            npool=1, # Set this arg for Bilby's internal multiprocessing that only works on CPU!
            flow_proposal_class='gwflowproposal',
            max_iteration=10000,    # Safety break to prevent infinite hangs
            reset_flow=16,          # Periodic reset to clear "stuck" AI states
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


    # # -- Injected waveform will be the native SEOBNRv4 implementation
    # injection_generator = WaveformGenerator(
    #     duration=DURATION,
    #     sampling_frequency=SAMPLE_RATE,
    #     # NOTE: The `lal_binary_black_hole` source model works basically FrequencyDomain approximants!
    #     frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    #     waveform_arguments=dict(
    #         waveform_approximant="SEOBNRv4",      #"IMRPhenomPv2",
    #         reference_frequency=FREF,
    #         minimum_frequency=FMIN,
    #         mode_array=[[2,2]],
    #         catch_waveform_errors=True, 
    #     )
    # )
    # logger.info("Injection generator initialized with SEOBNRv4 waveform model.")

    # -- initialize the ML waveform generator with the loaded model
    # NOTE: We will only use this for the likelihood evaluation in the sampler!
    waveform_generator = MLWaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLE_RATE,
        time_domain_source_model=None,   # We will load ML model at initialization!
        parameter_conversion=convert_to_ml_parameters,
        waveform_arguments={'model_path': model_path, 'config_path': args.model_config},
        )
    logger.info("MLWaveformGenerator initialized with the loaded model.")

    # -- For now, try using another instance of MLWaveformGenerator for the injection generator!
    injection_ml_generator = MLWaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLE_RATE,
        time_domain_source_model=None,   # We will load ML model at initialization!
        parameter_conversion=convert_to_ml_parameters,
        waveform_arguments={'model_path': model_path, 'config_path': args.model_config},
        )
    logger.info("Injection MLWaveformGenerator initialized with the loaded model.")

    results = run_injection_campaign(N_injections=20, 
                                     base_seed=42,
                                     injection_generator=injection_ml_generator, 
                                     waveform_generator=waveform_generator,
                                     label_base=label+f'_{NOW}', 
                                     outdir=outdir,
                                     sampler=sampler, **sampler_kwargs)
    logger.info("Completed injection campaign and sampling for all injections.")

    # Plot the inferred waveform superposed on the actual data.
    results[0].plot_waveform_posterior(n_samples=100, filename=f'{label}_waveform_posterior.png')

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
    parser.add_argument('--no-multiprocessing', action='store_true', dest='multiprocessing',
                        help="Whether to use multiprocessing for parallel sampling (default: False)")
    
    parser.add_argument('--with-original-model', action='store_true',
                        help="Whether to use the original CVAE model instead of the FlexCVAE (default: False)")
    
    parser = init_verbosity_args(parser)
    args = parser.parse_args()
    init_logging(args)

    logger.getLogger("bilby").setLevel(logging.INFO)  # Allow INFO level logs from Bilby to be printed, but suppress DEBUG logs
    logger.getLogger("nessai").setLevel(logging.INFO)  # Allow INFO level logs from nessai to be printed, but suppress DEBUG logs

    # Force PyTorch's spawn context globally
    mp.set_start_method('spawn', force=True)

    main(args, label=args.label, multiprocessing=not args.multiprocessing)