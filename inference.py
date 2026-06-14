
"""
Use trained surrogate ML model and do Bayesian parameter estimation on this
using Bilby, and obtain a Posterior Probability plot.
"""

import os
import json
import time
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

from utils.io import save_json, save_pickle, save_txt, write_DONE_file, ensure_dir

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
LUMINOSITY_DISTANCE = 400.0  # Mpc, should be same as for the ML waveform training data, to avoid bias in amplitudes!


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

def signed_chi_to_bilby_spins(params):
    p = dict(params)

    chi1 = p.pop("spin_1z")
    chi2 = p.pop("spin_2z")

    p["a_1"] = abs(chi1)
    p["tilt_1"] = 0.0 if chi1 >= 0 else np.pi

    p["a_2"] = abs(chi2)
    p["tilt_2"] = 0.0 if chi2 >= 0 else np.pi

    p.setdefault("phi_12", 0.0)
    p.setdefault("phi_jl", 0.0)

    return p

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
        luminosity_distance=LUMINOSITY_DISTANCE,
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
        start_time=injection_parameters["geocent_time"] - DURATION / 2,   # NOTE: Injection signal should be within data segment!
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

    # -- Save IFOs with injected signal and noise to file
    print(outdir, this_label)
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

    # Make a corner plot.
    result.plot_corner(save=True, filename=outdir+f'{this_label}_corner.png')
    return result


def run_injection_campaign(num_injections=50, base_seed=1234, 
                           label_base='umamipe', outdir=f'../{PROJECT_DIR}/results/',
                           injection_generator=None, waveform_generator=None, 
                           sampler=None, **sampler_kwargs):
    if injection_generator is None or waveform_generator is None:
        raise NotImplementedError("Both injection and waveform generators must be provided")
    results = []
    for i in tqdm(range(num_injections), desc="Running injections"):
        logger.info(f"Injection {i+1}/{num_injections}")
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
                                'config_path': wfkwargs.get('config_path'),
                                'distance_scale_factor': LUMINOSITY_DISTANCE},
        )
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
         pe_run_type: {'eob2eob', 'ml2ml', 'eob2ml'} = 'ml2ml', 
         sampler: {'nessai', 'dynesty', 'pocomc'} = 'nessai',):
    label = label + f'_{pe_run_type}_{sampler}'
    project_dir = f'../{args.project_dir}/'
    outdir = os.path.join(project_dir, f'results/{TODAY}/')
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

    # TODO: For EOB waveforms, priors should be [m_1, m_2, a_1, a_2, tilt_1, tilt_2], since otherwise the "spin1z" and "spin2z" parameters will be ignored by the EOB waveform generator, since internally Bilby requires aforementioned parameter names, and then uses its the `bilby_to_lal_bbh_...` function to convert them to LAL parameters, before calling the waveform model.

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

    sampler_kwargs = set_sampler_kwargs(args, sampler)

    if args.run_one_injection:
        outdir = os.path.join(outdir, f'inj_{args.injection_index}_{NOW}/')
        ensure_dir(outdir)
        logger.info(f"Running a single injection and PE with fixed seed 42, for index {args.injection_index}...")
        results = run_single_injection(args.injection_index, seed=42, 
                                        label_base=label, outdir=outdir,
                                       injection_generator=injection_generator, 
                                       waveform_generator=waveform_generator, 
                                       sampler=sampler, **sampler_kwargs)
    elif args.run_pe_campaign:
        logger.info(f"Running a PE campaign with {args.num_injections} injections...")

        results = run_injection_campaign(num_injections=args.num_injections, 
                                        base_seed=42,
                                        injection_generator=injection_generator, 
                                        waveform_generator=waveform_generator,
                                        label_base=label+f'_{NOW}', 
                                        outdir=outdir,
                                        sampler=sampler, 
                                        **sampler_kwargs)
    
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



def analyze_results(fname: str = None, 
                    results: Optional[List[bilby.gw.result.CBCResult]] = None,
                    label: str = 'umamipe',
                    outdir: str = f'../{PROJECT_DIR}/results/'):
    
    if results is None:
        if not fname.endswith('.json'):
            fname += '.json'
        results = bilby.gw.result.CBCResult.from_json(f"{outdir}/{fname}")
    if not isinstance(results, list):
        results = [results]

    # Bilby built-in PP plot
    fig, pvals = make_pp_plot(
        results,
        filename=f"{outdir}/{label}_pp-plot.png",
        save=True,
    )
    print("Combined p-value:", pvals.combined_pvalue)

    # Plot the inferred waveform superposed on the actual data.
    # results[0].plot_waveform_posterior(n_samples=100)
    logger.info("Saved PP plot, waveform posterior plot, and corner plot for the first injection result.")



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
    
    parser.add_argument('--results-fname', type=str, default=None,
                        help="Filename of the results JSON file to analyze in analyze-only mode (default: None, required if --analyze-only is set)")
    parser.add_argument('--results-run-index-start', type=int, default=1,
                        help="Starting index of the injection runs to analyze (default: %(default)s)")
    parser.add_argument('--results-run-index-end', type=int, default=50,
                        help="Ending index of the injection runs to analyze (default: %(default)s)")
    
    parser.add_argument('--num-injections', type=int, default=50,
                        help="Number of injections to run in the campaign (default: %(default)s)")
    parser.add_argument('--injection-index', type=int, default=1,
                        help="Index of the specific injection to run (default: %(default)s)")
    
    parser.add_argument('--pe-run-type', type=str, choices=['eob2eob', 'ml2ml', 'eob2ml'], default='ml2ml',
                        help="Type of PE run: 'eob2eob' for EOB injection and EOB recovery, 'ml2ml' for ML injection and ML recovery, 'eob2ml' for EOB injection and ML recovery (default: ml2ml)")
    parser.add_argument('--sampler', type=str, choices=['nessai', 'dynesty', 'pocomc'], default='nessai',
                        help="Sampler to use for parameter estimation: 'nessai' for neural density estimation sampler, 'dynesty' for nested sampling, 'pocomc' for preconditioned Monte Carlo (default: nessai)")
    
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
    
    methodargs = parser.add_mutually_exclusive_group(required=True)
    methodargs.add_argument('--run-one-injection', action='store_true',
                              help="Whether to run a single injection (default: False)")
    methodargs.add_argument('--run-pe-campaign', action='store_true', 
                              help="Whether to run the full PE campaign (default: False)")
    methodargs.add_argument('--analyze-only', action='store_true', 
                              help="Whether to only analyze results from a previous run, using the provided JSON file (default: False)")
    
    parser = init_verbosity_args(parser)
    args = parser.parse_args()
    init_logging(args, log_dir=f'../{PROJECT_DIR}/logs/{TODAY}')

    # Force PyTorch's spawn context globally
    mp.set_start_method('spawn', force=True)

    if args.analyze_only:
        logger.info("Running in analyze-only mode. Will analyze results from a previous run using the provided JSON file.")
        analyze_results(fname=args.results_fname, label=args.label, outdir=f'../{PROJECT_DIR}/results/{TODAY}')
    else:
        main(args, label=args.label, 
            pe_run_type=args.pe_run_type,
            sampler=args.sampler,)
    

    # analyze_results(fname='ml2ml-4d_20260607-153120_inj_0001_result.json', 
    #                 label=args.label, 
    #                 outdir=f'../{PROJECT_DIR}/results/{TODAY}')