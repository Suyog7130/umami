
"""
Routine to call my waveform models with Bilby
and estimate the posteriors! We will generate some simulated
data and call theoretical waveforms from my trained ML model
and then try to constrain the parameter posteriors!
Then we can work with real events and noise from the detectors,
and do the same for say the GW150914 etc.
"""

import os
import gc
import json
import logging
import argparse
import pandas as pd

from tqdm import tqdm
import datetime

import bilby
from bilby.core.utils import logger
from bilby.core import utils
from bilby.gw import WaveformGenerator

import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F
import numpy as np

from flexcvae import FlexTwoC2E1D, FlexCAE, FlexCAEPhase
from optimize import load_flex_model
from cvae import CVAE

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
    PRECISION = 'float64'  # Use double precision for CUDA if available
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    PRECISION = 'float32'  # Use float32 for MPS since it does not support float64 well
else:
    DEVICE = torch.device("cpu")
    PRECISION = 'float64'  # Use double precision for CPU
print(f"Using device: {DEVICE}, with precision: {PRECISION}")


bilby.core.utils.setup_logger(outdir=f'../logs/{TODAY}', label='umamipe', log_level="DEBUG")

# Set up a random seed for result reproducibility.  This is optional!
bilby.core.utils.random.seed(42)


def get_td_SEOBNRv4ml(time_array, **kwargs):
    """
    Generate a waveform using the ML model based on the input parameters.

    Arguments
    ---------
        time_array: np.ndarray
            The array of time points at which to evaluate the waveform.
            This input is ignored by the waveform generator!

    NOTE: Bilby sent a list of `*parameters` to the waveform generator function!
    We assume this list contains either [m1, m2, chi1z, chi2z] or [mass_1, mass_2, spin_1z, spin_2z],
    or ['mass_ratio', 'chirp_mass', 'a_1', 'a_2'] depending on how the parameters are formatted!
    
    NOTE: Additionally, the `**kwargs` should contain the `modelpath` and `configpath` for the ML model, 
    which we will use to load the model and generate the waveform!

    Returns
    -------
        np.ndarray
            The generated time-domain strain waveform as a 1D numpy array.
    """
    print("Received parameters for waveform generation:", kwargs)
    # mlmodel=f'../{PROJECT_DIR}/trained-models/model-20251004_072338-10'
    if any(key not in kwargs for key in ['model_path', 'config_path']):
        raise ValueError("Missing 'model_path' or 'config_path' in kwargs for waveform generation.")
    model = load_flex_model(model_path=kwargs['model_path'], 
                            configpath=kwargs['config_path'], device=DEVICE, precision=PRECISION)
    parameters = {model_param: kwargs[model_param] for model_param in ['mass_1', 'mass_2', 'spin_1z', 'spin_2z']}
    labels = torch.tensor([parameters[key] for key in sorted(parameters.keys())], 
                            dtype=torch.float32).unsqueeze(0).to(DEVICE)
    generated_waveform = model.generate(labels)  # has shape (1, 2=[hp,hc], sequence_length)!
    print("Generated waveform from ML model with shape:", generated_waveform.shape)

    hplus, hcross = generated_waveform[0][0], generated_waveform[0][1]

    # -- add two dummy repeated value at the start to makeup for length req by Bilby Interferometer.
    hplus = np.concatenate([[hplus[0],hplus[1]], hplus])
    hcross = np.concatenate([[hcross[0],hcross[1]], hcross])
    print(f"Waveform shapes after adding dummy element at the start: {hplus.shape}, {hcross.shape}")

    waveforms = {'plus': hplus, 'cross': hcross}

    # fig, ax = plt.subplots(figsize=(12, 5))
    # ax.plot(np.arange(len(waveforms['plus'])), waveforms['plus'], label='hp')
    # ax.plot(np.arange(len(waveforms['cross'])), waveforms['cross'], label='hc')
    # ax.legend()
    # plt.savefig('check-global-denorming-outputs.png', dpi=300)
    # plt.show()
    return waveforms

def convert_to_ml_parameters(parameters):
    """
    Convert the input parameters from the format expected by Bilby to the format expected by the ML model.
    The ML model expects parameters in the format [mass_1, mass_2, spin_1z, spin_2z], whereas Bilby may provide them in a different format (e.g., mass_ratio, chirp_mass, a_1, a_2, tilt_1, tilt_2, etc.).

    Arguments
    ---------
        parameters: dict
            The input parameters in the format expected by Bilby (e.g., mass_1, mass_2, spin_1z, spin_2z, etc.)
    Returns
        dict
            The converted parameters in the format expected by the ML model (e.g., mass_1, mass_2, spin_1z, spin_2z)
    """
    if not isinstance(parameters, dict):
        raise TypeError('"parameters" must be a dictionary.')
    new_parameters = parameters.copy()

    m1 = parameters.get("mass_1", None)
    m2 = parameters.get("mass_2", None)
    if m1 is None or m2 is None:
        if "mass_ratio" not in parameters or "chirp_mass" not in parameters:
            raise ValueError("Missing 'mass_ratio' or 'chirp_mass' in parameters for mass conversion.")
        q = parameters["mass_ratio"]
        M_chirp = parameters["chirp_mass"]
        M_total = M_chirp * (q**(-3/5) + q**(2/5))**(5/3)
        m1 = M_total / (1 + q)
        m2 = M_total - m1
        print(f"Converted (mass_ratio, chirp_mass) to (mass_1, mass_2): {m1}, {m2}")
    new_parameters["mass_1"] = m1
    new_parameters["mass_2"] = m2
    # new_parameters.pop("mass_ratio", None)
    # new_parameters.pop("chirp_mass", None)

    if "a_1" in parameters and "a_2" in parameters and "tilt_1" in parameters and "tilt_2" in parameters and \
        "phi_12" in parameters and "phi_jl" in parameters and "theta_jn" in parameters:
        from bilby.gw.conversion import bilby_to_lalsimulation_spins
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
            theta_jn=parameters["theta_jn"],
            phi_jl=parameters["phi_jl"],
            tilt_1=parameters["tilt_1"],
            tilt_2=parameters["tilt_2"],
            phi_12=parameters["phi_12"],
            a_1=parameters["a_1"],
            a_2=parameters["a_2"],
            mass_1=new_parameters["mass_1"] * utils.solar_mass,
            mass_2=new_parameters["mass_2"] * utils.solar_mass,
            reference_frequency=FREF,
            phase=parameters.get("phase", 0.0),
        )
        new_parameters["spin_1z"] = spin_1z
        new_parameters["spin_2z"] = spin_2z
    print(f"Converted (a_i, tilt_i, phi_i) to (spin_i_z): {spin_1z}, {spin_2z}")
    # TODO: Can pop out original parameters that are not needed by ML model.
    return (new_parameters, None)

class MLWaveformGenerator(WaveformGenerator):
    """
    Custom waveform generator that uses a trained ML model to generate waveforms 
    based on input parameters. For now, it is assumed that the model generates
    time-domain SEOBNRv4 waveforms sampled at 8192 Hz and of duration 1 second.
    Further only the 22 mode is generated to things simple for now!
    We will use the `generate()` method of the ML model to produce the waveform, 
    and then return it in the format expected by Bilby.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def time_domain_strain(self, parameters=None):
        """
        Override the time_domain_strain method to use the ML model for waveform generation.
        This method is called by Bilby to get the strain for given parameters.
        """
        print("Generating waveform using ML model for parameters:", parameters)
        return self._calculate_strain(model=self.time_domain_source_model,
                                      model_data_points=self.time_array,
                                      parameters=parameters,
                                      transformation_function=utils.infft,
                                      transformed_model=self.frequency_domain_source_model,
                                      transformed_model_data_points=self.frequency_array)

    def frequency_domain_strain(self, parameters=None):
        print("Generating waveform using ML model for parameters:", parameters)
        return self._calculate_strain(model=self.frequency_domain_source_model,
                                      model_data_points=self.frequency_array,
                                      parameters=parameters,
                                      transformation_function=utils.nfft,
                                      transformed_model=self.time_domain_source_model,
                                      transformed_model_data_points=self.time_array)
    
    def _strain_from_model(self, model_data_points, model, parameters):
        print("Generating waveform using ML model for parameters:", parameters)
        return model(model_data_points, **parameters)
    
    def _calculate_strain(self, model, model_data_points, transformation_function, transformed_model,
                          transformed_model_data_points, parameters):
        if parameters is None:
            parameters = self.parameters
        if parameters == self._cache['parameters'] and self._cache['model'] == model and \
                self._cache['transformed_model'] == transformed_model:
            return self._cache['waveform']
        else:
            self._cache['parameters'] = parameters.copy()
            self._cache['model'] = model
            self._cache['transformed_model'] = transformed_model
        parameters = self._format_parameters(parameters)
        print("Generating waveform using ML model for parameters:", parameters)
        print(f"Using model: {model} and transformed model {transformed_model}")
        if model is not None:
            model_strain = self._strain_from_model(model_data_points, model, parameters)
        elif transformed_model is not None:
            model_strain = self._strain_from_transformed_model(transformed_model_data_points, transformed_model,
                                                               transformation_function, parameters)
        else:
            raise RuntimeError("No source model given")
        self._cache['waveform'] = model_strain
        return model_strain

    def _format_parameters(self, parameters):
        """
        Removes any parameters that are not in the source model's expected parameter keys, 
        and adds any additional parameters that are needed for waveform generation (e.g., waveform_arguments)
        """
        if not isinstance(parameters, dict):
            raise TypeError('"parameters" must be a dictionary.')
        new_parameters = parameters.copy()
        # convert parameters to lal BBH parameters using the provided conversion function
        new_parameters, _ = self.parameter_conversion(new_parameters)
        print("Formatted parameters for waveform generation:", new_parameters)

        # from bilby.gw.conversion import bilby_to_lalsimulation_spins

        # iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        #     theta_jn=parameters["theta_jn"],
        #     phi_jl=parameters["phi_jl"],
        #     tilt_1=parameters["tilt_1"],
        #     tilt_2=parameters["tilt_2"],
        #     phi_12=parameters["phi_12"],
        #     a_1=parameters["a_1"],
        #     a_2=parameters["a_2"],
        #     mass_1=new_parameters["mass_1"] * utils.solar_mass,
        #     mass_2=new_parameters["mass_2"] * utils.solar_mass,
        #     reference_frequency=FREF,
        #     phase=parameters["phase"],
        # )
        # print("Converted spins and inclination:", iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)

        # ml_parameters = {
        #     "mass_1": new_parameters["mass_1"],
        #     "mass_2": new_parameters["mass_2"],
        #     "spin_1z": spin_1z,
        #     "spin_2z": spin_2z,
        # }

        # for key in self.source_parameter_keys.symmetric_difference(
        #     ml_parameters.keys()):
        #     new_parameters.pop(key)
        # new_parameters.update(ml_parameters)
        new_parameters.update(self.waveform_arguments)
        print("Final parameters for waveform generation:", new_parameters)
        return new_parameters

    def _strain_from_transformed_model(
        self, transformed_model_data_points, transformed_model, transformation_function, parameters
    ):
        print("Generating waveform using ML model for parameters:", parameters)
        transformed_model_strain = self._strain_from_model(
            transformed_model_data_points, transformed_model, parameters
        )

        if isinstance(transformed_model_strain, np.ndarray):
            return transformation_function(transformed_model_strain, self.sampling_frequency)

        model_strain = dict()
        for key in transformed_model_strain:
            if transformation_function == utils.nfft:
                model_strain[key], _ = \
                    transformation_function(transformed_model_strain[key], self.sampling_frequency)
            else:
                model_strain[key] = transformation_function(transformed_model_strain[key], self.sampling_frequency)
        return model_strain




def main(args, label='umamipe'):
    project_dir = '../' + args.project_dir + '/'
    outdir = os.path.join(project_dir, f'results/{TODAY}')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    model_path = os.path.join(project_dir, 'trained-models', args.model_name)
    if not os.path.isfile(model_path):
        logging.error(f"Provided MODEL_PATH does not exist: {model_path}")
        raise FileNotFoundError(f"MODEL_PATH file not found at {model_path}")
    logging.info(f"Using MODEL_PATH: {model_path}")

    # configpath = args.model_config
    # if configpath is not None:
    #     if not configpath.endswith('.json'):
    #         configpath += '.json'
    #     if not os.path.isfile(configpath):
    #         logging.error(f"Provided MODEL_CONFIG path does not exist: {configpath}")
    #         raise FileNotFoundError(f"MODEL_CONFIG file not found at {configpath}")
    #     logging.info(f"Using MODEL_CONFIG: {configpath}")
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
    #     print("Model loaded and set to evaluation mode.")

    # else:
    model = load_flex_model(model_path=model_path, 
                            configpath=args.model_config)

    # -- initialize the ML waveform generator with the loaded model
    # NOTE: We will only use this for the likelihood evaluation in the sampler!
    # Whereas, the injection is performed using LAL waveform!
    waveform_generator = MLWaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLE_RATE,
        time_domain_source_model=get_td_SEOBNRv4ml,
        parameter_conversion=convert_to_ml_parameters,
        waveform_arguments={'model_path': model_path, 'config_path': args.model_config},
        )
    print("MLWaveformGenerator initialized with the loaded model.")

    # -- Injected waveform will be the native SEOBNRv4 implementation
    injection_generator = WaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLE_RATE,
        # NOTE: The `lal_binary_black_hole` source model works basically FrequencyDomain approximants!
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=FREF,
            minimum_frequency=FMIN,
            mode_array=[[2,2]],
            catch_waveform_errors=True, 
        )
    )
    print("Injection generator initialized with SEOBNRv4 waveform model.")
    
    # -- Our ML model is only for [m1,m2,chi1z,chi2z], 
    # so we will just set all other parameters to some default values for now!
    # injection_parameters = dict(
    #     mass_1=60.0,
    #     mass_2=60.0,
    #     a_1=0.5,  # spin-magnitude of the primary black hole
    #     a_2=0.5,  # spin-magnitude of the secondary black hole
    #     tilt_1=0.0,  # tilt angle of the primary black hole's spin vector with respect to the orbital angular momentum
    #     tilt_2=0.0,  # tilt angle of the secondary black hole's spin vector with respect to the orbital angular momentum
    #     phi_12=0.0,  # azimuthal angle between the two spin vectors in the plane of the orbit
    #     phi_jl=0.0,  # azimuthal angle between the total angular momentum and the orbital angular momentum in the plane of the orbit
    #     luminosity_distance=400.0,
    #     theta_jn=0.0,  # angle between the total angular momentum and the line of sight, aka inclination angle
    #     # psi=2.659,
    #     phase=0.0,
    #     geocent_time=1126259642.413,
    #     # ra=1.375,
    #     # dec=-1.2108,
    # )
    injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
    luminosity_distance=2000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    )

    # Set up interferometers.  In this case we'll use two interferometers
    # (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
    # sensitivity
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLE_RATE,
        duration=DURATION,
        start_time=injection_parameters["geocent_time"] - 2,
    )
    ifos.inject_signal(
        waveform_generator=injection_generator,
        parameters=injection_parameters,
    )
    print("Signal injected into interferometer data.")

    # Set up a PriorDict, which inherits from dict.
    # By default we will sample all terms in the signal models.  However, this will
    # take a long time for the calculation, so for this example we will set almost
    # all of the priors to be equall to their injected values.  This implies the
    # prior is a delta function at the true, injected value.  In reality, the
    # sampler implementation is smart enough to not sample any parameter that has
    # a delta-function prior.
    # The above list does *not* include mass_1, mass_2, theta_jn and luminosity
    # distance, which means those are the parameters that will be included in the
    # sampler.  If we do nothing, then the default priors get used.
    priors = bilby.gw.prior.BBHPriorDict()
    for key in [
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "psi",
        "ra",
        "dec",
        "geocent_time",
        "phase",
    ]:
        priors[key] = injection_parameters[key]


    # Perform a check that the prior does not extend to a parameter space longer than the data
    priors.validate_prior(DURATION, FMIN)

    # Initialise the likelihood by passing in the interferometer data (ifos)
    # -- For each likelihood, we use the ML generated waveform!
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, 
        waveform_generator=waveform_generator
    )

    # Run sampler. In this case we're going to use the `dynesty` sampler
    # Note that the `nlive`, `naccept`, and `sample` parameters are specified
    # to ensure sufficient convergence of the analysis.
    # We set `npool=16` to parallelize the analysis over 16 cores.
    # The conversion function will determine the distance posterior in post processing
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=1000,
        naccept=60,
        sample="acceptance-walk",
        npool=16,
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
    )

    # Plot the inferred waveform superposed on the actual data.
    result.plot_waveform_posterior(n_samples=1000)

    # Make a corner plot.
    result.plot_corner()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CVAE model on GW waveforms")
    parser.add_argument('--label', type=str, default='umamipe',
                        help="Label for the analysis (default: umamipe)")
    parser.add_argument('--project-dir', type=str, choices=['cvae@taiwan', 'v0p1', '@alvin', '@korea'], 
                        default=PROJECT_DIR, help="Base directory for the project (default: current directory)")
    parser.add_argument('--model-config', type=str, default='modelconfig-cvae-paper-I',
                        help="Name of the model configuration JSON file (default: None)")
    parser.add_argument('--model-name', type=str, default='model-20251004_072338-10',
                        help="Name of the trained model checkpoint (default: None)")
    
    parser.add_argument('--with-original-model', action='store_true',
                        help="Whether to use the original CVAE model instead of the FlexCVAE (default: False)")
    
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()


    # if args.debug:
    #     log_level = logging.DEBUG
    # elif args.verbose:
    #     log_level = logging.INFO
    # else:
    #     log_level = logging.WARNING

    # logfname = f"umamipe-{NOW}.log"
    # log_dir = f'../logs/{TODAY}/'
    # os.makedirs(log_dir, exist_ok=True)
    # log_file = os.path.join(log_dir, logfname)
    # logging.basicConfig(
    #     format='%(asctime)s: %(levelname)s: %(message)s',
    #     level=log_level,
    #     datefmt='%y-%m-%d %H:%M:%S',
    #     force=True,
    #     handlers=[
    #         logging.StreamHandler(),  # Log to console
    #         logging.FileHandler(log_file)  # Log to file
    #     ]
    # )
    
    # # Set FileHandler to always be at least INFO level
    # for handler in logging.root.handlers:
    #     if isinstance(handler, logging.FileHandler):
    #         handler.setLevel(max(handler.level, logging.INFO))

    print(f'Working on device: {DEVICE}, with precision: {PRECISION}')

    main(args, label=args.label)

