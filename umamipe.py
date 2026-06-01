
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


import torch
import torch.nn.functional as F
import numpy as np

from flexcvae import FlexTwoC2E1D, FlexCAE, FlexCAEPhase
from optimize import load_flex_model
from cvae import CVAE

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


def get_td_SEOBNRv4ml(time_array, mass_1, mass_2, spin_1z, spin_2z):
    """
    Generate a waveform using the ML model based on the input parameters.

    Arguments
    ---------
        time_array: np.ndarray
            The array of time points at which to evaluate the waveform.
            This input is ignored by the waveform generator!

    NOTE: Bilby send a list of `*parameters` to the waveform generator function!
    We assume this list contains the following parameters (after conversion to LAL format):
        mass_1: float
            Mass of the primary black hole in solar masses.
        mass_2: float
            Mass of the secondary black hole in solar masses.
        spin_1z: float
            Dimensionless spin of the primary black hole along the z-axis.
        spin_2z: float
            Dimensionless spin of the secondary black hole along the z-axis.

    Returns
    -------
        np.ndarray
            The generated time-domain strain waveform as a 1D numpy array.
    """
    mlmodel='../trained-models/model-20251004_072338-10'
    print("Generating waveform using ML model for parameters:", locals())
    print("Time array shape:", time_array.shape)
    parameters = {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "spin_1z": spin_1z,
        "spin_2z": spin_2z,
    }
    # -- convert parameters to tensor and move to model device
    if type(parameters) is not dict:
        raise TypeError("Expected parameters to be a dictionary, got {}".format(type(parameters)))
    labels = torch.tensor([parameters[key] for key in sorted(parameters.keys())], 
                            dtype=torch.float32).unsqueeze(0).to(mlmodel.MODEL_CONFIG.device)
    exit(0)
    generated_waveform = mlmodel.generate(labels)
    return generated_waveform.cpu().numpy().flatten()  # Return as 1D numpy array

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
            phase=parameters["phase"],
        )
        print("Converted spins and inclination:", iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)

        ml_parameters = {
            "mass_1": new_parameters["mass_1"],
            "mass_2": new_parameters["mass_2"],
            "spin_1z": spin_1z,
            "spin_2z": spin_2z,
        }

        for key in self.source_parameter_keys.symmetric_difference(
            ml_parameters.keys()):
            new_parameters.pop(key)
        new_parameters.update(ml_parameters)
        new_parameters.update(self.waveform_arguments)
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




def main(args, outdir='../results/{TODAY}', label='umamipe'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not args.model_path.startswith('../trained-models/'):
        model_path = os.path.join('../trained-models/', args.model_path)
    else:
        model_path = args.model_path

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
    parser.add_argument('--model-config', type=str, default=None,
                        help="Path to the model configuration JSON file (default: None)")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Path to the trained model checkpoint (default: None)")
    
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

