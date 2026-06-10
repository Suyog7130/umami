
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
from cvae import CVAE

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


# bilby.core.utils.setup_logger(outdir=f'../logs/{TODAY}', label='umamipe', log_level="INFO")

# Set up a random seed for result reproducibility.  This is optional!
bilby.core.utils.random.seed(42)


def get_td_SEOBNRv4ml(time_array, model=None, **kwargs):
    """
    Generate a waveform using the ML model based on the input parameters.

    Arguments
    ---------
        time_array: np.ndarray
            The array of time points at which to evaluate the waveform.
            This input is ignored by the waveform generator!
        model: torch.nn.Module
            The trained ML model to use for waveform generation. 
            If None, the model will be initialized using the provided model_path 
            and config_path in kwargs.
        distance_scale_factor: float
            A distance scale factor to apply to the waveform amplitude to correct for the fact that
            the ML model generates waveforms at a fixed luminosity distance of 1.0 MPc!

    Returns
    -------
        np.ndarray
            The generated time-domain strain waveform as a 1D numpy array.
    
    Note
    ----
    Bilby sent a list of `*parameters` to the waveform generator function!
    We assume this list contains either [m1, m2, chi1z, chi2z] or [mass_1, mass_2, spin_1z, spin_2z],
    or ['mass_ratio', 'chirp_mass', 'a_1', 'a_2'] depending on how the parameters are formatted!
    
    Additionally, the `**kwargs` should contain the `modelpath` and `configpath` for the ML model, 
    which we will use to load the model and generate the waveform!
    """
    logger.info(f"Received parameters for waveform generation: {kwargs}")
    if model is None:
        logger.warning("No ML model provided to get_td_SEOBNRv4ml. We will initialize the model using the provided model_path and config_path in kwargs!")
        # mlmodel=f'../{PROJECT_DIR}/trained-models/model-20251004_072338-10'
        if any(key not in kwargs for key in ['model_path', 'config_path']):
            raise ValueError("Missing 'model_path' or 'config_path' in kwargs for waveform generation.")
        model = load_flex_model(model_path=kwargs['model_path'], 
                                configpath=kwargs['config_path'], 
                                device=DEVICE, precision=PRECISION)
    
    parameters = {model_param: kwargs[model_param] for model_param in ['mass_1', 'mass_2', 'spin_1z', 'spin_2z']}
    labels = torch.tensor([parameters[key] for key in sorted(parameters.keys())], 
                            dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    generated_waveform = model.generate(labels)  # has shape (1, 2=[hp,hc], sequence_length)!
    logger.info(f"Generated waveform from ML model with shape: {generated_waveform.shape}")

    hplus, hcross = generated_waveform[0][0], generated_waveform[0][1]

    # -- add two dummy repeated value at the start to makeup for length req by Bilby Interferometer.
    hplus = np.concatenate([[hplus[0],hplus[1]], hplus])
    hcross = np.concatenate([[hcross[0],hcross[1]], hcross])
    logger.info(f"Waveform shapes after adding dummy element at the start: {hplus.shape}, {hcross.shape}")

    waveforms = {'plus': hplus, 'cross': hcross}

    distance_scale_factor = kwargs.get('distance_scale_factor', None)
    if distance_scale_factor is not None:
        waveforms['plus'] /= distance_scale_factor
        waveforms['cross'] /= distance_scale_factor

    # fig, ax = plt.subplots(figsize=(12, 5))
    # ax.plot(np.arange(len(waveforms['plus'])), waveforms['plus'], label='hp')
    # ax.plot(np.arange(len(waveforms['cross'])), waveforms['cross'], label='hc')
    # ax.legend()
    # plt.savefig(f'check-global-denorming-outputs_{NOW}.png', dpi=300)
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
        logger.info(f"Converted (mass_ratio, chirp_mass) to (mass_1, mass_2): {m1}, {m2}")
    new_parameters["mass_1"] = m1
    new_parameters["mass_2"] = m2
    # new_parameters.pop("mass_ratio", None)
    # new_parameters.pop("chirp_mass", None)

    spin_1z = parameters.get("chi_1z", parameters.get("spin_1z", None))
    spin_2z = parameters.get("chi_2z", parameters.get("spin_2z", None))
    if spin_1z is None and spin_2z is None:
        if "a_1" in parameters and "a_2" in parameters and "tilt_1" in parameters and "tilt_2" in parameters and "phi_12" in parameters and "phi_jl" in parameters and "theta_jn" in parameters:
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
        else:
            raise ValueError("Missing spin parameters for conversion. Please provide either (spin_1z, spin_2z) or (a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, theta_jn). Provided parameters: ", parameters)
    new_parameters["spin_1z"] = spin_1z
    new_parameters["spin_2z"] = spin_2z
    logger.info(f"Converted (a_i, tilt_i, phi_i) to (spin_i_z): {spin_1z}, {spin_2z}")
    if "spin_1z" in parameters or "chi_1z" in parameters:
        spin_1z_input = parameters.get("chi_1z", parameters.get("spin_1z", None))
        assert new_parameters["spin_1z"]==spin_1z_input, "Spin parameter conversion failed!"
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
        time_domain_source_model = kwargs.get('time_domain_source_model', None)
        frequency_domain_source_model = kwargs.get('frequency_domain_source_model', None)
        model_path = kwargs['waveform_arguments'].get('model_path', None) if 'waveform_arguments' in kwargs else None
        config_path = kwargs['waveform_arguments'].get('config_path', None) if 'waveform_arguments' in kwargs else None

        if time_domain_source_model is None and frequency_domain_source_model is None:
            logger.warning("No source model provided to MLWaveformGenerator. We will initialize our ML model!")
            if model_path is None or config_path is None:
                raise ValueError("Missing 'model_path' or 'config_path' in waveform_arguments for ML model initialization.")
            self.init_mlmodel(model_path=model_path, config_path=config_path)

        # -- add `time_domain_source_model` to kwargs so that they can be used in the super class!
        kwargs['time_domain_source_model'] = self.time_domain_source_model
        # -- init __super__ class after initializing the ML model, so that the model can be used in the time_domain_strain method!
        super().__init__(**kwargs)

    def init_mlmodel(self, model_path, config_path):
        """
        Initialize the ML model for waveform generation. This method can be called to load the model after the generator is initialized.
        """
        logger.info(f"Initializing ML model with model_path: {model_path} and config_path: {config_path}")
        self.loaded_mlmodel = load_flex_model(model_path=model_path, configpath=config_path, 
                                              device=None, precision=None)
        self.time_domain_source_model = self.get_ml_waveform
        self.frequency_domain_source_model = None  # We will only use the time-domain model for now!
        logger.info("ML model initialized for waveform generation in MLWaveformGenerator.")

    def check_model_weights_on_device(self, device=DEVICE, precision=PRECISION):
        if self.loaded_mlmodel is None:
            logger.warning("ML model not initialized yet. Please call init_mlmodel() first.")
            return
        # -- Check if model weights loaded are of the same precision as our initialized model.
        # -- If not, then convert loaded model to the correct precision before moving to device.
        for name, param in self.loaded_mlmodel.named_parameters():
            if param.dtype != precision:
                logger.info(f"Converting model parameter '{name}' from {param.dtype} to {precision} for consistency with initialized model precision.")
                param.data = param.data.to(getattr(torch, precision))
        # -- Check if model weights are already on the correct device before moving.
        for name, param in self.loaded_mlmodel.named_parameters():
            if param.device != device:
                logger.info(f"Moving model parameter '{name}' from {param.device} to {device}.")
                param.data = param.data.to(device)
            else:
                logger.info(f"Model parameter '{name}' is already on the correct device: {device}.")

    def get_ml_waveform(self, time_array, **kwargs):
        return get_td_SEOBNRv4ml(time_array, model=self.loaded_mlmodel, **kwargs)
    
    def time_domain_strain(self, parameters=None):
        """
        Override the time_domain_strain method to use the ML model for waveform generation.
        This method is called by Bilby to get the strain for given parameters.
        """
        return self._calculate_strain(model=self.time_domain_source_model,
                                      model_data_points=self.time_array,
                                      parameters=parameters,
                                      transformation_function=utils.infft,
                                      transformed_model=self.frequency_domain_source_model,
                                      transformed_model_data_points=self.frequency_array)

    def frequency_domain_strain(self, parameters=None):
        return self._calculate_strain(model=self.frequency_domain_source_model,
                                      model_data_points=self.frequency_array,
                                      parameters=parameters,
                                      transformation_function=utils.nfft,
                                      transformed_model=self.time_domain_source_model,
                                      transformed_model_data_points=self.time_array)
    
    def _strain_from_model(self, model_data_points, model, parameters):
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
        logger.info(f"Generating waveform using ML model for parameters: {parameters}")
        logger.info(f"Using model: {model} and transformed model {transformed_model}")
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
        # logger.info("Converted spins and inclination:", iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z)

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
        return new_parameters

    def _strain_from_transformed_model(
        self, transformed_model_data_points, transformed_model, transformation_function, parameters
    ):
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


