
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
import argparse
import pandas as pd

from tqdm import tqdm
from datetime import datetime

import bilby
from bilby.core.utils import logger
from bilby.gw import WaveformGenerator


import torch
import torch.nn.functional as F
import numpy as np

from flexcvae import FlexTwoC2E1D, FlexCAE, FlexCAEPhase
from optimize import load_flex_model

TODAY = datetime.now().strftime("%Y-%m-%d")

# -- define some constants for waveform generation
SAMPLE_RATE = 8192  # Hz
DURATION = 1.0  # seconds
FMIN = 20.0  # Hz
FREF = 20.0  # Hz


bilby.core.utils.setup_logger(outdir=f'../results/{TODAY}', label='umamipe', log_level="DEBUG")

# Set up a random seed for result reproducibility.  This is optional!
bilby.core.utils.random.seed(42)


def get_td_SEOBNRv4ml(parameters, mlmodel):
    """
    Generate a waveform using the ML model based on the input parameters.
    """
    # -- convert parameters to tensor and move to model device
    labels = torch.tensor([parameters[key] for key in sorted(parameters.keys())], 
                            dtype=torch.float32).unsqueeze(0).to(mlmodel.MODEL_CONFIG.device)
    # -- generate waveform using the model's generate method
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
    def __init__(self, mlmodel, **kwargs):
        super().__init__(**kwargs)
        self.mlmodel = mlmodel
        self.mlmodel.MODEL_CONFIG['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mlmodel.to(self.mlmodel.MODEL_CONFIG.device)
        logger.info(f"ML model loaded and moved to device: {self.mlmodel.MODEL_CONFIG['device']}")
    
    def time_domain_strain(self, parameters: dict):
        """
        Override the time_domain_strain method to use the ML model for waveform generation.
        This method is called by Bilby to get the strain for given parameters.
        """
        return self.time_domain_source_model(parameters, self.mlmodel)



def main(args):
    model = load_flex_model(model_path=args.model_config, 
                            config_path=args.model_path)
    
    # -- initialize the ML waveform generator with the loaded model
    # NOTE: We will only use this for the likelihood evaluation in the sampler!
    # Whereas, the injection is performed using LAL waveform!
    waveform_generator = MLWaveformGenerator(
        mlmodel=model,
        duration=DURATION,
        sampling_frequency=SAMPLE_RATE,
        time_domain_source_model=get_td_SEOBNRv4ml
        )
    
    # -- Our ML model is only for [m1,m2,chi1z,chi2z], 
    # so we will just set all other parameters to some default values for now!
    injection_parameters = dict(
        mass_1=36.0,
        mass_2=29.0,
        a_1=0.4,
        a_2=0.3,
        tilt_1=0.5,
        tilt_2=1.0,
        phi_12=1.7,
        phi_jl=0.3,
        luminosity_distance=100.0,
        theta_jn=0.4,
        psi=2.659,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
    )

    # Fixed arguments passed into the source model
    waveform_arguments = dict(
        waveform_approximant="SEOBNRv4",
        reference_frequency=FREF,
        minimum_frequency=FMIN,
    )





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CVAE model on GW waveforms")
    
    parser.add_argument('--model-config', type=str, default=None,
                        help="Path to the model configuration JSON file (default: None)")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Path to the trained model checkpoint (default: None)")
    
    args = parser.parse_args()
    main(args)

