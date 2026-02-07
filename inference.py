
"""
Use trained surrogate ML model and do Bayesian parameter estimation on this
using Bilby, and obtain a Posterior Probability plot.
"""

import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch

import bilby
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.detector import InterferometerList
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.core.prior import Uniform
from bilby.gw.prior import PriorDict
from bilby.core.result import make_pp_plot

from maincvae import Test
from cvae import CVAE



class SEOBNRv4ml:
    def __init__(self, model_path="../trained-models/model-20251004_072338-10"):
        self.load_model(model_path)
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def load_model(self, model_path):
        model = CVAE(input_shape=(2, 8191), num_classes=4, 
                    key_shape=(2,2)).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        self.model = model

    def __call__(self, **wfkwargs):
        mass_1 = wfkwargs["m1"]
        mass_2 = wfkwargs["m2"]
        chi_1 = wfkwargs["chi1z"]
        chi_2 = wfkwargs["chi2z"]

        params = np.array([[mass_1, mass_2, chi_1, chi_2]])

        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, dtype=torch.float32)

        params = params.to(self.device)
        num_samples = params.shape[0]
        logging.info(f'Generating {num_samples} samples conditioned on provided labels.')
        with torch.no_grad():
            z1_mean, z1_log_var = self.model.encode_label_for_x(params)
            z1p_mean, z1p_log_var = self.model.encode_label_for_key(params)
            z1 = self.model.reparameterize(z1_mean, z1_log_var)
            z1p = self.model.reparameterize(z1p_mean, z1p_log_var)
            generated = self.model.decode(z1, z1p, params)

        # TODO: the waveform outputs here from the trained model
        # are normalized. Denormalization cannot be performed, if
        # the parameters are the not the ones from the training,
        # validation or test sets. The model training targets should
        # always have been unnormalized waveforms. This has to be done!
        generated = generated.cpu().numpy()
        return generated[0], generated[1]



duration = 4.0
sampling_frequency = 1024.0
outdir = "toy_pp_out"
label_base = "toy_surrogate_pp"

bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)

ifos = InterferometerList(["H1", "L1"])
for ifo in ifos:
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        psd_file=None
    )

priors = PriorDict()
priors["mass_1"] = Uniform(5, 75, "mass_1")
priors["mass_2"] = Uniform(5, 75, "mass_2")
priors["chi_1"] = Uniform(-0.99, 0.99, "chi_1")
priors["chi_2"] = Uniform(-0.99, 0.99, "chi_2")
# priors["luminosity_distance"] = Uniform(500, 1500, "luminosity_distance")


def run_single_injection(run_idx, seed=None, generator=None):
    if seed is not None:
        np.random.seed(seed + run_idx)

    injection_parameters = priors.sample()
    injection_parameters.setdefault("geocent_time", 0.0)

    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"],
    )

    ifos.inject_signal(
        waveform_generator=generator,
        parameters=injection_parameters,
    )

    likelihood = GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=generator,
    )

    this_label = f"{label_base}_inj_{run_idx:04d}"
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=300,
        dlogz=0.2,
        outdir=outdir,
        label=this_label,
        injection_parameters=injection_parameters,
        resume=False,
        clean=True,
    )

    return result

def run_injection_campaign(N_injections=50, base_seed=1234,
                           generator=None):
    if generator is None:
        raise NotImplementedError("Waveform generator must be provided")
    results = []
    for i in range(N_injections):
        print(f"Injection {i+1}/{N_injections}")
        res = run_single_injection(i, seed=base_seed,
                                   generator=generator)
        results.append(res)
    return results


def main():
    logging.info("Starting Bayesian Parameter Estimation using Bilby...")

    mlmodel = Test().generate()

    generator = WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        time_domain_source_model=mlmodel,
        waveform_arguments=dict(),
    )

    results = run_injection_campaign(N_injections=50,
                                      base_seed=1234,
                                      generator=generator)

    # Bilby built-in PP plot
    fig, pvals = make_pp_plot(
        results,
        filename=f"{outdir}/{label_base}_pp.png",
        save=True,
    )
    print("Combined p-value:", pvals.combined_pvalue)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Parameter Estimation using Bilby")
    parser.add_argument("-v", "--verbose" , action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    main()