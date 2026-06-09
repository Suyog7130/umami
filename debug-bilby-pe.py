#!/usr/bin/env python3
"""
Minimal 4D Bilby parameter-estimation test:

    IMRPhenomD injection -> IMRPhenomD recovery

Sampled parameters:
    mass_1, mass_2, spin_1z, spin_2z

Fixed parameters:
    luminosity_distance, theta_jn, phase, geocent_time,
    ra, dec, psi, and spin-precession angles.

This script is meant as a clean sanity check before using ML waveforms.
"""

import os
import argparse
import logging
import numpy as np
import bilby


# ----------------------------------------------------------------------
# Basic configuration
# ----------------------------------------------------------------------

APPROXIMANT = "SEOBNRv4" #"IMRPhenomD"
DURATION = 1.0
SAMPLING_FREQUENCY = 8192.0
FMIN = 20.0
FREF = 50.0

if APPROXIMANT == "IMRPhenomD":
    DEFAULT_OUTDIR = "out_phenomd_4d"
    DEFAULT_LABEL = "phenomd_4d"
elif APPROXIMANT == "SEOBNRv4":
    DEFAULT_OUTDIR = "out_seobnr4_4d"
    DEFAULT_LABEL = "seobnr4_4d"


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def setup_logger(outdir, label):
    os.makedirs(outdir, exist_ok=True)
    bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level="INFO")
    logging.getLogger("bilby").setLevel(logging.INFO)


# ----------------------------------------------------------------------
# Signed aligned-spin wrapper
# ----------------------------------------------------------------------

def signed_chi_to_a_tilt(chi_z):
    """
    Convert signed aligned spin chi_z into Bilby's spin magnitude + tilt.

    chi_z >= 0:
        a = |chi_z|, tilt = 0
    chi_z < 0:
        a = |chi_z|, tilt = pi

    This lets us sample spin_1z and spin_2z directly while still using
    bilby.gw.source.lal_binary_black_hole.
    """
    a = abs(float(chi_z))
    tilt = 0.0 if chi_z >= 0.0 else np.pi
    return a, tilt


def aligned_spin_source(
    frequency_array,
    mass_1,
    mass_2,
    spin_1z,
    spin_2z,
    luminosity_distance,
    theta_jn,
    phase,
    **kwargs,
):
    """
    Frequency-domain source model for {APPROXIMANT} with signed aligned spins.

    Bilby/LAL wants:
        a_1, tilt_1, a_2, tilt_2

    We expose:
        spin_1z, spin_2z

    and internally map signed aligned spins to magnitude + tilt.
    """

    a_1, tilt_1 = signed_chi_to_a_tilt(spin_1z)
    a_2, tilt_2 = signed_chi_to_a_tilt(spin_2z)

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
        waveform_approximant=APPROXIMANT,
        reference_frequency=FREF,
        minimum_frequency=FMIN,
        catch_waveform_errors=True,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Injection and priors
# ----------------------------------------------------------------------

def make_injection_parameters():
    """
    One clean GW150914-like-ish BBH injection.

    The sampled parameters are:
        mass_1, mass_2, spin_1z, spin_2z

    Everything else is fixed.
    """
    return dict(
        mass_1=36.0,
        mass_2=32.0,
        spin_1z=0.30,
        spin_2z=-0.20,
        luminosity_distance=400.0,
        theta_jn=0.4,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
        psi=2.659,
    )


def make_4d_priors(injection_parameters):
    """
    Priors for a 4D PE run.

    Only mass_1, mass_2, spin_1z, spin_2z are sampled.
    All extrinsic parameters are fixed at injected values.

    Keep the injected values safely inside the prior ranges.
    """
    priors = bilby.core.prior.PriorDict()

    priors["mass_1"] = bilby.core.prior.Uniform(
        minimum=30.0,
        maximum=45.0,
        name="mass_1",
        latex_label="$m_1$",
        unit="$M_\\odot$",
    )

    priors["mass_2"] = bilby.core.prior.Uniform(
        minimum=25.0,
        maximum=40.0,
        name="mass_2",
        latex_label="$m_2$",
        unit="$M_\\odot$",
    )

    priors["spin_1z"] = bilby.core.prior.Uniform(
        minimum=-0.8,
        maximum=0.8,
        name="spin_1z",
        latex_label="$\\chi_{1z}$",
    )

    priors["spin_2z"] = bilby.core.prior.Uniform(
        minimum=-0.8,
        maximum=0.8,
        name="spin_2z",
        latex_label="$\\chi_{2z}$",
    )

    # Fixed parameters
    for key in [
        "luminosity_distance",
        "theta_jn",
        "phase",
        "geocent_time",
        "ra",
        "dec",
        "psi",
    ]:
        priors[key] = injection_parameters[key]

    return priors


# ----------------------------------------------------------------------
# Waveform generator, detectors, likelihood
# ----------------------------------------------------------------------

def make_waveform_generator():
    """
    Use the same waveform generator for injection and recovery.

    This is the cleanest sanity check:
        {APPROXIMANT} -> {APPROXIMANT}
    """
    return bilby.gw.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=aligned_spin_source,
        waveform_arguments=dict(),
    )


def make_interferometers(injection_parameters, waveform_generator, zero_noise=False):
    """
    Build H1 and L1 design-sensitivity data and inject the signal.

    zero_noise=True is recommended for first debugging.
    zero_noise=False uses Gaussian noise drawn from the design PSD.
    """
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])

    start_time = injection_parameters["geocent_time"] - DURATION / 2.0

    if zero_noise:
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=DURATION,
            start_time=start_time,
        )
    else:
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=DURATION,
            start_time=start_time,
        )

    ifos.inject_signal(
        waveform_generator=waveform_generator,
        parameters=injection_parameters,
    )

    return ifos


def make_likelihood(ifos, waveform_generator):
    return bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
    )


# ----------------------------------------------------------------------
# Debug diagnostics
# ----------------------------------------------------------------------

def print_injection_snr(ifos):
    print("\n========== INJECTION SNR DEBUG ==========")
    for ifo in ifos:
        print(f"\n{ifo.name}")
        for key, value in ifo.meta_data.items():
            print(f"  {key}: {value}")
    print("========================================\n")


def debug_likelihood(likelihood, priors, injection_parameters, n_random=8):
    """
    Check likelihood values before sampling.

    The log_likelihood may contain a large constant.
    The log_likelihood_ratio is usually more interpretable.
    """
    print("\n========== LIKELIHOOD DEBUG ==========")

    likelihood.parameters.update(injection_parameters)
    ll = likelihood.log_likelihood()
    llr = likelihood.log_likelihood_ratio()

    print("At injected parameters:")
    print(f"  log_likelihood       = {ll:.6e}")
    print(f"  log_likelihood_ratio = {llr:.6e}")

    print("\nAt random prior samples:")
    for i in range(n_random):
        sample = priors.sample()
        likelihood.parameters.update(sample)
        try:
            ll_i = likelihood.log_likelihood()
            llr_i = likelihood.log_likelihood_ratio()
            print(f"  random {i:02d}: logL={ll_i:.6e}, logLR={llr_i:.6e}")
        except Exception as exc:
            print(f"  random {i:02d}: failed with {exc}")

    print("======================================\n")


def debug_waveform_scale(waveform_generator, injection_parameters):
    """
    Print waveform array shape and scale.
    """
    print("\n========== WAVEFORM DEBUG ==========")

    fd = waveform_generator.frequency_domain_strain(injection_parameters)

    for pol in ["plus", "cross"]:
        arr = np.asarray(fd[pol])
        print(f"{pol}:")
        print(f"  shape   = {arr.shape}")
        print(f"  finite  = {np.all(np.isfinite(arr))}")
        print(f"  max abs = {np.max(np.abs(arr)):.6e}")

    print("====================================\n")


# ----------------------------------------------------------------------
# Sampler
# ----------------------------------------------------------------------

def run_pe(
    sampler,
    nlive,
    dlogz,
    npool,
    outdir,
    label,
    zero_noise,
    debug_only,
):
    setup_logger(outdir, label)

    injection_parameters = make_injection_parameters()
    priors = make_4d_priors(injection_parameters)

    waveform_generator = make_waveform_generator()

    ifos = make_interferometers(
        injection_parameters=injection_parameters,
        waveform_generator=waveform_generator,
        zero_noise=zero_noise,
    )

    likelihood = make_likelihood(
        ifos=ifos,
        waveform_generator=waveform_generator,
    )

    print("\n========== RUN CONFIG ==========")
    print(f"sampler                 = {sampler}")
    print(f"nlive                   = {nlive}")
    print(f"dlogz                   = {dlogz}")
    print(f"npool                   = {npool}")
    print(f"zero_noise              = {zero_noise}")
    print(f"duration                = {DURATION}")
    print(f"sampling_frequency      = {SAMPLING_FREQUENCY}")
    print(f"minimum_frequency       = {FMIN}")
    print("sampled parameters      = mass_1, mass_2, spin_1z, spin_2z")
    print(f"waveform approximant    = {APPROXIMANT}")
    print("================================\n")

    print("Injection parameters:")
    for key, value in injection_parameters.items():
        print(f"  {key}: {value}")

    print_injection_snr(ifos)
    debug_waveform_scale(waveform_generator, injection_parameters)
    debug_likelihood(likelihood, priors, injection_parameters)

    if debug_only:
        print("Debug-only mode requested. Not running sampler.")
        return None

    # Bilby docs describe run_sampler as the core sampler interface.
    # Dynesty is a good default for this simple test.
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=sampler,
        nlive=nlive,
        dlogz=dlogz,
        npool=npool,
        sample="acceptance-walk",
        naccept=10,
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
        result_class=bilby.gw.result.CBCResult,
        conversion_function=None,
        resume=False,
        clean=True,
    )

    # Save useful plots.
    corner_file = os.path.join(outdir, f"{label}_corner.png")
    result.plot_corner(
        parameters=["mass_1", "mass_2", "spin_1z", "spin_2z"],
        save=True,
        filename=corner_file,
    )

    try:
        waveform_file = os.path.join(outdir, f"{label}_waveform_posterior.png")
        result.plot_waveform_posterior(
            n_samples=100,
            save=True,
            filename=waveform_file,
        )
    except Exception as exc:
        print(f"Waveform posterior plot failed, but PE result is saved. Reason: {exc}")

    print("\n========== RESULT SUMMARY ==========")
    print(f"Result saved in: {outdir}")
    print(f"Corner plot: {corner_file}")
    print("Posterior head:")
    print(result.posterior[["mass_1", "mass_2", "spin_1z", "spin_2z"]].head())
    print("====================================\n")

    return result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Minimal 4D Bilby IMRPhenomD injection-recovery test."
    )

    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--label", default=DEFAULT_LABEL)

    parser.add_argument(
        "--sampler",
        default="dynesty",
        choices=["dynesty"],
        help="Keep dynesty for this first clean baseline test.",
    )

    parser.add_argument(
        "--nlive",
        type=int,
        default=200,
        help="Use 100 to 200 for debugging, 500 or more for nicer posteriors.",
    )

    parser.add_argument(
        "--dlogz",
        type=float,
        default=0.5,
        help="Stopping criterion. Use 1.0 for quick debug, 0.1 for stricter runs.",
    )

    parser.add_argument(
        "--npool",
        type=int,
        default=1,
        help="Dynesty worker pool. Keep 1 first, then try more if stable.",
    )

    parser.add_argument(
        "--zero-noise",
        action="store_true",
        help="Use zero-noise injection. Recommended for first understanding PE.",
    )

    parser.add_argument(
        "--debug-only",
        action="store_true",
        help="Build waveform/data/likelihood and print diagnostics, but do not sample.",
    )

    args = parser.parse_args()

    bilby.core.utils.random.seed(1234)
    np.random.seed(1234)

    run_pe(
        sampler=args.sampler,
        nlive=args.nlive,
        dlogz=args.dlogz,
        npool=args.npool,
        outdir=args.outdir,
        label=args.label,
        zero_noise=args.zero_noise,
        debug_only=args.debug_only,
    )


if __name__ == "__main__":
    main()