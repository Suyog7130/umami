#!/usr/bin/env python3
"""
Minimal 4D Bilby PE script for:

    ML waveform injection -> ML waveform recovery

Sampled parameters:
    mass_1, mass_2, spin_1z, spin_2z

This is designed to debug a custom time-domain MLWaveformGenerator before
running large PP-plot campaigns.
"""

import os
import json
import time
import datetime
import argparse
import logging
import importlib
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt

import bilby
from bilby.gw.detector import InterferometerList
from bilby.gw.likelihood import GravitationalWaveTransient

import torch
import torch.multiprocessing as mp

from plotutils import putils

from inference import (
    base_injection,
    make_wf_generator,
    make_analysis_priors,
    sample_injection_from_priors,
)

NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

DEFAULT_DURATION = 1.0
DEFAULT_SAMPLING_FREQUENCY = 8192.0
DEFAULT_FMIN = 20.0
DEFAULT_FREF = 50.0
DEFAULT_OUTDIR = f"out_ml2ml_pe_debug_{NOW}"
DEFAULT_LABEL = "ml2ml_pe_debug"


def import_symbol(path: str) -> Any:
    if ":" in path:
        module_name, symbol_name = path.split(":", 1)
    else:
        module_name, symbol_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def setup_logger(outdir: str, label: str, log_level: str = "INFO") -> None:
    os.makedirs(outdir, exist_ok=True)
    bilby.core.utils.setup_logger(outdir=outdir, label=label, log_level=log_level)
    logging.getLogger("bilby").setLevel(getattr(logging, log_level.upper()))
    logging.getLogger("nessai").setLevel(getattr(logging, log_level.upper()))


def to_jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.complex128):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def save_json(data: Dict[str, Any], filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=True)


def finite_summary(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr)
    out = {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "all_finite": bool(np.all(np.isfinite(arr))),
    }
    if arr.size == 0:
        out.update({"min": None, "max": None, "max_abs": None, "rms": None})
        return out
    abs_arr = np.abs(arr)
    out.update(
        {
            "min": float(np.nanmin(arr.real)) if np.iscomplexobj(arr) else float(np.nanmin(arr)),
            "max": float(np.nanmax(arr.real)) if np.iscomplexobj(arr) else float(np.nanmax(arr)),
            "max_abs": float(np.nanmax(abs_arr)),
            "rms": float(np.sqrt(np.nanmean(abs_arr**2))),
        }
    )
    return out


def print_dict(title: str, data: Dict[str, Any]) -> None:
    print(f"\n========== {title} ==========")
    for key, value in data.items():
        print(f"{key}: {value}")
    print("=" * (22 + len(title)) + "\n")




def assert_injection_inside_priors(priors, injection_parameters, keys=("mass_1", "mass_2", "chi_1", "chi_2")):
    for key in keys:
        p = priors[key].prob(injection_parameters[key])
        if not np.isfinite(p) or p <= 0:
            raise ValueError(f"Injected value {key}={injection_parameters[key]} is outside prior {priors[key]}")



def clear_waveform_cache(generator) -> None:
    if hasattr(generator, "_cache"):
        try:
            generator._cache = {"parameters": None, "waveform": None, "model": None, "transformed_model": None}
        except Exception:
            pass

def check_noise_projection_before_injection(ifos, waveform_generator, params):
    pols = waveform_generator.frequency_domain_strain(params)
    print("\n========== CHECKING NOISE PROJECTION BEFORE INJECTION ==========")
    for ifo in ifos:
        h = ifo.get_detector_response(pols, params)
        d0 = ifo.frequency_domain_strain.copy()

        mf_before = ifo.matched_filter_snr(signal=h)
        opt = np.sqrt(ifo.optimal_snr_squared(signal=h))

        print()
        print("IFO:", ifo.name)
        print("optimal SNR:", opt)
        print("matched filter SNR before injection:", mf_before)
        print("noise projection real:", np.real(mf_before))
        print("noise projection imag:", np.imag(mf_before))
    print("===============================================================\n")


def make_interferometers(args, injection_parameters, injection_generator):
    ifos = InterferometerList(list(args.ifos))
    start_time = injection_parameters["geocent_time"] - args.duration / 2.0
    if args.zero_noise:
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=args.sampling_frequency,
            duration=args.duration,
            start_time=start_time,
        )
    else:
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=args.sampling_frequency,
            duration=args.duration,
            start_time=start_time,
        )

    check_noise_projection_before_injection(
        ifos,
        injection_generator,
        injection_parameters,
    )
    ifos.inject_signal(waveform_generator=injection_generator, parameters=injection_parameters)
    return ifos


def make_likelihood(ifos, recovery_generator):
    return GravitationalWaveTransient(interferometers=ifos, waveform_generator=recovery_generator)


def waveform_debug_summary(generator, params, name: str) -> Dict[str, Any]:
    out = {"name": name}
    try:
        clear_waveform_cache(generator)
        td = generator.time_domain_strain(params)
        out["time_domain"] = {pol: finite_summary(np.asarray(td[pol])) for pol in ["plus", "cross"]}
    except Exception as exc:
        out["time_domain_error"] = repr(exc)
    try:
        clear_waveform_cache(generator)
        fd = generator.frequency_domain_strain(params)
        out["frequency_domain"] = {pol: finite_summary(np.asarray(fd[pol])) for pol in ["plus", "cross"]}
    except Exception as exc:
        out["frequency_domain_error"] = repr(exc)
    return out


def compare_generators(generator_a, generator_b, params, name_a="injection", name_b="recovery") -> Dict[str, Any]:
    out = {"name_a": name_a, "name_b": name_b}
    for domain, method in [("time_domain", "time_domain_strain"), ("frequency_domain", "frequency_domain_strain")]:
        out[domain] = {}
        try:
            clear_waveform_cache(generator_a)
            clear_waveform_cache(generator_b)
            ha = getattr(generator_a, method)(params)
            hb = getattr(generator_b, method)(params)
            for pol in ["plus", "cross"]:
                a = np.asarray(ha[pol])
                b = np.asarray(hb[pol])
                diff = a - b
                out[domain][pol] = {
                    "shape_a": tuple(a.shape),
                    "shape_b": tuple(b.shape),
                    "max_abs_a": float(np.max(np.abs(a))),
                    "max_abs_b": float(np.max(np.abs(b))),
                    "max_abs_diff": float(np.max(np.abs(diff))),
                    "relative_diff": float(np.max(np.abs(diff)) / (np.max(np.abs(a)) + 1e-300)),
                    "all_finite_a": bool(np.all(np.isfinite(a))),
                    "all_finite_b": bool(np.all(np.isfinite(b))),
                }
        except Exception as exc:
            out[domain]["error"] = repr(exc)
    return out


def test_ml_determinism(make_generator_fn, params, n_trials=3) -> Dict[str, Any]:
    out = {"n_trials": n_trials, "comparisons": []}
    base = make_generator_fn()
    for i in range(n_trials):
        other = make_generator_fn()
        out["comparisons"].append(compare_generators(base, other, params, name_a="base", name_b=f"trial_{i}"))
    return out


def print_injection_snr(ifos) -> Dict[str, Any]:
    out = {}
    print("\n========== INJECTION METADATA / SNR ==========")
    for ifo in ifos:
        print(f"\n{ifo.name}")
        out[ifo.name] = dict(ifo.meta_data)
        for key, value in ifo.meta_data.items():
            print(f"  {key}: {value}")
    print("==============================================\n")
    return out


def perform_1d_likelihood_scan(args, likelihood, injection_parameters, n_points=200,
                               fontsize=15, labelsize=12):
    print("\n========== 1D LIKELIHOOD SCAN ==========")
    # m1_min = priors["mass_1"].minimum
    # m1_max = priors["mass_1"].maximum
    # m1_grid = np.linspace(m1_min, m1_max, n_points)

    truth = injection_parameters.copy()
    truth_logl_direct = likelihood.log_likelihood(parameters=truth)
    print("truth logL direct:", truth_logl_direct)
    # -- Make grid around the injection value
    m1_inj = injection_parameters["mass_1"]
    m2_inj = injection_parameters["mass_2"]
    m1_grid = np.linspace(max(m2_inj, m1_inj * 0.8), m1_inj * 1.2, n_points)
    logls = []
    for m1 in m1_grid:
        p = injection_parameters.copy()
        p["mass_1"] = float(m1)
        logls.append(likelihood.log_likelihood(parameters=p))
    logls = np.array(logls)
    print("best m1:", m1_grid[np.argmax(logls)])
    print("injected m1:", injection_parameters["mass_1"])
    print("log likelihood at best m1:", logls[np.argmax(logls)])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.sort(m1_grid), logls[np.argsort(m1_grid)], label=f"{args.pe_run_type} log likelihood")
    ax.axvline(injection_parameters["mass_1"], linestyle="--", label="injected $m_1$", color="red")
    ax.set_xlabel("$m_1$", fontsize=fontsize)
    ax.set_ylabel(f"{args.pe_run_type} log likelihood")
    ax.legend()
    plt.tight_layout()
    putils.beautifyPlot(ax, labelsize=labelsize, minor=True, tickDirection="in", top=True, right=True)
    scan_plot_file = os.path.join(args.outdir, f"{args.label}_1d_likelihood_scan_mass1.png")
    if not args.no_save:
        plt.savefig(scan_plot_file, bbox_inches="tight", dpi=200)
        print(f"Saved 1D likelihood scan plot: {scan_plot_file}")
    if not args.no_show:
        plt.show()
    print("=========================================\n")
    plt.close()
    return {"m1_grid": m1_grid.tolist(), "log_likelihoods": logls.tolist()}

def perform_2d_likelihood_scan(args, likelihood, injection_parameters, n_points=200, fontsize=15, labelsize=12):
    print("\n========== 2D LIKELIHOOD SCAN ==========")
    print("injected m1:", injection_parameters["mass_1"])
    print("injected m2:", injection_parameters["mass_2"])
    # -- Make grids around the injection values
    m1_inj = injection_parameters["mass_1"]
    m2_inj = injection_parameters["mass_2"]
    m1_grid = np.linspace(max(m2_inj, m1_inj * 0.8), m1_inj * 1.2, n_points)
    m2_grid = np.linspace(m2_inj * 0.8, min(m2_inj * 1.2, m1_inj), n_points)
    logls = np.zeros((n_points, n_points))
    for i, m1 in enumerate(m1_grid):
        for j, m2 in enumerate(m2_grid):
            p = injection_parameters.copy()
            p["mass_1"] = float(m1)
            p["mass_2"] = float(m2)
            logls[i, j] = likelihood.log_likelihood(parameters=p)
    print("best m1:", m1_grid[np.unravel_index(np.argmax(logls), logls.shape)[0]])
    print("best m2:", m2_grid[np.unravel_index(np.argmax(logls), logls.shape)[1]])
    print("max log likelihood:", np.max(logls))
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(logls, extent=[m1_grid[0], m1_grid[-1], m2_grid[0], m2_grid[-1]], origin="lower")
    ax.set_xlabel("$m_1$", fontsize=fontsize)
    ax.set_ylabel("$m_2$", fontsize=fontsize)
    plt.colorbar(im, ax=ax, label=f"{args.pe_run_type} log likelihood")
    putils.beautifyPlot(ax, labelsize=labelsize, minor=True, tickDirection="in", top=True, right=True)
    scan_plot_file = os.path.join(args.outdir, f"{args.label}_2d_likelihood_scan.png")
    if not args.no_save:
        plt.savefig(scan_plot_file, bbox_inches="tight", dpi=300)
        print(f"Saved 2D likelihood scan plot: {scan_plot_file}")
    if not args.no_show:
        plt.show()
    print("=========================================\n")
    plt.close()
    return {"m1_grid": m1_grid.tolist(), "m2_grid": m2_grid.tolist(), "log_likelihoods": logls.tolist()}

def test_ml_likelihood(args, likelihood, injection_parameters):
    truth = injection_parameters.copy()
    wrong = injection_parameters.copy()
    wrong["mass_1"] = 43.0  # deliberately wrong mass_1
    print("\n========== LIKELIHOOD TEST ==========")
    print("truth:", truth)
    ll_truth = likelihood.log_likelihood(parameters=truth)
    ll_wrong = likelihood.log_likelihood(parameters=wrong)
    print(f"{args.pe_run_type} logL at truth:", ll_truth)
    print(f"{args.pe_run_type} logL at wrong m1=43:", ll_wrong)
    print("truth - wrong:", ll_truth - ll_wrong)
    print("======================================\n")
    return {
        "truth": {"parameters": truth, "log_likelihood": float(ll_truth)},
        "wrong": {"parameters": wrong, "log_likelihood": float(ll_wrong)},
        "delta_log_likelihood": float(ll_truth - ll_wrong),
    }


def debug_likelihood(likelihood, priors, injection_parameters, n_random=8) -> Dict[str, Any]:
    out = {"random_points": []}
    print("\n========== LIKELIHOOD DEBUG ==========")
    likelihood.parameters.update(injection_parameters)
    ll = likelihood.log_likelihood()
    llr = likelihood.log_likelihood_ratio()
    out["injection"] = {"log_likelihood": float(ll), "log_likelihood_ratio": float(llr)}
    print("At injected parameters:")
    print(f"  log_likelihood       = {ll:.6e}")
    print(f"  log_likelihood_ratio = {llr:.6e}")
    print("\nAt random prior points:")
    for i in range(n_random):
        sample = priors.sample()
        likelihood.parameters.update(sample)
        try:
            ll_i = likelihood.log_likelihood()
            llr_i = likelihood.log_likelihood_ratio()
            row = {"i": i, "sample": dict(sample), "log_likelihood": float(ll_i), "log_likelihood_ratio": float(llr_i)}
            print(f"  random {i:02d}: logL={ll_i:.6e}, logLR={llr_i:.6e}")
        except Exception as exc:
            row = {"i": i, "sample": dict(sample), "error": repr(exc)}
            print(f"  random {i:02d}: failed with {exc}")
        out["random_points"].append(row)
    print("======================================\n")
    return out


def compare_inj_recover_at_same_params(args, injection_generator, recovery_generator, injection_parameters,
                                       plot_waveforms=True, fontsize=15, labelsize=12) -> None:
    h_inj = injection_generator.frequency_domain_strain(injection_parameters)
    h_rec = recovery_generator.frequency_domain_strain(injection_parameters)
    print("\n========== INJECTION VS RECOVERY AT SAME PARAMETERS ==========")
    for pol in ["plus", "cross"]:
        print(f"\n{pol} polarization:")
        print(f"  injection max abs: {np.max(np.abs(h_inj[pol])):.6e}")
        print(f"  recovery max abs: {np.max(np.abs(h_rec[pol])):.6e}")
        diff = h_inj[pol] - h_rec[pol]
        rel = np.max(np.abs(diff)) / np.max(np.abs(h_inj[pol]))
        print(pol, "relative max difference:", rel)
        print(pol, "all finite in injection:", np.all(np.isfinite(h_inj[pol])))
        print(pol, "all finite in recovery:", np.all(np.isfinite(h_rec[pol])))
    # -- plot of injection vs recovery waveforms
    if plot_waveforms:
        freqs = injection_generator.frequency_array
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        for i, (pol, label) in enumerate(zip(["plus", "cross"], ["$h_+$", "$h_\\times$"])):
            ax[i].plot(freqs, np.abs(h_inj[pol]), label=f"inj {label}")
            ax[i].plot(freqs, np.abs(h_rec[pol]), label=f"rec {label}", linestyle="--")
            ax[i].set_xlabel("Frequency [Hz]", fontsize=fontsize)
            ax[i].set_ylabel("Strain amplitude", fontsize=fontsize)
            ax[i].legend()
        plt.tight_layout()
        putils.beautifyPlot([ax[0], ax[1]], labelsize=labelsize, minor=True, tickDirection="in", top=True, right=True)
        waveform_plot_file = os.path.join(args.outdir, f"{args.label}_injection_vs_recovery_waveforms.png")
        if not args.no_save:
            plt.savefig(waveform_plot_file, bbox_inches="tight", dpi=300)
            print(f"Saved injection vs recovery waveform plot: {waveform_plot_file}")
            # -- save another plot with log-scaling
            for i in range(2):
                ax[i].set_yscale("log")
            waveform_plot_file_log = os.path.join(args.outdir, f"{args.label}_injection_vs_recovery_waveforms_log.png")
            plt.savefig(waveform_plot_file_log, bbox_inches="tight", dpi=300)
            print(f"Saved injection vs recovery waveform plot (log scale): {waveform_plot_file_log}")
        if not args.no_show:
            plt.show()
        plt.close()
    print("==============================================================\n")


def check_wf_interferometer_compatibility(injection_generator, recovery_generator, ifos) -> dict:
    out = {}
    for wfg in [injection_generator, recovery_generator]:
        wfg_name = "injection" if wfg == injection_generator else "recovery"
        out[wfg_name] = {
            'duration': wfg.duration,
            'sampling_frequency': wfg.sampling_frequency,
            'start_time': wfg.start_time,
            'N frequency_array': len(wfg.frequency_array),
            'df': wfg.frequency_array[1] - wfg.frequency_array[0],
            'fmin': wfg.frequency_array[0],
            'fmax': wfg.frequency_array[-1]
        }
    for ifo in ifos:
        out[ifo.name] = {
            'duration': ifo.strain_data.duration,
            'sampling_frequency': ifo.strain_data.sampling_frequency,
            'start_time': ifo.strain_data.start_time
        }
    return out


def check_sampled_params_follow_constraints(priors, n_samples=100, constraints=("mass_ratio",)):
    print("\n========== CHECKING SAMPLED PARAMETERS AGAINST CONSTRAINTS ==========")
    print(f"Checking {n_samples} random samples from priors {list(priors.keys())} against constraints: {constraints}")
    samples = priors.sample(n_samples)
    for constraint in constraints:
        if constraint not in priors:
            print(f"Warning: constraint {constraint} not in priors, skipping check.")
            continue
        prior_constraint = priors[constraint]
        if not isinstance(prior_constraint, bilby.gw.prior.Constraint):
            print(f"Warning: prior for {constraint} is not a Constraint, skipping check.")
            continue
        min_val = prior_constraint.minimum
        max_val = prior_constraint.maximum
        print(f"Checking constraint {constraint}: min={min_val}, max={max_val}")
        m1s = samples["mass_1"]
        m2s = samples["mass_2"]
        if constraint == "mass_ratio":
            mass_ratios = m2s / m1s  # Bilby convention: mass_ratio = m2/m1, whereas my convention is m1/m2<10.0, so m2/m1>0.1!
            for i, mr in enumerate(mass_ratios):
                if not (min_val <= mr <= max_val):
                    print(f"[BAD] Sample {i}: mass_1={m1s[i]:.3f}, mass_2={m2s[i]:.3f}, mass_ratio={mr:.3f} violates constraint!")
                else:
                    print(f"[GOOD] Sample {i}: mass_1={m1s[i]:.3f}, mass_2={m2s[i]:.3f}, mass_ratio={mr:.3f} satisfies constraint.")
    print("====================================================================\n")


def check_sampled_params_within_priors(priors, n_samples=100):
    print("\n========== CHECKING SAMPLED PARAMETERS WITHIN PRIORS ==========")
    print(f"Checking {n_samples} random samples from priors {list(priors.keys())}")
    samples = priors.sample(n_samples)
    for key, prior in priors.items():
        if isinstance(prior, bilby.gw.prior.Constraint):
            continue  # skip constraints
        min_val = prior.minimum
        max_val = prior.maximum
        vals = samples[key]
        for i, val in enumerate(vals):
            if not (min_val <= val <= max_val):
                print(f"[BAD] Sample {i}: {key}={val:.3f} violates prior range [{min_val}, {max_val}]!")
            else:
                print(f"[GOOD] Sample {i}: {key}={val:.3f} is within prior range [{min_val}, {max_val}].")
    print("===============================================================\n")



def run_pre_sampler_debug(args, injection_generator, recovery_generator, ifos, likelihood,
                           priors, injection_parameters):
    debug = {"run_config": vars(args), "injection_parameters": injection_parameters, 
             "priors": {k: str(v) for k, v in priors.items()}}
    print_dict("INJECTION PARAMETERS", injection_parameters)
    print_dict("PRIORS", {k: str(v) for k, v in priors.items()})
    debug["injection_generator_summary"] = waveform_debug_summary(injection_generator, injection_parameters, "injection_generator")
    debug["recovery_generator_summary"] = waveform_debug_summary(recovery_generator, injection_parameters, "recovery_generator")
    print_dict("INJECTION GENERATOR SUMMARY", debug["injection_generator_summary"])
    print_dict("RECOVERY GENERATOR SUMMARY", debug["recovery_generator_summary"])
    debug["generator_comparison"] = compare_generators(injection_generator, recovery_generator, injection_parameters)
    print_dict("INJECTION VS RECOVERY GENERATOR COMPARISON", debug["generator_comparison"])
    if args.check_determinism:
        def factory():
            return make_wf_generator("ml", wfkwargs={'wfmodel_modelpath': args.model_path, 
                                                     'wfmodel_configpath': args.config_path,
                                                     'calibrator_modelpath': args.calmodel_path})
        debug["determinism"] = test_ml_determinism(factory, injection_parameters, n_trials=args.determinism_trials)
        print_dict("DETERMINISM TEST", debug["determinism"])
    debug["ifo_metadata"] = print_injection_snr(ifos)
    debug["likelihood_test"] = test_ml_likelihood(args, likelihood, injection_parameters)
    debug["likelihood_scan"] = perform_1d_likelihood_scan(args, likelihood, injection_parameters, 
                                                          n_points=args.n_debug_random)
    debug["likelihood_scan_2d"] = perform_2d_likelihood_scan(args, likelihood, injection_parameters, 
                                                          n_points=args.n_debug_random)
    debug["likelihood"] = debug_likelihood(likelihood, priors, injection_parameters, n_random=args.n_debug_random)
    compare_inj_recover_at_same_params(args, injection_generator, recovery_generator, injection_parameters)

    if args.without_mass_ratio_constraint:
        check_sampled_params_within_priors(priors, n_samples=args.n_debug_random)
    else:
        check_sampled_params_follow_constraints(priors, 
                    n_samples=args.n_debug_random, constraints=("mass_ratio",))

    debug["wf_interferometer_compatibility"] = check_wf_interferometer_compatibility(injection_generator, recovery_generator, ifos)
    print_dict("WAVEFORM GENERATOR AND INTERFEROMETER COMPATIBILITY", debug["wf_interferometer_compatibility"])

    print("CHECK IMMEDIATELY AFTER INJECTION")
    diagnose_data_template_alignment_bilby(ifos, injection_generator, injection_parameters)

    # -- convert any non-JSON-serializable objects to JSON-serializable forms and save debug report
    for key, value in debug.items():
        debug[key] = to_jsonable(value)
    debug_path = os.path.join(args.outdir, f"{args.label}_debug.json")
    if not args.no_save:
        save_json(debug, debug_path)
        print(f"Saved debug report: {debug_path}")
    if args.strict_debug:
        enforce_strict_debug(debug, args)
    return debug


def enforce_strict_debug(debug: Dict[str, Any], args) -> None:
    comp = debug.get("generator_comparison", {})
    fd = comp.get("frequency_domain", {})
    for pol in ["plus", "cross"]:
        row = fd.get(pol, {})
        rel = row.get("relative_diff", None)
        if rel is not None and rel > args.max_relative_generator_diff:
            raise RuntimeError(f"Strict debug failed: FD {pol} relative difference {rel:.3e} > {args.max_relative_generator_diff:.3e}")
    inj_llr = debug.get("likelihood", {}).get("injection", {}).get("log_likelihood_ratio", None)
    if inj_llr is not None and abs(inj_llr) > args.max_abs_log_likelihood_ratio:
        raise RuntimeError(
            f"Strict debug failed: abs(log_likelihood_ratio at injection) {abs(inj_llr):.3e} > {args.max_abs_log_likelihood_ratio:.3e}."
        )
    for summary_key in ["injection_generator_summary", "recovery_generator_summary"]:
        td = debug.get(summary_key, {}).get("time_domain", {})
        for pol in ["plus", "cross"]:
            max_abs = td.get(pol, {}).get("max_abs", None)
            if max_abs is not None and max_abs > args.max_td_abs:
                raise RuntimeError(f"Strict debug failed: {summary_key} TD {pol} max_abs {max_abs:.3e} > {args.max_td_abs:.3e}.")


def make_sampler_kwargs(args) -> Dict[str, Any]:
    if args.sampler == "dynesty":
        return dict(
            nlive=args.nlive,
            dlogz=args.dlogz,
            sample=args.dynesty_sample,
            naccept=args.naccept,
            npool=args.npool,
            resume=args.resume,
        )
    if args.sampler == "nessai":
        return dict(
            nlive=args.nlive,
            n_pool=args.n_pool,
            pytorch_threads=args.pytorch_threads,
            resume=args.resume,
            flow_proposal_class=args.flow_proposal_class,     # 'gwflowproposal' instead reparameterisation full 15D space!
            analytic_priors=True,
            stopping=1.0,   # Stop if log evidence `dlogZ` value is this much or less!
        )
    raise ValueError(f"Unsupported sampler: {args.sampler}")


def run_sampler(args, likelihood, priors, injection_parameters):
    sampler_kwargs = make_sampler_kwargs(args)
    print_dict("SAMPLER KWARGS", sampler_kwargs)
    t0 = time.time()
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=args.sampler,
        injection_parameters=injection_parameters,
        outdir=args.outdir,
        label=args.label,
        result_class=bilby.gw.result.CBCResult,
        conversion_function=None,
        clean=args.clean,
        **sampler_kwargs,
    )
    print(f"Sampler finished in {(time.time() - t0) / 3600:.3f} hours")
    return result


def make_plots(args, result):
    corner_file = os.path.join(args.outdir, f"{args.label}_corner.png")
    injection_parameters = result.injection_parameters
    truths = [
        injection_parameters["mass_1"],
        injection_parameters["mass_2"],
        injection_parameters["chi_1"],
        injection_parameters["chi_2"],
    ]
    result.plot_corner(parameters=["mass_1", "mass_2", "chi_1", "chi_2"],
                       save=True is not args.no_save, 
                       filename=corner_file, truths=truths)
    print(f"Saved corner plot: {corner_file}")
    if args.plot_waveform_posterior:
        try:
            waveform_file = os.path.join(args.outdir, f"{args.label}_waveform_posterior.png")
            result.plot_waveform_posterior(n_samples=args.waveform_plot_samples, save=True, filename=waveform_file)
            print(f"Saved waveform posterior plot: {waveform_file}")
        except Exception as exc:
            print(f"Waveform posterior plot failed: {exc}")


def print_likelihood_setup(likelihood):
    wg = likelihood.waveform_generator
    print("\n========== LIKELIHOOD SETUP ==========")
    print("waveform generator duration:", getattr(wg, "duration", None))
    print("waveform generator sampling_frequency:", getattr(wg, "sampling_frequency", None))
    print("waveform generator start_time:", getattr(wg, "start_time", None))
    print("waveform generator frequency_array[0:5]:", wg.frequency_array[:5])
    print("waveform generator frequency_array[-5:]:", wg.frequency_array[-5:])

    for ifo in likelihood.interferometers:
        print()
        print("IFO:", ifo.name)
        print("ifo start_time:", ifo.strain_data.start_time)
        print("ifo duration:", ifo.strain_data.duration)
        print("ifo sampling_frequency:", ifo.strain_data.sampling_frequency)
        print("ifo min/max frequency:", ifo.frequency_array[0], ifo.frequency_array[-1])
        print("ifo strain shape:", ifo.frequency_domain_strain.shape)
        print("ifo PSD shape:", ifo.power_spectral_density_array.shape)
    print("======================================\n")



def inner_product_frequency_domain(a, b, psd, df, mask=None):
    if mask is not None:
        a = a[mask]
        b = b[mask]
        psd = psd[mask]

    good = np.isfinite(psd) & (psd > 0)
    a = a[good]
    b = b[good]
    psd = psd[good]
    return 4.0 * np.real(np.sum(a * np.conjugate(b) / psd)) * df


DERIVED_KEYS = {
    "mass_ratio",
    "chirp_mass",
    "total_mass",
    "symmetric_mass_ratio",
    "chi_eff",
    "mass_triangle_u",
    "mass_triangle_v",
}


def clean_params_for_likelihood(params):
    p = dict(params)

    for key in DERIVED_KEYS:
        p.pop(key, None)

    return p


def evaluate_likelihood(likelihood, params):
    """
    Robust evaluator for Bilby-style likelihoods.

    Works for likelihoods where parameters are assigned through
    likelihood.parameters before calling log_likelihood().
    """
    p = clean_params_for_likelihood(params)

    old_params = getattr(likelihood, "parameters", {}).copy()

    try:
        likelihood.parameters.update(p)
        return likelihood.log_likelihood()
    finally:
        likelihood.parameters = old_params


def compare_truth_and_posterior_mode(result, likelihood):
    posterior = result.posterior.copy()
    truth = clean_params_for_likelihood(result.injection_parameters)

    if "log_likelihood" in posterior.columns:
        idx = posterior["log_likelihood"].idxmax()
    elif "log_likelihood_evaluated" in posterior.columns:
        idx = posterior["log_likelihood_evaluated"].idxmax()
    else:
        raise ValueError("No log_likelihood column found in posterior.")

    mode = clean_params_for_likelihood(posterior.loc[idx].to_dict())

    ll_truth = evaluate_likelihood(likelihood, truth)
    ll_mode = evaluate_likelihood(likelihood, mode)

    print("truth:")
    print({k: truth[k] for k in ["mass_1", "mass_2", "chi_1", "chi_2"] if k in truth})

    print("posterior mode:")
    print({k: mode[k] for k in ["mass_1", "mass_2", "chi_1", "chi_2"] if k in mode})

    print("logL truth:", ll_truth)
    print("logL posterior mode:", ll_mode)
    print("truth - mode:", ll_truth - ll_mode)

    return {
        "ll_truth": ll_truth,
        "ll_mode": ll_mode,
        "delta": ll_truth - ll_mode,
        "truth": truth,
        "mode": mode,
    }

def check_detector_residual_at_truth(ifos, waveform_generator, injection_parameters):
    params = clean_params_for_likelihood(injection_parameters)

    waveform_polarizations = waveform_generator.frequency_domain_strain(params)
    rows = []
    for ifo in ifos:
        response = ifo.get_detector_response(waveform_polarizations, params)

        data = ifo.frequency_domain_strain
        residual = data - response

        psd = ifo.power_spectral_density_array

        # -- calculate frequency spacing
        df = ifo.frequency_array[1] - ifo.frequency_array[0]

        mask = getattr(ifo, "frequency_mask", None)

        rr = inner_product_frequency_domain(residual, residual, psd, df, mask)
        hh = inner_product_frequency_domain(response, response, psd, df, mask)
        dd = inner_product_frequency_domain(data, data, psd, df, mask)

        rows.append(
            {
                "ifo": ifo.name,
                "residual_snr": np.sqrt(max(rr, 0.0)),
                "signal_snr": np.sqrt(max(hh, 0.0)),
                "data_norm": np.sqrt(max(dd, 0.0)),
                "residual_over_signal": np.sqrt(max(rr, 0.0)) / np.sqrt(max(hh, 1e-300)),
            }
        )
    for row in rows:
        print(row)
    return rows

def nwip(a, b, psd, duration):
    """
    Bilby-style noise-weighted inner product.

    Uses the same normalization convention as bilby.gw.utils.noise_weighted_inner_product.
    """
    val = bilby.gw.utils.noise_weighted_inner_product(
        aa=a,
        bb=b,
        power_spectral_density=psd,
        duration=duration,
    )
    return float(np.real(val))


def diagnose_data_template_alignment_bilby(ifos, waveform_generator, parameters):
    print("\n========== DATA-TEMPLATE ALIGNMENT DIAGNOSTICS ==========")

    waveform_pols = waveform_generator.frequency_domain_strain(parameters)

    for ifo in ifos:
        h_full = ifo.get_detector_response(waveform_pols, parameters)
        d_full = ifo.frequency_domain_strain

        mask = ifo.frequency_mask

        h = h_full[mask]
        d = d_full[mask]
        psd = ifo.power_spectral_density_array[mask]
        duration = ifo.strain_data.duration

        r = d - h

        hh = nwip(h, h, psd, duration)
        dd = nwip(d, d, psd, duration)
        dh = nwip(d, h, psd, duration)
        rr = nwip(r, r, psd, duration)

        rr_from_identity = dd + hh - 2.0 * dh
        dh_from_identity = 0.5 * (dd + hh - rr)

        signal_snr = np.sqrt(max(hh, 0.0))
        data_norm = np.sqrt(max(dd, 0.0))
        residual_snr = np.sqrt(max(rr, 0.0))

        matched_rho = dh / np.sqrt(max(hh, 1e-300))
        normalized_overlap = dh / np.sqrt(max(dd * hh, 1e-300))
        amplitude_factor = dh / max(hh, 1e-300)

        print(f"\nIFO: {ifo.name}")
        print("<h,h>:", hh)
        print("<d,d>:", dd)
        print("<d,h>:", dh)
        print("<d-h,d-h>:", rr)
        print("identity residual dd + hh - 2dh:", rr_from_identity)
        print("identity dh from dd,hh,rr:", dh_from_identity)
        print("identity error rr - identity:", rr - rr_from_identity)
        print("signal_snr sqrt<h,h>:", signal_snr)
        print("data_norm sqrt<d,d>:", data_norm)
        print("residual_snr sqrt<d-h,d-h>:", residual_snr)
        print("matched rho <d,h>/sqrt<h,h>:", matched_rho)
        print("normalized data-template overlap:", normalized_overlap)
        print("best-fit amplitude factor alpha=(d|h)/(h|h):", amplitude_factor)
    print("=========================================================\n")


def read_ifos_from_file(fname: str, outdir: str = f'../v0p1/results/'):
    if not fname.endswith('.pkl'):
        fname += '.pkl'
    ifos = bilby.gw.detector.InterferometerList.from_pickle(f"{outdir}/{fname}")
    print(f"Loaded interferometers from {outdir}/{fname} for analysis...")
    return ifos


def run_post_sampler_debug(args, wfgenerator):
    fname = os.path.join(args.result_dir, args.fname)
    outdir = args.outdir
    ifos = read_ifos_from_file(args.fname.replace('_result.json', '_ifos.pkl'), 
                               args.result_dir)
    likelihood = make_likelihood(ifos, wfgenerator)

    result = bilby.result.read_in_result(fname)
    print(f"Loaded result from {fname} for analysis...")

    print_likelihood_setup(likelihood)
    diagnostic = compare_truth_and_posterior_mode(result, likelihood)
    residual_rows = check_detector_residual_at_truth(
        likelihood.interferometers,
        likelihood.waveform_generator,
        result.injection_parameters,
    )
    truth = dict(result.injection_parameters)
    diagnose_data_template_alignment_bilby(
        likelihood.interferometers,
        likelihood.waveform_generator,
        result.injection_parameters
    )


def parse_args():
    parser = argparse.ArgumentParser(description="4D ML waveform Bilby PE debug script: ML injection -> ML recovery.")
    parser.add_argument("--model-path", required=True, help="Path to the ML waveform model file.")
    parser.add_argument("--config-path", required=True, help="Path to the ML waveform model configuration file.")
    parser.add_argument("--calmodel-path", required=True, help="Path to the ML calibrator model file.")
    parser.add_argument("--pe-run-type", default="ml2ml", choices=["ml2ml", "eob2ml"], help="Type of PE run to perform.")
    parser.add_argument("--distance-factor", type=float, default=None, help="Factor to scale the injected luminosity distance by.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--sampling-frequency", type=float, default=DEFAULT_SAMPLING_FREQUENCY)
    parser.add_argument("--fmin", type=float, default=DEFAULT_FMIN)
    parser.add_argument("--fref", type=float, default=DEFAULT_FREF)
    parser.add_argument("--ifos", nargs="+", default=["H1", "L1"])
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--inject-mass-1", type=float, default=36.0)
    parser.add_argument("--inject-mass-2", type=float, default=32.0)
    parser.add_argument("--inject-spin-1z", type=float, default=0.30)
    parser.add_argument("--inject-spin-2z", type=float, default=-0.20)
    parser.add_argument("--luminosity-distance", type=float, default=400.0)
    parser.add_argument("--theta-jn", type=float, default=0.4)
    parser.add_argument("--phase", type=float, default=1.3)
    parser.add_argument("--geocent-time", type=float, default=1126259642.413)
    parser.add_argument("--ra", type=float, default=1.375)
    parser.add_argument("--dec", type=float, default=-1.2108)
    parser.add_argument("--psi", type=float, default=2.659)
    parser.add_argument("--m1-min", type=float, default=30.0)
    parser.add_argument("--m1-max", type=float, default=75.0)
    parser.add_argument("--m2-min", type=float, default=30.0)
    parser.add_argument("--m2-max", type=float, default=75.0)
    parser.add_argument("--chi1-min", type=float, default=-0.8)
    parser.add_argument("--chi1-max", type=float, default=0.8)
    parser.add_argument("--chi2-min", type=float, default=-0.8)
    parser.add_argument("--chi2-max", type=float, default=0.8)
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda.")
    parser.add_argument("--precision", default="float32", choices=["float32", "float64"])
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--scale-amplitude", action="store_true", help="Whether to apply an overall amplitude scaling to the ML waveforms. This can be useful for debugging when the ML model was trained on whitened waveforms or waveforms with a different distance convention.")

    parser.add_argument("--debug-only", action="store_true")

    parser.add_argument("--debug-post-sampler", action="store_true")
    parser.add_argument("--result-dir", default=DEFAULT_OUTDIR,)
    parser.add_argument("--fname", default=f"{DEFAULT_LABEL}_result.json")

    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--without-mass-ratio-constraint", action="store_true")

    parser.add_argument("--strict-debug", action="store_true")
    parser.add_argument("--n-debug-random", type=int, default=8)
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--determinism-trials", type=int, default=2)
    parser.add_argument("--max-relative-generator-diff", type=float, default=1e-10)
    parser.add_argument("--max-abs-log-likelihood-ratio", type=float, default=1e6)
    parser.add_argument("--max-td-abs", type=float, default=1e-18)
    parser.add_argument("--sampler", default="nessai", choices=["nessai", "dynesty"])
    parser.add_argument("--nlive", type=int, default=100)
    parser.add_argument("--dlogz", type=float, default=1.0)
    parser.add_argument("--npool", type=int, default=1, help="dynesty pool")
    parser.add_argument("--dynesty-sample", default="acceptance-walk")
    parser.add_argument("--naccept", type=int, default=10)
    parser.add_argument("--n-pool", type=int, default=1, help="nessai likelihood pool")
    parser.add_argument("--pytorch-threads", type=int, default=1, help="nessai PyTorch threads")
    parser.add_argument("--flow-proposal-class", default="gwflowproposal", choices=["gwflowproposal", "flowproposal"], help="Nessai flow proposal class to use.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--plot-waveform-posterior", action="store_true")
    parser.add_argument("--waveform-plot-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-level", default="INFO")
    print("Command-line arguments:")
    for arg in vars(parser.parse_args()):
        print(f"  {arg}: {getattr(parser.parse_args(), arg)}")
    return parser.parse_args()


def main():
    args = parse_args()
    # if args.debug_only:
    #     print("Debug-only mode requested. Not running sampler.")
    #     args.no_save = True
    if args.pe_run_type == "eob2ml":
        print("PE run type is eob2ml: injection will be EOB, recovery will be ML.")
        args.outdir = args.outdir.replace("ml2ml", "eob2ml")
        args.label = args.label.replace("ml2ml", "eob2ml")

    if not args.no_save:
        os.makedirs(args.outdir, exist_ok=True)
        setup_logger(args.outdir, args.label, args.log_level)
    np.random.seed(args.seed)
    bilby.core.utils.random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)
        if args.torch_threads is not None and args.torch_threads > 0:
            torch.set_num_threads(args.torch_threads)
    if mp is not None:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    if args.plot_only:
        result = bilby.result.read_in_result(os.path.join(args.outdir, 
                                            f"{args.label}_result.json"))
        make_plots(args, result)
        return

    priors = make_analysis_priors(injection_parameters=base_injection,
                        with_mass_ratio_constraint=not args.without_mass_ratio_constraint)
    
    injection_parameters = sample_injection_from_priors(
        base_injection=base_injection, active_priors=priors
    )
    assert_injection_inside_priors(priors, injection_parameters)

    wfkwargs = {'wfmodel_modelpath': args.model_path, 
                'wfmodel_configpath': args.config_path,
                'calibrator_modelpath': args.calmodel_path}
    if args.distance_factor is not None:
        print(f"Scaling injected luminosity distance by factor {args.distance_factor}, which is also the luminosity distance for injection!")
        injection_parameters['luminosity_distance'] = args.distance_factor
        wfkwargs['distance_scale_factor'] = args.distance_factor
    if args.pe_run_type == "eob2ml":
        injection_generator = make_wf_generator("eob")
        recovery_generator = make_wf_generator("ml", wfkwargs=wfkwargs)
    else:
        injection_generator = make_wf_generator("ml", wfkwargs=wfkwargs)
        recovery_generator = make_wf_generator("ml", wfkwargs=wfkwargs)

    if args.debug_post_sampler:
        run_post_sampler_debug(args, recovery_generator)
        return

    ifos = make_interferometers(args, injection_parameters, injection_generator)
    likelihood = make_likelihood(ifos, recovery_generator)
    run_pre_sampler_debug(args, injection_generator, recovery_generator, ifos, likelihood, priors, injection_parameters)

    if args.debug_only:
        print("Debug-only mode was requested. Exiting.")
        return
    
    result = run_sampler(args, likelihood, priors, injection_parameters)
    if not args.no_save:
        posterior_csv = os.path.join(args.outdir, f"{args.label}_posterior.csv")
        result.posterior.to_csv(posterior_csv, index=False)
        print(f"Saved posterior CSV: {posterior_csv}")
    make_plots(args, result)


if __name__ == "__main__":
    main()
