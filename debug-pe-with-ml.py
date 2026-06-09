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
import argparse
import logging
import importlib
from typing import Any, Dict

import numpy as np

import bilby
from bilby.gw.detector import InterferometerList
from bilby.gw.likelihood import GravitationalWaveTransient

try:
    import torch
    import torch.multiprocessing as mp
except Exception:
    torch = None
    mp = None

DEFAULT_DURATION = 1.0
DEFAULT_SAMPLING_FREQUENCY = 8192.0
DEFAULT_FMIN = 20.0
DEFAULT_FREF = 50.0
DEFAULT_OUTDIR = "out_ml_4d_debug"
DEFAULT_LABEL = "ml_4d_debug"


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


def make_injection_parameters(args) -> Dict[str, float]:
    return dict(
        mass_1=args.inject_mass_1,
        mass_2=args.inject_mass_2,
        spin_1z=args.inject_spin_1z,
        spin_2z=args.inject_spin_2z,
        luminosity_distance=args.luminosity_distance,
        theta_jn=args.theta_jn,
        phase=args.phase,
        geocent_time=args.geocent_time,
        ra=args.ra,
        dec=args.dec,
        psi=args.psi,
    )


def make_4d_priors(args, injection_parameters: Dict[str, float]) -> bilby.core.prior.PriorDict:
    priors = bilby.core.prior.PriorDict()
    priors["mass_1"] = bilby.core.prior.Uniform(args.m1_min, args.m1_max, name="mass_1", latex_label="$m_1$")
    priors["mass_2"] = bilby.core.prior.Uniform(args.m2_min, args.m2_max, name="mass_2", latex_label="$m_2$")
    priors["spin_1z"] = bilby.core.prior.Uniform(args.chi1_min, args.chi1_max, name="spin_1z", latex_label="$\\chi_{1z}$")
    priors["spin_2z"] = bilby.core.prior.Uniform(args.chi2_min, args.chi2_max, name="spin_2z", latex_label="$\\chi_{2z}$")
    for key in ["luminosity_distance", "theta_jn", "phase", "geocent_time", "ra", "dec", "psi"]:
        priors[key] = injection_parameters[key]
    return priors


def assert_injection_inside_priors(priors, injection_parameters, keys=("mass_1", "mass_2", "spin_1z", "spin_2z")):
    for key in keys:
        p = priors[key].prob(injection_parameters[key])
        if not np.isfinite(p) or p <= 0:
            raise ValueError(f"Injected value {key}={injection_parameters[key]} is outside prior {priors[key]}")


def build_ml_waveform_generator(args):
    MLWaveformGenerator = import_symbol(args.ml_generator)
    parameter_conversion = import_symbol(args.parameter_conversion)
    waveform_arguments = {"model_path": args.model_path, 
                          "config_path": args.model_config,}
    if args.scale_amplitude is not None:
        waveform_arguments["scale_amplitude"] = args.scale_amplitude
    generator = MLWaveformGenerator(
        duration=args.duration,
        sampling_frequency=args.sampling_frequency,
        time_domain_source_model=None,
        parameter_conversion=parameter_conversion,
        waveform_arguments=waveform_arguments,
    )
    maybe_prepare_torch_model(generator, args)
    return generator


def maybe_prepare_torch_model(generator, args) -> None:
    model = getattr(generator, "loaded_mlmodel", None)
    if model is None or torch is None:
        return
    model.eval()
    if args.torch_threads is not None and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.device is not None:
        device = torch.device(args.device)
        dtype = getattr(torch, args.precision)
        try:
            model.to(device=device, dtype=dtype)
        except TypeError:
            model.to(device)
            model.to(dtype)
    if hasattr(generator, "check_model_weights_on_device") and args.device is not None:
        try:
            generator.check_model_weights_on_device(device=torch.device(args.device), precision=getattr(torch, args.precision))
        except Exception as exc:
            print(f"WARNING: check_model_weights_on_device failed: {exc}")


def clear_waveform_cache(generator) -> None:
    if hasattr(generator, "_cache"):
        try:
            generator._cache = {"parameters": None, "waveform": None, "model": None, "transformed_model": None}
        except Exception:
            pass


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


def run_pre_sampler_debug(args, injection_generator, recovery_generator, ifos, likelihood, priors, injection_parameters):
    debug = {"run_config": vars(args), "injection_parameters": injection_parameters, "priors": {k: str(v) for k, v in priors.items()}}
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
            return build_ml_waveform_generator(args)
        debug["determinism"] = test_ml_determinism(factory, injection_parameters, n_trials=args.determinism_trials)
        print_dict("DETERMINISM TEST", debug["determinism"])
    debug["ifo_metadata"] = print_injection_snr(ifos)
    debug["likelihood"] = debug_likelihood(likelihood, priors, injection_parameters, n_random=args.n_debug_random)
    # -- convert any non-JSON-serializable objects to JSON-serializable forms and save debug report
    for key, value in debug.items():
        debug[key] = to_jsonable(value)
    debug_path = os.path.join(args.outdir, f"{args.label}_debug.json")
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
    result.plot_corner(parameters=["mass_1", "mass_2", "spin_1z", "spin_2z"], save=True, filename=corner_file)
    print(f"Saved corner plot: {corner_file}")
    if args.plot_waveform_posterior:
        try:
            waveform_file = os.path.join(args.outdir, f"{args.label}_waveform_posterior.png")
            result.plot_waveform_posterior(n_samples=args.waveform_plot_samples, save=True, filename=waveform_file)
            print(f"Saved waveform posterior plot: {waveform_file}")
        except Exception as exc:
            print(f"Waveform posterior plot failed: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="4D ML waveform Bilby PE debug script: ML injection -> ML recovery.")
    parser.add_argument("--ml-generator", default="mlwavegen:MLWaveformGenerator", help="Import path for MLWaveformGenerator.")
    parser.add_argument("--parameter-conversion", default="mlwavegen:convert_to_ml_parameters", help="Import path for parameter conversion function.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-config", required=True)
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
    parser.add_argument("--scale-amplitude", action="store_true", help="Scale the generated waveform amplitude.")
    parser.add_argument("--debug-only", action="store_true")
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
    injection_parameters = make_injection_parameters(args)
    priors = make_4d_priors(args, injection_parameters)
    assert_injection_inside_priors(priors, injection_parameters)
    injection_generator = build_ml_waveform_generator(args)
    recovery_generator = build_ml_waveform_generator(args)
    ifos = make_interferometers(args, injection_parameters, injection_generator)
    likelihood = make_likelihood(ifos, recovery_generator)
    run_pre_sampler_debug(args, injection_generator, recovery_generator, ifos, likelihood, priors, injection_parameters)
    if args.debug_only:
        print("Debug-only mode requested. Not running sampler.")
        return
    result = run_sampler(args, likelihood, priors, injection_parameters)
    make_plots(args, result)
    posterior_csv = os.path.join(args.outdir, f"{args.label}_posterior.csv")
    result.posterior.to_csv(posterior_csv, index=False)
    print(f"Saved posterior CSV: {posterior_csv}")


if __name__ == "__main__":
    main()
