"""
Module to analyze first PE run results, generate PP plots, and then
use the first PE posteriors as importance-weighted sample proposal for a second PE run.
Additionally contains methods and functions to use different kinds of sampling methods using
the first PE posteriors as proposal distribution, or as the actual target posterior!
"""

import os
import copy
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
from tqdm import tqdm
import corner


from plotutils import putils


# ============================================================
# Parameter utilities
# ============================================================

def complete_params_from_row(row, injection_parameters=None, extra_fixed=None):
    """
    Build a full Bilby parameter dictionary from one posterior row.

    Missing fixed parameters are filled from injection_parameters and/or extra_fixed.

    This is useful when the posterior only contains active sampled parameters,
    while the likelihood still needs fixed extrinsic parameters such as phase,
    geocent_time, ra, dec, theta_jn, psi, luminosity_distance, etc.
    """

    if injection_parameters is None:
        injection_parameters = {}

    if extra_fixed is None:
        extra_fixed = {}

    source_dicts = [extra_fixed, injection_parameters]

    wanted_keys = [
        "mass_1", "mass_2",
        "chi_1", "chi_2",
        "spin_1z", "spin_2z",
        "luminosity_distance",
        "theta_jn", "psi", "phase",
        "geocent_time", "ra", "dec",
        "mass_ratio", "chirp_mass",
    ]

    p = {}

    for key in wanted_keys:
        if key in row.index and pd.notnull(row[key]):
            p[key] = float(row[key])
            continue

        for src in source_dicts:
            if key in src and src[key] is not None:
                p[key] = float(src[key])
                break

    # Aligned-spin aliases
    if "chi_1" in p and "spin_1z" not in p:
        p["spin_1z"] = p["chi_1"]
    if "chi_2" in p and "spin_2z" not in p:
        p["spin_2z"] = p["chi_2"]

    if "spin_1z" in p and "chi_1" not in p:
        p["chi_1"] = p["spin_1z"]
    if "spin_2z" in p and "chi_2" not in p:
        p["chi_2"] = p["spin_2z"]

    return p


def evaluate_bilby_log_likelihood(likelihood, parameters):
    """
    Robustly evaluate a Bilby likelihood at a parameter dictionary.

    Some custom likelihood wrappers may accept log_likelihood(parameters=p).
    Standard Bilby likelihoods usually require:
        likelihood.parameters.update(p)
        likelihood.log_likelihood()
    """

    try:
        return float(likelihood.log_likelihood(parameters=parameters))
    except TypeError:
        pass

    old_params = copy.deepcopy(getattr(likelihood, "parameters", {}))

    try:
        if not hasattr(likelihood, "parameters") or likelihood.parameters is None:
            likelihood.parameters = {}

        likelihood.parameters.update(parameters)
        val = float(likelihood.log_likelihood())

    finally:
        likelihood.parameters = old_params

    return val


def get_injection_parameters(result, fallback=None):
    if hasattr(result, "injection_parameters") and result.injection_parameters is not None:
        return result.injection_parameters
    if fallback is not None:
        return fallback
    return {}


# ============================================================
# Manual importance reweighting
# ============================================================

def compute_eob_importance_weights(
    eob2ml_result,
    eob_likelihood,
    posterior=None,
    proposal_logl_column="log_likelihood",
    proposal_beta=1.0,
    proposal_logl_is_already_tempered=False,
    injection_parameters=None,
    extra_fixed=None,
    enforce_mass_order=False,
    mass_ratio_max=None,
    checkpoint_csv=None,
    save_every=25,
):
    """
    Compute manual EOB importance weights for an EOB2ML posterior.

    Proposal posterior:
        q(theta) proportional to prior(theta) * L_ML(theta)

    Target posterior:
        p(theta) proportional to prior(theta) * L_EOB(theta)

    Weight:
        log w_i = log L_EOB(theta_i) - log L_ML(theta_i)

    If the EOB2ML run used a tempered proposal:
        q(theta) proportional to prior(theta) * L_ML(theta)^beta

    and the stored log_likelihood column is the untempered ML log likelihood,
    set:
        proposal_logl_is_already_tempered=False
        proposal_beta=beta

    If the stored log_likelihood column is already the tempered value used by
    the sampler, leave:
        proposal_logl_is_already_tempered=True
        proposal_beta=1.0
    """

    if posterior is None:
        post = eob2ml_result.posterior.copy()
    else:
        post = posterior.copy()

    post = post.reset_index(drop=True)

    if injection_parameters is None:
        injection_parameters = get_injection_parameters(eob2ml_result)

    if extra_fixed is None:
        extra_fixed = {}

    if proposal_logl_column not in post.columns:
        raise ValueError(
            f"Missing column {proposal_logl_column}. "
            "The EOB2ML posterior must contain the proposal log likelihood."
        )

    n = len(post)

    # Load checkpoint, if possible
    if checkpoint_csv is not None:
        try:
            old = pd.read_csv(checkpoint_csv)
            if len(old) == n and "log_likelihood_eob" in old.columns:
                post["log_likelihood_eob"] = old["log_likelihood_eob"].to_numpy()
                print(f"Loaded checkpoint: {checkpoint_csv}")
            else:
                post["log_likelihood_eob"] = np.nan
        except FileNotFoundError:
            post["log_likelihood_eob"] = np.nan
    else:
        post["log_likelihood_eob"] = np.nan

    # Evaluate EOB likelihoods
    for i, row in tqdm(post.iterrows(), total=n, desc="EOB likelihoods"):
        if np.isfinite(post.loc[i, "log_likelihood_eob"]):
            continue

        params = complete_params_from_row(
            row,
            injection_parameters=injection_parameters,
            extra_fixed=extra_fixed,
        )

        if enforce_mass_order:
            if "mass_1" in params and "mass_2" in params:
                if params["mass_1"] < params["mass_2"]:
                    post.loc[i, "log_likelihood_eob"] = -np.inf
                    continue

        if mass_ratio_max is not None:
            if "mass_1" in params and "mass_2" in params:
                q_inv = params["mass_1"] / params["mass_2"]
                if q_inv > mass_ratio_max:
                    post.loc[i, "log_likelihood_eob"] = -np.inf
                    continue

        try:
            post.loc[i, "log_likelihood_eob"] = evaluate_bilby_log_likelihood(
                eob_likelihood,
                params,
            )
        except Exception as exc:
            print(f"Sample {i} failed: {repr(exc)}")
            post.loc[i, "log_likelihood_eob"] = -np.inf

        if checkpoint_csv is not None and (i + 1) % save_every == 0:
            post.to_csv(checkpoint_csv, index=False)
            print(f"Saved checkpoint at {i + 1}/{n}: {checkpoint_csv}")

    if checkpoint_csv is not None:
        post.to_csv(checkpoint_csv, index=False)

    logl_eob = post["log_likelihood_eob"].to_numpy(dtype=float)
    logl_prop = post[proposal_logl_column].to_numpy(dtype=float)

    if proposal_logl_is_already_tempered:
        logl_proposal_used = logl_prop
    else:
        logl_proposal_used = proposal_beta * logl_prop

    logw_raw = logl_eob - logl_proposal_used

    finite = np.isfinite(logw_raw)

    if finite.sum() == 0:
        raise RuntimeError("All importance weights are non-finite.")

    logw_norm = np.full_like(logw_raw, -np.inf, dtype=float)
    logw_norm[finite] = logw_raw[finite] - logsumexp(logw_raw[finite])

    weights = np.exp(logw_norm)
    weights[~finite] = 0.0

    neff = 1.0 / np.sum(weights[finite] ** 2)

    post["logw_raw"] = logw_raw
    post["logw_norm"] = logw_norm
    post["eob_weight"] = weights

    diagnostics = {
        "n_total": int(n),
        "n_finite": int(finite.sum()),
        "neff": float(neff),
        "neff_fraction": float(neff / finite.sum()),
        "max_weight": float(np.max(weights[finite])),
        "logw_min": float(np.min(logw_raw[finite])),
        "logw_median": float(np.median(logw_raw[finite])),
        "logw_max": float(np.max(logw_raw[finite])),
    }

    return post, diagnostics


def print_reweighting_diagnostics(diag):
    print("========== EOB IMPORTANCE WEIGHT DIAGNOSTICS ==========")
    print(f"N total:       {diag['n_total']}")
    print(f"N finite:      {diag['n_finite']}")
    print(f"N_eff:         {diag['neff']:.3f}")
    print(f"N_eff / N:     {diag['neff_fraction']:.6e}")
    print(f"max weight:    {diag['max_weight']:.6e}")
    print(f"logw min:      {diag['logw_min']:.3f}")
    print(f"logw median:   {diag['logw_median']:.3f}")
    print(f"logw max:      {diag['logw_max']:.3f}")
    print("=======================================================")


# ============================================================
# Weighted summary statistics
# ============================================================

def normalize_weights(w):
    w = np.asarray(w, dtype=float)
    finite = np.isfinite(w) & (w >= 0)
    out = np.zeros_like(w, dtype=float)
    if np.sum(w[finite]) <= 0:
        raise ValueError("Weights have zero or non-finite sum.")
    out[finite] = w[finite] / np.sum(w[finite])
    return out


def weighted_quantile(x, w, qs):
    """
    Weighted quantiles.

    qs should be in [0,1], e.g.
        [0.15865, 0.5, 0.84135] for 1 sigma
        [0.05, 0.5, 0.95] for 90 percent
    """

    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    finite = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[finite]
    w = w[finite]

    if len(x) == 0:
        raise ValueError("No finite values for weighted quantile.")

    w = w / np.sum(w)

    order = np.argsort(x)
    x = x[order]
    w = w[order]

    cdf = np.cumsum(w)

    return np.interp(qs, cdf, x)


def weighted_histogram_mode(x, w, bins=50, value_range=None):
    """
    Histogram-based marginal posterior mode.
    This depends on binning, so use consistently.
    """

    x = np.asarray(x, dtype=float)
    w = normalize_weights(w)

    finite = np.isfinite(x) & np.isfinite(w)
    x = x[finite]
    w = w[finite]
    w = w / np.sum(w)

    hist, edges = np.histogram(
        x,
        bins=bins,
        range=value_range,
        weights=w,
        density=True,
    )

    imax = int(np.argmax(hist))
    mode = 0.5 * (edges[imax] + edges[imax + 1])

    return mode, hist, edges


def weighted_kde_mode(x, w, gridsize=2000, padding_fraction=0.05):
    """
    KDE-based marginal posterior mode.
    Usually smoother and better than raw histogram mode.
    """

    x = np.asarray(x, dtype=float)
    w = normalize_weights(w)

    finite = np.isfinite(x) & np.isfinite(w)
    x = x[finite]
    w = w[finite]
    w = w / np.sum(w)

    if len(np.unique(x)) < 2:
        return float(x[0]), x, np.ones_like(x)

    xmin = np.min(x)
    xmax = np.max(x)

    if xmax == xmin:
        return float(xmin), np.array([xmin]), np.array([1.0])

    pad = padding_fraction * (xmax - xmin)
    grid = np.linspace(xmin - pad, xmax + pad, gridsize)

    kde = gaussian_kde(x, weights=w)
    density = kde(grid)

    mode = float(grid[int(np.argmax(density))])

    return mode, grid, density


def get_interval_quantiles(interval="90"):
    """
    Returns quantiles for summary titles.

    interval:
        "90" or "90%"
        "1sigma" or "68"
    """

    s = str(interval).lower().replace("%", "").replace("_", "").replace("-", "")

    if s in ["90"]:
        return [0.05, 0.5, 0.95], "90%"

    if s in ["1sigma", "1sig", "sigma", "68", "68.3", "68.27"]:
        return [0.158655, 0.5, 0.841345], "1 sigma"

    raise ValueError("interval must be '90' or '1sigma'.")


def get_corner_levels(interval="90"):
    """
    2D contour enclosed-probability levels for corner plots.
    """

    s = str(interval).lower().replace("%", "").replace("_", "").replace("-", "")

    if s in ["90"]:
        return (0.90,)

    if s in ["1sigma", "1sig", "sigma", "68", "68.3", "68.27"]:
        # 2D Gaussian 1-sigma enclosed probability
        return (0.393469,)

    raise ValueError("interval must be '90' or '1sigma'.")


def summarize_weighted_marginals(
    weighted_post,
    parameters,
    injection_parameters=None,
    interval="90",
    mode_method="kde",
    bins=50,
):
    """
    Make a summary table for weighted 1D marginalized posteriors.

    Important:
        median and credible interval are not the same as the marginal mode.
    """

    qs, interval_label = get_interval_quantiles(interval)

    w = weighted_post["eob_weight"].to_numpy(dtype=float)
    w = normalize_weights(w)

    rows = []

    for par in parameters:
        x = weighted_post[par].to_numpy(dtype=float)

        qlo, q50, qhi = weighted_quantile(x, w, qs)

        if mode_method == "kde":
            mode, _, _ = weighted_kde_mode(x, w)
        elif mode_method == "hist":
            mode, _, _ = weighted_histogram_mode(x, w, bins=bins)
        else:
            raise ValueError("mode_method must be 'kde' or 'hist'.")

        inj = None
        inj_inside = None
        bias_mode = None
        bias_median = None

        if injection_parameters is not None and par in injection_parameters:
            inj = float(injection_parameters[par])
            inj_inside = bool(qlo <= inj <= qhi)
            bias_mode = mode - inj
            bias_median = q50 - inj

        rows.append({
            "parameter": par,
            "mode": mode,
            "median": q50,
            "lower": qlo,
            "upper": qhi,
            "interval": interval_label,
            "injection": inj,
            "injection_inside_interval": inj_inside,
            "mode_minus_injection": bias_mode,
            "median_minus_injection": bias_median,
        })

    return pd.DataFrame(rows)


# ============================================================
# Weighted marginalized histogram plots
# ============================================================

def plot_weighted_marginal(
    weighted_post,
    parameter,
    injection_parameters=None,
    eob2ml_result=None,
    eob2eob_result=None,
    bins=50,
    interval="90",
    filename=None,
    outdir=None,
    title=None,
    show_kde_mode=False,
    fontsize=15,
    labelsize=12,
    _plot_density=False
):
    """
    Plot one weighted marginalized posterior.

    Shows:
        original EOB2ML proposal, if eob2ml_result is given
        EOB-reweighted posterior
        EOB2EOB reference, if eob2eob_result is given
        injection value, if available
    """

    x = weighted_post[parameter].to_numpy(dtype=float)
    w = normalize_weights(weighted_post["eob_weight"].to_numpy(dtype=float))

    finite = np.isfinite(x) & np.isfinite(w)
    x = x[finite]
    w = w[finite]
    w = w / np.sum(w)

    fig, ax = plt.subplots(1,1, figsize=(6.2, 4.2))

    if eob2ml_result is not None and parameter in eob2ml_result.posterior.columns:
        x_ml = eob2ml_result.posterior[parameter].to_numpy(dtype=float)
        x_ml = x_ml[np.isfinite(x_ml)]
        ax.hist(
            x_ml,
            bins=bins,
            density=_plot_density,
            histtype="step",
            linewidth=1.4,
            alpha=0.7,
            label="EOB2ML proposal",
        )

    ax.hist(
        x,
        bins=bins,
        weights=w,
        density=_plot_density,
        histtype="step",
        linewidth=2.2,
        label="EOB-reweighted",
    )

    if eob2eob_result is not None and parameter in eob2eob_result.posterior.columns:
        x_ref = eob2eob_result.posterior[parameter].to_numpy(dtype=float)
        x_ref = x_ref[np.isfinite(x_ref)]
        ax.hist(
            x_ref,
            bins=bins,
            density=_plot_density,
            histtype="step",
            linewidth=2.0,
            linestyle="--",
            label="EOB2EOB reference",
        )

    if show_kde_mode:
        mode, grid, dens = weighted_kde_mode(x, w)
        if len(grid) > 1:
            ax.plot(grid, dens, linewidth=1.5, alpha=0.8, label="Weighted KDE")
        ax.axvline(mode, linestyle="-.", linewidth=1.6, label="Weighted mode")

    if injection_parameters is not None and parameter in injection_parameters:
        ax.axvline(
            float(injection_parameters[parameter]),
            linestyle=":",
            linewidth=2.0,
            label="Injection",
        )

    latex_labels = {
        "mass_1": r"$m_1$",
        "mass_2": r"$m_2$",
        "chi_1": r"$\chi_1$",
        "chi_2": r"$\chi_2$",
        "spin_1z": r"$\chi_1(z)$",
        "spin_2z": r"$\chi_2(z)$",
        "luminosity_distance": r"$d_L$",
        "theta_jn": r"$\theta_{jn}$",
        "psi": r"$\psi$",
        "phase": r"$\phi$",
        "geocent_time": r"$t_c$",
        "ra": r"$\alpha$",
        "dec": r"$\delta$",
        "mass_ratio": r"$q$",
        "chirp_mass": r"$\mathcal{M}$",
    }

    ax.set_xlabel(latex_labels.get(parameter, parameter), fontsize=fontsize)
    ylabel = "density" if _plot_density else "weighted counts"
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)   # -- if some samples have all the weights, other bins have near zero counts!
    plt.tick_params(axis="both", which="major", labelsize=labelsize)
    plt.tight_layout()

    # if title is None:
    #     title = f"Weighted marginal posterior: {parameter}"
    # plt.title(title, fontsize=fontsize)

    plt.legend(fontsize=labelsize-2, loc="upper right")
    plt.tight_layout()

    if filename is not None:
        if outdir is not None:
            filename = os.path.join(outdir, filename)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_all_weighted_marginals(
    weighted_post,
    parameters,
    injection_parameters=None,
    eob2ml_result=None,
    eob2eob_result=None,
    bins=50,
    interval="90",
    out_prefix="weighted_marginal",
    outdir=None,
):
    for par in parameters:
        plot_weighted_marginal(
            weighted_post=weighted_post,
            parameter=par,
            injection_parameters=injection_parameters,
            eob2ml_result=eob2ml_result,
            eob2eob_result=eob2eob_result,
            bins=bins,
            interval=interval,
            filename=f"{out_prefix}_{par}.png",
            title=f"EOB-reweighted marginal posterior: {par}",
            outdir=outdir,
        )


# ============================================================
# Weighted corner plot
# ============================================================

def plot_weighted_corner(
    weighted_post,
    parameters,
    injection_parameters=None,
    interval="90",
    filename="weighted_corner.png",
    outdir=None,
    labels=None,
    title_fmt=".3g",
    smooth=1.0,
    bins=40,
    fontsize=15,
    labelsize=12,
):
    """
    Make a proper weighted corner plot using corner.corner.

    This is the correct way to visualize the manual reweighted posterior.

    Diagonal panels:
        1D weighted marginalized posteriors.

    Off-diagonal panels:
        2D weighted marginalized posteriors.

    Titles:
        median and credible interval, not necessarily histogram maxima.
    """

    samples = weighted_post[parameters].to_numpy(dtype=float)
    weights = normalize_weights(weighted_post["eob_weight"].to_numpy(dtype=float))

    finite = np.all(np.isfinite(samples), axis=1) & np.isfinite(weights) & (weights >= 0)
    samples = samples[finite]
    weights = weights[finite]
    weights = weights / np.sum(weights)

    if labels is None:
        labels = parameters

    truths = None
    if injection_parameters is not None:
        truths = []
        for p in parameters:
            if p in injection_parameters:
                truths.append(float(injection_parameters[p]))
            else:
                truths.append(None)

    qs, interval_label = get_interval_quantiles(interval)
    levels = get_corner_levels(interval)

    fig = corner.corner(
        samples,
        weights=weights,
        labels=labels,
        truths=truths,
        bins=bins,
        smooth=smooth,
        show_titles=True,
        title_quantiles=qs,
        quantiles=qs,
        title_fmt=title_fmt,
        levels=levels,
        plot_datapoints=False,
        fill_contours=True,
        density=True,
        hist_kwargs={"density": True},
    )

    fig.suptitle(
        f"EOB-reweighted posterior, titles show median and {interval_label} credible interval",
        y=1.02,
        fontsize=fontsize
    )

    if outdir is not None:
        filename = os.path.join(outdir, filename)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fig


# ============================================================
# Optional equal-weight resampling
# ============================================================

def multinomial_resample_to_equal_weight(
    weighted_post,
    n_samples=None,
    random_seed=1234,
):
    """
    Optional only.

    Converts weighted samples into equal-weight samples by resampling with replacement.
    This is useful only if some plotting or Bilby function cannot accept weights.
    For your main weighted histograms and weighted corner plot, you do not need this.
    """

    if n_samples is None:
        n_samples = len(weighted_post)

    rng = np.random.default_rng(random_seed)

    w = normalize_weights(weighted_post["eob_weight"].to_numpy(dtype=float))

    idx = rng.choice(
        np.arange(len(weighted_post)),
        size=n_samples,
        replace=True,
        p=w,
    )

    return weighted_post.iloc[idx].reset_index(drop=True).copy()


# ============================================================
# One-shot workflow
# ============================================================

def run_manual_eob_reweighting_workflow(
    eob2ml_result,
    eob_likelihood,
    parameters=("mass_1", "mass_2", "chi_1", "chi_2"),
    eob2eob_result=None,
    interval="90",
    proposal_logl_column="log_likelihood",
    proposal_beta=1.0,
    proposal_logl_is_already_tempered=False,
    extra_fixed=None,
    checkpoint_csv="eob_reweighting_checkpoint.csv",
    out_prefix="eob-reweighted",
    outdir=None,
):
    """
    Complete workflow:

    1. Compute EOB importance weights.
    2. Save weighted posterior.
    3. Print diagnostics.
    4. Make weighted corner plot.
    5. Make weighted marginalized histograms.
    6. Save weighted summary table.
    """

    inj = get_injection_parameters(eob2ml_result)

    weighted_post, diag = compute_eob_importance_weights(
        eob2ml_result=eob2ml_result,
        eob_likelihood=eob_likelihood,
        proposal_logl_column=proposal_logl_column,
        proposal_beta=proposal_beta,
        proposal_logl_is_already_tempered=proposal_logl_is_already_tempered,
        injection_parameters=inj,
        extra_fixed=extra_fixed,
        checkpoint_csv=checkpoint_csv,
        save_every=25,
    )

    print_reweighting_diagnostics(diag)

    weighted_csv = f"{out_prefix}_weighted_posterior.csv"
    if outdir is not None:
        weighted_csv = os.path.join(outdir, weighted_csv)
    weighted_post.to_csv(weighted_csv, index=False)
    print(f"Saved weighted posterior: {weighted_csv}")

    summary = summarize_weighted_marginals(
        weighted_post=weighted_post,
        parameters=list(parameters),
        injection_parameters=inj,
        interval=interval,
        mode_method="kde",
        bins=50,
    )

    summary_csv = f"{out_prefix}_summary_{interval}.csv"
    if outdir is not None:
        summary_csv = os.path.join(outdir, summary_csv)
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")
    print(summary)

    plot_weighted_corner(
        weighted_post=weighted_post,
        parameters=list(parameters),
        injection_parameters=inj,
        interval=interval,
        outdir=outdir,
        filename=f"{out_prefix}_corner_{interval}.png",
        bins=40,
        smooth=1.0,
    )

    plot_all_weighted_marginals(
        weighted_post=weighted_post,
        parameters=list(parameters),
        injection_parameters=inj,
        eob2ml_result=eob2ml_result,
        eob2eob_result=eob2eob_result,
        bins=50,
        interval=interval,
        out_prefix=f"{out_prefix}_marginal",
        outdir=outdir,
    )

    return weighted_post, summary, diag