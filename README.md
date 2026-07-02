
[![Static Badge](https://img.shields.io/badge/https%3A%2F%2Fdoi.org%2F10.1103%2Fh92m-k44j?style=flat-square&logo=doi&logoColor=white&label=Paper&color=blue)](https://doi.org/10.1103/h92m-k44j)


# UMAMI

UMAMI (Unified Models for Astrophysical Merger Inspirals) is a Python-based gravitational-wave inference toolkit for generating compact-binary coalescence waveforms and running Bayesian parameter estimation with Bilby.

## Overview

The repository supports several closely related workflows:

- Baseline Bilby parameter-estimation sanity checks using analytic waveform approximants such as SEOBNRv4 / IMRPhenomD
- Machine-learning waveform inference using a custom MLWaveformGenerator
- Mixed injection/recovery experiments, such as EOB injection with ML recovery
- Debugging and validation utilities for waveform compatibility, likelihood behavior, determinism, and posterior quality
- Importance reweighting of existing posteriors for comparing waveform models

## Features

- Bayesian parameter estimation for binary black hole systems
- Support for multiple PE modes:
  - `eob2eob`
  - `ml2ml`
  - `eob2ml`
- Bilby integration for:
  - waveform generation
  - interferometer simulation
  - likelihood evaluation
  - nested sampling (`dynesty`, `nessai`, `pocomc` depending on script)
- ML waveform generation and calibration via trained PyTorch models
- Command-line utilities for:
  - running single injections
  - running injection campaigns
  - plotting corner plots
  - producing waveform posterior plots
  - analyzing and reweighting existing results
- Debug tools for checking:
  - waveform finiteness and amplitude scale
  - generator consistency
  - likelihood values at injection and random prior points
  - prior constraint validity
  - deterministic behavior across repeated runs

## Repository layout

The codebase is organized around a few main entry points:

- `inference.py` — main PE driver for ML and EOB/ML comparison workflows
- `mlwavegen.py` — custom Bilby waveform generator that wraps trained ML waveform and calibration models
- `debug-pe-with-ml.py` — focused debugging script for ML injection/recovery
- `debug-bilby-pe.py` — minimal analytic-waveform PE sanity check

Additional support modules referenced by the scripts include utilities for logging, IO, plotting, calibration, CVAE/FlexCVAE models, and model loading.

## What this project does

At a high level, UMAMI lets you:

- Generate a simulated gravitational-wave signal from either:
  - an analytic approximant such as `SEOBNRv4`, or
  - a trained ML waveform model
- Inject that signal into synthetic detector noise for interferometers such as H1 and L1
- Recover the source parameters with Bilby sampling
- Compare recovered posteriors across waveform families or model choices
- Inspect diagnostics to understand whether waveform generation, priors, or likelihoods are behaving as expected

This makes the repository useful for:

- validating ML surrogate waveforms against standard waveform models
- testing end-to-end inference pipelines
- running controlled experiments before large-scale PE campaigns
- studying biases introduced by waveform approximations

## Main workflows

### 1. Baseline analytic PE sanity check

`debug-bilby-pe.py` is a minimal end-to-end example that performs:

- aligned-spin BBH injection
- injection into H1/L1 data
- Bilby nested sampling with `dynesty`
- corner plotting of `mass_1`, `mass_2`, `spin_1z`, `spin_2z`

It is intended as a clean reference run before using ML models.

### 2. ML waveform injection and recovery

`debug-pe-with-ml.py` is a richer debug script that:

- loads a trained waveform model and calibration model
- builds an `MLWaveformGenerator`
- runs PE with either:
  - `ml2ml`
  - `eob2ml`
- generates diagnostics before sampling
- optionally runs in debug-only or plot-only mode

### 3. Full ML-based inference

`inference.py` is the primary inference driver. It supports:

- constructing priors from a base injection
- sampling injections from those priors
- building ML or EOB waveform generators
- running single injections or campaigns
- analyzing results with PP plots
- reweighting posteriors from EOB to ML likelihoods

### 4. ML waveform generator

`mlwavegen.py` defines `MLWaveformGenerator`, a custom Bilby waveform generator that:

- loads a trained ML waveform model
- optionally loads a calibration model
- converts Bilby parameter conventions into the ML model’s expected inputs
- returns plus and cross polarizations in the format Bilby expects

## Requirements

This repository is Python-based and depends heavily on:

- `bilby`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `torch`
- `tqdm`

Depending on the script and model path you use, you may also need project-specific modules such as:

- `flexcvae`
- `cvae`
- `calibration`
- `optimize`
- local utils modules

You will also need access to trained model checkpoints and configuration files when running ML-based inference.

## Example usage

### Run the baseline Bilby sanity check

```bash
python debug-bilby-pe.py --zero-noise --debug-only
```

Then run a full sampler once the setup looks correct:

```bash
python debug-bilby-pe.py --zero-noise --nlive 200 --dlogz 0.5
```

### Run ML injection/recovery debug workflow

```bash
python debug-pe-with-ml.py \
  --model-path path/to/model.pt \
  --config-path path/to/model_config.json \
  --calmodel-path path/to/calibrator.pt \
  --pe-run-type ml2ml \
  --zero-noise
```

To test EOB injection with ML recovery:

```bash
python debug-pe-with-ml.py \
  --model-path path/to/model.pt \
  --config-path path/to/model_config.json \
  --calmodel-path path/to/calibrator.pt \
  --pe-run-type eob2ml
```

### Run the main inference pipeline

```bash
python inference.py \
  --run-one-injection \
  --model-name model.pt \
  --model-config model-config.json \
  --calmodel-name calibrator.pt
```

For an injection campaign:

```bash
python inference.py \
  --run-pe-campaign \
  --num-injections 50 \
  --model-name model.pt \
  --model-config model-config.json \
  --calmodel-name calibrator.pt
```

For importance reweighting:

```bash
python inference.py \
  --imp-reweight \
  --results-fname some_result.json
```

## Outputs

Depending on the script and mode, UMAMI may produce:

- result JSON files
- posterior CSV files
- corner plots
- waveform posterior plots
- PP plots
- debug JSON reports
- pickled interferometer data
- timing summaries
- `DONE` marker files

Outputs are typically written into a run-specific `outdir` or `results/` directory.

## Notes on waveform conventions

The codebase shows careful handling of waveform parameter conventions:

- Bilby/LAL often expects parameters like `a_1`, `a_2`, `tilt_1`, `tilt_2`
- The ML model often uses simpler aligned-spin inputs such as `spin_1z`, `spin_2z` or `chi_1`, `chi_2`
- Conversion utilities map between those conventions during waveform generation and likelihood evaluation

This is important for matching waveform interfaces across analytic and ML sources.

## Development status

The repository appears to be actively evolving and includes extensive debugging hooks. Some scripts are aimed at:

- proving correctness of the waveform interface
- checking sampler behavior
- ensuring ML and analytic generators are compatible with Bilby
- validating output serialization and reweighting

If you are extending the project, it is a good idea to start with the debug scripts before running large campaigns.

## Suggested next steps

If you plan to use or extend UMAMI, consider documenting:

- exact environment setup and package versions
- how to obtain trained waveform and calibration models
- expected directory structure for `trained-models/`
- supported waveform approximants and parameterizations
- example notebooks or plots for validating results

## License

No license file was visible from the available repository metadata. Add one if you want to clarify reuse and distribution terms.

## Acknowledgements

UMAMI builds on the excellent open-source gravitational-wave inference ecosystem, especially:

- Bilby
- PyTorch
- the broader gravitational-wave modeling and parameter-estimation community