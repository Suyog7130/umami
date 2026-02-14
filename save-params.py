#!/usr/bin/env python3
"""
Generate random gravitational-wave prior samples and write to multiple files.

Defaults:
- 1e6 total samples
- 10 files, 1e5 rows each
- Columns: m1_msun, m2_msun, chi1z, chi2z, dL_Mpc
- 5 decimal digits in output
- Crash resilience: reopen the file every `rows_per_commit` rows (default 100)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Literal, Optional

import numpy as np


def sample_distance(
    rng: np.random.Generator,
    n: int,
    dmin: float,
    dmax: float,
    prior: Literal["uniform", "uniform_in_volume"] = "uniform",
) -> np.ndarray:
    """
    Sample luminosity distance in Mpc.

    prior="uniform": d ~ U(dmin, dmax)
    prior="uniform_in_volume": p(d) ∝ d^2 on [dmin, dmax] (uniform in volume)
    """
    if dmin <= 0 or dmax <= 0 or dmax <= dmin:
        raise ValueError(f"Invalid distance range: dmin={dmin}, dmax={dmax}")

    if prior == "uniform":
        return rng.uniform(dmin, dmax, size=n)

    if prior == "uniform_in_volume":
        # If u ~ U(0,1), then d = (dmin^3 + u*(dmax^3 - dmin^3))^(1/3)
        u = rng.random(size=n)
        return (dmin**3 + u * (dmax**3 - dmin**3)) ** (1.0 / 3.0)

    raise ValueError(f"Unknown distance prior: {prior}")


def generate_gw_prior_samples_csv(
    out_dir: str | Path,
    total_samples: int = 1_000_000,
    n_files: int = 10,
    rows_per_file: int = 100_000,
    rows_per_commit: int = 100,  # reopen the file every 100 rows
    mass_min: float = 5.0,
    mass_max: float = 75.0,
    spin_min: float = -0.99,
    spin_max: float = 0.99,
    dL_min_mpc: float = 200.0,
    dL_max_mpc: float = 600.0,
    distance_prior: Literal["uniform", "uniform_in_volume"] = "uniform",
    enforce_m1_ge_m2: bool = True,
    seed: Optional[int] = None,
) -> None:
    """
    Writes CSV files:
      {prefix}_000.csv ... {prefix}_{n_files-1:03d}.csv

    Each file has a header and rows_per_file rows.
    Values are written with exactly 5 decimal digits.

    Crash safety behavior:
      - generates rows in chunks of `rows_per_commit`
      - opens file in append mode, writes the chunk, closes it
    """
    dpname = 'u' if distance_prior == 'uniform' else 'uvol'
    fname = f'm{int(mass_min)}-{int(mass_max)}_s{spin_min}-{spin_max}_dL{int(dL_min_mpc)}-{int(dL_max_mpc)}_{dpname}'
    fname = fname.replace('.', 'p')
    out_dir = out_dir + "/" + fname
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if total_samples != n_files * rows_per_file:
        raise ValueError(
            f"total_samples must equal n_files*rows_per_file. "
            f"Got total_samples={total_samples}, n_files={n_files}, rows_per_file={rows_per_file}."
        )

    if rows_per_commit <= 0:
        raise ValueError("rows_per_commit must be >= 1")

    if not (mass_max > mass_min):
        raise ValueError("Mass range invalid")
    if not (spin_max > spin_min):
        raise ValueError("Spin range invalid")

    rng = np.random.default_rng(seed)  # PCG64, good quality, low correlation

    header = "index,m1_msun,m2_msun,chi1x,chi1y,chi1z,chi2x,chi2y,chi2z,dL_Mpc"

    for file_idx in range(n_files):
        out_path = out_dir / f"seed{seed}-params_{file_idx:03d}.csv"

        # If file exists, remove it so we always get exactly rows_per_file rows.
        if out_path.exists():
            out_path.unlink()

        # Write in commits
        written = 0
        while written < rows_per_file:
            n_chunk = min(rows_per_commit, rows_per_file - written)

            m1 = rng.uniform(mass_min, mass_max, size=n_chunk)
            m2 = rng.uniform(mass_min, mass_max, size=n_chunk)

            if enforce_m1_ge_m2:
                m_hi = np.maximum(m1, m2)
                m_lo = np.minimum(m1, m2)
                m1, m2 = m_hi, m_lo

            chi1x = rng.uniform(spin_min, spin_max, size=n_chunk)
            chi1y = rng.uniform(spin_min, spin_max, size=n_chunk)
            chi1z = rng.uniform(spin_min, spin_max, size=n_chunk)
            chi2x = rng.uniform(spin_min, spin_max, size=n_chunk)
            chi2y = rng.uniform(spin_min, spin_max, size=n_chunk)
            chi2z = rng.uniform(spin_min, spin_max, size=n_chunk)

            dL = sample_distance(
                rng=rng,
                n=n_chunk,
                dmin=dL_min_mpc,
                dmax=dL_max_mpc,
                prior=distance_prior,
            )

            index = np.arange(written, written + n_chunk)
            data = np.column_stack([index, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dL])

            # Optional: round in-memory too (saving format already enforces 2 decimals)
            data = np.round(data, 2)
            data[:, 0] = data[:, 0].astype(int)

            # Open, append, write, close (crash resilient)
            is_first_write = (written == 0)
            with open(out_path, "a", newline="") as f:
                if is_first_write:
                    f.write(header + "\n")
                # Write index as integer, rest as float with 2 decimals
                fmt = ["%d"] + ["%.2f"] * (data.shape[1] - 1)
                np.savetxt(f, data, delimiter=",", fmt=fmt)
            written += n_chunk
            
        print(f"Wrote {rows_per_file} rows -> {out_path}")


# # Optional: HDF5 writer (faster and smaller than CSV for big datasets).
# # This version ALSO closes and reopens every commit, as you requested.
# def generate_gw_prior_samples_hdf5(
#     out_dir: str | Path,
#     total_samples: int = 1_000_000,
#     n_files: int = 10,
#     rows_per_file: int = 100_000,
#     rows_per_commit: int = 10_000,  # HDF5 is fine with bigger chunks
#     mass_min: float = 5.0,
#     mass_max: float = 75.0,
#     spin_min: float = -0.99,
#     spin_max: float = 0.99,
#     dL_min_mpc: float = 200.0,
#     dL_max_mpc: float = 600.0,
#     distance_prior: Literal["uniform", "uniform_in_volume"] = "uniform",
#     enforce_m1_ge_m2: bool = True,
#     seed: Optional[int] = None,
#     filename_prefix: str = "gw_prior_samples",
# ) -> None:
#     """
#     Requires: pip install h5py
#     Writes HDF5 files with datasets:
#       /params  shape=(rows_per_file, 5)
#       /columns stored as an attribute
#     """
#     import h5py  # local import so CSV-only users do not need it

#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     if total_samples != n_files * rows_per_file:
#         raise ValueError("total_samples must equal n_files*rows_per_file")

#     rng = np.random.default_rng(seed)
#     columns = ["m1_msun", "m2_msun", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z", "dL_Mpc"]

#     for file_idx in range(n_files):
#         out_path = out_dir / f"{filename_prefix}_{file_idx:03d}.h5"
#         if out_path.exists():
#             out_path.unlink()

#         written = 0
#         while written < rows_per_file:
#             n_chunk = min(rows_per_commit, rows_per_file - written)

#             m1 = rng.uniform(mass_min, mass_max, size=n_chunk)
#             m2 = rng.uniform(mass_min, mass_max, size=n_chunk)

#             if enforce_m1_ge_m2:
#                 m_hi = np.maximum(m1, m2)
#                 m_lo = np.minimum(m1, m2)
#                 m1, m2 = m_hi, m_lo

#             chi1x = rng.uniform(spin_min, spin_max, size=n_chunk)
#             chi1y = rng.uniform(spin_min, spin_max, size=n_chunk)
#             chi1z = rng.uniform(spin_min, spin_max, size=n_chunk)
#             chi2x = rng.uniform(spin_min, spin_max, size=n_chunk)
#             chi2y = rng.uniform(spin_min, spin_max, size=n_chunk)
#             chi2z = rng.uniform(spin_min, spin_max, size=n_chunk)
#             dL = sample_distance(rng, n_chunk, dL_min_mpc, dL_max_mpc, distance_prior)

#             data = np.column_stack([m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dL])
#             data = np.round(data, 2).astype(np.float32)  # float32 is plenty for 2 decimals
#             data[:, 0] = data[:, 0].astype(int)

#             # Close and reopen every commit, as requested
#             with h5py.File(out_path, "a") as f:
#                 if "params" not in f:
#                     dset = f.create_dataset(
#                         "params",
#                         shape=(rows_per_file, 9),
#                         maxshape=(rows_per_file, 9),
#                         dtype=np.float32,
#                         chunks=(min(rows_per_commit, rows_per_file), 9),
#                         compression="gzip",
#                         compression_opts=4,
#                     )
#                     dset.attrs["columns"] = np.array(columns, dtype="S")
#                 else:
#                     dset = f["params"]

#                 dset[written:written + n_chunk, :] = data

#             written += n_chunk

#         print(f"Wrote {rows_per_file} rows -> {out_path}")


if __name__ == "__main__":

    DEFAULTS = {
        "out_dir": "data/params",
        "total_samples": 1_000_000,
        "mass_min": 5.0,
        "mass_max": 200.0,
        "spin_min": -0.99,
        "spin_max": 0.99,
        "dL_min_mpc": 100.0,
        "dL_max_mpc": 1000.0,
        "distance_prior": "uniform_in_volume",
        "random_seed": 100,
    }

    parser = argparse.ArgumentParser(description="Generate gravitational-wave prior samples and save to CSV files.")
    parser.add_argument( "--out-dir", type=str, default=DEFAULTS["out_dir"],
                        help="Output directory to save parameter files.")
    parser.add_argument( "--random-seed", type=int, default=DEFAULTS["random_seed"],
                        help="Random seed for reproducibility (default: 100).")
    parser.add_argument('--mass-min', type=float, default=DEFAULTS["mass_min"], 
                        help='Minimum component mass (Msun).')
    parser.add_argument('--mass-max', type=float, default=DEFAULTS["mass_max"], 
                        help='Maximum component mass (Msun).')
    parser.add_argument('--spin-min', type=float, default=DEFAULTS["spin_min"], 
                        help='Minimum spin component.')
    parser.add_argument('--spin-max', type=float, default=DEFAULTS["spin_max"], 
                        help='Maximum spin component.')
    parser.add_argument('--dL-min-mpc', type=float, default=DEFAULTS["dL_min_mpc"], 
                        help='Minimum luminosity distance (Mpc).')
    parser.add_argument('--dL-max-mpc', type=float, default=DEFAULTS["dL_max_mpc"], 
                        help='Maximum luminosity distance (Mpc).')
    parser.add_argument('--distance-prior', type=str, choices=['uniform', 'uniform_in_volume'], 
                        default=DEFAULTS["distance_prior"],
                        help='Distance prior type.')
    args = parser.parse_args()

    generate_gw_prior_samples_csv(
        out_dir=args.out_dir,  # default="data/params"
        total_samples=DEFAULTS["total_samples"],  # 1_000_000
        n_files=10,
        rows_per_file=DEFAULTS["total_samples"] // 10,
        rows_per_commit=100,
        mass_min=args.mass_min,
        mass_max=args.mass_max,
        spin_min=args.spin_min,
        spin_max=args.spin_max,
        dL_min_mpc=args.dL_min_mpc,
        dL_max_mpc=args.dL_max_mpc,
        distance_prior=args.distance_prior,
        enforce_m1_ge_m2=True,
        seed=args.random_seed,
    )