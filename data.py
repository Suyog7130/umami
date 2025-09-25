
import os
import h5py
import logging
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt


SAMPLE_RATE = 8192.0  # n_samples = duration(s) / sample_rate
DURATION = 1.00
sample_len = int(DURATION * SAMPLE_RATE)
DELTA_T = DURATION / SAMPLE_RATE   # delta_t is just 1/sample_rate!
delta_f = 1.0 / DURATION  # delta_f = 1.0 / duration(s)
f_lower = 40.0
f_len = sample_len // 2 + 1  # upper frequency

np_gen = np.random.default_rng()

# TODO: Why is the array size not 8192 !!
PRESET_ARRAY_SIZE = 8191


INPUT_SHAPE = int(SAMPLE_RATE)
WFKW_SHAPE = 6
OUTPUT_SHAPE = INPUT_SHAPE


def tttdatasets(nsamples=1e5):
    """
    Gets the training, validation and test datasets for (m1,m2)
    \in [5,75] uniformly with qlim=10
    """
    m1 = np.random.uniform(5, 75, int(nsamples))
    m2 = np.random.uniform(5, 75, int(nsamples))
    q = m1/m2
    m1, m2 = m1[q <= 10], m2[q <= 10]
    print(f"Number of samples after qlim=10: {len(m1), len(m2)}")
    masses = np.vstack((m1, m2)).T
    np.random.shuffle(masses)
    train_split = int(0.7 * len(masses))
    val_split = int(0.80 * len(masses))
    train_masses = masses[:train_split]
    val_masses = masses[train_split:val_split]
    test_masses = masses[val_split:]
    print(f"Train/Val/Test sizes: {len(train_masses), len(val_masses), len(test_masses)}")
    return train_masses, val_masses, test_masses


def checkwaveform(masses):
    m1, m2 = masses
    


def main():
    train_masses, val_masses, test_masses = tttdatasets()
    checkwaveform(train_masses[0])

if __name__=="__main__":
    main()
