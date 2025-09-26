
import os
import h5py
import logging
import argparse
import numpy as np
import pandas as pd

import pycbc.waveform

from tqdm import tqdm
import matplotlib.pyplot as plt

from datacvae import calc_cutoffconst
from plotutils import putils

APPROXIMANT = 'SEOBNRv4'


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



class CheckWaveform:
    def __init__(self, masses, aligned=True, nosave=False):
        self.masses = masses
        self.nosave = nosave
        self.fname = 'waveforms'
        self.init_plot()
        self.waveform(aligned=aligned)

    def savefig(self, axes):
        plt.tight_layout()
        putils.beautifyPlot(axes, tickNum=6)
        if not self.nosave:
            plt.savefig(self.fname+'.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        plt.close()

    def init_plot(self):
        self.fig = plt.figure(figsize=(7,4.5))
        self.ax = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=self.fig)
        self.ax1 = plt.subplot2grid((2, 2), (1, 0), fig=self.fig)
        self.ax2 = plt.subplot2grid((2, 2), (1, 1), fig=self.fig)

    def plot_single_wf(self, m1, m2, hp, hc, amp, phase, freq):
        self.ax.plot(hp.sample_times, hp, label='$h_{+}$')
        self.ax.plot(hc.sample_times, hc, label='$h_{\\times}$')
        self.ax.set_xlabel('Sample Times', fontsize=12)
        self.ax.set_ylabel('h(t)', fontsize=12)
        self.ax.legend(loc='upper left', fontsize=10, ncols=2)
        self.ax1.plot(amp.sample_times, amp)
        self.ax1.set_xlabel('Sample Times', fontsize=12)
        self.ax1.set_ylabel('Amplitude', fontsize=12)
        self.ax2.plot(freq.sample_times, freq)
        self.ax2.set_xlabel('Sample Times', fontsize=12)
        self.ax2.set_ylabel('Frequency', fontsize=12)
        self.fig.suptitle(f'$m_1$={m1:.2f}, $m_2$={m2:.2f}, approx={APPROXIMANT}')
        self.savefig([self.ax, self.ax1, self.ax2])

    def plot_two_wfs(self, *kwargs):
        m1s, m2s, hps, amps, phases, freqs = kwargs
        for m1, m2, hp, amp, phase, freq in \
            zip(m1s, m2s, hps, amps, phases, freqs):
            self.ax.plot(hp.sample_times, hp, label=f'{m1:.2f} & {m2:.2f}'+' $M_{\\odot}$')
            self.ax.set_xlabel('Sample Times', fontsize=12)
            self.ax.set_ylabel('$h_{+}(t)$', fontsize=12)
            self.ax.legend(loc='upper left', fontsize=10, ncols=2)
            self.ax1.plot(amp.sample_times, amp)
            self.ax1.set_xlabel('Sample Times', fontsize=12)
            self.ax1.set_ylabel('Amplitude', fontsize=12)
            self.ax2.plot(freq.sample_times, freq)
            self.ax2.set_xlabel('Sample Times', fontsize=12)
            self.ax2.set_ylabel('Frequency', fontsize=12)
        self.savefig([self.ax, self.ax1, self.ax2])

    def get_waveform(self, m1, m2, aligned=True):
        wfkwargs = {
            "approximant": APPROXIMANT,
            "delta_t": DELTA_T,
            "f_lower": f_lower,
            "mass1": m1,
            "mass2": m2,
        }
        if aligned:
            wfkwargs["spin1z"] = 0.5 # np.random.uniform(-0.999, 0.999, 1)
            wfkwargs["spin2z"] = 0.5 # np.random.uniform(-0.999, 0.999, 1)
        logging.debug(f"Masses: {m1}, {m2}")
        print(wfkwargs)
        hp, hc = pycbc.waveform.get_td_waveform(**wfkwargs)
        hp, hc = hp.trim_zeros(), hc.trim_zeros()
        amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
        phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc)
        freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)
        logging.debug(f'Length of hp: {len(hp)}, hc: {len(hc)}')
        logging.debug(f'Length of amp: {len(amp)}, phase: {len(phase)}, freq: {len(freq)}')
        # print(hp.__dict__)
        return m1, m2, hp, hc, amp, phase, freq

    def waveform(self, aligned=True):
        self.fname += '-aligned' if aligned else ''
        m1s, m2s = self.masses
        if len(m1s) == 2:
            hps, amps, phases, freqs = [], [], [], []
            for m1, m2 in zip(m1s, m2s):
                m1, m2, hp, hc, amp, phase, freq = self.get_waveform(m1, m2, aligned=aligned)
                hps.append(hp)
                amps.append(amp)
                phases.append(phase)
                freqs.append(freq)
            self.plot_two_wfs(m1s, m2s, hps, amps, phases, freqs)
        else:
            m1, m2, hp, hc, amp, phase, freq = self.get_waveform(m1s[0], m2s[0], aligned=aligned)
            self.fname += f'_m1_{m1:.2f}_m2_{m2:.2f}'
            self.plot_single_wf(m1, m2, hp, hc, amp, phase, freq)



def main(args):
    train_masses, val_masses, test_masses = tttdatasets()
    CheckWaveform(masses=[[50, 30], [15, 5]], 
                  aligned=args.aligned,
                  nosave=args.nosave)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate and plot gravitational waveforms.")
    parser.add_argument("--aligned", action="store_true", help="Generate aligned-spin waveforms.")
    parser.add_argument("--fname", type=str, default="waveforms", help="Filename for saving the plots.")
    parser.add_argument("--nosave", action="store_true", help="Do not save the plots.")
    args = parser.parse_args()
    main(args)
