"""
2025/03/15
For Phenom models, since the freq-amp calculation is,
in fact, the first step in the evaluating the waveform,
we should not be converting the strain to freq-amp.
Instead, for the Phenom models atleast, we should be
able to directly use the freq-amp values from lalsuite.
However, as the initial work, this code is alrighto.

2026/02/13
This code is to read already saved waveforms from HDF5 files,
and then load them as PyTorch Datasets and DataLoaders.
These waveforms and input files are created using `data.py`.
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('Agg')   # non GUI backend
import matplotlib.pyplot as plt

# import seaborn as sns
# sns.set()
import pylab

from sklearn import metrics
from tqdm import tqdm

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

from torch import nn

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

import pycbc.noise
import pycbc.psd
from pycbc.waveform import get_td_waveform
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, \
    pycbc.detector
import pycbc.filter
import pycbc.filter

from sklearn import metrics

from datetime import datetime

import logging

import sys
sys.path.append('~Dropbox/plotutils-work/plotutils/')
from plotutils import putils


APPROXIMANTS = ['IMRPhenomD', 'SEOBNRv4', 'NRSur7dq4', 'EccentricTD']


SAMPLE_RATE = 8192.0  # n_samples = duration(s) / sample_rate
DURATION = 1.00
sample_len = int(DURATION * SAMPLE_RATE)
DELTA_T = DURATION / SAMPLE_RATE   # delta_t is just 1/sample_rate!
DELTA_T = DURATION / SAMPLE_RATE   # delta_t is just 1/sample_rate!
delta_f = 1.0 / DURATION  # delta_f = 1.0 / duration(s)
f_lower = 40.0
f_len = sample_len // 2 + 1  # upper frequency

np_gen = np.random.default_rng()

# TODO: Why is the array size not 8192 !!
PRESET_ARRAY_SIZE = 8191

# Create the detectors
detectors_abbr = ('H1', 'L1')
detectors = []
for det_abbr in detectors_abbr:
    detectors.append(pycbc.detector.Detector(det_abbr))

nts_func = pycbc.noise.gaussian.frequency_noise_from_psd
psd_func = pycbc.psd.analytical.aLIGOZeroDetHighPower
psds = [psd_func(f_len, delta_f, f_lower) for _ in range(len(detectors))]
psd = psds[0]

skylocation_dist = pycbc.distributions.sky_location.UniformSky()

INPUT_SHAPE = int(SAMPLE_RATE)
WFKW_SHAPE = 6
OUTPUT_SHAPE = INPUT_SHAPE


NOW = datetime.now().strftime("%Y-%m-%d-%H%M%S")

def get_mass(m1start=5, m1end=75, m1delta=0.25, m2end=None, m2start=None,
             m2delta=None, criterion=True, plot=False, splitTT=True,
             splitq=False, ntraining=0.7, nvald=0.1, ntest=0.2, qlim=10,
             transparent=True):
    """
    Generate mass range for different specified criterion.
    Default is the one used is:
        5<=m1<=75, 5<=m2<=75, m2/m1<=10 and m2>=m1
    The GH18 criterion is
        5>=m1=m2<=75, m1/m2<10 and m2<m1

    TODO: This mass distro should be motivated by the actual
    distribution of the masses in the LVK observations. Kipp
    talked about this sometime back and I gotta mention this
    in the paper.

    Arguments
    ---------
        m1start : float
            Starting mass of the larger (primary) binary by default.
            Default is 5 Msolar.
        m1end : float
            Ending mass of the larger (primary)  binary by default.
            Default is 75 Msolar.
        m1delta : float
            Step size for the larger (primary)  binary by default.
            Default is 0.25 Msolar.
        m2end : float
            Ending mass of the smaller binary by default.
            Default is None.
        m2start : float
            Starting mass of the smaller binary by default.
            Default
            Default is None.
        m2delta : float
            Step size for the smaller binary by default.
            Default is None.
        criterion : str
            Use the mass ratio rule or not. Default is True.
        qlim : float
            Mass ratio limit. Default is 10 Msolar.
            Any binary below this limit is labeled Low Mass Binary.
            Here, the mass ratio q is defined as m1/m2.
    
    Returns
    -------
        masses : list
            List of masses (m1,m2) generated based on the
            specified criterion.
    """
    fname = 'mass-plot'
    # select range of masses
    m2start = m2start if m2start is not None else m1start
    m2end = m2end if m2end is not None else m1end
    m2delta = m2delta if m2delta is not None else m1delta

    masses = []
    if criterion:
        fname += '-gh18like'
        for m1 in np.arange(m1end, m1start-m1delta, -m1delta):
            for m2 in np.arange(m2start, m2end+m2delta, m2delta):
                if (m1/m2 <= qlim and m1>=m2):
                    masses.append([m1,m2])
    else:
        for m1 in np.arange(m1end, m1start, -m1delta):
            for m2 in np.arange(m2start+(m2delta/2), m2end+(m2delta/2), m2delta):
                    masses.append([m1,m2])

    masses = np.random.permutation(np.array(masses))
    logging.debug(f'type(masses)={type(masses)}')
    if not splitTT:
        logging.debug(f'type(masses)={type(masses)}')
        return masses

    if splitTT:
        ntraining = int(ntraining*len(masses))
        nvald = int(nvald*len(masses))
        ntest = int(ntest*len(masses))
        logging.debug(f'Training: {ntraining}, Validation: {nvald}, Testing: {ntest}')

    logging.debug(f'type(masses)={type(masses)}')
    if not splitTT:
        logging.debug(f'type(masses)={type(masses)}')
        return masses

    if splitTT:
        ntraining = int(ntraining*len(masses))
        nvald = int(nvald*len(masses))
        ntest = int(ntest*len(masses))
        logging.debug(f'Training: {ntraining}, Validation: {nvald}, Testing: {ntest}')

        ttsplits = np.split(masses, [ntraining,ntraining+nvald,ntraining+nvald+ntest])
        logging.debug(f"Training set shape: {ttsplits[0].shape}")
        logging.debug(f"Validation set shape: {ttsplits[1].shape}")
        logging.debug(f"Testing set shape: {ttsplits[2].shape}")
        logging.debug(f'type(ttsplits)={type(ttsplits)}')
        logging.debug(f"Training set shape: {ttsplits[0].shape}")
        logging.debug(f"Validation set shape: {ttsplits[1].shape}")
        logging.debug(f"Testing set shape: {ttsplits[2].shape}")
        logging.debug(f'type(ttsplits)={type(ttsplits)}')

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(5,5))
            if splitTT:
                ax.plot(ttsplits[0][:,0], ttsplits[0][:,1], '.', 
                        color='darkgray', label='Training')
                ax.plot(ttsplits[1][:,0], ttsplits[1][:,1], '.', 
                        color='blue', label='Validation', alpha=0.5)
                ax.plot(ttsplits[2][:,0], ttsplits[2][:,1], '.', 
                        color='red', label='Testing', alpha=0.5,)
                ax.legend()
            else:
                ax.plot(masses[:,0], masses[:,1], '.')
            #if not criterion=='gh18':
            #    ax.set_xlim([m1start-m1delta, m1end+m1delta])
            #    ax.set_ylim([m2start-m2delta, m2end+m2delta])
            ax.set_xlabel('$m_1$ ($M_{\\odot}$)')
            ax.set_ylabel('$m_2$ ($M_{\\odot}$)')
            putils.beautifyPlot([ax], grid=True, tickNum=8)
            plt.tight_layout()
            fname = fname+f'-{str(len(masses))}-qlim{qlim}'
            if transparent:
                fname += '-transparent'
                plt.savefig(fname+'.png', dpi=300, transparent=True)
            else:
                plt.savefig(fname+'.png', dpi=300)
            logging.info(f"Mass plot saved to {fname+'.png'}")
            plt.show()
        logging.debug(f'type(ttsplits)={type(ttsplits)}')
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(5,5))
            if splitTT:
                ax.plot(ttsplits[0][:,0], ttsplits[0][:,1], '.', 
                        color='darkgray', label='Training')
                ax.plot(ttsplits[1][:,0], ttsplits[1][:,1], '.', 
                        color='blue', label='Validation', alpha=0.5)
                ax.plot(ttsplits[2][:,0], ttsplits[2][:,1], '.', 
                        color='red', label='Testing', alpha=0.5,)
                ax.legend()
            else:
                ax.plot(masses[:,0], masses[:,1], '.')
            #if not criterion=='gh18':
            #    ax.set_xlim([m1start-m1delta, m1end+m1delta])
            #    ax.set_ylim([m2start-m2delta, m2end+m2delta])
            ax.set_xlabel('$m_1$ ($M_{\\odot}$)')
            ax.set_ylabel('$m_2$ ($M_{\\odot}$)')
            putils.beautifyPlot([ax], grid=True, tickNum=8)
            plt.tight_layout()
            fname = fname+f'-{str(len(masses))}-qlim{qlim}'
            if transparent:
                fname += '-transparent'
                plt.savefig(fname+'.png', dpi=300, transparent=True)
            else:
                plt.savefig(fname+'.png', dpi=300)
            logging.info(f"Mass plot saved to {fname+'.png'}")
            plt.show()
        logging.debug(f'type(ttsplits)={type(ttsplits)}')
        return ttsplits


def get_strain (m1, m2, approximant='IMRPhenomD', convert=False,
                plot=False, nokeys=False, plotfreqamp=False, nosave=False):
    """
    TODO: Why is the waveform truncated ?
    NOTE: Generated `hp` is of different length ~
          Perhaps, due to freq->time domain conversion ~
          -> Only take the last `OUTPUT_LENGTH` values,
             initial ones are zeros

    Projects the plus and cross polarizations to obtain the strain
    as visible at each detector. Only the input masses are specified.
    The right ascension, declination, and polarization angle are
    randomly sampled for each signal.

    Instead of using `time_slice`, have simply sliced the list manually.
    Because using that function can give variable lengths of the array.

    TODO: Whiten the waveforms!
    
    Arguments
    ---------
        m1 : float
            Mass of the larger Binary
        m2 : float
            Mass of the smaller binary (m2>m1 and m2<3*M_solar for NS)
        approximant : str
            Approximant to use for the waveform
        convert : bool, optional
            Convert the strains to phase & amplitude. Default is True.
        nokeys : bool, optional
            Return only the strain values. Default is False.

    Returns
    -------
        strains : array
            Strain values for the two detector.
    """
    logging.debug('logging started!')

    # Initialize random distributions.
    angles = np_gen.uniform(0., 2*np.pi, 3)

    # Generate 1.00 sec signal at SAMPLE_RATE Hz.
    waveform_kwargs = {'delta_t': DELTA_T, 
                       'f_lower': f_lower}
    waveform_kwargs['approximant'] = approximant
    waveform_kwargs['mass1'] = m1
    waveform_kwargs['mass2'] = m2

    # Have random COA phase and Inclination for each sample
    # TODO: Having additional parameters also seems to complicate
    # the freq-amp conversion.
    # waveform_kwargs['coa_phase'] = angles[0]
    # waveform_kwargs['inclination'] = angles[1]

    # pol_angle = angles[2]
    # declination, right_ascension = skylocation_dist.rvs()[0]

    hp, hc = pycbc.waveform.get_td_waveform(**waveform_kwargs)

    logging.debug(f'waveform_kwargs: {waveform_kwargs}')

    logging.debug(f"Type of hp: {type(hp)}, Type of hc: {type(hc)}")
    logging.debug(f"Length of hp: {len(hp)}, Length of hc: {len(hc)}")
    # logging.debug(dir(hp))
    # logging.debug(dir(hp))
    logging.debug(f'duration={hp.duration}')
    logging.debug(f'sample-rate={hp.sample_rate}')

    hp = hp.trim_zeros()
    hc = hc.trim_zeros()

    # # Slice the Ringdown part, assuming all mergers occur @T=0
    # hp = hp.time_slice(hp.start_time, 0.00)
    # hc = hc.time_slice(hc.start_time, 0.00)

    # Truncate or pad the waveform to ensure it is exactly 1 second long
    # Merger time is always 80% of duration away from start for waveforms
    # whose original length is > 1 second.
    if len(hp) > SAMPLE_RATE * DURATION:
        hp = hp[len(hp) - int(SAMPLE_RATE * DURATION):]
        hc = hc[len(hc) - int(SAMPLE_RATE * DURATION):]
    elif len(hp) < SAMPLE_RATE * DURATION:
        hp.append_zeros(int(SAMPLE_RATE * DURATION) - len(hp))
        hc.append_zeros(int(SAMPLE_RATE * DURATION) - len(hc))

    logging.debug(f'hp slice duration={hp.duration}')

    # # Don't need to project this ~

    # strains = [det.project_wave(hp, hc, right_ascension, declination, pol_angle) for det in detectors]
    
    # waveform_kwargs['declination'] = declination
    # waveform_kwargs['right_ascension'] = right_ascension
    # waveform_kwargs['pol_angle'] = pol_angle

    # shift_time = np_gen.uniform(0.0, 0.2)
    # logging.debug(shift_time)
    # logging.debug(SAMPLE_RATE)

    # for strain in strains:
    #     logging.debug('hi', strain.sample_rate, len(strain))
    #     # `append_zeros` is a `pycbc.TimeSeries` function
    #     strain.append_zeros(strain.sample_rate*shift_time) #time shift
    #     strain.prepend_zeros(max(0,sample_len-len(strain))) #prepend zeros if the length is shorter than 1.25 seconds
    #     logging.debug(len(strain))

    # # TODO: I cut-off initial part here ~
    # strains = [strain[len(strain)-sample_len:] for strain in strains if len(strain) > sample_len]
    # logging.debug(len(strains[int(np.random.randint(len(strains)))]))
    
    if plot:
        # Plot original strain data
        fig, ax = plt.subplots(1, 1, figsize=(7,3))
        ax.plot(hp.sample_times, hc, label=f'duration={hp.duration}')
        # ax.plot(hc.sample_times, hc, label='hc')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('$h_{\\plus}$ Strain')
        ax.secondary_xaxis('top', functions=(lambda x: (x - hp.sample_times[0]) / hp.duration * len(hc), 
                             lambda x: x * hp.duration / len(hc)))
        ax.legend()
        ax.set_title(f'{approximant}: {m1} & {m2}'+'$M_{\\odot}$')
        plt.tight_layout()
        putils.beautifyPlot([ax])
        if not nosave:
            fname = f'strain-{approximant}-plot'
            fname += '-jsps'
            plt.savefig(fname+'.png', dpi=300)
        plt.show()

    if convert:
        get_strain_convert(hp, hc, nokeys=nokeys, plotfreqamp=plotfreqamp,
                           m1=m1, m2=m2)

    # If not `convert` to `freq` and `amp`
    # Have correct input lengths!
    # It should be ensured that the merger is always within the data.
    if len(hp) > PRESET_ARRAY_SIZE:
        diff = len(hp) - PRESET_ARRAY_SIZE
        logging.debug(f'len(freq) > {PRESET_ARRAY_SIZE} by {diff} ele \
                            \n So truncating array from the left!')
        hp = hp[diff:]
    if len(hc) > PRESET_ARRAY_SIZE:
        diff = len(hc) - PRESET_ARRAY_SIZE
        logging.debug(f'len(amp) > {PRESET_ARRAY_SIZE} by {diff} ele \
                            \n So truncating array from the left!')
        hc = hc[diff:]

    # # Rescale the amplitude
    # # No need to rescale if we are normalizing the data.
    # hp = hp * 10**20
    # hc = hc * 10**20

    # Normalize inputs wrt to the keys
    hpkeys = [np.mean(hp), np.std(hp)]
    hckeys = [np.mean(hc), np.std(hc)]
    hp_orig = hp
    hp = (hp - np.mean(hp)) / np.std(hp)
    hc = (hc - np.mean(hc)) / np.std(hc)
    strains = np.vstack((hp,hc)).astype(np.float64)

    # Plot normalized and original data side-by-side
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(7,5))
        ax[0].plot(hp_orig.sample_times, hp_orig, label='Original')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('$h_{\\plus}$ Strain')
        ax[0].legend()

        ax[1].plot(hp.sample_times, hp, label='Normalized')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('$h_{\\plus}$ Strain')
        ax[1].legend()
        plt.tight_layout()
        putils.beautifyPlot(ax)
        if not nosave:
            fname = f'normalized-{approximant}-strain-plot'
            fname += '-jsps'
            plt.savefig(fname+'.png', dpi=300)
        plt.show()
        plt.close()

    return (strains, np.array([m1,m2]), np.array([hpkeys,hckeys]))


def get_strain_convert(hp, hc, nokeys=False, plotfreqamp=False, 
                       m1=None, m2=None):
    # No need to manually do this, `pycbc` has a function for this ~
    # See : https://pycbc.org/pycbc/latest/html/waveform.html
    # `arctan` or `arctan2(hp,hc)` ? --> use arctan2 which takes care of the
    # quadrant of the angle.
    # theta = np.arctan(hc, hp)
    # amp = np.sqrt(hp**2 + hc**2)
    #hp, hc = hp.trim_zeros(), hc.trim_zeros()
    logging.debug('Converting `hp` & `hc` to Freq Amp!')
    amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
    phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc)
    freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)

    # phase = np.arctan2(hc, hp)
    # amp = np.sqrt(hp**2 + hc**2)
    # freq = np.diff(phase) / 2 * np.pi * DELTA_T

    logging.debug(f'length of amp: {len(amp)}')
    logging.debug(f'length of phase: {len(phase)}')
    logging.debug(f'length of freq: {len(freq)}')
    # logging.debug(freq.__dict__)
    logging.debug(f'length of freq.data: {len(freq.data)}')

    if plotfreqamp:
        logging.debug('# printing Freq Amp plot')
        fig, axes = plt.subplots(1, 4, figsize=(20,5))
        axes[0].plot(amp.sample_times, amp)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')

        axes[1].plot(phase, amp)
        axes[1].set_xlabel('Phase')
        axes[1].set_ylabel('Amplitude')

        axes[2].plot(range(len(freq)), freq)
        axes[2].set_xlabel('Index')
        axes[2].set_ylabel('Frequency')

        axes[3].plot(freq.sample_times, freq)
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Frequency')
        plt.tight_layout()
        putils.beautifyPlot(axes)
        # plt.savefig('amp-phase-freq-plot.png', dpi=300)
        plt.show()

    # Ensure freq and amp are numeric arrays
    amp = np.array(amp.data, dtype=np.float64)
    freq = np.array(freq.data, dtype=np.float64)
    logging.debug(f'shape of freq: {freq.shape}')

    # Rescale the amp by 10^20
    logging.debug(f'Original Amp: {amp}')
    logging.debug(f'Type of Amp: {type(amp)}, Type of Amp[0]: {type(amp[0])}')
    amp = amp * 10**20
    logging.debug(f'Rescaled Amp: {amp}')
    logging.debug(f'Type of Rescaled Amp: {type(amp)}, Type of Rescaled Amp[0]: {type(amp[0])}')

    # Have correct input lengths
    # amp = amp[1:] # to have equal sized arrays
    if len(amp) > PRESET_ARRAY_SIZE:
        diff = len(amp) - PRESET_ARRAY_SIZE
        logging.debug(f'len(amp) > {PRESET_ARRAY_SIZE} by {diff} ele \
                            \n So truncating array from the left!')
        amp = amp[diff:]
    logging.debug(amp.shape)
    if len(freq) > PRESET_ARRAY_SIZE:
        diff = len(freq) - PRESET_ARRAY_SIZE
        logging.debug(f'len(freq) > {PRESET_ARRAY_SIZE} by {diff} ele \
                            \n So truncating array from the left!')
        freq = freq[diff:]

    # Stack freq and amp into strains
    strains = np.vstack((amp, freq)).astype(np.float64)
    logging.debug(strains.shape)
    logging.debug(amp.shape)
    logging.debug(freq.shape)

    if nokeys:
        return (np.array([amp, freq]), np.array([m1,m2]))
    else:
        # calculate the keys
        # the returned values are the freq/amp, labels and keys
        # for a single data sample.
        amp_keys = [np.mean(amp), np.std(amp)]
        freq_keys = [np.mean(freq), np.std(freq)]
        logging.debug(f"Amplitude Keys: {amp_keys}")
        logging.debug(f"Frequency Keys: {freq_keys}")
        
        # Normallize the input data
        # This is the standard normalization, centers the distro
        # at the mean with a std of 1
        amp = (amp - np.mean(amp)) / np.std(amp)
        freq = (freq - np.mean(freq)) / np.std(freq)

        # # print(amp)
        # plt.plot(range(len(freq)), freq, '-')
        # plt.show()
        # plt.close()
        logging.debug(type(amp), type(amp[0]))
        return (strains, np.array([m1,m2]), np.array([amp_keys, freq_keys]))


def get_fd_strain(m1, m2, approximant='IMRPhenomD', plot=False):
    """
    """
    logging.debug('logging started!')

    # Initialize random distributions.
    angles = np_gen.uniform(0., 2*np.pi, 3)

    # Generate 1.00 sec signal at SAMPLE_RATE Hz.
    waveform_kwargs = {'delta_t': DELTA_T, 'f_lower': f_lower}
    waveform_kwargs['approximant'] = approximant
    waveform_kwargs['mass1'] = m1
    waveform_kwargs['mass2'] = m2
    # Have random COA phase and Inclination for each sample
    waveform_kwargs['coa_phase'] = angles[0]
    waveform_kwargs['inclination'] = angles[1]
    # waveform_kwargs['duration'] = 1.00   # no keyword like this
    pol_angle = angles[2]
    declination, right_ascension = skylocation_dist.rvs()[0]
    
    waveform_kwargs['delta_f'] = delta_f
    sptilde, sctilde = pycbc.waveform.get_fd_waveform(**waveform_kwargs)
    # list all Attributes in `sptilde`
    # print(dir(sptilde))
    # print(sptilde.sample_rate)
    # print(sptilde.delta_t)
    # print(sptilde.sample_frequencies)
    # print(sptilde.duration)
    # print(len(sptilde))

    amp = pycbc.waveform.utils.amplitude_from_frequencyseries(sptilde)

    if plot:
        x = range(len(sptilde.sample_frequencies))
        # plt.plot(amp.sample_frequencies, amp, '-')
        plt.plot(sptilde.sample_frequencies, sptilde, '-')
        plt.show()

    strains = np.vstack((sptilde, sctilde)).astype(np.float64)
    spkeys = [np.mean(sptilde), np.var(sptilde)]
    sckeys = [np.mean(sctilde), np.var(sctilde)]
    return (strains, np.array([m1,m2]), np.array([spkeys, sckeys]))


def get_vals_for_hdf(m1, m2, approximant='SEOBNRv4', eccentricity=None,
                    otherparams=False):
    """
    It is taken care of that the `hp` and `hc` are of the same length
    and the `amp` and `phase` and `freq` are of the same length.
    However, the data is not normalized or whitened. And thus, the keys
    are not calculated!
    These are the full waveforms, i.e. the ringdown part is not cut-off.
    The amplitudes are not rescaled to 10^20.
    """
    # Initialize random distributions.
    angles = np_gen.uniform(0., 2*np.pi, 3)

    # Initialize random distributions.
    angles = np_gen.uniform(0., 2*np.pi, 3)

    extra = {}
    waveform_kwargs = {'approximant': approximant,
                        'mass1': m1,
                        'mass2': m2,
                        'f_lower': f_lower,
                        'delta_t': DELTA_T,
                        # 'right_ascension': angles[2],
                        # 'declination': angles[3],
                        # 'pol_angle': angles[4],
                        }
    if eccentricity is not None:
        waveform_kwargs['eccentricity'] = eccentricity
    if otherparams:
        waveform_kwargs['coa_phase'] = np_gen.uniform(0., 2*np.pi)
        # TODO: check if the inclination has to be in this range ??
        waveform_kwargs['inclination'] = np_gen.uniform(0., np.pi)
    if otherparams:
        waveform_kwargs['coa_phase'] = np_gen.uniform(0., 2*np.pi)
        # TODO: check if the inclination has to be in this range ??
        waveform_kwargs['inclination'] = np_gen.uniform(0., np.pi)
    logging.info(waveform_kwargs)
    hp, hc = pycbc.waveform.get_td_waveform(**waveform_kwargs)

    plt.plot(hp.sample_times, hp, label='hp')
    plt.plot(hc.sample_times, hc, label='hc')
    plt.show()

    hp = hp.trim_zeros()
    hc = hc.trim_zeros()

    # Truncate or pad the waveform to ensure it is exactly 1 second long
    # Merger time is always 80% of duration away from start for waveforms
    # whose original length is > 1 second.
    if len(hp) > SAMPLE_RATE * DURATION:
        extra['truncated'] = True
        extra['truncated_len'] = len(hp) - int(SAMPLE_RATE * DURATION)
        hp = hp[len(hp) - int(SAMPLE_RATE * DURATION):]
        hc = hc[len(hc) - int(SAMPLE_RATE * DURATION):]
    elif len(hp) < SAMPLE_RATE * DURATION:
        extra['padded'] = True
        extra['padded_at'] = len(hp)
        hp.append_zeros(int(SAMPLE_RATE * DURATION) - len(hp))
        hc.append_zeros(int(SAMPLE_RATE * DURATION) - len(hc))
    logging.debug(f'hp slice duration={hp.duration}')

    # Calculate the amplitude and phase from the polarizations.
    logging.debug('Converting `hp` & `hc` to Freq Amp!')
    amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
    phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc)
    freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)

    # Have correct input lengths!
    # It should be ensured that the merger is always within the data.
    if len(hp) > PRESET_ARRAY_SIZE:
        diff = len(hp) - PRESET_ARRAY_SIZE
        logging.debug(f'len(freq) > {PRESET_ARRAY_SIZE} by {diff} ele \
                            \n So truncating array from the left!')
        hp = hp[diff:]
    if len(hc) > PRESET_ARRAY_SIZE:
        diff = len(hc) - PRESET_ARRAY_SIZE
        logging.debug(f'len(amp) > {PRESET_ARRAY_SIZE} by {diff} ele \
                            \n So truncating array from the left!')
        hc = hc[diff:]
    # Have correct input lengths
    if len(amp) > PRESET_ARRAY_SIZE:
        diff = len(amp) - PRESET_ARRAY_SIZE
        logging.debug(f'len(amp) > {PRESET_ARRAY_SIZE} by {diff} ele \
                            \n So truncating array from the left!')
        amp = amp[diff:]
    logging.debug(f'amp.shape: {amp.shape}')
    logging.debug(f'amp.shape: {amp.shape}')
    if len(phase) > PRESET_ARRAY_SIZE:
        diff = len(phase) - PRESET_ARRAY_SIZE
        logging.debug(f'len(phase) > {PRESET_ARRAY_SIZE} by {diff} ele \
                            \n So truncating array from the left!')
        phase = phase[diff:]
    if len(freq) > PRESET_ARRAY_SIZE:
        diff = len(freq) - PRESET_ARRAY_SIZE
        logging.debug(f'len(freq) > {PRESET_ARRAY_SIZE} by {diff} ele \
                            \n So truncating array from the left!')
        freq = freq[diff:]

    hp = np.array(hp, dtype=np.float64)
    hc = np.array(hc, dtype=np.float64)
    amp = np.array(amp.data, dtype=np.float64)
    phase = np.array(phase.data, dtype=np.float64)
    freq = np.array(freq.data, dtype=np.float64)

    # # TODO: Check why rescaling the amp leads to HDF save error.
    # # Rescale the amp by 10^20
    # hp = hp * 10**20
    # hc = hc * 10**20
    # amp = amp * 10**20

    # append the extra info to the end of the data
    extra['truncated'] = extra.get('truncated', False)
    extra['padded'] = extra.get('padded', False)
    extra['truncated_len'] = extra.get('truncated_len', None)
    extra['padded_at'] = extra.get('padded_at', None)
    extra['eccentricity'] = waveform_kwargs.get('eccentricity', None)
    extra['coa_phase'] = waveform_kwargs.get('coa_phase', None)
    extra['inclination'] = waveform_kwargs.get('inclination', None)
    # append the extra info to the end of the data
    extra['truncated'] = extra.get('truncated', False)
    extra['padded'] = extra.get('padded', False)
    extra['truncated_len'] = extra.get('truncated_len', None)
    extra['padded_at'] = extra.get('padded_at', None)
    extra['eccentricity'] = waveform_kwargs.get('eccentricity', None)
    extra['coa_phase'] = waveform_kwargs.get('coa_phase', None)
    extra['inclination'] = waveform_kwargs.get('inclination', None)
    return [hp, hc, amp, phase, freq, extra]


def get_vals(m1, m2, approximant='SEOBNRv4', eccentricity=None,
            otherparams=False, dataset='raw', cutoffconst=None,
            calc_duration_mean=False):
    """
    Generate the time-domain waveform for the given masses and
    approximant. The waveform is generated with variable length (duration)
    and this is directly converted to Amp/Freq and saved.
    There are 3 different datasets that can be generated:
    - `raw`: The raw time-domain waveform is generated, converted to Amp/Freq
            and then saved, without appending zeros in hp/hc. The equal duration
            waveforms of 1 second lenght for this dataset are the ones generated
            in `get_vals_for_hdf`.
    - `f_cutoff`: The raw time-domain waveform is generated, and is made to be of
            the desired duration of 1 second, by changing the lower freq cutoff.
            Then it is converted to Amp/Freq and saved.
    - `f_sample`: The raw time-domain waveform is generated, and is made to be
            of the desired duration of 1 second, by changing the sample rate.
            Then it is converted to Amp/Freq and saved.
    """
    # Initialize random distributions.
    angles = np_gen.uniform(0., 2*np.pi, 3)

    extra = {}
    waveform_kwargs = {'approximant': approximant,
                        'mass1': m1,
                        'mass2': m2,
                        'f_lower': f_lower,
                        'delta_t': DELTA_T,
                        # 'right_ascension': angles[2],
                        # 'declination': angles[3],
                        # 'pol_angle': angles[4],
                        }
    if eccentricity is not None:
        waveform_kwargs['eccentricity'] = eccentricity
    if otherparams:
        waveform_kwargs['coa_phase'] = np_gen.uniform(0., 2*np.pi)
        # TODO: check if the inclination has to be in this range ??
        waveform_kwargs['inclination'] = np_gen.uniform(0., np.pi)
    logging.info(f"waveform_kwargs: {waveform_kwargs}")
    hp, hc = pycbc.waveform.get_td_waveform(**waveform_kwargs)

    # plt.plot(hp.sample_times, hp, label='hp')
    # plt.plot(hc.sample_times, hc, label='hc')
    # plt.show()

    hp = hp.trim_zeros()
    hc = hc.trim_zeros()

    if dataset=='f_cutoff' and hp.duration < DURATION:
        logging.info(f'f_low={f_lower}, duration={hp.duration}')

        # calculate new f_lower
        mchirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        new_fcutoff = ( DURATION / (cutoffconst * mchirp ** (-5/3)) )**(-3/8)
        extra['f_lower'] = new_fcutoff
        logging.info(f'New f_lower={new_fcutoff}')
        # adjust the new f_lower to allow for some error
        new_fcutoff -= 0.2*new_fcutoff

        # generate a second waveform
        waveform_kwargs['f_lower'] = new_fcutoff
        hp, hc = pycbc.waveform.get_td_waveform(**waveform_kwargs)
        hp = hp.trim_zeros()
        hc = hc.trim_zeros()
        extra['delta_t'] = hp.delta_t
        logging.info(f'New f_lower={new_fcutoff}, duration={hp.duration}')

        # plt.plot(hp.sample_times, hp, label='hp')
        # plt.plot(hc.sample_times, hc, label='hc')
        # plt.show()
        logging.info(f'sample_len={len(hp)}')
        if calc_duration_mean:
            return hp.duration

    if dataset=='f_sample':
        # DEPRECATED!

        # # Resample the waveform to the desired sample rate
        # logging.debug('Resampling the waveform to the desired sample rate')
        # sample_rate = len(hp) / 1.0
        # logging.info(f'Sample rate: {sample_rate}')
        # delta_t = 1 / sample_rate
        # hp = resample_to_delta_t(hp, delta_t)
        # hc = resample_to_delta_t(hc, delta_t)
        sample_len = len(hp)
        logging.info(f'Sample length: {sample_len}, duration: {hp.duration}')
        logging.info(f'Sample rate: {hp.sample_rate}, delta_t: {hp.delta_t}')
        new_sample_rate = 0.25 * hp.sample_rate
        new_delta_t = 1 / (hp.duration * SAMPLE_RATE)
        waveform_kwargs['delta_t'] = new_delta_t
        logging.info(f'New sample rate: {new_sample_rate}, new delta_t: {new_delta_t}')
        hp, hc = pycbc.waveform.get_td_waveform(**waveform_kwargs)
        logging.info(f'Sample rate: {hp.sample_rate}, duration: {hp.duration}')

        hp = hp.trim_zeros()
        hc = hc.trim_zeros()
        plt.plot(hp.sample_times, hp, label='hp')
        plt.plot(hc.sample_times, hc, label='hc')
        plt.show()

        hp = pycbc.filter.resample.resample_to_delta_t(hp, new_delta_t)
        hc = pycbc.filter.resample.resample_to_delta_t(hc, new_delta_t)
        logging.info(f'Sample rate: {hp.sample_rate}, duration: {hp.duration}')

    # Have correct input lengths!
    # It should be ensured that the merger is always within the data.
    if len(hp) > PRESET_ARRAY_SIZE:
        extra['truncated'] = True
        extra['truncated_len'] = diff = len(hp) - PRESET_ARRAY_SIZE
        logging.debug(f'len(freq) > {PRESET_ARRAY_SIZE} by {diff} ele \
                            \n So truncating array from the left!')
        hp = hp[diff:]
    if len(hc) > PRESET_ARRAY_SIZE:
        extra['truncated'] = True
        extra['truncated_len'] = diff = len(hc) - PRESET_ARRAY_SIZE
        logging.debug(f'len(amp) > {PRESET_ARRAY_SIZE} by {diff} ele \
                            \n So truncating array from the left!')
        hc = hc[diff:]

    # Calculate the amplitude and phase from the polarizations.
    logging.debug('Converting `hp` & `hc` to Freq Amp!')
    amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
    phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc)
    freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)

    hp = np.array(hp, dtype=np.float64)
    hc = np.array(hc, dtype=np.float64)
    amp = np.array(amp.data, dtype=np.float64)
    phase = np.array(phase.data, dtype=np.float64)
    freq = np.array(freq.data, dtype=np.float64)

    # append the extra info to the end of the data
    extra['truncated'] = extra.get('truncated', False)
    extra['padded'] = extra.get('padded', False)
    extra['truncated_len'] = extra.get('truncated_len', None)
    extra['padded_at'] = extra.get('padded_at', None)
    extra['eccentricity'] = waveform_kwargs.get('eccentricity', None)
    extra['coa_phase'] = waveform_kwargs.get('coa_phase', None)
    extra['inclination'] = waveform_kwargs.get('inclination', None)
    return [hp, hc, amp, phase, freq, extra]


def write_hdf_grp(hf, data, grpname):
    """
    Write the data to the HDF5 file.
    `data` is a list or dictionary of arrays.
        'hp', 'hc', 'amp', 'phase', 'freq'
    Either passed as a list or a dictionary.
    """
    if isinstance(data, np.ndarray):
        logging.info('Assuming data is the `hp` strain.')
        hf[grpname].create_dataset('hp', data=data)

    elif isinstance(data, list):
        dsnames = ['hp', 'hc', 'amp', 'phase', 'freq']
        logging.info(f'Assuming data array is in the form {dsnames}')
        # Do not write the extra info since it was already written!
        for name, tsdata in zip(dsnames, data[:-1]):
            logging.info(name, tsdata.shape)
            ds = hf[grpname].create_dataset(name, data=tsdata)
            # ds.attrs['sample_rate'] = tsdata.sample_rate
            # ds.attrs['delta_t'] = tsdata.delta_t
            # ds.attrs['duration'] = tsdata.duration
            # ds.attrs['merger_time'] = tsdata.merger_time

    elif isinstance(data, dict):
        logging.info(f'Assuming data array is in the form {data.keys()}')
        for name, d in data.items():
            if isinstance(d, np.ndarray):
                hf[grpname].create_dataset(name, data=d)
            else:
                raise ValueError(f"Data for {name} is not a numpy array.")
    else:
        raise ValueError("Data must be a numpy array or a list of arrays or \
                          a dictionary of arrays.")

def write_data_to_hdf(fname='SEOBNRv4', masses=None, approximant='SEOBNRv4',
                      otherparams=False, dataset=None):
    logging.info(f'Writing data to HDF5 file {fname}')
    if os.path.exists(fname+'.hdf'):
        logging.info(f'File {fname}.hdf already exists. Using an incremented name.')
    if os.path.exists(fname+'.hdf'):
        logging.info(f'File {fname}.hdf already exists. Using an incremented name.')
        fname = fname.split('.hdf')[0] + '-1.hdf'
    if dataset is not None:
        fname += f'-{dataset}'
    with h5py.File(fname+'.hdf', 'w') as hf:
        # Create a group for each mass
        for i, mass in tqdm(enumerate(masses),
                            total=len(masses),
                            desc='samples-written',
                            ncols=100,):
            m1, m2 = mass
            grpname = f'sample{i}'

            if approximant=='EccentricTD':
                ecc = np.random.choice(np.random.uniform(0.01, 0.25, 200), size=1)[0]
            else:
                ecc = None
            if dataset is None:
                data = get_vals_for_hdf(m1, m2, approximant, eccentricity=ecc,
                                        otherparams=otherparams)
            else:
                cutoffconst = calc_cutoffconst()
                data = get_vals(m1, m2, approximant=approximant,
                                eccentricity=ecc, otherparams=otherparams,
                                dataset=dataset, cutoffconst=cutoffconst)
            if dataset is None:
                data = get_vals_for_hdf(m1, m2, approximant, eccentricity=ecc,
                                        otherparams=otherparams)
            else:
                cutoffconst = calc_cutoffconst()
                data = get_vals(m1, m2, approximant=approximant,
                                eccentricity=ecc, otherparams=otherparams,
                                dataset=dataset, cutoffconst=cutoffconst)
            # plt.plot(range(len(data[0])), data[0], label=f'{m1} & {m2}')
            # plt.legend()
            # plt.show()
            # plt.plot(range(len(data[2])), data[2], label=f'{m1} & {m2}')
            # plt.legend()
            # plt.show()
            hfgrp = hf.create_group(grpname)
            hfgrp.attrs['mass1'] = m1
            hfgrp.attrs['mass2'] = m2
            hfgrp.attrs['approximant'] = approximant
            hfgrp.attrs['sample_rate'] = SAMPLE_RATE
            hfgrp.attrs['delta_t'] = data[-1].get('delta_t', DELTA_T)
            hfgrp.attrs['f_lower'] = data[-1].get('f_lower', f_lower)
            hfgrp.attrs['delta_t'] = data[-1].get('delta_t', DELTA_T)
            hfgrp.attrs['f_lower'] = data[-1].get('f_lower', f_lower)
            if approximant == 'EccentricTD':
                hfgrp.attrs['eccentricity'] = ecc

            if otherparams:
                hfgrp.attrs['coa_phase'] = data[-1]['coa_phase']
                hfgrp.attrs['inclination'] = data[-1]['inclination']

            # extra info like truncated or padded
            # save these for all samples, with `False` vals when no padding.
            # save these for all samples, with `False` vals when no padding.
            logging.info(f'extra: {data[-1]}')
            hfgrp.attrs['truncated'] = data[-1]['truncated']
            hfgrp.attrs['padded'] = data[-1]['padded']
            # This is the length of the original waveform
            # before padding.
            # This is useful to know how much padding was done.
            # If the waveform was truncated, this will not be present.
            # If the waveform was padded, this will be present.
            if data[-1]['truncated_len'] is not None:
                hfgrp.attrs['truncated_len'] = data[-1]['truncated_len']
            if data[-1]['padded_at'] is not None:
                hfgrp.attrs['padded_at'] = data[-1]['padded_at']
            write_hdf_grp(hf, data, grpname)
        logging.info(f"Data written to {fname+'.hdf'} successfully.")
        hf.close()

def check_hdf(fname, noshow=False):
    """
    Read the data from the HDF5 file.
    """
    with h5py.File(fname, 'r') as hf:
        keys = list(hf.keys())[:10]
        for key in keys:
            # print(key)
            grp = hf[key]
            print(dict(grp.attrs))
            for name in grp.keys():
                print(grp[name])
                print(name, grp[name].shape)
                # print(list(grp[name].attrs.keys()))
                # print(list(grp[name].attrs.keys()))
                # print(grp[name].__dict__)
                ts = grp[name]
                # print(np.array(ts))
                # print(ts[10:20])
                if not noshow or len(ts) < 8190:
                    plt.plot(range(len(ts)), np.array(ts), label=name)
                    plt.legend()
                    plt.savefig(f'checkhdf-{key}_{name}_plot.png', dpi=300)
                    plt.show()
    return hf



def _phase_from_frequency(freq, dt, theta0=0.0):
    """
    Compute gravitational-wave phase from a frequency time series.

    Parameters
    ----------
    freq : array_like
        Instantaneous frequency time series (Hz).
    dt : float
        Time step between samples (seconds).
    phi0 : float, optional
        Initial phase (radians). Default is 0.
        
    Returns
    -------
    phase : ndarray
        Phase time series (radians).
    """
    from scipy.integrate import cumulative_trapezoid
    # Integrate frequency using trapezoidal rule
    theta_integral = cumulative_trapezoid(freq, dx=dt, initial=0.0)
    print(f'freq shape: {freq.shape}, theta_integral shape: {theta_integral.shape}')
    # Multiply by 2π and add initial phase
    return theta0 + 2 * np.pi * theta_integral

def _compute_best_phase_alignment(h_orig, h_recon):
    """
    Compute the best phase alignment between the original and reconstructed waveforms.
    This is done by finding the phase shift that maximizes the cross-correlation
    between the two waveforms. The aligned waveform is then given by:

    :math:
        h_{recon} = h_{recon} \\cdot e^{i \\Delta\\phi}
        \\Delta\\phi = \\argmax_{ \\Sum_t h_{orig}(t) \\cdot h_{recon}^{*}(t) }
    """
    # # Compute the cross-correlation between h_orig and h_recon
    # correlation = np.correlate(h_orig, h_recon, mode='full')
    # # Find the index of the maximum correlation
    # max_index = np.argmax(correlation)
    # # Calculate the corresponding phase shift
    # phase_shift = 2 * np.pi * (max_index - len(h_orig) + 1) / len(h_orig)
    # # Apply the phase shift to the reconstructed waveform
    # h_recon_aligned = h_recon * np.exp(1j * phase_shift)
    # return h_recon_aligned

    # do h = hp + 1j*hc
    dphi = np.angle(np.sum(h_orig * np.conj(h_recon)))
    return h_recon * np.exp(1j * dphi)

def _phase_from_freq_intervals(freq, dt, theta0=0.0):
    """
    freq: length N-1, interpreted as interval frequency between samples.
    returns theta: length N

    NOTE: the returned phase has one extra element, while the output of my
    neural network will have the size of the trucated frequency series, that
    we obtain when we originally convert the polarization strains to amplitude
    and frequency.
    """
    dtheta = 2 * np.pi * freq * dt              # length N-1
    theta = np.empty(freq.size + 1, dtype=np.float64)
    theta[0] = theta0
    theta[1:] = theta0 + np.cumsum(dtheta)      # length N
    return theta

def _polarizations_from_ampfreq(amp, freq, theta0=0.0):
    """
    Convert amplitude and frequency to hplus and hcross polarizations.
    Phase array will have one element less than the amp, since the freq array
    is derived from the phase array by differentiation originally!
    Well, this certainly seems to be a mess now and it would definitely be
    better that I directly work with the phase and amplitude instead of the frequency.
    """
    print(f'amp shape: {amp.shape}, freq shape: {freq.shape}')
    # phase = _phase_from_frequency(freq, dt=1.0/SAMPLE_RATE)
    theta = _phase_from_freq_intervals(freq, dt=1.0/SAMPLE_RATE, theta0=theta0)
    print(f'amp shape: {amp.shape}, freq shape: {freq.shape}, phase shape: {theta.shape}')
    # amp = amp[1:] # to have equal sized arrays
    # print(f'amp shape after slicing: {amp.shape}')
    hplus = amp * np.cos(theta)
    hcross = amp * np.sin(theta)
    return hplus, hcross

def check_ampfreq(fname, noshow=False, usephase=False):
    """
    Calculate the mismatch using noise-weighted inner product between the
    reconstructed waveform obtained from the `amp` and `freq` 
    and the original waveform saved in the HDF5 file. Ideally,
    this difference should be near-zero, about 1e-14 ish.
    Use the `aLIGOZeroDetHighPower` PSD for the noise-weighting.

    This basically simply checks if the conversion from `hp` and `hc` to `amp` and `freq`
    and back is consistent and does not lead to any significant loss of information.
    """
    if usephase:
        logging.warning('Using the original phase saved in the HDF5 file for reconstruction.')
    datadir = '../data/'
    mismatchs = [[],[]] # for hp and hc respectively
    with h5py.File(datadir+fname+'.hdf', 'r') as hf:
        for i, key in enumerate(hf.keys()):
            grp = hf[key]
            print(dict(grp.attrs))
            hp = np.array(grp['hp'])
            hc = np.array(grp['hc'])
            amp = np.array(grp['amp'])
            phase_hdf = np.array(grp['phase'])
            freq = np.array(grp['freq'])
            delta_t = grp.attrs.get('delta_t')
            f_lower = grp.attrs.get('f_lower')

            # Use the `aLIGOZeroDetHighPower` PSD for the noise-weighting.
            # NOTE: All pycbc.types.TimeSeries objects have the same `delta_t` 
            # and also the `delta_f` which is 1/duration or 1/(len(hp)*delta_t).
            psd = pycbc.psd.aLIGOZeroDetHighPower(len(hp), 
                                                  delta_f=1/(len(hp)*delta_t), 
                                                  low_freq_cutoff=f_lower)
            print(f'PSD length: {len(psd)}, PSD delta_f: {psd.delta_f}')

            # Get the correct phase from the original polarization strains, since
            # the assholic `pycbc` function set the start phase to zero, without
            # any warnings.
            logging.info('Calculating the phase from the original polarizations using `arctan2` for better accuracy.')
            phase = np.unwrap(np.arctan2(hc, hp))
            print(f'Original start phase value from HDF5: {phase_hdf[0]}, correct start phase: {phase[0]}')
            
            # Reconstruct the waveform from the amp and freq
            # TODO: Use `arctan2(hcross[0], hplus[0])` as the initial phase for the reconstruction, 
            # since this is more accurate than using the original phase saved in the HDF5 file,
            # which is derived from the original waveform and thus may have some numerical errors.
            print(f'Reference phase: {phase[0]}')
            if usephase:
                logging.info('Using the original phase saved in the HDF5 file for reconstruction.')
                recon_hp = amp * np.cos(phase)
                recon_hc = amp * np.sin(phase)
            else:
                recon_hp, recon_hc = _polarizations_from_ampfreq(amp, freq, theta0=phase[0])

            # recon_hp, recon_hc = _compute_best_phase_alignment(hp, recon_hp), _compute_best_phase_alignment(hc, recon_hc)
            
            # Convert all waveforms to float64 numpy arrays for consistency in mismatch calculation
            recon_hp = np.asarray(recon_hp, dtype=np.float64)
            recon_hc = np.asarray(recon_hc, dtype=np.float64)
            hp = np.asarray(hp, dtype=np.float64)
            hc = np.asarray(hc, dtype=np.float64)
            psd = psd.astype(np.float64)

            # Resize hp to recon_hp length
            if len(hp) > len(recon_hp):
                hp = hp[1:]
                hc = hc[1:]
            
            # Convert the reconstructed waveforms to `pycbc` TimeSeries objects for mismatch calculation
            recon_hp = pycbc.types.TimeSeries(recon_hp, delta_t=delta_t)
            recon_hc = pycbc.types.TimeSeries(recon_hc, delta_t=delta_t)
            hp = pycbc.types.TimeSeries(hp, delta_t=delta_t)
            hc = pycbc.types.TimeSeries(hc, delta_t=delta_t)
            print(f'hp delta_f: {hp.delta_f}, recon_hp delta_f: {recon_hp.delta_f}, psd delta_f: {psd.delta_f}')

            # -- This still gives the same delta_f not matching error --#
            # Resample PSD at specific frequencies to match `delta_f` of the 
            # waveforms to the `delta_f` of the PSD. This is necessary for the mismatch calculation.
            hp_fs = hp.to_frequencyseries(delta_f=hp.delta_f)
            hc_fs = hc.to_frequencyseries(delta_f=hc.delta_f)
            recon_hp_fs = recon_hp.to_frequencyseries(delta_f=recon_hp.delta_f)
            recon_hc_fs = recon_hc.to_frequencyseries(delta_f=recon_hc.delta_f)
            freqs = hp_fs.sample_frequencies
            psd_interp = np.interp(freqs, psd.sample_frequencies, psd.data)
            psd_resampled = pycbc.types.FrequencySeries(psd_interp, delta_f=hp.delta_f, dtype=psd.dtype)

            # Calculate the mismatch using pycbc function
            # match_hp, i = pycbc.filter.optimized_match(hp, recon_hp, psd=psd_resampled, low_frequency_cutoff=f_lower)
            # match_hc, j = pycbc.filter.optimized_match(hc, recon_hc, psd=psd_resampled, low_frequency_cutoff=f_lower)
            match_hp, a = pycbc.filter.match(hp, recon_hp, psd=psd_resampled, low_frequency_cutoff=f_lower)
            match_hc, b = pycbc.filter.match(hc, recon_hc, psd=psd_resampled, low_frequency_cutoff=f_lower)
            mismatchs[0].append(1 - match_hp)
            mismatchs[1].append(1 - match_hc)
            print(f"Mismatch for sample {key}: {1 - match_hp}, {1 - match_hc}")

            # Plot the original and reconstructed waveforms for a few samples to visually check the reconstruction quality.
            if not noshow and i < 5:
                fig, axes = plt.subplots(2, 1, figsize=(10,5))
                axes[0].plot(range(len(hp)), hp, label='Original hp')
                axes[0].plot(range(len(recon_hp)), recon_hp, label='Recombined hp', linestyle='dashed')
                axes[0].set_xlabel('Time (s)')
                axes[0].set_ylabel('Strain')
                axes[0].legend()

                axes[1].plot(range(len(hc)), hc, label='Original hc')
                axes[1].plot(range(len(recon_hc)), recon_hc, label='Recombined hc', linestyle='dashed')
                axes[1].set_xlabel('Time (s)')
                axes[1].set_ylabel('Strain')
                axes[1].legend()

                plt.tight_layout()
                plt.savefig(f'checkampfreq-{fname}_{key}.png', dpi=300)
                plt.show()

    # Calculate the mean and std of the mismatchs
    mismatchs = np.array(mismatchs)
    mean_mismatch_hp = np.mean(mismatchs[0])
    std_mismatch_hp = np.std(mismatchs[0])
    mean_mismatch_hc = np.mean(mismatchs[1])
    std_mismatch_hc = np.std(mismatchs[1])
    print(f"Mean mismatch for hp: {mean_mismatch_hp}, std: {std_mismatch_hp}")
    print(f"Mean mismatch for hc: {mean_mismatch_hc}, std: {std_mismatch_hc}")

    # Save these mismatch values to a text file
    with open('checkampfreq-'+fname+f'_mismatch-{NOW}.txt', 'w') as f:
        f.write(f"Mean mismatch for hp: {mean_mismatch_hp}, std: {std_mismatch_hp}\n")
        f.write(f"Mean mismatch for hc: {mean_mismatch_hc}, std: {std_mismatch_hc}\n")
        for i, (mismatch_hp, mismatch_hc) in enumerate(zip(mismatchs[0], mismatchs[1])):
            f.write(f"Sample {i}: Mismatch for hp: {mismatch_hp}, Mismatch for hc: {mismatch_hc}\n")
    logging.info(f"Checked amp-freq reconstruction for {fname}.hdf successfully.")


def check_ampfreq_via_wavegen(noshow=False, usephase=False, f_lower=20.0):
    """
    This is a more direct check of the amp-freq reconstruction, where we
    directly generate a waveform using `pycbc.waveform.get_td_waveform`, 
    convert it to amp-freq and then back to hp-hc and check the mismatch between the 
    original and reconstructed waveforms.
    """
    # Generate a random waveform
    m1, m2 = np.random.uniform(5, 100, size=2)
    chi1z, chi2z = np.random.uniform(-0.99, 0.99, size=2)
    hp, hc = pycbc.waveform.get_td_waveform(approximant='SEOBNRv4',
                                            mass1=m1,
                                            mass2=m2,
                                            chi1z=chi1z,
                                            chi2z=chi2z,
                                            f_lower=f_lower,
                                            delta_t=1/(DURATION*SAMPLE_RATE))
    hp = hp.trim_zeros()
    hc = hc.trim_zeros()

    # Convert to amp-freq and back to hphc
    # NOTE: The dumbass pycbc code `phase_from_polarization`
    # func by default set the starting phase to 0, without even
    # informing the user a warning that this was being done.
    # Correctly it brings the mismatch value to 1e-16 !
    amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
    phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc, remove_start_phase=False)
    freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)
    # amp = np.sqrt(hp**2 + hc**2)
    # phase = np.unwrap(np.arctan2(hc, hp))
    # freq = np.gradient(phase) / (2 * np.pi * hp.delta_t)
    if usephase:
        logging.info('Using the original phase for reconstruction.')
        recon_hp = amp * np.cos(phase)
        recon_hc = amp * np.sin(phase)
    else:
        recon_hp, recon_hc = _polarizations_from_ampfreq(amp.data, freq.data, theta0=phase[0])
    print(f'Original hp shape: {hp.shape}, recon_hp shape: {recon_hp.shape}')

    # Convert the reconstructed waveforms to `pycbc` TimeSeries objects for mismatch calculation
    recon_hp = pycbc.types.TimeSeries(recon_hp, delta_t=hp.delta_t)
    recon_hc = pycbc.types.TimeSeries(recon_hc, delta_t=hc.delta_t)
    hp = pycbc.types.TimeSeries(hp, delta_t=hp.delta_t)
    hc = pycbc.types.TimeSeries(hc, delta_t=hc.delta_t)

    psd = pycbc.psd.aLIGOZeroDetHighPower(len(hp), delta_f=1/(len(hp)*hp.delta_t), low_freq_cutoff=f_lower)
    hp_fs = hp.to_frequencyseries(delta_f=hp.delta_f)
    hc_fs = hc.to_frequencyseries(delta_f=hc.delta_f)
    recon_hp_fs = recon_hp.to_frequencyseries(delta_f=recon_hp.delta_f)
    recon_hc_fs = recon_hc.to_frequencyseries(delta_f=recon_hc.delta_f)
    freqs = hp_fs.sample_frequencies
    psd_interp = np.interp(freqs, psd.sample_frequencies, psd.data)
    psd_resampled = pycbc.types.FrequencySeries(psd_interp, delta_f=hp.delta_f, dtype=psd.dtype)

    # Calculate the mismatch using pycbc function
    match_hp, i = pycbc.filter.match(hp, recon_hp, psd=psd_resampled, low_frequency_cutoff=f_lower)
    match_hc, j = pycbc.filter.match(hc, recon_hc, psd=psd_resampled, low_frequency_cutoff=f_lower)
    mismatch_hp = 1 - match_hp
    mismatch_hc = 1 - match_hc
    print(f"Mismatch for hp: {mismatch_hp}, Mismatch for hc: {mismatch_hc}")

    if not noshow:
        fig, axes = plt.subplots(2, 1, figsize=(10,5))
        axes[0].plot(range(len(hp)), hp, label='Original hp')
        axes[0].plot(range(len(recon_hp)), recon_hp, label='Recombined hp', linestyle='dashed')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Strain')
        axes[0].legend()
        axes[1].plot(range(len(hc)), hc, label='Original hc')
        axes[1].plot(range(len(recon_hc)), recon_hc, label='Recombined hc', linestyle='dashed')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Strain')
        axes[1].legend()
        plt.tight_layout()
        plt.show()
    logging.info(f"Checked amp-freq reconstruction via direct waveform generation successfully.")


# -- Save param values in CSV by reading pre-saved HDF files for training set
def save_params_from_hdf(hdf_fname, txt_fname='params'):
    """
    Read the parameters from the HDF5 file and save them to a text file in a readable format.
    These output parameters files will be used to calculate the mean and std of the parameters 
    in the training set, which will be used for normalization and also for calculating the 
    error metrics for the test set.
    """
    if '.hdf' not in hdf_fname:
        hdf_fname += '.hdf'
    if '.csv' not in txt_fname:
        txt_fname += '.csv'
    os.makedirs(os.path.dirname(txt_fname), exist_ok=True)
    with h5py.File(hdf_fname, 'r') as hf, open(txt_fname, 'w') as f:
        logging.info(f'Reading parameters from {hdf_fname} and saving to {txt_fname}')
        f.write("key,mass1,mass2,chi1z,chi2z\n")
        for key in hf.keys():
            grp = hf[key]
            m1 = grp.attrs['mass1']
            m2 = grp.attrs['mass2']
            chi1 = grp.attrs['spin1z']
            chi2 = grp.attrs['spin2z']
            f.write(f"{key},{m1},{m2},{chi1},{chi2}\n")
            logging.info(f'Wrote: {key},{m1},{m2},{chi1},{chi2}')
    logging.info(f"Saved parameters to {txt_fname} successfully.")


class CustomDataset(Dataset):
    """
    Custom Dataset to generate the data for training or testing
    models. This will use `masses` to choose the mass values and
    then use `get_strain` to first generate the `h_p` and `h_c`
    and then convert them to the frequency and amplitude array.

    Thus, there is no need to pass on the masses array or the labels
    etc., since these are internally calculated.

    TODO: For Phenom models instead directly import the original
    frequency and amplitude from lalsuite~

    Attributes
    ----------
    train_device : str
        The device where the training will take place.
    store_device : str
        The device where the data will be stored.
    inputsize : int
        The size of the input data.
    outputsize : int
        The size of the output data.
    forwhat : str
        Specifies the purpose of the dataset.
    masses : np.ndarray
        The mass values used to generate the data.
    nsamples : int
        The number of samples in the dataset.

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(idx)
        Returns the sample, label, and keys for the given index.
    set_masses(forwhat='train', labelsonly=False)
        Sets the mass values for the dataset based on the purpose.
    _get_data_old(n_samples=1000, masses=get_mass(), \
        approximant='IMRPhenomD', paramsonly=False)
        Generates data using the specified parameters (legacy method).
    """
    def __init__(self, store_device='cpu', train_device='mps', plot=False,
                 inputsize=2048, outputsize=1994, forwhat='train', approximant='IMRPhenomD',
                 convert=False, nokeys=False, hdf_fname=None, **kwargs):
        super().__init__()
        self.train_device = train_device
        self.store_device = store_device
        self.inputsize = inputsize
        self.outputsize = outputsize
        self.plot = plot
        self.convert = convert
        self.nokeys = nokeys
        self.hdf_fname = hdf_fname
        self.returnattr = kwargs.get('returnattr', False)
        self.unnorm_target = kwargs.get('unnorm_target', False)
        self.precision = kwargs.get('precision', 'float64')
        self.phase_target = kwargs.get('phase_target', False)

        self.forwhat = forwhat
        if hdf_fname is None:
            self.set_masses(forwhat=self.forwhat)
            self.nsamples = len(self.masses)
        else:
            self._find_nsamples()

        if approximant not in APPROXIMANTS:
            if approximant.split('-')[0][-7:]=='padinfo':
                # If the approximant is of the form `IMRPhenomDpadinfo-<something>.hdf`,
                # self.approximant = approximant.split('-')[0].removesuffix('padinfo')
                # logging.warning(f"Approximant {approximant} is a padded info \
                #                 approximant. Using {self.approximant} instead.")
                logging.info('Approximant is a padded info approximant.')
            else:
                raise ValueError(f"Invalid approximant: {approximant}. \
                                Choose from {APPROXIMANTS}")
            if approximant.split('-')[0][-7:]=='padinfo':
                # If the approximant is of the form `IMRPhenomDpadinfo-<something>.hdf`,
                # self.approximant = approximant.split('-')[0].removesuffix('padinfo')
                # logging.warning(f"Approximant {approximant} is a padded info \
                #                 approximant. Using {self.approximant} instead.")
                logging.info('Approximant is a padded info approximant.')
            else:
                raise ValueError(f"Invalid approximant: {approximant}. \
                                Choose from {APPROXIMANTS}")
        else:
            self.approximant = approximant
        # # Automatically initialize kwargs as attributes
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        
        
    def __len__(self):
        return self.nsamples
    
    def _find_nsamples(self):
        with h5py.File(self.hdf_fname+'.hdf', 'r') as hf:
            self.nsamples = len(hf.keys())
            logging.info(f'Set nsamples to {self.nsamples}')
        return self.nsamples
    
    def _find_nsamples(self):
        with h5py.File(self.hdf_fname+'.hdf', 'r') as hf:
            self.nsamples = len(hf.keys())
            logging.info(f'Set nsamples to {self.nsamples}')
    
    def _get_data_old(self, n_samples=1000, masses=get_mass(), approximant='IMRPhenomD', paramsonly=False):
        """
        TODO: Put this function inside `CustomDataset` class,
        If this is here, it doesn't make much sense using Datasets ~
        """
        inputsize, outputsize = self.inputsize, self.outputsize
        outputs = np.empty((n_samples,outputsize))
        inputs = np.empty((n_samples,inputsize+WFKW_SHAPE))
        for i in range(n_samples):
            m1, m2 = np.random.choice(masses.flatten()), np.random.choice(masses.flatten())
            bbhWaves, wfkw = get_strain(m1, m2, approximant)
            if len(bbhWaves)>outputsize:
                bbhWaves = bbhWaves[len(bbhWaves)-outputsize:]
            wfkw = [
                wfkw['mass1'],
                wfkw['mass2'],
                wfkw['delta_t'],
                wfkw['f_lower'],
                wfkw['coa_phase'],
                wfkw['inclination'],
                # wfkw['right_ascension']
                # + wfkw['declination'],
                # wfkw['pol_angle'],
            ]
            wfkw = np.array(wfkw)
            wfkw = (wfkw - np.var(wfkw)) / np.mean(wfkw)
            #wfkw = wfkw.reshape((1,wfkw.shape[0]))

            # nts_func = pycbc.noise.gaussian.frequency_noise_from_psd
            # nts = nts_func(psd)
            # nts = np.stack(nts, axis=0)
            # step input
            #nts = np.append(np.zeros(inputsize//2),np.ones(inputsize//2))
            # delta input
            if paramsonly:
                input = np.append(wfkw,np.zeros(inputsize))
            else:
                nts = np.zeros(inputsize)
                nts[np.random.choice(inputsize)] = 1.0
                input = np.append(wfkw, nts)
            inputs[i] = input
            # There may be numerical errors happening due to too low values ~
            outputs[i] = bbhWaves #* 10**19
        #inputs = inputs.reshape((1,inputs.shape[0],inputs.shape[1]))
        #outputs = outputs.reshape((1,outputs.shape[0],outputs.shape[1]))
        return inputs, outputs


    def set_masses(self, forwhat='train', labelsonly=False):
        """
        TODO: Make this more efficient ~

        Generate data using set number of mass values,
        to align with the papers and prev studies.

        Parameters
        ----------
            forwhat : str, optional
                Generate data for what. Default is 'train'.
                Choices are 'train', 'vald', 'test'.
            labelsonly : bool, optional
                Generate only the labels. Default is False.
        """
        ttsplits = get_mass(splitTT=True, plot=False)
        if forwhat=='vald' or forwhat=='valid' or forwhat=='val':
            masses = ttsplits[1]
        elif forwhat=='test':
            masses = ttsplits[2]
        else:
            # default is for `train` data
            masses = ttsplits[0]
        self.n_samples = len(masses)
        self.masses = masses

    def make_strain(self, idx, custom_batch=None):
        if custom_batch is not None:
            m1, m2 = custom_batch[idx]
        else:
            m1, m2 = self.masses[idx]
    def make_strain(self, idx, custom_batch=None):
        if custom_batch is not None:
            m1, m2 = custom_batch[idx]
        else:
            m1, m2 = self.masses[idx]
        strains, labels, keys = get_strain(m1, m2, approximant=self.approximant,
                                    convert=self.convert, nokeys=self.nokeys, plot=self.plot)
        # strains, labels, keys = get_fd_strain(m1, m2, plot=True)
        logging.debug(f"Strains shape: {strains.shape}")
        logging.debug(f"Type of strains: {type(strains)}, Type of strains[0]: {type(strains[0])}, Type of strains[0][0]: {type(strains[0][0])}")
        # use `torch.float64` dtype b'cuz mps only supports that!
        sample = torch.from_numpy(strains).to(device=self.train_device, dtype=getattr(torch, self.precision))
        label = torch.from_numpy(labels).to(device=self.train_device, dtype=getattr(torch, self.precision))
        keys = torch.from_numpy(keys).to(device=self.train_device, dtype=getattr(torch, self.precision))
        return (sample, label, keys)
    
    # TODO: This function should not be necessary, if I have the correct length
    # of waveforms are saved into a new HDF5 file.
    def _regenerate_sample(self, data, write_access=False):
        """
        Regenerate the sample with a lower fcutoff to ensure
        that the waveform is of the desired length.
        And replace the sample content in the HDF file.
        """
        extra = {}
        f_lower = 0.75 * data.attrs['f_lower']
        logging.info(f'new f_lower: {f_lower}')
        wfkwargs = {'approximant': data.attrs['approximant'],
                    'mass1': data.attrs['mass1'],
                    'mass2': data.attrs['mass2'],
                    'spin1z': data.attrs['spin1z'],
                    'spin2z': data.attrs['spin2z'],
                    'delta_t': data.attrs['delta_t'],
                    'f_lower': f_lower,
                    }
        hp, hc = pycbc.waveform.get_td_waveform(**wfkwargs)
        hp = hp.trim_zeros()
        hc = hc.trim_zeros()
        while len(hc) < PRESET_ARRAY_SIZE:
            f_lower -= 0.2 * f_lower
            wfkwargs['f_lower'] = f_lower
            hp, hc = pycbc.waveform.get_td_waveform(**wfkwargs)
            hp = hp.trim_zeros()
            hc = hc.trim_zeros()
        logging.info(f'new sample duration: {hp.duration}')

        if len(hc) > PRESET_ARRAY_SIZE:
            extra['truncated'] = True
            extra['truncated_len'] = diff = len(hc) - PRESET_ARRAY_SIZE
            logging.debug(f'len(amp) > {PRESET_ARRAY_SIZE} by {diff} ele \
                                \n So truncating array from the left!')
            hc = hc[diff:]
            hp = hp[diff:]

        amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
        freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)
        phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc, remove_start_phase=False)

        hp = np.array(hp, dtype=getattr(np, self.precision))
        hc = np.array(hc, dtype=getattr(np, self.precision))
        amp = np.array(amp.data, dtype=getattr(np, self.precision))
        phase = np.array(phase.data, dtype=getattr(np, self.precision))
        freq = np.array(freq.data, dtype=getattr(np, self.precision))

        if write_access:
            data.attrs['f_lower'] = wfkwargs['f_lower']
            data.attrs['truncated'] = extra.get('truncated', False)
            data.attrs['truncated_len'] = extra.get('truncated_len', None)
        else:
            data = {}
        # Replace the content of the waveform with the new values
        # Overwrite the datasets in the HDF5 group if write_access is True
        if write_access:
            for key, arr in zip(['hp', 'hc', 'amp', 'freq', 'phase'], [hp, hc, amp, freq, phase]):
                if key in data:
                    # Backup the old data before deletion
                    data[f"{key}_backup"] = data[key][:]
                    del data[key]
                data.create_dataset(key, data=arr)
            logging.info(f'Sample regenerated and updated in the HDF5 file successfully.')
        else:
            data['hp'] = hp
            data['hc'] = hc
            data['amp'] = amp
            data['freq'] = freq
            data['phase'] = phase
        return data

    def read_strain_hdf(self, idx, write_access=False):
        """
        Read the strain data from the HDF5 file.

        Parameters
        ----------
        idx : int
            The index of the sample to read.

        Returns
        -------
        tuple
            A tuple containing:
            - strain: np.ndarray
                The strain data as a 2D numpy array with shape (2, N).
            - labels: np.ndarray
                The labels as a 1D numpy array with shape (2,).
            - keys: np.ndarray
                The keys as a 2D numpy array with shape (2, 2).
        """
        logging.debug(f'Reading strain data from HDF5 file {self.hdf_fname}.hdf for sample {idx}')
        if write_access:
            open_mode = 'r+'
        else:
            open_mode = 'r'
        with h5py.File(self.hdf_fname+'.hdf', open_mode) as hf:
            data = hf[f'sample{idx}']
            logging.debug(f'keys: {data.keys()}')

            logging.debug(f'keys: {data.keys()}')

            m1, m2 = data.attrs['mass1'], data.attrs['mass2']
            labels = [m1,m2]
            spin1z = data.attrs.get('spin1z', None)
            spin2z = data.attrs.get('spin2z', None)
            if spin1z is not None and spin2z is not None:
                labels.append(spin1z)
                labels.append(spin2z)
            # logging.debug(f'Labels: {labels}')

            if not "regen" in self.hdf_fname:
                # Regenerate sample if it is shorter duration, but is not padded!
                # NOTE: Once this is checked, since I also write the data into the
                # HDF file, the next epoch onwards, this check will not be necessary, 
                # since the data will already have been regenerated. However, if the
                # HDF file is not closed properly after writing, the changes may not be saved, 
                # so this check will still be necessary for the next epoch, until the file is
                # properly closed and the changes are saved.
                if len(data['amp']) < PRESET_ARRAY_SIZE and not data.attrs.get('padded', False):
                    logging.info(f"\nSample {idx} is shorter than {PRESET_ARRAY_SIZE} and not padded. Regenerating!")
                    data = self._regenerate_sample(data, write_access=write_access)

            hp, hc = np.array(data['hp']), np.array(data['hc'])
            amp, freq = np.array(data['amp']), np.array(data['freq'])
            phase = np.array(data['phase'])
            logging.debug(f'Phase shape: {phase.shape}')

            # # freq array will be one less in length than amp
            # logging.debug(f'len(amp)={len(amp)}, len(freq)={len(freq)}')
            # if len(freq) < len(amp):
            #     amp = amp[1:]
            # assert len(amp) == len(freq), "Amplitude and Frequency arrays must be of the same length."

            # # check length for the phase
            # if len(phase) > len(freq):
            #     phase = phase[1:]
            # # -- so these are now of length 8191!
            # assert len(phase) == len(freq) == len(amp)

            # -- By definition, freq array will be one element less,
            # -- So, add a dummy value (repeated first element) to the 
            # beginning of the freq array to make it of the same length 
            # as amp and phase. Later on, this value will be removed when
            # calculating the mismatch later on during testing.
            freq = np.insert(freq, 0, freq[0])
            assert len(amp) == len(freq) == len(phase), "Amplitude, Frequency, and Phase arrays must be of the same length after adjustment."
            logging.debug(f'Adjusted len(amp)={len(amp)}, len(freq)={len(freq)}, len(phase)={len(phase)}')

            # Rescale the amp by 10^20
            logging.debug(f'Original Amp: {amp}')
            logging.debug(f'Type of Amp: {type(amp)}, Type of Amp[0]: {type(amp[0])}')
            amp = amp * 10**20
            logging.debug(f'Rescaled Amp: {amp}')
            logging.debug(f'Type of Rescaled Amp: {type(amp)}, Type of Rescaled Amp[0]: {type(amp[0])}')

            amp_keys = [np.mean(amp), np.std(amp)]
            freq_keys = [np.mean(freq), np.std(freq)]
            logging.debug(f"Amplitude Keys: {amp_keys}")
            logging.debug(f"Frequency Keys: {freq_keys}")
            unnorm_amp, unnorm_freq = amp.copy(), freq.copy()
            unnorm_amp, unnorm_freq = amp.copy(), freq.copy()
            amp = (amp - np.mean(amp)) / np.std(amp)
            freq = (freq - np.mean(freq)) / np.std(freq)

            out_normed = np.vstack((amp, freq)).astype(getattr(np, self.precision))
            out_unnormed = np.vstack((unnorm_amp, unnorm_freq)).astype(getattr(np, self.precision))
            out_labels = np.array(labels).astype(getattr(np, self.precision))
            out_keys = np.array([amp_keys, freq_keys]).astype(getattr(np, self.precision))
            out_phase = np.array(phase).astype(getattr(np, self.precision))
            out_strains = np.vstack((hp, hc)).astype(getattr(np, self.precision))
            out_attr = data.attrs if type(data) is not dict else data.get('attrs', {})
            out_attr = dict(out_attr)  # Convert HDF5 attributes to a regular dictionary for easier handling
                        
            if self.returnattr:
                if self.forwhat=='test':
                    return (out_normed,
                            out_labels,
                            out_keys,
                            out_phase,
                            out_strains,
                            out_attr)
                else:
                    logging.debug(f"Attributes: {out_attr}")
                    # also return the loc of padding or truncation
                    return (out_normed, 
                            out_normed, # -- target are normed amp & freq.
                            out_labels, 
                            out_keys,
                            out_strains,
                            out_attr)
            if self.unnorm_target:
                return (out_normed, 
                        out_unnormed, 
                        out_labels, 
                        out_keys,
                        out_strains)
            if self.phase_target:
                logging.debug("Returning normalized amp and freq as input, and phase as target since `phase_target` is True.")
                out_amp_phase = np.vstack((amp, phase)).astype(getattr(np, self.precision))
                return (out_amp_phase, 
                        out_amp_phase, # -- target is phase.
                        out_labels, 
                        out_keys,
                        out_strains)
            logging.debug("Returning normalized amp and freq as both input and target since `unnorm_target` is False.")
            return (out_normed, 
                    out_normed, # -- target are normed amp & freq.
                    out_labels, 
                    out_keys,
                    out_strains)
        
    def collate_fn(self, batch):
        """ 
        Custom collate function to handle the batch data.
        Because `KeyError` arose when 'padded' is not found in the `data.attr`
        for some samples. The feature_batch and tag_batch should have the same
        length, so we can use the default collate function for both.

        One batch consists of ([amp, freq], labels, [amp_keys, freq_keys]),
        where `labels` is [m1, m2] or [m1, m2, spin1z, spin2z] depending on
        the type of data used.
        """
        tag_batches = []
        feat_dict_batch = {}
        logging.debug(f'Batch size: {len(batch)}')

        # Determine the maximum number of tags in the batch
        if self.forwhat=='test' or self.returnattr:
            max_tags = max(len(sample) - 1 for sample in batch)  # Exclude the feature dict
        else:
            max_tags = max(len(sample) for sample in batch)

        # Initialize lists for each tag dynamically
        for _ in range(max_tags):
            tag_batches.append([])

        for sample in batch:

            # Append the feature dict to the batch dict
            if self.forwhat=='test' or self.returnattr:
                *tags, feat_dict = sample
                logging.debug(f'Number of tags: {len(tags)}')
                for key, value in feat_dict.items():
                    if key not in feat_dict_batch:
                        feat_dict_batch[key] = []
                    feat_dict_batch[key].append(value)
            else:
                tags = sample

            # Append tags to their respective lists
            for i, tag in enumerate(tags):
                tag = torch.tensor(tag, device=self.train_device, dtype=getattr(torch, self.precision))
                tag_batches[i].append(tag)

        # Convert lists of tags to tensors
        for i in range(len(tag_batches)):
            tag_batches[i] = torch.stack(tag_batches[i]).to(device=self.train_device, dtype=getattr(torch, self.precision))

        # Ensure all tensors are of the same shape
        if self.forwhat=='test' or self.returnattr:
            return (*tag_batches, feat_dict_batch)
        return tag_batches

    def __getitem__(self, idx, custom_batch=None):
        # logging.debug(idx)
        if idx>self.nsamples:
            raise IndexError('Index out of range')
        if self.hdf_fname is not None:
            if "regen" in self.hdf_fname:
                logging.debug("No need to have write-access, since waveforms in HDF file are already regenerated.")
                return self.read_strain_hdf(idx, write_access=False)
            return self.read_strain_hdf(idx, write_access=True)
        else:
            return self.make_strain(idx, custom_batch=custom_batch)
        

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=pin_memory,
                         collate_fn=dataset.collate_fn)
    

def example_input_plot():
    inputs, outputs = get_data(n_samples=3)
    logging.debug(inputs.shape,outputs.shape)
    fig, ax = plt.subplots(1,1,figsize=(7,3))
    for i in range(len(outputs)):
        #ax[0].plot(np.arange(len(inputs[i])),inputs[i])
        xarr = np.linspace(0,1,len(outputs[i]))
        ismax = np.argmax(outputs[i])
        # should do max of |outputs[i]|
        xarr = xarr - xarr[ismax]
        ax.plot(xarr,outputs[i])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('$h_{\\plus}$ ($\\times 10^{19}$)')
    putils.beautifyPlot([ax],grid=True)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    savename = 'gw-waveform-hplus-example-'+timestamp+'.png'
    plt.savefig(savename, dpi=300)
    #plt.show()

    fig, ax1 = plt.subplots(1,1,figsize=(2,2))
    impulse = inputs[len(inputs)//2][6:]
    # logging.debug(impulse)
    # logging.debug(impulse)
    ax1.plot(np.arange(len(impulse)),impulse,)
    #plotAnal.beautifyPlot([ax1],xTicks=False,yTicks=False)
    ax1.set_axis_off()
    plt.savefig('example-impulse-response.png', dpi=300)
    plt.show()


def example (m1=10, m2=10, approxs: list = []):
    approxs = ['IMRPhenomB', 'SEOBNRv4', 'TaylorT4']
    # approxs.append('IMRPhenomD')
    # approxs.append('IMRPhenomXPHM')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for apx in approxs:
        hp, hc = get_td_waveform(approximant=apx, mass1=m1, mass2=m2,
                                 delta_t=DELTA_T, f_lower=f_lower)
        hp, hc = hp.trim_zeros(), hc.trim_zeros()
        amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
        phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc)
        freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)
        # print(f'Length of {apx} hp: {len(hp)}, hc: {len(hc)}')
        # print(f'Length of {apx} amp: {len(amp)}, phase: {len(phase)}, freq: {len(freq)}')
        axes[0].plot(phase, amp, label=apx)
        axes[1].plot(freq.sample_times, freq, label=apx)
    axes[0].set_xlabel('Phase')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[1].set_xlabel('Sample Times')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    plt.suptitle(f'm1={m1} & m2={m2}'+' $M_{\\odot}$')
    plt.tight_layout()
    plt.show()
    plt.close()


def example2 (appoximant='SEONRv4', nsamples=10, qlim=5, m1end=75):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    masses = get_mass(qlim=5, m1end=m1end)[0]
    for i in range(nsamples):
        m1 = np.random.choice(masses[:,0])
        m2 = np.random.choice(masses[:,1])
        logging.debug(f"Masses: {m1}, {m2}")
        hp, hc = get_td_waveform(approximant=appoximant, mass1=m1, mass2=m2,
                                 delta_t=DELTA_T, f_lower=f_lower)
        hp, hc = hp.trim_zeros(), hc.trim_zeros()
        amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
        phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc)
        freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)
        logging.debug(f'Length of {appoximant} hp: {len(hp)}, hc: {len(hc)}')
        logging.debug(f'Length of {appoximant} amp: {len(amp)}, phase: {len(phase)}, freq: {len(freq)}')
        axes[0].plot(phase, amp, label=f'{m1} & {m2}'+'$M_{\\odot}$')
        axes[1].plot(freq.sample_times, freq, label=f'{m1} & {m2}'+'$M_{\\odot}$')
        axes[2].plot(range(len(amp)), amp, label=f'{m1} & {m2}'+'$M_{\\odot}$')
    axes[0].set_xlabel('Phase')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[1].set_xlabel('Sample Times')
    axes[1].set_ylabel('Frequency')
    axes[2].set_xlabel('Sample Count')
    axes[2].set_ylabel('Amplitude')
    # axes[1].legend()
    plt.suptitle(f'{appoximant} qlim={qlim} m1end={m1end}')
    plt.tight_layout()
    fname = f'freqamp-plot-{appoximant}-qlim{qlim}-m1end{m1end}-nsamples{nsamples}'
    plt.savefig(fname+'.png', dpi=300)
    plt.show()
    plt.close()

def example3 (approximant='SEONRv4', ecc=True, transparent=True):
    fig, axes = plt.subplots(1, 2, figsize=(5, 2))
    # masses = get_mass(qlim=5, m1end=m1end)[0]
    masses = np.array([[15, 50],[5, 30]])  # [5,10]])
    masses = np.array([[15, 50],[5, 30]])  # [5,10]])
    for i in range(len(masses)):
        m1 = masses[i,0]
        m2 = masses[i,1]
        logging.info(f"Masses: {m1}, {m2}")
        waveform_kwargs = {'delta_t':DELTA_T, 
                           'f_lower':f_lower}
        if ecc:
            waveform_kwargs['eccentricity'] = 0.2
        hp, hc = get_td_waveform(approximant=approximant, mass1=m1, mass2=m2,
                                 **waveform_kwargs)
        logging.info('Strain polarizations generated!')
        hp, hc = hp.trim_zeros(), hc.trim_zeros()
        amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
        phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc)
        freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)
        logging.debug(f'Length of {approximant} hp: {len(hp)}, hc: {len(hc)}')
        logging.debug(f'Length of {approximant} amp: {len(amp)}, phase: {len(phase)}, freq: {len(freq)}')
        axes[0].plot(amp.sample_times, amp, label=f'{m1} & {m2}'+'$M_{\\odot}$')
        axes[1].plot(freq.sample_times, freq, label=f'{m1} & {m2}'+'$M_{\\odot}$')
    axes[0].set_xlabel('Sample Times')
    axes[0].set_ylabel('Amplitude')
    # axes[0].legend(loc='lower center', bbox_to_anchor=(1, 1), ncol=3)
    # axes[0].legend(loc='best', fontsize=8)
    axes[0].text(0.075, 0.80, f'{approximant}', fontsize=10,
                 ha='left', transform=axes[0].transAxes, fontweight='bold')
    axes[1].set_xlabel('Sample Times')
    axes[1].set_ylabel('Frequency')
    axes[1].legend(loc='best', fontsize=8)
    # plt.tight_layout(rect=[0, 0, 1.0, 0.75])
    plt.tight_layout()
    fname = f'freqamp-plot-{approximant}'
    # fname += '-jsps'
    # plt.savefig(fname+'.png', dpi=300, bbox_inches='tight')
    if transparent:
        fname += '-transparent'
        plt.savefig(fname+'.png', dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.savefig(fname+'.png', dpi=300, bbox_inches='tight')
    if transparent:
        fname += '-transparent'
        plt.savefig(fname+'.png', dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.savefig(fname+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def example4(approximant='SEOBNRv4', ecc=True, transparent=True):
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    masses = np.array([[15, 50],[5, 30]]) # [5,10]])
    masses = np.array([[15, 50],[5, 30]]) # [5,10]])
    for i in range(len(masses)):
        m1 = masses[i, 0]
        m2 = masses[i, 1]
        logging.info(f"Masses: {m1}, {m2}")
        waveform_kwargs = {'delta_t':DELTA_T, 
                           'f_lower':f_lower}
        if ecc:
            waveform_kwargs['eccentricity'] = 0.2
        hp, hc = get_td_waveform(approximant=approximant, mass1=m1, mass2=m2,
                                 **waveform_kwargs)
        logging.info('Strain polarizations generated!')
        hp, hc = hp.trim_zeros(), hc.trim_zeros()
        ax.plot(hp.sample_times, hp, label=f'{m1} & {m2}' + '$M_{\\odot}$')
    ax.set_xlabel('Sample Times')
    ax.set_ylabel('$h_{\\plus}$ Strain')
    ax.text(0.075, 0.80, f'{approximant}', fontsize=10,
                    ha='left', transform=ax.transAxes, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    fname = f'strain-plot-{approximant}'
    # fname += '-jsps'
    if transparent:
        fname += '-transparent'
        plt.savefig(fname + '.png', dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.savefig(fname + '.png', dpi=300, bbox_inches='tight')
    if transparent:
        fname += '-transparent'
        plt.savefig(fname + '.png', dpi=300, bbox_inches='tight', transparent=True)
    else:
        plt.savefig(fname + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def example4b(approximant='EccentricTD', m1=15, m2=10, e=0.2):
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    f_lowers = [10, 15, 20, 25, 30, 40]
    for f_lower in f_lowers:
        print(f'f_lower = {f_lower}')
        waveform_kwargs = {'delta_t':DELTA_T, 
                           'f_lower':f_lower,
                           'eccentricity':0.2}
        hp, hc = get_td_waveform(approximant=approximant, mass1=m1, mass2=m2,
                                **waveform_kwargs)
        logging.info('Strain polarizations generated!')
        print(f'duration={hp.duration}')
        # hp, hc = hp.trim_zeros(), hc.trim_zeros()
        ax.plot(hp.sample_times, hp, label='$f_{min}$='+str(f_lower))
    ax.set_xlabel('Sample Times')
    ax.set_ylabel('$h_{\\plus}$ Strain')
    ax.set_title(f'm1={m1} and m2={m2}')
    ax.text(0.075, 0.80, f'{approximant}', fontsize=10,
                    ha='left', transform=ax.transAxes, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=3)
    plt.tight_layout()
    fname = f'strain-plot-{approximant}-fmin-{m1}-{m2}'
    # fname += '-jsps'
    plt.savefig(fname + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def flower_duration_plot(approximant='EccentricTD', noshow=False):
    """
    Plot the duration of the waveform as a function of the f_lower value,
    with contours for different mass pairs and eccentricities.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    masses = np.array([np.arange(5, 50, 5), np.arange(5, 50, 5)]).T
    f_lowers = np.arange(10, 50, 5)
    xloc, yloc, ha = 0.05, 0.05, 'left'
    for m1, m2 in masses:
        durations = []
        for f_lower in f_lowers:
            logging.info(f'm1={m1}, m2={m2}, f_lower={f_lower}')
            label = f'm1={m1}, m2={m2}'
            waveform_kwargs = {'delta_t':DELTA_T,
                            'f_lower':f_lower,}
            if approximant == 'EccentricTD':
                ecc = np.random.uniform(0.0001, 0.25, 1)[0]
                waveform_kwargs['eccentricity'] = ecc
                label += f', e={ecc:.2f}'
                xloc, yloc, ha = 0.95, 0.85, 'right'
            hp, hc = get_td_waveform(approximant=approximant, mass1=m1, mass2=m2,
                                    **waveform_kwargs)
            durations.append(hp.duration)
            logging.info(f'Duration: {hp.duration}')
            # hp, hc = hp.trim_zeros(), hc.trim_zeros()
        ax.plot(durations, f_lowers, '-', label=label)
        # # Add text annotation for the last point
        # ax.text(durations[-1], f_lowers[-1], f"m1={m1}, m2={m2}, e={ecc:.2f}", 
        #         fontsize=8, va='bottom', ha='right')
        # Place rotated annotation in the middle of each curve
        # mid_idx = len(durations) // 2
        # ax.text(durations[mid_idx], f_lowers[mid_idx], f"m1={m1}, m2={m2}, e={ecc:.2f}",
        #     fontsize=6, va='top', ha='left', rotation=150)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=7)
    ax.set_xlabel('Duration (s)', fontsize=12)
    ax.set_ylabel('$f_{min}$ (Hz)', fontsize=12)
    # ax.set_title(f'{approximant}')
    for axis in ['x', 'y']:
        ax.tick_params(axis=axis, which='both', length=5, width=1.0)
    ax.text(xloc, yloc, f'{approximant}', fontsize=15,
            ha=ha, transform=ax.transAxes, fontweight='bold')
    plt.tight_layout()
    fname = f'flower-duration-plot-{approximant}'
    plt.savefig(fname + '.png', dpi=300, bbox_inches='tight')
    if not noshow:
        plt.show()
    plt.close()


def srate_duration_plot(approximant='SEOBNRv4'):
    """
    Plot the duration of the waveform as a function of the sample rate,
    with contours for different mass pairs and eccentricities.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    masses = np.array([np.arange(5, 50, 5), np.arange(5, 50, 5)]).T
    sample_rates = np.array([4196, 8192])  # 1024, 2048, 4196, 8192, 16384, 32768
    deltaTs = 1.0 / sample_rates
    xloc, yloc, ha = 0.05, 0.05, 'left'
    f_lower = 20.0  # fixed f_lower for this plot
    for m1, m2 in masses:
        durations = []
        for delta_t in deltaTs:
            logging.info(f'm1={m1}, m2={m2}, f_lower={f_lower}')
            label = f'm1={m1}, m2={m2}'
            waveform_kwargs = {'delta_t':delta_t,
                            'f_lower':f_lower,}
            if approximant == 'EccentricTD':
                ecc = np.random.uniform(0.0001, 0.25, 1)[0]
                waveform_kwargs['eccentricity'] = ecc
                label += f', e={ecc:.2f}'
                xloc, yloc, ha = 0.95, 0.85, 'right'
            hp, hc = get_td_waveform(approximant=approximant, mass1=m1, mass2=m2,
                                    **waveform_kwargs)
            durations.append(hp.duration)
            logging.info(f'Duration: {hp.duration}')
            # hp, hc = hp.trim_zeros(), hc.trim_zeros()
        ax.plot(durations, sample_rates, '-', label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=7)
    ax.set_xlabel('Duration (s)', fontsize=12)
    ax.set_ylabel('Sample Rate (Hz)', fontsize=12)
    # ax.set_title(f'{approximant}')
    for axis in ['x', 'y']:
        ax.tick_params(axis=axis, which='both', length=5, width=1.0)
    ax.text(xloc, yloc, f'{approximant}', fontsize=15,
            ha=ha, transform=ax.transAxes, fontweight='bold')
    plt.tight_layout()
    fname = f'srate-duration-plot-{approximant}'
    plt.savefig(fname + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def m1_duration_vs_srate_plot(approximant='SEOBNRv4', m2=10):
    """
    Plot the duration of the waveform as a function of m1,
    for different sample rates, keeping m2 constant.
    Also show example time-series for each sample rate.
    Now, show at least 4 different time-series examples in the second subplot.
    """
    fig = plt.figure(figsize=(16, 8))
    # Main duration plot
    ax = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    # 4 subplots for time-series
    ax_ts_list = [
        plt.subplot2grid((2, 3), (1, i)) for i in range(3)
    ]
    # # Add a 4th subplot if needed
    # ax_ts_list.append(plt.subplot2grid((2, 3), (1, 2)))

    m1s = np.arange(5, 50, 5)
    sample_rates = np.array([1024, 2048, 4096, 8192, 16384])
    deltaTs = 1.0 / sample_rates
    f_lower = 20.0  # fixed f_lower for this plot

    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_rates)))

    ts_examples = []

    for idx, (delta_t, srate) in enumerate(zip(deltaTs, sample_rates)):
        durations = []
        logging.info
        for m1 in m1s:
            waveform_kwargs = {'delta_t': delta_t, 'f_lower': f_lower}
            if approximant == 'EccentricTD' or approximant == 'SEOBNRE':
                ecc = np.random.uniform(0.0001, 0.25, 1)[0]
                waveform_kwargs['eccentricity'] = ecc
            try:
                hp, hc = get_td_waveform(approximant=approximant, mass1=m1, mass2=m2, **waveform_kwargs)
                durations.append(hp.duration)
                # Save a time-series example for the first 4 sample rates at the middle m1
                if m1 == m1s[len(m1s)//2] and idx < 4:
                    ts_examples.append((idx, hp.sample_times, hp, srate))
            except Exception as e:
                durations.append(np.nan)
        ax.plot(m1s, durations, '-o', label=f'{srate} Hz', color=colors[idx])

    ax.set_xlabel('$m_1$ ($M_{\\odot}$)', fontsize=12)
    ax.set_ylabel('Duration (s)', fontsize=12)
    ax.set_title(f'Duration vs $m_1$ for $m_2={m2}$, {approximant}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_xscale('log')

    # Plot up to 4 time-series examples in the subplots
    for i, (idx, t, h, srate) in enumerate(ts_examples[1:4]):
        ax_ts = ax_ts_list[i]
        ax_ts.plot(t, h, label=f'{srate} Hz', color=colors[idx], alpha=0.8)
        ax_ts.set_xlabel('Time (s)', fontsize=10)
        ax_ts.set_ylabel('$h_{\\plus}$ Strain', fontsize=10)
        ax_ts.set_title(f'$f_s$={srate} Hz, $m_1$={m1s[len(m1s)//2]}, $m_2$={m2}')
        ax_ts.legend(loc='best', fontsize=8)
        ax_ts.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fname = f'm1-duration-vs-srate-{approximant}-m2-{m2}'
    plt.savefig(fname + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    
    


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def flower_duration_3d_plot(approximant='EccentricTD', type=None, noshow=False, meshprecision=5):
    """
    3D plot of duration as a function of m1, m2, and f_lower,
    with random eccentricity in [0.001, 0.25].
    """
    m1s = np.arange(5, 50, meshprecision)
    m2s = np.arange(5, 50, meshprecision)
    f_lowers = np.arange(10, 50, meshprecision)

    m1arr, m2arr, f_arr = [], [], []
    durations = []

    for m1 in m1s:
        for m2 in m2s:
            for f_lower in f_lowers:
                waveform_kwargs = {
                    'delta_t': DELTA_T,
                    'f_lower': f_lower,}
                if approximant == 'EccentricTD':
                    ecc = np.random.uniform(0.0001, 0.25, 1)[0]
                    waveform_kwargs['eccentricity'] = ecc
                try:
                    hp, hc = get_td_waveform(
                        approximant=approximant,
                        mass1=m1,
                        mass2=m2,
                        **waveform_kwargs
                    )
                    duration = hp.duration
                except Exception as e:
                    duration = np.nan  # skip failed waveforms
                m1arr.append(m1)
                m2arr.append(m2)
                f_arr.append(f_lower)
                durations.append(duration)

    m1arr = np.array(m1arr)
    m2arr = np.array(m2arr)
    f_arr = np.array(f_arr)
    durations = np.array(durations)


    if type is None or type=='full':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Remove NaNs for plotting
        mask = ~np.isnan(DURATION)

        ax.scatter(m1arr[mask], m2arr[mask], f_arr[mask], c=durations[mask], cmap='viridis', marker='o')
        ax.set_xlabel('$m_1$ ($M_{\\odot}$)')
        ax.set_ylabel('$m_2$ ($M_{\\odot}$)')
        ax.set_zlabel('$f_{min}$ (Hz)')
        sc = ax.scatter(m1arr[mask], m2arr[mask], f_arr[mask], c=durations[mask], cmap='viridis')
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
        # cbar._set_scale('log')
        cbar.set_label('Duration (s)')
        ax.set_title(f'3D Duration Plot: {approximant}')
        plt.tight_layout()
        if not noshow:
            plt.show()
        # Set the viewing angle for better visualization
        ax.view_init(elev=30, azim=120)
        fig.savefig(f'flower-duration-3d-plot-{approximant}.png', dpi=300, bbox_inches='tight')

    # Truncate data to have maximum duration of only 50 sec
    elif type=='max50s':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        max_duration = 50.0
        mask = durations <= max_duration

        ax.scatter(m1arr[mask], m2arr[mask], f_arr[mask], c=durations[mask], cmap='viridis', marker='o')
        ax.set_xlabel('$m_1$ ($M_{\\odot}$)')
        ax.set_ylabel('$m_2$ ($M_{\\odot}$)')
        ax.set_zlabel('$f_{min}$ (Hz)')
        sc = ax.scatter(m1arr[mask], m2arr[mask], f_arr[mask], c=durations[mask], cmap='viridis')
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label('Duration (s)')

        ax.set_title(f'3D Duration Plot: {approximant} (max 50s)')
        plt.tight_layout()
        if not noshow:
            plt.show()
        ax.view_init(elev=30, azim=120)
        fig.savefig(f'flower-duration-3d-plot-{approximant}-max50s.png', dpi=300, bbox_inches='tight')

    # Save the 3D plot using plotly for interactive visualization
    if type=='plotly':
        import plotly.graph_objs as go
        import plotly.io as pio
        from scipy.interpolate import griddata

        # Remove NaNs for plotly
        max_duration = 50.0
        mask = durations <= max_duration
        fig_plotly = go.Figure(data=[go.Scatter3d(
            x=m1arr[mask],
            y=m2arr[mask],
            z=f_arr[mask],
            mode='markers',
            marker=dict(
                size=7,
                color=durations[mask],
                colorscale='Viridis',
                colorbar=dict(title='Duration (s)'),
                opacity=0.8
            ),
            text=[f"Duration: {d:.2f}s" for d in durations[mask]]
        )])

        # Add iso-surfaces at duration ≈ 10s, using plotly's isosurface
        # Reshape data into a 3D grid for Isosurface
        unique_m1 = np.unique(m1arr)
        unique_m2 = np.unique(m2arr)
        unique_f = np.unique(f_arr)
        grid_shape = (len(unique_m1), len(unique_m2), len(unique_f))
        # Create meshgrid
        m1_grid, m2_grid, f_grid = np.meshgrid(unique_m1, unique_m2, unique_f, indexing='ij')
        # Fill the grid with durations
        duration_grid = np.full(grid_shape, np.nan)
        for i in range(len(m1arr)):
            idx_m1 = np.where(unique_m1 == m1arr[i])[0][0]
            idx_m2 = np.where(unique_m2 == m2arr[i])[0][0]
            idx_f = np.where(unique_f == f_arr[i])[0][0]
            duration_grid[idx_m1, idx_m2, idx_f] = durations[i]
        # Plot the iso-surfaced for different durations
        duration_planes = [1.0, 5.0, 10.0, 20.0, 30.0]
        for iso in duration_planes:
            fig_plotly.add_trace(go.Isosurface( 
                # Flatten the grids for plotly
                x=m1_grid.flatten(),
                y=m2_grid.flatten(),
                z=f_grid.flatten(),
                value=duration_grid.flatten(),
                isomin=iso,
                isomax=iso,
                surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,
                opacity=0.3,
                colorscale='Greens',
                name='10s Iso-surface',
                hoverinfo='skip',
                legendgrouptitle_text=f'Duration Iso-surface at {iso}s'
            ))

        fig_plotly.update_layout(
            scene=dict(
                # WebGL cannot handle latex labels!
                xaxis_title='Primary Mass',
                yaxis_title='Secondary Mass',
                zaxis_title='Lower Frequency Cutoff (Hz)'
            ),
            title=f'3D Duration Plot: {approximant}',
            margin=dict(l=0, r=0, b=0, t=40)
        )

        if not noshow:
            fig_plotly.show()

        # Save as HTML for interactive viewing
        plotly_fname = f'flower-duration-3d-plot-{approximant}-max50s.html'
        pio.write_html(fig_plotly, file=plotly_fname, auto_open=False)
        print(f"Plotly 3D plot saved to {plotly_fname}")


def example5(approximants=['SEOBNRv4', 'EccentricTD']):
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    m1, m2 = 5, 10
    e = 0.2

    label0 = f'{approximants[0]}, e={0.0}'
    label1 = f'{approximants[1]}, e={e}'
    hp, hc = get_td_waveform(approximant=approximants[0], mass1=m1, mass2=m2,
                                delta_t=DELTA_T, f_lower=f_lower, eccentricity=e,)
    hp, hc = hp.trim_zeros(), hc.trim_zeros()
    ax.plot(hp.sample_times, hp, label=label0)
    hp, hc = get_td_waveform(approximant=approximants[1], mass1=m1, mass2=m2,
                                delta_t=DELTA_T, f_lower=f_lower, eccentricity=e)
    ax.plot(hp.sample_times, hp, label=label1)
    ax.set_xlabel('Sample Times')
    ax.set_ylabel('$h_{\\plus}$ Strain')
    ax.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    fname = 'strain-plot-eccentric'
    # fname += '-jsps'
    plt.savefig(fname + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def example5b(approximants=['SEOBNRv4', 'EccentricTD']):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2), sharey=True)
    m1, m2 = 5, 10
    e = 0.2

    label0 = f'{approximants[0]}, e={0.0}'
    label1 = f'{approximants[1]}, e={e}'

    # Plot for SEOBNRv4
    hp, hc = get_td_waveform(approximant=approximants[0], mass1=m1, mass2=m2,
                                delta_t=DELTA_T, f_lower=f_lower)
    hp, hc = hp.trim_zeros(), hc.trim_zeros()
    axes[0].plot(hp.sample_times, hp, label=label0)
    axes[0].set_xlabel('Sample Times')
    axes[0].set_ylabel('$h_{\\plus}$ Strain')
    axes[0].legend(loc='best', fontsize=8)

    # Plot for EccentricTD
    hp, hc = get_td_waveform(approximant=approximants[1], mass1=m1, mass2=m2,
                                delta_t=DELTA_T, f_lower=f_lower, eccentricity=e)
    hp, hc = hp.trim_zeros(), hc.trim_zeros()
    axes[1].plot(hp.sample_times, hp, label=label1)
    axes[1].set_xlabel('Sample Times')
    # axes[1].set_ylabel('$h_{\\plus}$ Strain')
    axes[1].set_ylabel('')  # Hide the y-axis label by setting it to an empty string
    axes[1].tick_params(axis='y', which='both', left=False, labelleft=False)  # Remove y-axis ticks and labels
    # axes[1].set_position([axes[0].get_position().x1, axes[1].get_position().y0, 
                        #   axes[1].get_position().width, axes[1].get_position().height])  # Touch axes[1] to axes[0]
    axes[1].legend(loc='best', fontsize=8)
    plt.subplots_adjust(wspace=-0.05)  # Adjust horizontal space between subplots

    plt.tight_layout()
    fname = 'strain-plot-eccentric-twoaxes'
    # fname += '-jsps'
    plt.savefig(fname + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def calc_cutoffconst(approximant='SEOBNRv4', nsamples=500, aligned=False):
    """
    Calculates the proportionality constant relating the duration
    of the signal to the lower frequency cutoff.

    ```math
        T \\propto f_{low}^{-8/3} M_{chirp}^{-5/3}
        T = k * f_{low}^{-8/3} M_{chirp}^{-5/3}
    ```

    We keep `f_low` fixed at 40 Hz, generate samples for different
    mass values and then find the proportionality constant based on the
    formula described above.
    """
    m1 = m2 = np.arange(5, 75, nsamples)
    mchirp = (m1 * m2) ** (3/5) / (m1 + m2) ** (1/5)
    consts = np.zeros(len(mchirp))
    wfkwargs = {"approximant": approximant, 
                "delta_t": DELTA_T, 
                "f_lower": f_lower}
    if aligned:
        spin1z, spin2z = np.random.uniform(-0.99, 0.99, 2)
        wfkwargs["spin1z"] = spin1z
        wfkwargs["spin2z"] = spin2z
    for i in range(len(mchirp)):
        wfkwargs["mass1"] = m1[i]
        wfkwargs["mass2"] = m2[i]
        hp, hc = get_td_waveform(**wfkwargs)
        consts[i] = hp.duration / (40 ** (-8/3) * mchirp[i] ** (-5/3))
    const = np.mean(consts)
    logging.info(f"Proportionality constant (k) for {approximant}: {const}")
    return const


def check_datasets(nsamples=10,):
    """
    Checks the datasets obtained via changing f_cutoff and f_sample.
    """
    logging.info("Checking datasets with different f_cutoff and f_sample values.")
    masses = get_mass(splitTT=False)
    cutoffconst = calc_cutoffconst()
    durations = np.zeros(nsamples)
    for i in tqdm(range(nsamples)):
        m1 = np.random.choice(masses[:,0])
        m2 = np.random.choice(masses[:,1])
        logging.info(f"Masses: {m1}, {m2}")
        durations[i] = get_vals(m1, m2, dataset='f_cutoff', cutoffprop=cutoffconst, 
                                calc_duration_mean=True)
        print(f"Mean duration: {np.mean(durations)}")



def calc_cutoffconst(approximant='SEOBNRv4', nsamples=500, aligned=False):
    """
    Calculates the proportionality constant relating the duration
    of the signal to the lower frequency cutoff.

    ```math
        T \\propto f_{low}^{-8/3} M_{chirp}^{-5/3}
        T = k * f_{low}^{-8/3} M_{chirp}^{-5/3}
    ```

    We keep `f_low` fixed at 40 Hz, generate samples for different
    mass values and then find the proportionality constant based on the
    formula described above.
    """
    m1 = m2 = np.arange(5, 75, nsamples)
    mchirp = (m1 * m2) ** (3/5) / (m1 + m2) ** (1/5)
    consts = np.zeros(len(mchirp))
    wfkwargs = {"approximant": approximant, 
                "delta_t": DELTA_T, 
                "f_lower": f_lower}
    if aligned:
        spin1z, spin2z = np.random.uniform(-0.99, 0.99, 2)
        wfkwargs["spin1z"] = spin1z
        wfkwargs["spin2z"] = spin2z
    for i in range(len(mchirp)):
        wfkwargs["mass1"] = m1[i]
        wfkwargs["mass2"] = m2[i]
        hp, hc = get_td_waveform(**wfkwargs)
        consts[i] = hp.duration / (40 ** (-8/3) * mchirp[i] ** (-5/3))
    const = np.mean(consts)
    logging.info(f"Proportionality constant (k) for {approximant}: {const}")
    return const


def check_datasets(nsamples=10,):
    """
    Checks the datasets obtained via changing f_cutoff and f_sample.
    """
    logging.info("Checking datasets with different f_cutoff and f_sample values.")
    masses = get_mass(splitTT=False)
    cutoffconst = calc_cutoffconst()
    durations = np.zeros(nsamples)
    for i in tqdm(range(nsamples)):
        m1 = np.random.choice(masses[:,0])
        m2 = np.random.choice(masses[:,1])
        logging.info(f"Masses: {m1}, {m2}")
        durations[i] = get_vals(m1, m2, dataset='f_cutoff', cutoffprop=cutoffconst, 
                                calc_duration_mean=True)
        print(f"Mean duration: {np.mean(durations)}")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate Data for training a CVAE for GW data')

    parser.add_argument('--nsamples', type=int, default=1,
                        help='Number of samples to generate for the specified operation.')
    parser.add_argument('--qlim', type=int, default=5,
                        help='Maximum mass ratio limit for generating waveforms.')
    parser.add_argument('--qlim', type=int, default=5,
                        help='Maximum mass ratio limit for generating waveforms.')

    parser.add_argument('--plotmass', action='store_true', default=False,
                        help='Plot mass distribution')
    parser.add_argument('--getstrain', action='store_true', default=False,
                        help='Generate example strain data to check code.')
    parser.add_argument('--trial', action='store_true', default=False,
                        help='Run a trial to check the CustomDataset class.')
    parser.add_argument('--fdstrain', action='store_true', default=False,
                        help='Generate example frequency-domain strain data to check code.')
    parser.add_argument('--approximant', nargs='+', default=['IMRPhenomD'],
                        help='Approximant(s) to use. Can be a single value or a list.')
    parser.add_argument('--otherparams', action='store_true', default=False,
                        help='Use other parameters for the waveform generation.')
    
    parser.add_argument('--otherparams', action='store_true', default=False,
                        help='Use other parameters for the waveform generation.')
    
    parser.add_argument('--example', action='store_true', default=False,
                        help='Reproduce PyCBC documentation example.')
    parser.add_argument('--example2', action='store_true', default=False,
                        help='Plot Freq-Amp plots for different mass ratios.')
    parser.add_argument('--example3', action='store_true', default=False,
                        help='Plot Freq-Amp plots for different mass ratios.')
    parser.add_argument('--example4', action='store_true', default=False,
                        help='Plot Strain plots for different mass ratios.')
    parser.add_argument('--example4b', action='store_true', default=False,
                        help='Plot Strain plots for different f_lower vals for EccentricTD.')
    parser.add_argument('--example5', action='store_true', default=False,
                        help='Plot Eccentric Strain plots for different mass ratios.')
    parser.add_argument('--example5b', action='store_true', default=False,
                        help='Plot Eccentric Strain plots for different mass ratios.')
    parser.add_argument('--flower_duration_plot', '-flower', action='store_true', default=False,
                        help='Plot the duration of the waveform as a function of the f_lower value.')
    parser.add_argument('--srate_duration_plot', '-srate', action='store_true', default=False,
                        help='Plot the duration of the waveform as a function of the sample rate.')
    parser.add_argument('--flower_duration_3d_plot', '-flower3d', action='store_true', default=False,
                        help='3D plot of duration as a function of m1, m2, and f_lower.')
    parser.add_argument('--checkdatasets', action='store_true', default=False,
                        help='Check the datasets obtained via changing f_cutoff and f_sample.')
    parser.add_argument('--checkdatasets', action='store_true', default=False,
                        help='Check the datasets obtained via changing f_cutoff and f_sample.')

    parser.add_argument('--plot', action='store_true', default=False,
                        help='Output and save a Plot!')
    parser.add_argument('--plotfreqamp', action='store_true', default=False,
                        help='Plot frequency and amplitude data')
    parser.add_argument('--convert', action='store_true', default=False,
                        help='Convert the strain to phase and amplitude')
    parser.add_argument('--fname', type=str, default='',
                        help='Filename suffix for saving data to HDF5 file.')

    parser.add_argument('--save-params-from-hdf', action='store_true', default=False,
                        help='Save the parameters from the HDF5 file to a text file for reference.')
    
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Increase verbosity of the output.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode for detailed logging.')
    
    parser.add_argument('--nosave', action='store_true', default=False,
                        help='Do not save the plot.')
    parser.add_argument('--noshow', action='store_true', default=False,
                        help='Do not show the plot.')
    parser.add_argument('--savedata', action='store_true', default=False,
                        help='Save the data to HDF5 file.')
    parser.add_argument('--checkhdf', action='store_true', default=False,
                        help='Read the data from HDF5 file.')
        
    # Subparsers for checkampfreq
    checkampfreq_parser = parser.add_subparsers(dest='checkampfreq').add_parser('checkampfreq', 
                        help='Check the amplitude and frequency data' \
                        'by calculating the noise-weighted inner product of the recombined strain' \
                        'waveform with the original strain waveform. Ideally, the difference' \
                        'should only be due to numerical errors and should be very small, about 1e-14 ish.')
    checkampfreq_parser.add_argument('--usephase', action='store_true', default=False,
                        help='Use phase data instead of frequency data to check round-trip error.')
    checkampfreq_parser.add_argument('--wavegen', action='store_true', default=False,
                        help='Check the round-trip reconstruction error by calling in the `get_td_waveform`' \
                            'function to get the original strain waveforms, converting them to amplitude and frequency' \
                            'and then reconstructing the strain waveforms again to check the error.'   \
                            'Instead of reading the waveforms data from the saved HDF5 datafile.')
        
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s ln%(lineno)d @ %(funcName)s : %(message)s', 
                            force=True)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s ln%(lineno)d @ %(funcName)s : %(message)s', 
                            force=True)
        
    
        
    if args.trial:
        ds =  CustomDataset(forwhat='valid', plot=args.plot, 
                            approximant=args.approximant, convert=args.convert)
        for i, x in enumerate(ds):
            logging.debug(f'x: {x}')
            logging.debug(f'x: {x}')
            if i==args.nsamples:
                break
        logging.debug(f"Length of dataset: {len(ds)}")
        logging.debug(f"Length of dataset: {len(ds)}")
    
    if args.plotmass:
        get_mass(plot=args.plot, qlim=10)
        get_mass(plot=args.plot, qlim=10)

    if args.getstrain:
        # masses = get_mass(qlim=5)[0]
        masses = np.array([[5,30],[15,50]])
        if args.plotfreqamp:
            args.convert = True
        for i in range(args.nsamples):
            m1 = np.random.choice(masses[:,0])
            m2 = np.random.choice(masses[:,1])
            get_strain(m1, m2, convert=args.convert, plot=args.plot, nosave=args.nosave,
                   plotfreqamp=args.plotfreqamp, approximant=args.approximant)
            
    if args.fdstrain:
        masses = get_mass()[0]
        for i in range(args.nsamples):
            m1 = np.random.choice(masses[:,0])
            m2 = np.random.choice(masses[:,1])
            get_fd_strain(m1, m2, plot=args.plot)

    if args.example:
        masses = get_mass(qlim=5)[0]
        for i in range(args.nsamples):
            m1 = np.random.choice(masses[:,0])
            m2 = np.random.choice(masses[:,1])
            # print(f"Masses: {m1}, {m2}")
            example(m1, m2)

    if args.example2:
        example2(appoximant=args.approximant, nsamples=args.nsamples,
                 qlim=5, m1end=75)
        
    if args.example3:
        approximant = args.approximant
        if type(approximant) is str:
            approximant = [approximant]
        for apx in approximant:
            example3(approximant=apx)
        approximant = args.approximant
        if type(approximant) is str:
            approximant = [approximant]
        for apx in approximant:
            example3(approximant=apx)
    if args.example4:
        approximant = args.approximant
        if type(approximant) is str:
            approximant = [approximant]
        for apx in approximant:
            example4(approximant=apx)
        approximant = args.approximant
        if type(approximant) is str:
            approximant = [approximant]
        for apx in approximant:
            example4(approximant=apx)
    if args.example4b:
        example4b()
    if args.example5:
        example5()
    if args.example5b:
        example5b()

    if args.flower_duration_plot:
        flower_duration_plot(approximant=args.approximant)
    if args.srate_duration_plot:
        for approximant in args.approximant:
            # srate_duration_plot(approximant=approximant)
            m1_duration_vs_srate_plot(approximant=approximant, m2=10)
            m1_duration_vs_srate_plot(approximant=approximant, m2=25)
            m1_duration_vs_srate_plot(approximant=approximant, m2=50)

    if args.flower_duration_3d_plot:
        approximant = args.approximant
        if type(approximant) is str:
            approximant = [approximant]
        for apx in approximant:
            if apx not in APPROXIMANTS:
                logging.error(f"Approximant {apx} not found in the list of supported approximants.")
                continue
            flower_duration_3d_plot(approximant=apx, type='full', noshow=args.noshow)
            flower_duration_3d_plot(approximant=apx, type='max50s', noshow=args.noshow)
            flower_duration_3d_plot(approximant=apx, type='plotly', noshow=args.noshow,
                                    meshprecision=2)
        
    if args.savedata:
        if type(args.approximant) is list:
            args.approximant = args.approximant[0]
        fname = '../data/' + args.approximant
        if args.otherparams:
            fname += '-coa&incli'
        fname += args.fname
        fname = '../data/' + args.approximant
        if args.otherparams:
            fname += '-coa&incli'
        fname += args.fname
        ttsplits = get_mass(splitTT=True, plot=False)

        dataset = 'f_cutoff'
        # `dataset` can be `None`, `raw`, `f_cutoff`, `f_sample`!
        write_data_to_hdf(fname+'-train', masses=ttsplits[0], dataset=dataset,
                          approximant=args.approximant, otherparams=args.otherparams)
        write_data_to_hdf(fname+'-valid', masses=ttsplits[1], dataset=dataset,
                          approximant=args.approximant, otherparams=args.otherparams)
        write_data_to_hdf(fname+'-test', masses=ttsplits[2], dataset=dataset,
                          approximant=args.approximant,
                          otherparams=args.otherparams)

        dataset = 'f_cutoff'
        # `dataset` can be `None`, `raw`, `f_cutoff`, `f_sample`!
        write_data_to_hdf(fname+'-train', masses=ttsplits[0], dataset=dataset,
                          approximant=args.approximant, otherparams=args.otherparams)
        write_data_to_hdf(fname+'-valid', masses=ttsplits[1], dataset=dataset,
                          approximant=args.approximant, otherparams=args.otherparams)
        write_data_to_hdf(fname+'-test', masses=ttsplits[2], dataset=dataset,
                          approximant=args.approximant,
                          otherparams=args.otherparams)

    if args.checkhdf:
        if type(args.approximant) is list:
            args.approximant = args.approximant[0]
        dataset = '-f_cutoff'
        if args.fname == '':
            fname = '../data/' + args.approximant + '-train' + dataset
        else:
            fname = '../data/' + args.fname
        print(fname)
        check_hdf(fname+'.hdf', noshow=args.noshow)

    if args.checkdatasets:
        check_datasets()

    if args.checkampfreq:
        fname = 'SEOBNRv4-train-100-fcutoff-uniform-aligned'
        # NOTE: The values I saved for phase, used the default params
        # for the `phase_from_polarizations` func, which made
        # the starting phase zero by default. Thus, my saved
        # HDF data files do not contain the correct phase values.
        check_ampfreq(fname, noshow=args.noshow, usephase=args.usephase)
        if args.wavegen:
            check_ampfreq_via_wavegen(noshow=args.noshow, usephase=args.usephase)

    if args.save_params_from_hdf:
        if type(args.approximant) is list:
            args.approximant = args.approximant[0]
        if args.fname == '':
            fname = args.approximant + '-train-100000-fcutoff-uniform-aligned-regen'
        else:
            fname = args.fname
        hdf_fname = '../data/' + fname
        txt_fname = '../data/' + 'params-' + fname
        save_params_from_hdf(hdf_fname, txt_fname=txt_fname)
