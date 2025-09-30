
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


class Waveform:
    """
    Main class to load and 
    """
    def __init__(self, masses, fcutoff=True):
        self.masses = masses
        if fcutoff:
            self.cutoffconst = calc_cutoffconst()
        else:
            self.cutoffconst == None
        self.otherparams = False

        self.approximant = APPROXIMANT
        self.sample_rate = SAMPLE_RATE
        self.delta_t = DELTA_T
        self.f_lower = f_lower
        self.duration = DURATION
        self.preset_array_size = PRESET_ARRAY_SIZE

        self.fname = str(self.approximant) + '-fcutoff-uniform-aligned'
        if self.otherparams:
            self.fname += '-otherparam'

    def get_aligned_vals(self, m1, m2, spin):
        """
        Generate the time-domain waveform for the given masses, aligned spins and
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
        """
        extra = {}
        wfkwargs = {
            'approximant': self.approximant,
            'mass1': m1,
            'mass2': m2,
            'f_lower': self.f_lower,
            'delta_t': self.delta_t,
            'spin1z': spin,
            'spin2z': spin,
        }
        wfkwargs["spin1z"] = np.random.uniform(-0.999, 0.999, 1)
        wfkwargs["spin2z"] = np.random.uniform(-0.999, 0.999, 1)
        if self.otherparams:
            angles = np_gen.uniform(0., 2*np.pi, 3)
            wfkwargs['coa_phase'] = np_gen.uniform(0., 2*np.pi)
            # TODO: check if the inclination has to be in this range ??
            wfkwargs['inclination'] = np_gen.uniform(0., np.pi)
        logging.info(f"waveform_kwargs: {wfkwargs}")
        hp, hc = pycbc.waveform.get_td_waveform(**wfkwargs)

        hp = hp.trim_zeros()
        hc = hc.trim_zeros()

        if self.cutoffconst is not None:
            logging.info(f'f_low={f_lower}, duration={hp.duration}')

            # calculate new f_lower
            mchirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
            new_fcutoff = ( self.duration / (self.cutoffconst * mchirp ** (-5/3)) )**(-3/8)
            extra['f_lower'] = new_fcutoff
            logging.info(f'New f_lower={new_fcutoff}')
            # adjust the new f_lower to allow for some error
            new_fcutoff -= 0.2*new_fcutoff

            # generate a second waveform
            wfkwargs['f_lower'] = new_fcutoff
            hp, hc = pycbc.waveform.get_td_waveform(**wfkwargs)
            hp = hp.trim_zeros()
            hc = hc.trim_zeros()
            extra['delta_t'] = hp.delta_t
            logging.info(f'New f_lower={new_fcutoff}, duration={hp.duration}')
            logging.info(f'sample_len={len(hp)}')

        # Have correct input lengths!
        # It should be ensured that the merger is always within the data.
        if len(hp) > self.preset_array_size:
            extra['truncated'] = True
            extra['truncated_len'] = diff = len(hp) - self.preset_array_size
            logging.debug(f'len(freq) > {self.preset_array_size} by {diff} ele \
                                \n So truncating array from the left!')
            hp = hp[diff:]
        if len(hc) > self.preset_array_size:
            extra['truncated'] = True
            extra['truncated_len'] = diff = len(hc) - self.preset_array_size
            logging.debug(f'len(amp) > {self.preset_array_size} by {diff} ele \
                                \n So truncating array from the left!')
            hc = hc[diff:]

        # Calculate the amplitude and phase from the polarizations.
        logging.debug('Converting `hp` & `hc` to Freq Amp!')
        amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
        phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc)
        freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)

        # NOTE: There's no need to pad or truncate amp/freq b'cuz the
        # length will be the same with the original hp/hc
        hp = np.array(hp, dtype=np.float32)
        hc = np.array(hc, dtype=np.float32)
        amp = np.array(amp.data, dtype=np.float32)
        phase = np.array(phase.data, dtype=np.float32)
        freq = np.array(freq.data, dtype=np.float32)

        # append the extra info to the end of the data
        extra['truncated'] = extra.get('truncated', False)
        extra['padded'] = extra.get('padded', False)
        extra['truncated_len'] = extra.get('truncated_len', None)
        extra['padded_at'] = extra.get('padded_at', None)
        extra['eccentricity'] = wfkwargs.get('eccentricity', None)
        extra['coa_phase'] = wfkwargs.get('coa_phase', None)
        extra['inclination'] = wfkwargs.get('inclination', None)
        return [hp, hc, amp, phase, freq, extra]


    def write_hdf_grp(self, hf, data, grpname):
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
        

    def write_data_to_hdf(self):
        logging.info(f'Writing data to HDF5 file {self.fname}.hdf')
        if os.path.exists(fname+'.hdf'):
            logging.info(f'File {fname}.hdf already exists. Using an incremented name.')
            fname = fname.split('.hdf')[0] + '-1.hdf'
        with h5py.File(fname+'.hdf', 'w') as hf:
            # Create a group for each mass
            for i, mass in tqdm(enumerate(self.masses),
                                total=len(self.masses),
                                desc='samples-written',
                                ncols=100,):
                m1, m2 = mass
                grpname = f'sample{i}'

                data = self.get_vals(m1, m2)
                hfgrp = hf.create_group(grpname)
                hfgrp.attrs['mass1'] = m1
                hfgrp.attrs['mass2'] = m2
                hfgrp.attrs['approximant'] = self.approximant
                hfgrp.attrs['sample_rate'] = self.sample_rate
                hfgrp.attrs['delta_t'] = data[-1].get('delta_t', self.delta_t)
                hfgrp.attrs['f_lower'] = data[-1].get('f_lower', self.f_lower)

                if self.otherparams:
                    hfgrp.attrs['coa_phase'] = data[-1]['coa_phase']
                    hfgrp.attrs['inclination'] = data[-1]['inclination']

                # extra info like truncated or padded
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

                self.write_hdf_grp(hf, data, grpname)
            logging.info(f"Data written to {fname+'.hdf'} successfully.")
            hf.close()

    def check_hdf(self, fname):
        """
        Read the data from the HDF5 file.
        """
        print(fname)
        with h5py.File(fname, 'r') as hf:
            for key in hf.keys():
                # print(key)
                grp = hf[key]
                print(dict(grp.attrs))
                for name in grp.keys():
                    print(grp[name])
                    print(name, grp[name].shape)
                    print(list(grp[name].attrs.keys()))
                    # print(grp[name].__dict__)
                    ts = grp[name]
                    print(np.array(ts))
                    print(ts[10:20])
                    plt.plot(range(len(ts)), np.array(ts), label=name)
                    plt.legend()
                    plt.show()
        return hf



    



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
