"""
2026/02/13
Module to generate and save gravitational waveforms using different approximants.
The output is saved in HDF5 format for training/validation/testing datasets.
However, the exact use happens in `datacvae.py` where the datasets are loaded into
PyTorch Datasets and DataLoaders and then passed to the training routines and calls.
"""

import os
import glob
import h5py
import logging
import argparse
import numpy as np
import pandas as pd

import lalsimulation as lalsim
import lal

import pycbc.waveform

from tqdm import tqdm
import matplotlib.pyplot as plt

from datacvae import calc_cutoffconst
from plotutils import putils


# TODO: remove dependence on these global params!
# TODO: have separate class for each interested approximant.

APPROXIMANT = 'SEOBNRv4'

SAMPLE_RATE = 8192.0  # n_samples = duration(s) / sample_rate
DURATION = 1.00
sample_len = int(DURATION * SAMPLE_RATE)
DELTA_T = DURATION / SAMPLE_RATE   # delta_t is just 1/sample_rate!
delta_f = 1.0 / DURATION  # delta_f = 1.0 / duration(s)
f_lower = FMIN = 40.0
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
    in [5,75] uniformly with qlim=10
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


def splitspins(nsamples=1e5):
    s1 = np.random.uniform(-0.999, 0.999, int(nsamples))
    s2 = np.random.uniform(-0.999, 0.999, int(nsamples))
    spins = np.vstack((s1, s2)).T
    np.random.shuffle(spins)
    print(f"Number of spin samples: {len(spins)}")
    train_split = int(0.7 * len(spins))
    val_split = int(0.80 * len(spins))
    train_spins = spins[:train_split]
    val_spins = spins[train_split:val_split]
    test_spins = spins[val_split:]
    print(f"Train/Val/Test sizes: {len(train_spins), len(val_spins), len(test_spins)}")
    return train_spins, val_spins, test_spins


class BaseWaveform:
    def __init__(self,
                 approximant=APPROXIMANT,
                 fcutoff: bool = False, 
                 aligned: bool = True, 
                 precess: bool = False,
                 baseparams : dict = {},
                 nosave: bool = False,
                 wflibname='pycbc',
                 fname='waveforms'):
        # Base source param distributions
        self.mass_range = [5,75]
        self.mtot_range = [10,200]
        self.q_range = [1,10]
        self.chi_range = [-0.8,0.8]

        self.f_lower = f_lower

        self.baseparams = {
            'f_lower': baseparams.get('f_lower', FMIN),
            'delta_t': baseparams.get('delta_t', DELTA_T),
            'approximant': baseparams.get('approximant', approximant),
        }

        self.fcutoff = fcutoff
        if self.fcutoff:
            self.fname += '-fcutoff'
            self.cutoffconst = self.calc_cutoffconst()
        else:
            self.cutoffconst = None

        # higher-level options
        self.nosave = nosave
        self.approximant = approximant
        self.aligned = aligned
        self.precess = precess
        self.use_lal_sim = (wflibname == 'lalsim')

        self.wflibname = wflibname
        self.wfloader = self._set_waveform_loader()

        self.fname = fname + '-' + self.approximant
        if self.aligned:
            self.fname += '-aligned'
        self.init_plot()
        # self.waveform()

    def get_params(self, index):
        """
        Get the parameters for the given index from the parameter space.
        """
        params = self.baseparams.copy()
        for param in self.param_space:
            params[param] = getattr(self, param+'s')[index]
        if 'q' in self.param_space:
            params['m1'] = params['m2'] * params['q']
        return params
    
    def calc_cutoffconst(self, nsamples=1000):
        """
        Calculate the cutoff constant for the waveform duration
        calculation based on the approximate relation:

            DURATION = C * fcutoff^(-8/3) * mchirp^(-5/3)
        
        where, C is the cutoff constant to be calculated.
        """
        logging.info("Calculating cutoff constant for waveform duration...")
        consts = np.zeros(nsamples)
        for i in range(nsamples):
            param = self.get_params(i)
            data = self.get_waveform(*param)
            hp, hc = data[0], data[1]
            duration = hp.duration
            m1, m2 = param['m1'], param['m2']
            mchirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
            fcutoff = param['f_lower']
            consts[i] = duration * fcutoff**(8/3) * mchirp**(5/3)
        cutoffconst = np.mean(consts)
        logging.info(f"Cutoff constant calculated: {cutoffconst}")
        return cutoffconst

    def _set_masses(self, num=100):
        m1s = np.random.uniform(self.mass_range[0], self.mass_range[1], num)
        m2s = np.random.uniform(self.mass_range[0], self.mass_range[1], num)
        return np.vstack((m1s, m2s))

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

    def _set_waveform_loader(self):
        """
        Set the waveform loader function based on the chosen library.
        By default import only Time-Domain waveforms!
        1. 'pycbc' : pycbc.waveform.get_td_waveform
        2. 'lalsim' : lalsim TD or FD
        4. 'pyseobnr' / 'pySEOBNR' : 
        5. 'teobresums' / 'TEOBResumS' : 
        6. 'gwsurrogate' / 'GWSurrogate'
        """
        if self.wflibname == 'pycbc':
            logging.info("Using PyCBC waveform generator.")
            return pycbc.waveform.get_td_waveform
        if self.wflibname == 'lalsim':
            logging.info("Using LALSimulation waveform generator.")
            return lalsim.SimInspiralChooseTDWaveform
        if self.wflibname == 'pyseobnr' or self.wflibname == 'pySEOBNR':
            raise NotImplementedError("PySEOBNR waveform loader not implemented yet.")
        if self.wflibname == 'teobresums' or self.wflibname == 'TEOBResumS':
            raise NotImplementedError("TEOBResumS waveform loader not implemented yet.")
        if self.wflibname == 'gwsurrogate' or self.wflibname == 'GWSurrogate':
            raise NotImplementedError("GWSurrogate waveform loader not implemented yet.")
        else:
            raise ValueError(f"Waveform loader '{self.wflibname}' not implemented yet.")
        
    def _set_lal_wfkwargs(self, params: dict, wfkwargs={}):
        wfkwargs["deltaT"] = params.get('delta_t')
        wfkwargs["f_min"] = params.get('f_lower')
        wfkwargs["f_ref"] = 0.0 # Reference frequency
        wfkwargs["distance"] = 400 * lal.PC_SI # Distance in parsecs
        wfkwargs["inclination"] = 0.0 # Inclination angle
        wfkwargs["phiRef"] = 0.0 # Reference phase
        wfkwargs["longAscNodes"] = 0.0 # Longitude of ascending nodes
        wfkwargs["eccentricity"] = 0.0 # Eccentricity
        wfkwargs["meanPerAno"] = 0.0 # Mean anomaly of pericenter
        wfkwargs["params"] = lal.CreateDict() # Additional params
        wfkwargs["m1"] = params.get('m1') * lal.MSUN_SI
        wfkwargs["m2"] = params.get('m2') * lal.MSUN_SI
        wfkwargs["approximant"] = getattr(lalsim, self.approximant)
        if self.aligned or self.precess:
            wfkwargs["s1z"] = 0.5 # np.random.uniform(-0.999, 0.999, 1)
            wfkwargs["s2z"] = 0.5 # np.random.uniform(-0.999, 0.999, 1)
        if self.precess:
            wfkwargs["s1x"] = 0.0
            wfkwargs["s1y"] = 0.0
            wfkwargs["s2x"] = 0.0
            wfkwargs["s2y"] = 0.0
        return wfkwargs
    
    def _set_pycbc_wfkwargs(self, params: dict, wfkwargs={}):
        wfkwargs["delta_t"] = params.get('delta_t')
        wfkwargs["f_lower"] = params.get('f_lower')
        wfkwargs["mass1"] = params.get('m1_msun', params.get('m1'))
        wfkwargs["mass2"] = params.get('m2_msun', params.get('m2'))
        wfkwargs["approximant"] = params.get('approximant', self.approximant)
        if self.aligned or self.precess:
            wfkwargs["spin1z"] = params.get('chi1z', 0.5) # np.random.uniform(-0.999, 0.999, 1)
            wfkwargs["spin2z"] = params.get('chi2z', 0.5) # np.random.uniform(-0.999, 0.999, 1)
        if self.precess:
            wfkwargs["spin1x"] = params.get('chi1x', 0.1)
            wfkwargs["spin1y"] = params.get('chi1y', 0.5)
            wfkwargs["spin2x"] = params.get('chi2x', -0.4)
            wfkwargs["spin2y"] = params.get('chi2y', 0.2)
        return wfkwargs

    def _set_wfkwargs(self, params: dict):
        if self.wflibname=='lalsim':
            return self._set_lal_wfkwargs(params)
        else:
            return self._set_pycbc_wfkwargs(params)
    
    # def get_lal_waveform(self, m1, m2):
    #     wfkwargs = self._set_wfkwargs(m1, m2)
    #     logging.debug(f"Masses: {m1}, {m2}")
    #     print(wfkwargs)
    #     print(lalsim.SimInspiralChooseTDWaveform.__doc__)
    #     # print(lalsim.__dict__)
    #     print(lalsim.SimInspiralChooseTDWaveform.__dir__)
    #     hp, hc = lalsim.SimInspiralChooseTDWaveform(**wfkwargs)
    #     # epoch = hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds * 1e-9
    #     return m1, m2, hp, hc, amp, phase, freq


    def get_waveform(self, params: dict):
        """
        Get the waveform for one set of source parameters.

        Returns
        -------
        tuple
            A tuple containing the masses and the waveforms (hp, hc, amp, phase, freq).
        """
        logging.info(f"Getting waveform for parameters: {params}")
        wfkwargs = self._set_wfkwargs(params)
        for param in self.param_space:
            logging.debug(f"{param}: {params[param]}")

        # Call PyCBC function by default!
        hp, hc = self.wfloader(**wfkwargs)
        logging.info(f"Generated waveform with {wfkwargs}")

        if self.wflibname=='lalsim':
            logging.info("Generated lal wavefroms!")
            logging.debug(f"hp: {hp.__dict__}")
            logging.debug(f"hp.data: {hp.data.__dir__()}")
            hp, hc = np.array(hp.data.data, copy=False), np.array(hp.data.data, copy=False)
            logging.debug(f"hp as array: {np.asarray(hp.data)}")
            logging.debug(f"hp shape: {hp.shape}, hc shape: {hc.shape}")

            # TODO: Implement Ampl/Phase/Freq conversion for lal waveforms!
            hp = pycbc.types.TimeSeries(hp, delta_t=wfkwargs["deltaT"])
            hc = pycbc.types.TimeSeries(hc, delta_t=wfkwargs["deltaT"])
            # hp, hc = hp.trim_zeros(), hc.trim_zeros()
            amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
            phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc)
            freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)
            logging.debug(f'Length of hp: {len(hp)}, hc: {len(hc)}')
            logging.debug(f'Length of amp: {len(amp)}, phase: {len(phase)}, freq: {len(freq)}')
            # print(hp.__dict__)

        if self.wflibname=='pycbc':
            hp, hc = hp.trim_zeros(), hc.trim_zeros()

            m1 = wfkwargs.get('mass1', params.get('m1_msun', wfkwargs['m1']))
            m2 = wfkwargs.get('mass2', params.get('m2_msun', wfkwargs['m2']))

            if self.cutoffconst is not None:
                logging.info(f'f_low={f_lower}, duration={hp.duration}')
                # calculate new f_lower
                mchirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
                new_fcutoff = ( DURATION / (self.cutoffconst * mchirp ** (-5/3)) )**(-3/8)
                logging.info(f'New f_lower={new_fcutoff}')
                # adjust the new f_lower to allow for some error
                new_fcutoff -= 0.2*new_fcutoff
                # generate a second waveform
                wfkwargs['f_lower'] = new_fcutoff
                hp, hc = pycbc.waveform.get_td_waveform(**wfkwargs)
                hp, hc = hp.trim_zeros(), hc.trim_zeros()
                logging.info(f'New f_lower={new_fcutoff}, duration={hp.duration}')
                logging.info(f'sample_len={len(hp)}')

            amp = pycbc.waveform.utils.amplitude_from_polarizations(hp, hc)
            phase = pycbc.waveform.utils.phase_from_polarizations(hp, hc)
            freq = pycbc.waveform.utils.frequency_from_polarizations(hp, hc)
            logging.debug(f'Length of hp: {len(hp)}, hc: {len(hc)}')
            logging.debug(f'Length of amp: {len(amp)}, phase: {len(phase)}, freq: {len(freq)}')
            # print(hp.__dict__)
        return (hp, hc, amp, phase, freq)

    def waveform(self, num=1):
        """
        Get N number of waveforms for the given param ranges.
        """
        params = self.baseparams.copy()
        params['m1'] = 10.0
        params['m2'] = 15.0
        m1s, m2s = self._set_masses(num)
        if len(m1s) == 2:
            hps, amps, phases, freqs = [], [], [], []
            for m1, m2 in zip(m1s, m2s):
                m1, m2, hp, hc, amp, phase, freq = self.get_waveform(m1, m2)
                hps.append(hp)
                amps.append(amp)
                phases.append(phase)
                freqs.append(freq)
            self.plot_two_wfs(m1s, m2s, hps, amps, phases, freqs)
        else:
            hp, hc, amp, phase, freq = self.get_waveform(params)
            self.fname += f'_m1_{params["m1"]:.2f}_m2_{params["m2"]:.2f}'
            self.plot_single_wf(params['m1'], params['m2'], hp, hc, amp, phase, freq)


class Waveform(BaseWaveform):
    """
    Main class to generate, save, and load training / test waveforms.
    """
    def __init__(self, nsamples=1e5, fcutoff=None,
                 param_space=['m1_msun', 'm2_msun', 'chi1z', 'chi2z'],
                 use_params_file=True, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nsamples = nsamples
        self.param_space = param_space
        self.tttratio = [0.7, 0.1, 0.2]  # train, val, test
        if use_params_file:
            self.load_params_from_file()
        else:
            self.set_parameter_space()

    def set_parameter_space(self):
        """
        Set the parameter space for waveform generation using the
        parameter names passed in `self.param_space` and the range
        defined in `self.mass_range`, `self.q_range`, `self.chi_range`.
        """
        mmin, mmax = self.mass_range
        qmin, qmax = self.q_range
        smin, smax = self.chi_range
        masses = np.random.uniform(mmin, mmax, (int(self.nsamples), 2))
        if 'm1' in self.param_space and 'm2' in self.param_space:
            self.m1s = masses[:,0]
            self.m2s = masses[:,1]
        if 'q' in self.param_space:
            q = np.random.uniform(qmin, qmax, int(self.nsamples))
            m2 = masses[:,0] / q
            self.m2s = m2
            self.qs = q
        if 's1' in self.param_space and 's2' in self.param_space:
            self.s1s = np.random.uniform(smin, smax, int(self.nsamples))
            self.s2s = np.random.uniform(smin, smax, int(self.nsamples))
        if self.precess:
            self.s1zs = self.s1s
            self.s2zs = self.s2s
            self.s1xs = np.random.uniform(smin, smax, int(self.nsamples))
            self.s1ys = np.random.uniform(smin, smax, int(self.nsamples))
            self.s2xs = np.random.uniform(smin, smax, int(self.nsamples))
            self.s2ys = np.random.uniform(smin, smax, int(self.nsamples))

    def load_params_from_file(self, 
                              params_file_dir='data/params/m5-200_s-0p99-0p99_dL100-1000_uvol/seed42/',
                              load_file='seed42-params_000'):
        """
        Load the parameters for waveform generation from a CSV file.
        The CSV file should have columns corresponding to the parameters in `self.param_space`.
        """
        fnames = glob.glob(params_file_dir + '*.csv')
        if load_file is not None:
            fnames = [params_file_dir+load_file+'.csv']
        self.nsamples = 0
        for fname in fnames:
            logging.info(f'Loading file {fname}')
            df = pd.read_csv(fname)
            self.nsamples += len(df)
            for param in self.param_space:
                if param not in df.columns:
                    raise ValueError(f"Parameter {param} not found in the CSV file.")
                setattr(self, param+'s', df[param].values)
        logging.info(f"Parameters loaded from {params_file_dir} successfully.")

    def _tttsplits(self):
        indices = np.arange(self.nsamples)
        train_indices = indices[:int(self.nsamples * self.tttratio[0])]
        val_indices = indices[int(self.nsamples * self.tttratio[0]):int(self.nsamples * self.tttratio[1])]
        test_indices = indices[int(self.nsamples * self.tttratio[1]):]
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)
        return (train_indices, val_indices, test_indices)
    
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
                logging.debug(f"{name}, {tsdata.shape}")
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
    
    def write_data_to_hdf(self, which='train'):
        """
        Write the data to HDF5 file for the given split: train, val, test.
        """
        self.fname += f'-{self.wflibname}-{int(DURATION)}sec-{int(SAMPLE_RATE)}Hz'
        self.fname += f'-{which}'
        logging.info(f'Writing {which} data to HDF5 file {self.fname}.hdf')
        if os.path.exists(self.fname+'.hdf'):
            logging.info(f'File {self.fname}.hdf already exists. Using an incremented name.')
            self.fname = self.fname.split('.hdf')[0] + '-1'

        split_indices = self._tttsplits()[{'train':0, 'val':1, 'test':2}[which]]

        with h5py.File(self.fname+'.hdf', 'w') as hf:
            # Create a group for each mass
            for i in tqdm(split_indices, desc='samples-written', ncols=100):
                params = self.get_params(i)
                # print(params)
                grpname = f'sample{i}'
                data = self.get_waveform(params)
                hfgrp = hf.create_group(grpname)
                for param, value in params.items():
                    hfgrp.attrs[param] = value
                self.write_hdf_grp(hf, data, grpname)
        logging.info(f"Data written to {self.fname+'.hdf'} successfully.")

    def read_data_from_hdf(self, which='train'):
        raise NotImplementedError("Reading data from HDF5 file not implemented yet.")



class SEOBNRv4:
    """
    DEPRECATED! NOTE: Please use the generic `Waveforms` class!
    Class to generate SEOBNRv4 waveforms.
    """
    def __init__(self, masses=None, spins=None, fcutoff=True, fname='',
                 preset_array_size=PRESET_ARRAY_SIZE):
        super().__init__()
        self.masses = masses if masses is not None else np.random.uniform(5, 75, (1000, 2))
        self.spins = spins if (spins is not None and masses is not None) else np.random.uniform(-0.999, 0.999, (1000, 2))
        if fcutoff:
            self.cutoffconst = calc_cutoffconst()
        else:
            self.cutoffconst = None
        self.otherparams = False

        self.approximant = APPROXIMANT
        self.sample_rate = SAMPLE_RATE
        self.delta_t = DELTA_T
        self.f_lower = f_lower
        self.duration = DURATION
        self.preset_array_size = preset_array_size

        self.fname = str(self.approximant) + '-' + fname + 'fcutoff-uniform-aligned'
        if self.otherparams:
            self.fname += '-otherparam'

    def get_aligned_vals(self, m1, m2, s1, s2):
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

        Returns:
            hp, hc, amp, phase, freq: numpy arrays of the waveform polarizations,
                amplitude, phase and frequency.
        """
        extra = {}
        wfkwargs = {
            'approximant': self.approximant,
            'mass1': m1,
            'mass2': m2,
            'f_lower': self.f_lower,
            'delta_t': self.delta_t,
            'spin1z': s1,
            'spin2z': s2,
        }
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

        # TODO: Add the `_regenerate_waveform` function here to ensure
        # that shorter waveforms are regenerated with still lower fcutoff.

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
                logging.debug(f"{name}, {tsdata.shape}")
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
        if os.path.exists(self.fname+'.hdf'):
            logging.info(f'File {self.fname}.hdf already exists. Using an incremented name.')
            self.fname = self.fname.split('.hdf')[0] + '-1'
        with h5py.File(self.fname+'.hdf', 'w') as hf:
            # Create a group for each mass
            for i in tqdm(range(len(self.masses)), desc='samples-written', ncols=100):
                m1, m2 = self.masses[i]
                s1, s2 = self.spins[i]
                grpname = f'sample{i}'

                data = self.get_aligned_vals(m1, m2, s1, s2)
                hfgrp = hf.create_group(grpname)
                hfgrp.attrs['mass1'] = m1
                hfgrp.attrs['mass2'] = m2
                hfgrp.attrs['spin1z'] = s1
                hfgrp.attrs['spin2z'] = s2
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
            logging.info(f"Data written to {self.fname+'.hdf'} successfully.")
            hf.close()


def check_hdf(fname):
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



def get_SEOBNRv4_data(args):
    nsample = args.nsample  # default 1e5
    print(f"Generating {nsample} samples.")
    train_masses, val_masses, test_masses = tttdatasets(nsamples=nsample)
    train_spins, val_spins, test_spins = splitspins(nsamples=nsample)
    trainwf = SEOBNRv4(masses=train_masses, spins=train_spins, fname=f'train-{int(nsample)}-')
    trainwf.write_data_to_hdf()
    valwf = SEOBNRv4(masses=val_masses, spins=val_spins, fname=f'val-{int(nsample)}-')
    valwf.write_data_to_hdf()
    testwf = SEOBNRv4(masses=test_masses, spins=test_spins, fname=f'test-{int(nsample)}-')
    testwf.write_data_to_hdf()


def get_NRSur_data(args):
    wave = Waveform(approximant=args.approximant,
                    wflibname='pycbc', precess=args.precess,)
    wave.waveform()


def save_SEOBNRv4_data():
    wave = Waveform(approximant='SEOBNRv4',
                    fcutoff=True,
                    baseparams={'f_lower': 15.0},
                    wflibname='pycbc', 
                    precess=False,
                    fname='dummy')
    wave.write_data_to_hdf('train')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate and plot gravitational waveforms.")
    
    parser.add_argument('--approximant', type=str, default=APPROXIMANT,
                        help='Waveform approximant to use.')

    parser.add_argument("--aligned", action="store_true", help="Generate aligned-spin waveforms.")
    parser.add_argument("--precess", action="store_true", help="Generate precessing-spin waveforms.")

    parser.add_argument("--use-lal-sim", action="store_true", help="Use LALSimulation for waveform generation.")
    parser.add_argument("--fname", type=str, default="waveforms", help="Filename for saving the plots.")
    parser.add_argument("--nosave", action="store_true", help="Do not save the plots.")
    parser.add_argument('--nsample', type=int, default=1e5,
                        help='Number of samples to generate.')
    parser.add_argument('--checkhdf', action='store_true', default=False,
                        help='Read the data from HDF5 file.')

    parser.add_argument('--fcutoff', action='store_true', default=False,
                        help='Use fcutoff to generate waveforms of equal duration.')
    parser.add_argument('--check-waveform', action='store_true', default=False,
                        help='Check waveform generation and plotting.')
    
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Increase verbosity of the output.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode for detailed logging.')
    
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s ln%(lineno)d @ %(funcName)s : %(message)s', 
                            force=True)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s ln%(lineno)d @ %(funcName)s : %(message)s', 
                            force=True)
        
    if args.checkhdf:
        fname = 'SEOBNRv4-train-100-fcutoff-uniform-aligned.hdf'
        check_hdf(fname)
    elif args.check_waveform:
        BaseWaveform(masses=[[50], [15]],
                      #masses=[[50, 30], [15, 5]],
                      approximant=args.approximant,
                      aligned=args.aligned,
                      precess=args.precess,
                      nosave=args.nosave,
                      fcutoff=args.fcutoff,
                      f_lower=20.0,
                      use_lal_sim=args.use_lal_sim)
    else:
        # get_SEOBNRv4_data(args)
        # get_NRSur_data(args)
        save_SEOBNRv4_data()