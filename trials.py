
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import time
import logging
from datetime import datetime
from tqdm import tqdm

from pycbc.waveform import get_td_waveform



SAMPLE_RATE = 8192.0  # n_samples = duration(s) / sample_rate
DURATION = 1.00
sample_len = int(DURATION * SAMPLE_RATE)
DELTA_T = DURATION / SAMPLE_RATE   # delta_t is just 1/sample_rate!
delta_f = 1.0 / DURATION  # delta_f = 1.0 / duration(s)
f_lower = 40.0
f_len = sample_len // 2 + 1  # upper frequency


# TODO: Why is the array size not 8192 !!
PRESET_ARRAY_SIZE = 8191



def test_timecomplexity_compare(self, iters=100):
    """
    Test the time complexity of different waveform generators for generating a 1-10e4 ish number of samples.
    Only compares SEOBNRv4, SEOBNRv4_ROM, and SEOBNRv4_opt variants.
    """
    Nruns = np.arange(1, iters)
    np.random.shuffle(Nruns)
    logging.info(f"Using {Nruns=}")

    device = getattr(self, 'device', 'cpu')
    args = getattr(self, 'args', self)
    preset_array_size = 8190 if getattr(args, 'fcutoff', False) or getattr(self, 'aligned', False) else PRESET_ARRAY_SIZE

    basetimes, romtimes, opttimes = [], [], []
    massratios, chieffs = [], []
    for Nr in tqdm(Nruns, ncols=100):
        # Generate random labels within the training range
        m1 = np.random.uniform(5, 75, Nr)
        q = np.random.uniform(1, 10, Nr)
        m2 = m1 / q
        if self.aligned:
            spin1z = np.random.uniform(-0.9, 0.9, Nr)
            spin2z = np.random.uniform(-0.9, 0.9, Nr)
        else:
            spin1z = spin2z = np.zeros(Nr)
        massratios = q
        chieffs = (m1 * spin1z + m2 * spin2z) / (m1 + m2) if self.aligned else np.zeros(Nr)

        # SEOBNRv4 (base)
        waveform_kwargs = {}
        start_time = time.time()
        for i in range(Nr):
            waveform_kwargs['mass1'] = m1[i]
            waveform_kwargs['mass2'] = m2[i]
            if self.aligned:
                waveform_kwargs['spin1z'] = spin1z[i]
                waveform_kwargs['spin2z'] = spin2z[i]
            waveform_kwargs.update({
                'approximant': 'SEOBNRv4',
                'delta_t': DELTA_T,
                'f_lower': 20.0,
            })
            get_td_waveform(**waveform_kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        basetimes.append(elapsed_time)
        logging.info(f'SEOBNRv4 time taken to generate {Nr} samples: {elapsed_time:.4f} seconds')

        # SEOBNRv4_ROM
        waveform_kwargs['approximant'] = 'SEOBNRv4_ROM'
        start_time = time.time()
        for i in range(Nr):
            waveform_kwargs['mass1'] = m1[i]
            waveform_kwargs['mass2'] = m2[i]
            if self.aligned:
                waveform_kwargs['spin1z'] = spin1z[i]
                waveform_kwargs['spin2z'] = spin2z[i]
            waveform_kwargs.update({
                'approximant': 'SEOBNRv4_ROM',
                'delta_t': DELTA_T,
                'f_lower': 20.0,
            })
            get_td_waveform(**waveform_kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        romtimes.append(elapsed_time)
        logging.info(f'SEOBNRv4_ROM time taken to generate {Nr} samples: {elapsed_time:.4f} seconds')

        # SEOBNRv4_opt
        waveform_kwargs['approximant'] = 'SEOBNRv4_opt'
        start_time = time.time()
        for i in range(Nr):
            waveform_kwargs['mass1'] = m1[i]
            waveform_kwargs['mass2'] = m2[i]
            if self.aligned:
                waveform_kwargs['spin1z'] = spin1z[i]
                waveform_kwargs['spin2z'] = spin2z[i]
            waveform_kwargs.update({
                'approximant': 'SEOBNRv4_opt',
                'delta_t': DELTA_T,
                'f_lower': 20.0,
            })
            get_td_waveform(**waveform_kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        opttimes.append(elapsed_time)
        logging.info(f'SEOBNRv4_opt time taken to generate {Nr} samples: {elapsed_time:.4f} seconds')

    # Save data to csv file
    logging.info(f"Nruns shape: {np.shape(Nruns)}, basetimes shape: {np.shape(basetimes)}, romtimes shape: {np.shape(romtimes)}, opttimes shape: {np.shape(opttimes)}, massratios shape: {np.shape(massratios)}, chieffs shape: {np.shape(chieffs)}")
    df_time = pd.DataFrame({
        'Nruns': Nruns,
        'base_time': basetimes,
        'rom_time': romtimes,
        'opt_time': opttimes,
        # 'mass_ratio': massratios, # TODO
        # 'chi_eff': chieffs
    })
    csvname = self.savedir + 'timecomplexity_compare_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
    df_time.to_csv(csvname, index=False)
    logging.info(f"Time complexity comparison data saved to {csvname}")

    # Plot time taken comparison between base, ROM, and opt
    logging.info("Plotting time complexity comparison between base, ROM, and opt.")
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(Nruns, basetimes, '.', color='black', markersize=10,
            markeredgewidth=0.5, markeredgecolor='black')
    ax.plot(Nruns, romtimes, '^', color='red', markersize=6,
            markeredgewidth=0.5, markeredgecolor='black')
    ax.plot(Nruns, opttimes, 'v', color='green', markersize=6,
            markeredgewidth=0.5, markeredgecolor='black')
    ax.legend(['SEOBNRv4', 'SEOBNRv4_ROM', 'SEOBNRv4_opt'], loc='upper left')
    ax.set_xlabel('Number of Waveforms Generated', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.xaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.yaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    plt.tight_layout()
    figname = f'{self.savedir}/timecomplexity-compare-' + f'{self.device}-' + datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(figname+'.png', dpi=300, transparent=True)
    plt.savefig(figname+'-white.png', dpi=300)
    plt.close()
    logging.info("Time complexity comparison plot saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='../results/', help='Directory to save results')
    parser.add_argument('--aligned', action='store_true', help='Use aligned-spin waveforms')
    parser.add_argument('--fcutoff', action='store_true', help='Use frequency cutoff in waveforms')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()


    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        print('Verbose mode!')
        log_level = logging.INFO
    else:
        log_level = logging.WARN

    # Create results directory for today if it doesn't exist
    today = datetime.today().strftime('%Y%m%d')
    log_dir = f'{args.savedir}/{today}/'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Set up logging to both console and file
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'trial_log_{now}.log')
    logging.basicConfig(
        format='%(levelname)s | %(asctime)s: %(message)s',
        level=log_level,
        datefmt='%y-%m-%d %H:%M:%S',
        force=True,
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(log_file)  # Log to file
        ]
    )

    class Dummy:
        pass

    trial = Dummy()
    trial.savedir = args.savedir + today + '/'
    trial.aligned = args.aligned
    trial.args = args
    trial.device = 'cpu'

    test_timecomplexity_compare(trial, iters=100)
