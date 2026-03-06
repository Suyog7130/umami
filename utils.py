from init import *

def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        logging.info(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        logging.info(f"Inf detected in {name}")

def calculate_cnn_output_size(n_layers, input_length, kernel_size,
                              dilation, padding=0, stride=1):
    """ Assumed that the Pooling layer always follows a CNN layer. """
    seq_len = input_length
    for _ in range(n_layers):
        seq_len = (seq_len - dilation*(kernel_size - 1) - padding - 1) // kernel_size + 1
        seq_len = (seq_len - kernel_size) * stride // kernel_size + 1
    return seq_len



import numpy as np
import pandas as pd


# import matplotlib
# matplotlib.use('Agg')   # non GUI backend
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.colors import LogNorm



import pycbc

from datacvae import PRESET_ARRAY_SIZE, SAMPLE_RATE, DELTA_T

markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'X', 'h', '1', '2', '3', '4', '8']



# TODO: f_lower is different for diff waveforms, and that is one
# of the main features of my code. So, I need to make sure that the f_lower 
# used in the mismatch calculation is consistent with the one used 
# in the waveform generation.
def calc_polarization_mismatch(hp_orig, hp_recon, resample_psd=True, delta_t=DELTA_T, f_lower=20.0):
    """
    Calculate the mismatch between the original and reconstructed hplus/hcross waveforms.
    This calls the `match` function from `pycbc.filter` to compute the match between the two waveforms, 
    and then returns the mismatch as 1 - match. The `match` function computes the optimum match
    between the waveforms by maximizing over time shifts and phase shifts, and it uses the 
    power spectral density (PSD) of the noise to weight the match calculation. 
    The `resample_psd` option allows you to resample the PSD to match the frequency resolution of the waveforms, 
    which is necessary for accurate mismatch calculation.

    Parameters:
    -----------
    hp_orig : np.ndarray or torch.Tensor
        The original hplus waveform.
    hp_recon : np.ndarray or torch.Tensor
        The reconstructed hplus waveform.
    resample_psd : bool, optional
        Whether to resample the PSD to match the waveform's delta_f. Default is True.
    delta_t : float, optional
        The time step between samples in seconds. Default is DELTA_T.
    f_lower : float, optional
        The lower frequency cutoff in Hz. Default is 20.0.

    Returns:
    --------
    mismatch : float
        The mismatch value, 0 means perfect match, 1 means orthogonal.
    """
    from pycbc.filter import match as matchfunc
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.types import TimeSeries

    if isinstance(hp_orig, torch.Tensor):
        hp_orig = hp_orig.detach().cpu().numpy()
    if isinstance(hp_recon, torch.Tensor):
        hp_recon = hp_recon.detach().cpu().numpy()
    
    psd = pycbc.psd.aLIGOZeroDetHighPower(len(hp_orig), delta_f=1/(len(hp_orig)*delta_t), low_freq_cutoff=f_lower)
    
    logging.debug(f"PSD delta_f: {1.0/(len(hp_orig)*delta_t)}")
    # Ensure all arrays are float64 for precision match
    hp_orig = np.asarray(hp_orig, dtype=np.float64)
    hp_recon = np.asarray(hp_recon, dtype=np.float64)
    psd = psd.astype(np.float64)
    logging.debug(f'len(hp_orig), len(hp_recon), len(psd) = {len(hp_orig)}, {len(hp_recon)}, {len(psd)}')
    assert len(hp_orig) == len(hp_recon), "Original and reconstructed waveforms must have the same length."

    hp_orig = TimeSeries(hp_orig, delta_t=delta_t)
    hp_recon = TimeSeries(hp_recon, delta_t=delta_t)
    logging.debug(f"hp_orig sample rate: {hp_orig.sample_rate}, hp_recon sample rate: {hp_recon.sample_rate}")
    logging.debug(f"hp_orig delta_f: {hp_orig.delta_f}")
    logging.debug(f'len(hp_orig)={len(hp_orig)}, len(hp_recon)={len(hp_recon)}, \
                  len(psd)={len(psd)}')
    assert hp_orig.delta_f == hp_recon.delta_f, "Delta_f of original and reconstructed waveforms must match."
    
    # -- This still gives the same delta_f not matching error --#
    # Resample PSD at specific frequencies to match `delta_f` of the 
    # waveforms to the `delta_f` of the PSD. This is necessary for the mismatch calculation.
    hp_fs = hp_recon.to_frequencyseries(delta_f=hp_recon.delta_f)
    freqs = hp_fs.sample_frequencies
    psd_interp = np.interp(freqs, psd.sample_frequencies, psd.data)
    psd_resampled = pycbc.types.FrequencySeries(psd_interp, delta_f=hp_recon.delta_f, dtype=psd.dtype)

    if resample_psd:
        logging.debug(f"Resampled PSD delta_f: {psd_resampled.delta_f}")
        match, i = matchfunc(hp_orig, hp_recon, psd=psd_resampled, low_frequency_cutoff=f_lower)
    else:
        match, i = matchfunc(hp_orig, hp_recon, psd=psd, low_frequency_cutoff=f_lower)
    logging.debug(f"Match value: {match}, Index: {i}")
    mismatch = 1 - match
    return mismatch


def phase_from_frequency(freq, dt, theta0=0.0):
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
    # Multiply by 2π and add initial phase
    return theta0 + 2 * np.pi * theta_integral


def _phase_from_freq_intervals(freq, dt, theta0=0.0):
    """
    freq: length N-1, interpreted as interval frequency between samples.
    returns theta: length N
    """
    logging.debug(f'freq shape: {freq.shape}, dt: {dt}, theta0: {theta0}')
    dtheta = 2 * np.pi * freq * dt              # length N-1
    theta = np.empty(freq.size+1, dtype=np.float64)
    theta[0] = theta0
    theta[1:] = theta0 + np.cumsum(dtheta)  # length N
    # theta = np.cumsum(dtheta)  # length N-1
    # theta[0] = theta0  # Set the initial phase at the first sample
    return theta

def polarizations_from_ampfreq(amp, freq, theta0=0.0):
    """
    Convert amplitude and frequency to hplus and hcross polarizations.
    Phase array will have one element less than the amp, since the freq array
    is derived from the phase array by differentiation originally!
    Well, this certainly seems to be a mess now and it would definitely be
    better that I directly work with the phase and amplitude instead of the frequency.
    """
    logging.debug(f'amp shape: {amp.shape}, freq shape: {freq.shape}')
    theta = _phase_from_freq_intervals(freq, dt=1.0/SAMPLE_RATE, theta0=theta0)
    logging.debug(f'amp shape: {amp.shape}, freq shape: {freq.shape}, phase shape: {theta.shape}')

    # NOTE: Unwantedly, I removed the first element from the 'amp' array in the 
    # `CustomDataset` when I calculated the amp-freq from hp-hc, to have the same
    # length of amplitude and frequency array as an input to the network.
    # However, by definition, frequency will have one less element than the phase or
    # the amplitude, since it is derived from the phase by differentiation. 
    # So, I need to make sure that the length of the 'amp' array is consistent with 
    # the length of the 'freq' array when I convert them back to hplus and hcross.
    # if len(amp) != len(theta):
    #     # repeat the first element of the 'amp' array to make it the same length as the 'theta' array.
    #     amp = np.insert(amp, 0, amp[0])
    #     logging.debug(f'After inserting the first element, amp shape: {amp.shape}, theta shape: {theta.shape}')
    hplus = amp * np.cos(theta)
    hcross = amp * np.sin(theta)
    return hplus, hcross