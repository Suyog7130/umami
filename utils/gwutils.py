
import numpy as np

import torch

import logging
logger = logging.getLogger(__name__)

import pycbc

PRESET_ARRAY_SIZE = 8191

SAMPLE_RATE = 8192.0  # n_samples = duration(s) / sample_rate
DURATION = 1.00
DELTA_T = DURATION / SAMPLE_RATE   # delta_t is just 1/sample_rate!


def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        logger.info(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        logger.info(f"Inf detected in {name}")

def calculate_cnn_output_size(n_layers, input_length, kernel_size,
                              dilation, padding=0, stride=1):
    """ Assumed that the Pooling layer always follows a CNN layer. """
    seq_len = input_length
    for _ in range(n_layers):
        seq_len = (seq_len - dilation*(kernel_size - 1) - padding - 1) // kernel_size + 1
        seq_len = (seq_len - kernel_size) * stride // kernel_size + 1
    return seq_len


def calc_chirp_mass(m1, m2):
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

def calc_chieff(m1, m2, chi1z, chi2z):
    return (m1 * chi1z + m2 * chi2z) / (m1 + m2)



def calc_time_array(n, sample_rate=SAMPLE_RATE):
    """
    Calculate the time array for a given number of samples and sample rate.
    """
    return torch.linspace(0, n / sample_rate, steps=n)


# TODO: f_lower is different for diff waveforms, and that is one
# of the main features of my code. So, I need to make sure that the f_lower 
# used in the mismatch calculation is consistent with the one used 
# in the waveform generation.
def calc_polarization_mismatch(hp_orig, hp_recon, delta_t=DELTA_T, f_lower=20.0, resample_psd=True):
    """
    Calculate the mismatch between the original and reconstructed hplus/hcross waveforms.
    This calls the `match` function from `pycbc.filter` to compute the match between the two waveforms, 
    and then returns the mismatch as `1 - match`. The `match` function computes the optimum match
    between the waveforms by maximizing over time shifts and phase shifts, and it uses the 
    power spectral density (PSD) of the noise to weight the match calculation. 
    The `resample_psd` option allows you to resample the PSD to match the frequency resolution of the waveforms, 
    which is necessary for accurate mismatch calculation.

    Parameters
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

    Returns
    --------
    mismatch : float
        The mismatch value, 0 means perfect match, 1 means orthogonal.
    """
    from pycbc.filter import match as matchfunc
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.types import TimeSeries
    logger.debug(f"Calculating mismatch with delta_t={delta_t}, f_lower={f_lower}, resample_psd={resample_psd}")

    if isinstance(hp_orig, torch.Tensor):
        hp_orig = hp_orig.detach().cpu().numpy()
    if isinstance(hp_recon, torch.Tensor):
        hp_recon = hp_recon.detach().cpu().numpy()
    
    psd = pycbc.psd.aLIGOZeroDetHighPower(len(hp_orig), delta_f=1/(len(hp_orig)*delta_t), low_freq_cutoff=f_lower)
    
    logger.debug(f"PSD delta_f: {1.0/(len(hp_orig)*delta_t)}")
    # Ensure all arrays are float64 for precision match
    hp_orig = np.asarray(hp_orig, dtype=np.float64)
    hp_recon = np.asarray(hp_recon, dtype=np.float64)
    psd = psd.astype(np.float64)
    logger.debug(f'len(hp_orig), len(hp_recon), len(psd) = {len(hp_orig)}, {len(hp_recon)}, {len(psd)}')
    assert len(hp_orig) == len(hp_recon), "Original and reconstructed waveforms must have the same length."

    hp_orig = TimeSeries(hp_orig, delta_t=delta_t)
    hp_recon = TimeSeries(hp_recon, delta_t=delta_t)
    logger.debug(f"hp_orig sample rate: {hp_orig.sample_rate}, hp_recon sample rate: {hp_recon.sample_rate}")
    logger.debug(f"hp_orig delta_f: {hp_orig.delta_f}")
    logger.debug(f'len(hp_orig)={len(hp_orig)}, len(hp_recon)={len(hp_recon)}, \
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
        logger.debug(f"Resampled PSD delta_f: {psd_resampled.delta_f}")
        match, i = matchfunc(hp_orig, hp_recon, psd=psd_resampled, low_frequency_cutoff=f_lower)
    else:
        match, i = matchfunc(hp_orig, hp_recon, psd=psd, low_frequency_cutoff=f_lower)
    logger.debug(f"Match value: {match}, Index: {i}")
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
    logger.debug(f'freq shape: {freq.shape}, dt: {dt}, theta0: {theta0}')
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
    logger.debug(f'amp shape: {amp.shape}, freq shape: {freq.shape}')
    theta = _phase_from_freq_intervals(freq, dt=1.0/SAMPLE_RATE, theta0=theta0)
    logger.debug(f'amp shape: {amp.shape}, freq shape: {freq.shape}, phase shape: {theta.shape}')

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
    #     logger.debug(f'After inserting the first element, amp shape: {amp.shape}, theta shape: {theta.shape}')
    hplus = amp * np.cos(theta)
    hcross = amp * np.sin(theta)
    return hplus, hcross


def calculate_cosine_distance(target, reconstructed):
    """
    Calculate the normalized mismatch between the target and reconstructed waveforms.
    Mismatch = 1 - ( <a|b> / sqrt(<a|a> * <b|b>) )
    where <a|b> is the inner product (dot product).

    NOTE: We assume last axis is the waveform axis!

    Parameters
    -----------
    target : np.ndarray or torch.Tensor
        The correct/original waveform, shape (..., N)
    reconstructed : np.ndarray or torch.Tensor
        The reconstructed waveform, shape (..., N)

    Returns
    --------
    cos_dist : float or np.ndarray
        The cosine distance value(s), 0 means perfect match, 1 means orthogonal.
    """
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    assert target.shape[-1] == PRESET_ARRAY_SIZE, "Target waveform length must match the preset array size."
    assert reconstructed.shape[-1] == PRESET_ARRAY_SIZE, "Reconstructed waveform length must match the preset array size."
    inner_product = np.sum(target * reconstructed, axis=-1)
    target_norm = np.sqrt(np.sum(target * target, axis=-1))
    reconstructed_norm = np.sqrt(np.sum(reconstructed * reconstructed, axis=-1))
    cos_sim = inner_product / (target_norm * reconstructed_norm)
    cos_dist = 1 - cos_sim
    return cos_dist

def polarizations_from_amp_phase(amp, phase, scale_factor=None,
                                 scale_polarizations_instead=False, phase_zero=None):
    """
    Convert amplitude and phase to hplus and hcross polarizations.

    Arguments
    ---------
    amp: array_like
        The amplitude time series.
    phase: array_like
        The phase time series (radians).
    scale_factor: float, optional
        A factor to scale down the amplitude, if the target amplitude was scaled up!
    scale_polarizations_instead: bool, optional
        If True, scale down the polarizations instead of the amplitude. This may be useful for numerical accuracy, however, floats are already in 32-bit precision, so it may not make a difference. Default is False.
    phase_zero: float, optional
        Replace the starting value of the phase with this value (radians), to perhaps account for the starting phase in the original phase series been put to zero. However, again, we would assume that the training input data, e.g. the 'regen.hdf' files, would have already correctly this by regenerating the phase series. So, it is recommended to set this to None!

    Returns
    -------
    hp: ndarray
        The hplus polarization time series.
    hc: ndarray
        The hcross polarization time series.
    """
    if scale_factor is not None and not scale_polarizations_instead:
        logger.info(f"Scaling down amplitude by factor {scale_factor}, current max amp: {amp.max().item()}")
        amp = amp / float(scale_factor)
    if phase_zero is not None:
        phase[0] = torch.tensor(phase_zero)
        logger.info(f"Setting phase[0] to phase_zero: {phase_zero}")
    hp = amp * torch.cos(phase)
    hc = amp * torch.sin(phase)
    if scale_factor is not None and scale_polarizations_instead:
        logger.info(f"Scaling down polarizations by factor {scale_factor}, current max hp: {hp.max().item()}, current max hc: {hc.max().item()}")
        hp = hp / float(scale_factor)
        hc = hc / float(scale_factor)
    return (hp, hc)