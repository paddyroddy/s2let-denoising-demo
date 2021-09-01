import numpy as np
import pyssht as ssht
from numpy.random import default_rng

from denoising.utils.logger import logger
from denoising.utils.vars import RANDOM_SEED


def _signal_power(signal: np.ndarray) -> float:
    """Computes the power of the signal

    Args:
        signal (np.ndarray): the harmonic coefficients of the signal

    Returns:
        float: the energy of the signal
    """
    return (np.abs(signal) ** 2).sum()


def compute_snr(signal: np.ndarray, noise: np.ndarray, signal_type: str) -> float:
    """Computes the SNR of the input signal

    Args:
        signal (np.ndarray): the harmonic coefficients of the initial signal
        noise (np.ndarray): the harmonic coefficients of the Gaussian noise

    Returns:
        float: the SNR in decibels of the signal
    """
    snr = 10 * np.log10(_signal_power(signal) / _signal_power(noise))
    logger.info(f"SNR: {snr:.2f}")
    return snr


def create_noise(L: int, signal: np.ndarray, snr_in: int) -> np.ndarray:
    """Computes Gaussian white noise of the signal

    Args:
        L (int): bandlimit of the signal
        signal (np.ndarray): harmonic coefficients of the signal
        snr_in (int): parameter to control the noise level

    Returns:
        np.ndarray: the harmonic coefficients of the noise
    """
    # set random seed
    rng = default_rng(RANDOM_SEED)

    # initialise
    nlm = np.zeros(L ** 2, dtype=np.complex_)

    # std dev of the noise
    sigma_noise = compute_sigma_noise(signal, snr_in)

    # compute noise
    for ell in range(L):
        ind = ssht.elm2ind(ell, 0)
        nlm[ind] = sigma_noise * rng.standard_normal()
        for m in range(1, ell + 1):
            ind_pm = ssht.elm2ind(ell, m)
            ind_nm = ssht.elm2ind(ell, -m)
            nlm[ind_pm] = (
                sigma_noise
                / np.sqrt(2)
                * (rng.standard_normal() + 1j * rng.standard_normal())
            )
            nlm[ind_nm] = (-1) ** m * nlm[ind_pm].conj()
    return nlm


def compute_sigma_noise(
    signal: np.ndarray,
    snr_in: int,
) -> float:
    """Computes the standard deviation of the noise

    Args:
        signal (np.ndarray): harmonic coefficients of the signal
        snr_in (int): parameter to control the noise

    Returns:
        float: the standard deviation of the noise
    """
    return np.sqrt(10 ** (-snr_in / 10) * _signal_power(signal) / signal.shape[0])
