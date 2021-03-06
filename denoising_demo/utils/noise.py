from typing import Union

import numpy as np
import pyssht as ssht
from numpy.random import default_rng

from denoising_demo.utils.logger import logger
from denoising_demo.utils.vars import RANDOM_SEED


def _signal_power(signal: np.ndarray) -> float:
    """Computes the power of the signal

    Args:
        signal (np.ndarray): the harmonic coefficients of the signal

    Returns:
        float: the energy of the signal
    """
    return (np.abs(signal) ** 2).sum()


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
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


def compute_sigma_j(signal: np.ndarray, psi_j: np.ndarray, snr_in: int) -> np.ndarray:
    """Computes the wavlet noise standard deviation for each wavelet

    Args:
        signal (np.ndarray): the harmonic coefficients of the signal
        psi_j (np.ndarray): the wavelets
        snr_in (int): the level of noise

    Returns:
        np.ndarray: the sigma_j values for each wavelet
    """
    lm_axis = 1
    sigma_noise = compute_sigma_noise(signal, snr_in)
    wavelet_power = (np.abs(psi_j) ** 2).sum(axis=lm_axis)
    return sigma_noise * np.sqrt(wavelet_power)


def harmonic_hard_thresholding(
    L: int, wav_coeffs: np.ndarray, sigma_j: np.ndarray, n_sigma: int
) -> np.ndarray:
    """Thresholds the wavelet coefficients of the signal

    Args:
        L (int): bandlimit of the signal
        wav_coeffs (np.ndarray): the input wavelet coefficients
        sigma_j (np.ndarray): the noise level of each wavelet
        n_sigma (int): the number of sigma to threshold

    Returns:
        np.ndarray: the thresholded wavelet coefficients
    """
    logger.info("begin harmonic hard thresholding")
    # don't threshold the scaling function
    for j, coefficient in enumerate(wav_coeffs[1:]):
        logger.info(f"start Psi^{j + 1}/{len(wav_coeffs)-1}")

        # convert to pixel space
        f = ssht.inverse(coefficient, L)

        # threshold
        f_thresholded = _perform_hard_thresholding(f, sigma_j[j], n_sigma)

        # convert back
        wav_coeffs[j + 1] = ssht.forward(f_thresholded, L)
    return wav_coeffs


def _perform_hard_thresholding(
    f: np.ndarray, sigma_j: Union[float, np.ndarray], n_sigma: int
) -> np.ndarray:
    """Set pixels in real space to zero if the magnitude is less than
    the threshold

    Args:
        f (np.ndarray): the pixel values of the signal
        sigma_j (Union[float, np.ndarray]): the noise level of each wavelet
        n_sigma (int): number of sigma to threshold

    Returns:
        np.ndarray: the thresholded wavelt scale j
    """
    threshold = n_sigma * sigma_j
    return np.where(np.abs(f) < threshold, 0, f)
