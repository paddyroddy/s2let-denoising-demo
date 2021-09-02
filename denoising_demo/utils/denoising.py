import numpy as np

from denoising_demo.utils.noise import (
    compute_sigma_j,
    compute_snr,
    harmonic_hard_thresholding,
)
from denoising_demo.utils.wavelet_methods import (
    axisymmetric_wavelet_forward,
    axisymmetric_wavelet_inverse,
)


def perform_denoising(
    L: int,
    signal: np.ndarray,
    noised_signal: np.ndarray,
    axisymmetric_wavelets: np.ndarray,
    snr_in: int,
    n_sigma: int,
) -> np.ndarray:
    """Performs signal denoising through hard-thresholding

    Args:
        L (int): bandlimit of the signal
        signal (np.ndarray): harmonic coefficients of signal
        noised_signal (np.ndarray): noised harmonic coefficients of the signal
        axisymmetric_wavelets (np.ndarray): the axisymmetric wavelets
        snr_in (int): the desired level of noise
        n_sigma (int): how many sigmas of noise to threshold

    Returns:
        np.ndarray: the denoised harmonic coefficients of the noised signal
    """
    # compute wavelet coefficients
    w = axisymmetric_wavelet_forward(L, noised_signal, axisymmetric_wavelets)

    # compute wavelet noise
    sigma_j = compute_sigma_j(signal, axisymmetric_wavelets[1:], snr_in)

    # hard thresholding
    w_denoised = harmonic_hard_thresholding(L, w, sigma_j, n_sigma)

    # wavelet synthesis
    flm = axisymmetric_wavelet_inverse(L, w_denoised, axisymmetric_wavelets)

    # compute SNR
    compute_snr(signal, flm - signal)
    return flm
