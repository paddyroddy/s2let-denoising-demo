import numpy as np
import pyssht as ssht

from denoising_demo.utils.noise import (
    compute_sigma_j,
    compute_snr,
    harmonic_hard_thresholding,
)
from denoising_demo.utils.plot_methods import rotate_earth_to_south_america
from denoising_demo.utils.vars import SAMPLING_SCHEME
from denoising_demo.utils.wavelet_methods import (
    axisymmetric_wavelet_forward,
    axisymmetric_wavelet_inverse,
)


def denoising_axisym(
    L: int,
    signal: np.ndarray,
    noised_signal: np.ndarray,
    axisymmetric_wavelets: np.ndarray,
    snr_in: int,
    n_sigma: int,
) -> tuple[np.ndarray, float, float]:
    """
    reproduce the denoising demo from s2let paper
    """
    # compute wavelet coefficients
    w = axisymmetric_wavelet_forward(L, noised_signal, axisymmetric_wavelets)

    # compute wavelet noise
    sigma_j = compute_sigma_j(signal, axisymmetric_wavelets[1:], snr_in)

    # hard thresholding
    w_denoised = harmonic_hard_thresholding(L, w, sigma_j, n_sigma)

    # wavelet synthesis
    flm = axisymmetric_wavelet_inverse(L, w_denoised, axisymmetric_wavelets)

    # rotate to South America
    flm_rot = rotate_earth_to_south_america(flm, L)

    # compute SNR
    compute_snr(signal, flm - signal)

    return ssht.inverse(flm_rot, L, Method=SAMPLING_SCHEME)
