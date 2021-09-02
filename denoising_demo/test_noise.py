from numpy.testing import assert_array_less

from denoising_demo.test.constants import J_MIN, N_SIGMA, SNR_IN, B, L
from denoising_demo.utils.denoising import perform_denoising
from denoising_demo.utils.noise import compute_snr, create_noise
from denoising_demo.utils.wavelet_methods import create_axisymmetric_wavelets


def test_denoising_earth_axisymmetric_wavelets(earth_smoothed) -> None:
    """tests that hard thresholding improves the SNR over the map"""
    # compute harmonic coefficients of the noise to create noised signal
    nlm = create_noise(L, earth_smoothed, SNR_IN)
    noised_snr = compute_snr(earth_smoothed, nlm)
    noised_earth_flm = earth_smoothed + nlm

    # create axisymmetric wavelets for hard-thresholding
    wavelets = create_axisymmetric_wavelets(L, B, J_MIN)

    # denoise Earth signal
    denoised_earth_flm = perform_denoising(
        L, earth_smoothed, noised_earth_flm, wavelets, SNR_IN, N_SIGMA
    )
    denoised_snr = compute_snr(earth_smoothed, denoised_earth_flm - earth_smoothed)

    # check for SNR boost
    assert_array_less(noised_snr, denoised_snr)
