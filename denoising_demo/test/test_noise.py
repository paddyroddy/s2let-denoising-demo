from numpy.testing import assert_array_less

from denoising_demo.test.constants import J_MIN, L_SMALL, N_SIGMA, SNR_IN, B
from denoising_demo.utils.denoising import perform_denoising
from denoising_demo.utils.noise import compute_snr, create_noise
from denoising_demo.utils.wavelet_methods import create_axisymmetric_wavelets


def test_denoising_earth_axisymmetric_wavelets(earth) -> None:
    """tests that hard thresholding improves the SNR over the map"""
    # compute harmonic coefficients of the noise to create noised signal
    nlm = create_noise(L_SMALL, earth, SNR_IN)
    noised_snr = compute_snr(earth, nlm)
    noised_earth_flm = earth + nlm

    # create axisymmetric wavelets for hard-thresholding
    wavelets = create_axisymmetric_wavelets(L_SMALL, B, J_MIN)

    # denoise Earth signal
    denoised_earth_flm = perform_denoising(
        L_SMALL, earth, noised_earth_flm, wavelets, SNR_IN, N_SIGMA
    )
    denoised_snr = compute_snr(earth, denoised_earth_flm - earth)

    # check for SNR boost
    assert_array_less(noised_snr, denoised_snr)
