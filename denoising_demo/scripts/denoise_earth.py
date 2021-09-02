import pyssht as ssht

from denoising_demo.data.create_earth_flm import create_flm
from denoising_demo.plotting.create_plot_sphere import Plot
from denoising_demo.utils.denoising import perform_denoising
from denoising_demo.utils.noise import compute_snr, create_noise
from denoising_demo.utils.wavelet_methods import create_axisymmetric_wavelets

B = 2
J_MIN = 0
L = 128
N_SIGMA = 3
SMOOTHING = 2
SNR_IN = 10


def main() -> None:
    """[summary]"""
    # create initial smoothed Earth topography
    earth_flm = create_flm(L, SMOOTHING)

    # compute harmonic coefficients of the noise to create noised signal
    nlm = create_noise(L, earth_flm, SNR_IN)
    compute_snr(earth_flm, nlm)
    noised_earth_flm = earth_flm + nlm

    # create axisymmetric wavelets for hard-thresholding
    wavelets = create_axisymmetric_wavelets(L, B, J_MIN)

    # denoise Earth signal
    denoised_earth_flm = perform_denoising(
        L, earth_flm, noised_earth_flm, wavelets, SNR_IN, N_SIGMA
    )

    # create dict to loop over
    fields_dict = {
        "earth": earth_flm,
        "noised_earth": noised_earth_flm,
        "denoised_earth": denoised_earth_flm,
    }

    # produce three plots
    for name, flm in fields_dict.items():
        # convert to pixel space
        field = ssht.inverse(flm, L)

        # perform plot
        Plot(field, L, name).execute()


if __name__ == "__main__":
    main()
