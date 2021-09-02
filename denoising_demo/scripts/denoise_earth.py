import pyssht as ssht

from denoising_demo.data.create_earth_flm import create_flm
from denoising_demo.plotting.create_plot_sphere import Plot
from denoising_demo.utils.cli import read_args
from denoising_demo.utils.denoising import perform_denoising
from denoising_demo.utils.logger import logger
from denoising_demo.utils.noise import compute_snr, create_noise
from denoising_demo.utils.wavelet_methods import create_axisymmetric_wavelets


def main() -> None:
    """Performs a denoising akin to figure 5 of the S2LET paper"""
    # read in command line arguments
    args = read_args()

    # create initial smoothed Earth topography
    earth_flm = create_flm(args.bandlimit)

    # compute harmonic coefficients of the noise to create noised signal
    nlm = create_noise(args.bandlimit, earth_flm, args.noise)
    compute_snr(earth_flm, nlm)
    noised_earth_flm = earth_flm + nlm

    # create axisymmetric wavelets for hard-thresholding
    wavelets = create_axisymmetric_wavelets(args.bandlimit, args.parameter, args.jmin)

    # denoise Earth signal
    denoised_earth_flm = perform_denoising(
        args.bandlimit, earth_flm, noised_earth_flm, wavelets, args.noise, args.sigma
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
        field = ssht.inverse(flm, args.bandlimit)

        # perform plot
        logger.info(f"producing the '{name}' plot")
        Plot(field, args.bandlimit, name).execute()


if __name__ == "__main__":
    main()
