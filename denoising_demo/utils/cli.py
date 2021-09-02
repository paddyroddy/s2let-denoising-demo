from argparse import ArgumentParser, Namespace

from denoising_demo.utils.vars import (
    B_DEFAULT,
    J_MIN_DEFAULT,
    L_DEFAULT,
    N_SIGMA_DEFAULT,
    SMOOTHING_DEFAULT,
    SNR_IN_DEFAULT,
)


def read_args() -> Namespace:
    """Method to read arguments from the command line

    Returns:
        Namespace: an argparse Namespace object
    """
    parser = ArgumentParser(description="Create denoising plots")
    parser.add_argument(
        "--bandlimit", "-L", type=int, default=L_DEFAULT, help="bandlimit"
    )
    parser.add_argument(
        "--jmin",
        "-j",
        type=int,
        default=J_MIN_DEFAULT,
        help="the minimum wavelet scale",
    )
    parser.add_argument(
        "--parameter",
        "-B",
        type=int,
        default=B_DEFAULT,
        help="the positive real parameter",
    )
    parser.add_argument(
        "--noise",
        "-n",
        type=int,
        default=SNR_IN_DEFAULT,
        help="the SNR_IN of the noise level",
    )
    parser.add_argument(
        "--sigma",
        "-si",
        type=int,
        default=N_SIGMA_DEFAULT,
    )
    parser.add_argument(
        "--smoothing",
        "-sm",
        type=int,
        default=SMOOTHING_DEFAULT,
        help="the scaling of the sigma in Gaussian smoothing of the Earth",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        nargs="?",
        default="real",
        const="real",
        choices=["abs", "real", "imag", "sum"],
        help="plotting type: defaults to real",
    )
    return parser.parse_args()
