import numpy as np
import pyssht as ssht

from denoising_demo.utils.harmonic_methods import invert_flm_boosted
from denoising_demo.utils.logger import logger


def calc_plot_resolution(L: int) -> int:
    """Calculates a suitable resolution for a given bandlimit

    Args:
        L (int): bandlimit of the signal

    Returns:
        int: the new bandlimit
    """
    # found by manual tweaking
    res_dict = {1: 6, 2: 5, 3: 4, 7: 3, 9: 2, 10: 1}

    for log_bandlimit, exponent in res_dict.items():
        if L < 2 ** log_bandlimit:
            return L * 2 ** exponent

    # otherwise just use the bandlimit
    return L


def create_plot_type(field: np.ndarray, plot_type: str) -> np.ndarray:
    """Computes the appropriate value to plot

    Args:
        field (np.ndarray): pixel values of the signal
        plot_type (str): one of abs/imag/real/sum

    Returns:
        np.ndarray: the resultant plot type of the signal
    """
    logger.info(f"plotting type: '{plot_type}'")
    plot_dict = dict(
        abs=np.abs(field), imag=field.imag, real=field.real, sum=field.real + field.imag
    )
    return plot_dict[plot_type]


def boost_field(field: np.ndarray, L: int, resolution: int) -> np.ndarray:
    """Takes an input field and boosts the resolution of it

    Args:
        field (np.ndarray): the pixel values of a signal
        L (int): bandlimit of the signal
        resolution (int): the desired resolution

    Returns:
        np.ndarray: [description]
    """
    flm = ssht.forward(field, L)
    return invert_flm_boosted(flm, L, resolution)
