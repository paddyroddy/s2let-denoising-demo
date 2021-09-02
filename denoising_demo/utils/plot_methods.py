import numpy as np
import pyssht as ssht
from matplotlib import colors

from denoising_demo.utils.harmonic_methods import invert_flm_boosted
from denoising_demo.utils.logger import logger
from denoising_demo.utils.vars import (
    EARTH_ALPHA,
    EARTH_BETA,
    EARTH_GAMMA,
    SAMPLING_SCHEME,
)


def calc_plot_resolution(L: int) -> int:
    """
    calculate appropriate resolution for given L
    """
    res_dict = {1: 6, 2: 5, 3: 4, 7: 3, 9: 2, 10: 1}

    for log_bandlimit, exponent in res_dict.items():
        if L < 2 ** log_bandlimit:
            return L * 2 ** exponent

    # otherwise just use the bandlimit
    return L


def convert_colourscale(cmap: colors, pl_entries: int = 255) -> list[tuple[float, str]]:
    """
    converts cmocean colourscale to a plotly colourscale
    """
    h = 1 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append((k * h, f"rgb{(C[0], C[1], C[2])}"))

    return pl_colorscale


def create_plot_type(field: np.ndarray, plot_type: str) -> np.ndarray:
    """
    gets the given plot type of the field
    """
    logger.info(f"plotting type: '{plot_type}'")
    plot_dict = dict(
        abs=np.abs(field), imag=field.imag, real=field.real, sum=field.real + field.imag
    )
    return plot_dict[plot_type]


def rotate_earth_to_south_america(earth_flm: np.ndarray, L: int) -> np.ndarray:
    """
    rotates the flms of the Earth to a view centered on South America
    """
    return ssht.rotate_flms(earth_flm, EARTH_ALPHA, EARTH_BETA, EARTH_GAMMA, L)


def boost_field(
    field: np.ndarray, L: int, resolution: int, reality: bool = False
) -> np.ndarray:
    """
    inverts and then boosts the field before plotting
    """
    flm = ssht.forward(field, L, Reality=reality, Method=SAMPLING_SCHEME)
    return invert_flm_boosted(flm, L, resolution, reality=reality)
