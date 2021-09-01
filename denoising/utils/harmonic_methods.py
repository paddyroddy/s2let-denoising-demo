import numpy as np
import pyssht as ssht
from pys2sleplet.utils.vars import SAMPLING_SCHEME


def boost_coefficient_resolution(flm: np.ndarray, boost: int) -> np.ndarray:
    """
    calculates a boost in resolution for given flm
    """
    return np.pad(flm, (0, boost), "constant")


def invert_flm_boosted(
    flm: np.ndarray, L: int, resolution: int, reality: bool = False, spin: int = 0
) -> np.ndarray:
    """
    performs the inverse harmonic transform
    """
    boost = resolution ** 2 - L ** 2
    flm = boost_coefficient_resolution(flm, boost)
    return ssht.inverse(
        flm, resolution, Reality=reality, Spin=spin, Method=SAMPLING_SCHEME
    )
