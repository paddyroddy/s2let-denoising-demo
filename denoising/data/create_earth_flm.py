from pathlib import Path
from typing import Optional

import numpy as np
import pyssht as ssht
from scipy import io as sio

from denoising.utils.smoothing import apply_gaussian_smoothing

_file_location = Path(__file__).resolve()
_matfile = _file_location.parent / "EGM2008_Topography_flms_L2190.mat"


def create_flm(L: int, smoothing: Optional[int] = None) -> np.ndarray:
    """The harmonic coefficients of the topography of the Earth are read in
    and the missing values are filled in. Smoothing is optionally performed
    on the data.

    Args:
        L (int): bandlimit of the data
        smoothing (Optional[int], optional): the level of smoothing desired.
        Defaults to None.

    Returns:
        np.ndarray: the harmonic coefficients of the data
    """
    # load in data
    flm = _load_flm()

    # fill in negative m components so as to avoid confusion with zero values
    for ell in range(1, L):
        for m in range(1, ell + 1):
            ind_pm = ssht.elm2ind(ell, m)
            ind_nm = ssht.elm2ind(ell, -m)
            flm_pm = flm[ind_pm]
            flm[ind_nm] = (-1) ** m * flm_pm.conj()

    # don't take the full L, invert dataset as Earth backwards
    flm = flm[: L ** 2].conj()

    if isinstance(smoothing, int):
        flm = apply_gaussian_smoothing(flm, L, smoothing)
    return flm


def _load_flm() -> np.ndarray:
    """Reads in the MATLAB binary and converts it to numpy format

    Returns:
        np.ndarray: the numpy array of harmonic coefficients
    """
    mat_contents = sio.loadmat(str(_matfile))
    return np.ascontiguousarray(mat_contents["flm"][:, 0])
