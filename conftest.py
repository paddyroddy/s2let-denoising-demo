import numpy as np
import pytest
from numpy.random import default_rng

from denoising_demo.data.create_earth_flm import create_flm
from denoising_demo.test.constants import J_MIN, L_SMALL, B
from denoising_demo.utils.harmonic_methods import compute_random_signal
from denoising_demo.utils.vars import RANDOM_SEED
from denoising_demo.utils.wavelet_methods import (
    axisymmetric_wavelet_forward,
    create_axisymmetric_wavelets,
)


@pytest.fixture(scope="session")
def earth() -> np.ndarray:
    """Creates the harmonic coefficients of the topography of Earth

    Returns:
        np.ndarray: the harmonic coefficients
    """
    return create_flm(L_SMALL)


@pytest.fixture(scope="session")
def axisymmetric_wavelets() -> np.ndarray:
    """Creates some axisymmetric wavelets

    Returns:
        np.ndarray: the wavelets
    """
    return create_axisymmetric_wavelets(L_SMALL, B, J_MIN)


@pytest.fixture(scope="session")
def axisymmetric_wavelet_coefficients_earth(earth, axisymmetric_wavelets) -> np.ndarray:
    """Computes the axisymmetric wavelet coefficients of Earth

    Args:
        axisymmetric_wavelets ([type]): the wavelets

    Returns:
        np.ndarray: the wavelet coefficients
    """
    return axisymmetric_wavelet_forward(L_SMALL, earth, axisymmetric_wavelets)


@pytest.fixture(scope="session")
def random_flm() -> np.ndarray:
    """
    creates random flm
    """
    rng = default_rng(RANDOM_SEED)
    return compute_random_signal(L_SMALL, rng, 1)
