import numpy as np
import pytest

from denoising_demo.data.create_earth_flm import create_flm
from denoising_demo.test.constants import J_MIN, SMOOTHING, B, L
from denoising_demo.utils.wavelet_methods import (
    axisymmetric_wavelet_forward,
    create_axisymmetric_wavelets,
)


@pytest.fixture(scope="session")
def earth_smoothed() -> np.ndarray:
    """Creates the harmonic coefficients of a smoothed Earth

    Returns:
        np.ndarray: the harmonic coefficients
    """
    return create_flm(L, SMOOTHING)


@pytest.fixture(scope="session")
def axisymmetric_wavelets() -> np.ndarray:
    """Creates some axisymmetric wavelets

    Returns:
        np.ndarray: the wavelets
    """
    return create_axisymmetric_wavelets(L, B, J_MIN)


@pytest.fixture(scope="session")
def axisymmetric_wavelet_coefficients_earth(
    earth_smoothed, axisymmetric_wavelets
) -> np.ndarray:
    """Computes the axisymmetric wavelet coefficients of Earth

    Args:
        axisymmetric_wavelets ([type]): the wavelets

    Returns:
        np.ndarray: the wavelet coefficients
    """
    return axisymmetric_wavelet_forward(L, earth_smoothed, axisymmetric_wavelets)
