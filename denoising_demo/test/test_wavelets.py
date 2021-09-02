import numpy as np
from numpy.testing import assert_allclose, assert_equal
from pys2let import pys2let_j_max

from denoising_demo.test.constants import J_MIN, L_LARGE, L_SMALL, B
from denoising_demo.utils.wavelet_methods import (
    axisymmetric_wavelet_inverse,
    create_kappas,
)


def test_axisymmetric_synthesis(
    earth, axisymmetric_wavelets, axisymmetric_wavelet_coefficients_earth
) -> None:
    """Tests that the axisymmetric wavelet synthesis recoveres
    the signal
    """
    flm = axisymmetric_wavelet_inverse(
        L_SMALL, axisymmetric_wavelet_coefficients_earth, axisymmetric_wavelets
    )
    assert_allclose(np.abs(flm - earth).mean(), 0, atol=1e-13)


def test_create_kappas() -> None:
    """Checks that the method creates the scaling function and wavelets"""
    wavelets = create_kappas(L_LARGE ** 2, B, J_MIN)
    j_max = pys2let_j_max(B, L_LARGE ** 2, J_MIN)
    assert_equal(j_max - J_MIN + 2, wavelets.shape[0])
    assert_equal(L_LARGE ** 2, wavelets.shape[1])
