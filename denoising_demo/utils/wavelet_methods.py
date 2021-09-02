import numpy as np
import pyssht as ssht
from pys2let import axisym_wav_l


def axisymmetric_wavelet_forward(
    L: int, flm: np.ndarray, wavelets: np.ndarray
) -> np.ndarray:
    """
    computes the coefficients of the axisymmetric wavelets
    """
    w = np.zeros(wavelets.shape, dtype=np.complex_)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * wavelets[:, ind_m0].conj()
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            w[:, ind] = wav_0 * flm[ind]
    return w


def axisymmetric_wavelet_inverse(
    L: int, wav_coeffs: np.ndarray, wavelets: np.ndarray
) -> np.ndarray:
    """
    computes the inverse axisymmetric wavelet transform
    """
    flm = np.zeros(L ** 2, dtype=np.complex_)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * wavelets[:, ind_m0]
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = (wav_coeffs[:, ind] * wav_0).sum()
    return flm


def create_axisymmetric_wavelets(L: int, B: int, j_min: int) -> np.ndarray:
    """
    computes the axisymmetric wavelets
    """
    kappas = create_kappas(L, B, j_min)
    wavelets = np.zeros((kappas.shape[0], L ** 2), dtype=np.complex_)
    for ell in range(L):
        factor = np.sqrt((2 * ell + 1) / (4 * np.pi))
        ind = ssht.elm2ind(ell, 0)
        wavelets[:, ind] = factor * kappas[:, ell]
    return wavelets


def create_kappas(xlim: int, B: int, j_min: int) -> np.ndarray:
    """
    computes the Slepian wavelets
    """
    kappa0, kappa = axisym_wav_l(B, xlim, j_min)
    return np.concatenate((kappa0[np.newaxis], kappa.T))
