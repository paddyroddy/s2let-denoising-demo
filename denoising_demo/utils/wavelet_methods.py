import numpy as np
import pyssht as ssht
from pys2let import axisym_wav_l


def axisymmetric_wavelet_forward(
    L: int, flm: np.ndarray, wavelets: np.ndarray
) -> np.ndarray:
    """Computes the axisymmetric wavelet forward transform

    Args:
        L (int): bandlimit of signal
        flm (np.ndarray): harmonic coefficients of the signal
        wavelets (np.ndarray): the axisymmetric wavelets

    Returns:
        np.ndarray: the wavelet coefficients of the signal
    """
    # initialise wavelet coefficients
    w = np.zeros(wavelets.shape, dtype=np.complex_)

    # fill in wavelet coefficient values
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
    """Computes the axisymmetric wavelet inverse transform

    Args:
        L (int): bandlimit of the signal
        wav_coeffs (np.ndarray): the wavelet coefficients
        wavelets (np.ndarray): axisymmetric wavelets

    Returns:
        np.ndarray: the signal reconstructed from its wavelet coefficients
    """
    # initialise signal
    flm = np.zeros(L ** 2, dtype=np.complex_)

    # fill in the signal values
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * wavelets[:, ind_m0]
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = (wav_coeffs[:, ind] * wav_0).sum()
    return flm


def create_axisymmetric_wavelets(L: int, B: int, j_min: int) -> np.ndarray:
    """Construct wavelets from a tiling of the harmonic line

    Args:
        L (int): bandlimit of the signal
        B (int): positive real parameter
        j_min (int): controls the lowest wavelet scale

    Returns:
        np.ndarray: the harmonic coefficients of the axisymmetric wavelets
    """
    # create harmonic tiling
    kappas = create_kappas(L, B, j_min)

    # initialise empty wavelets
    wavelets = np.zeros((kappas.shape[0], L ** 2), dtype=np.complex_)

    # fill in wavelet values
    for ell in range(L):
        factor = np.sqrt((2 * ell + 1) / (4 * np.pi))
        ind = ssht.elm2ind(ell, 0)
        wavelets[:, ind] = factor * kappas[:, ell]
    return wavelets


def create_kappas(xlim: int, B: int, j_min: int) -> np.ndarray:
    """Computes the tiling of the wavelets

    Args:
        xlim (int): the bandlimit of the harmonic line
        B (int): positive real parameter
        j_min (int): minimum wavelet scale

    Returns:
        np.ndarray: a tiling of the harmonic line for scaling function
        and wavelets
    """
    kappa0, kappa = axisym_wav_l(B, xlim, j_min)
    return np.concatenate((kappa0[np.newaxis], kappa.T))
