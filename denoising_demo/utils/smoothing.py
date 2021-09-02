import numpy as np
import pyssht as ssht

from denoising_demo.utils.logger import logger


def apply_gaussian_smoothing(
    flm: np.ndarray, L: int, smoothing_factor: int
) -> np.ndarray:
    """Applies Gaussian smoothing to the given signal

    Args:
        flm (np.ndarray): the initial harmonic coefficients of the signal
        L (int): the bandlimit of the signal
        smoothing_factor (int): the level of smoothing used for the sigma

    Returns:
        np.ndarray: the smoothed harmonic coefficients
    """
    sigma = np.pi / (smoothing_factor * L)
    fwhm = 2 * np.sqrt(np.log(2)) * sigma
    logger.info(f"FWHM = {np.rad2deg(fwhm):.2f}degrees")
    return ssht.gaussian_smoothing(flm, L, sigma)
