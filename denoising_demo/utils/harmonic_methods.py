import numpy as np
import pyssht as ssht
from numpy.random import Generator


def boost_coefficient_resolution(flm: np.ndarray, boost: int) -> np.ndarray:
    """pads the harmonic coefficients with zeros which boosts the plot
    resolution after an inverse harmonic transform

    Args:
        flm (np.ndarray): harmonic coefficients of the signal
        boost (int): how many zeros to pad with

    Returns:
        np.ndarray: the original harmonic coefficients followed by some zeros
    """
    return np.pad(flm, (0, boost), "constant")


def invert_flm_boosted(flm: np.ndarray, L: int, resolution: int) -> np.ndarray:
    """Performs an inverse harmonic transform with a boost in resolution

    Args:
        flm (np.ndarray): harmonic coefficients of the signal
        L (int): bandlimit of the signal
        resolution (int): the desired final resolution

    Returns:
        np.ndarray: [description]
    """
    boost = resolution ** 2 - L ** 2
    flm = boost_coefficient_resolution(flm, boost)
    return ssht.inverse(flm, resolution)


def compute_random_signal(L: int, rng: Generator, var_signal: float) -> np.ndarray:
    """Generates a normally distributed random signal of a complex
    signal with mean 0 and variance 1

    Args:
        L (int): bandlimit of signal
        rng (Generator): random generator object
        var_signal (float): variance of the signal

    Returns:
        np.ndarray: the harmonic coefficients of a random signal
    """
    return np.sqrt(var_signal / 2) * (
        rng.standard_normal(L ** 2) + 1j * rng.standard_normal(L ** 2)
    )
