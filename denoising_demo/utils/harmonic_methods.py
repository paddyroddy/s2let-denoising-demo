import numpy as np
import pyssht as ssht


def _boost_coefficient_resolution(flm: np.ndarray, boost: int) -> np.ndarray:
    """pads the harmonic coefficients with zeros which boosts the plot
    resolution after an inverse harmonic transform

    Args:
        flm (np.ndarray): harmonic coefficients of the signal
        boost (int): how many zeros to pad with

    Returns:
        np.ndarray: the original harmonic coefficients followed by some zeros
    """
    return np.pad(flm, (0, boost), "constant")


def invert_flm_boosted(
    flm: np.ndarray, L: int, resolution: int, reality: bool = False
) -> np.ndarray:
    """Performs an inverse harmonic transform with a boost in resolution

    Args:
        flm (np.ndarray): harmonic coefficients of the signal
        L (int): bandlimit of the signal
        resolution (int): the desired final resolution
        reality (bool, optional): controls whether the signal is real or not.
        Defaults to False.

    Returns:
        np.ndarray: [description]
    """
    boost = resolution ** 2 - L ** 2
    flm = _boost_coefficient_resolution(flm, boost)
    return ssht.inverse(flm, resolution, Reality=reality)
