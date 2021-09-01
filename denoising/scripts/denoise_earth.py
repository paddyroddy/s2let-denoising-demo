from typing import Optional

import pyssht as ssht

from denoising.data.create_earth_flm import create_flm
from denoising.plotting.create_plot_sphere import Plot
from denoising.utils.vars import SAMPLING_SCHEME


def main(L: int, smoothing: Optional[int] = None) -> None:
    """[summary]"""
    flm = create_flm(L, smoothing)

    field = ssht.inverse(flm, L, Method=SAMPLING_SCHEME)

    Plot(
        field,
        L,
        "test",
    ).execute()


if __name__ == "__main__":
    main(128, 2)
