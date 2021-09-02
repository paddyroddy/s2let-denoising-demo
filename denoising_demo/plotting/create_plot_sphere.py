from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.offline as py
import pyssht as ssht
from plotly.graph_objs import Figure, Surface
from plotly.graph_objs.surface import Lighting

from denoising_demo.utils.plot_methods import (
    boost_field,
    calc_plot_resolution,
    create_plot_type,
)
from denoising_demo.utils.plotly_methods import (
    create_camera,
    create_colour_bar,
    create_layout,
    create_tick_mark,
)
from denoising_demo.utils.vars import UNSEEN

_file_location = Path(__file__).resolve()
_fig_path = _file_location.parents[1] / "figures"


@dataclass
class Plot:
    f: np.ndarray = field(repr=False)
    L: int
    filename: str
    plot_type: str = field(default="real", repr=False)

    def __post_init__(self) -> None:
        self.resolution = calc_plot_resolution(self.L)

    def execute(self) -> None:
        """Perfoms the plotly plot using a 3D surface
        the plot will open in a browser as a HTML
        """
        f = self._prepare_field(self.f)

        # get values from the setup
        x, y, z, f_plot, vmin, vmax = self._setup_plot(f, self.resolution)

        # appropriate zoom in on north pole
        camera = create_camera(-0.1, -0.1, 10, 7.88)

        # pick largest tick max value
        tick_mark = create_tick_mark(vmin, vmax)

        # create the plotly figure
        data = [
            Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=f_plot,
                cmax=tick_mark,
                cmid=0,
                cmin=-tick_mark,
                colorbar=create_colour_bar(tick_mark),
                lighting=Lighting(ambient=1),
                reversescale=True,
            )
        ]
        layout = create_layout(camera)
        fig = Figure(data=data, layout=layout)

        # create html and plot offline
        html_filename = str(_fig_path / f"{self.filename}.html")
        py.plot(fig, filename=html_filename)

    @staticmethod
    def _setup_plot(
        f: np.ndarray,
        resolution: int,
        method: str = "MW",
        close: bool = True,
        parametric: bool = False,
        parametric_scaling: list[float] = [0.0, 0.5],
        color_range: Optional[list[float]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Computes the values necessary to perform plot

        Args:
            f (np.ndarray): pixel values of the signal
            resolution (int): the desired resolution
            method (str, optional): the sampling scheme. Defaults to "MW".
            close (bool, optional): whether to close up the samples.
            Defaults to True.
            parametric (bool, optional): whether to plot the parametric
            version. Defaults to False.
            parametric_scaling (list[float], optional): scaling of the
            parametric plot. Defaults to [0.0, 0.5].
            color_range (Optional[list[float]], optional): control what range
            of the plot to be seen on the colour bar values. Defaults to None.

        Raises:
            ValueError: checks the signal size matches the resolution

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
            the x, y, z values of the samples in Cartesian space
            the signal over the samples,
            the min and max values of the signal
        """
        # handle this sampling scheme differently
        if method == "MW_pole":
            if len(f) == 2:
                f, _ = f
            else:
                f, _, _ = f

        # generate samples
        thetas, phis = ssht.sample_positions(resolution, Grid=True, Method=method)
        if thetas.size != f.size:
            raise ValueError("Bandlimit L deos not match that of f")

        # find colour range of plot
        f_plot = f.copy()
        f_max = f_plot.max()
        f_min = f_plot.min()

        if not isinstance(color_range, list):
            vmin = f_min
            vmax = f_max
        else:
            vmin = color_range[0]
            vmax = color_range[1]
            f_plot[f_plot < color_range[0]] = color_range[0]
            f_plot[f_plot > color_range[1]] = color_range[1]
            f_plot[f_plot == UNSEEN] = np.nan

        # Compute position scaling for parametric plot.
        f_normalised = (
            (f_plot - vmin / (vmax - vmin)) * parametric_scaling[1]
            + parametric_scaling[0]
            if parametric
            else np.zeros(f_plot.shape)
        )

        # Close plot.
        if close:
            first_row, phi_index = 0, 1
            _, n_phi = ssht.sample_shape(resolution, Method=method)
            f_plot = np.insert(f_plot, n_phi, f[:, first_row], axis=phi_index)
            if parametric:
                f_normalised = np.insert(
                    f_normalised, n_phi, f_normalised[:, first_row], axis=phi_index
                )
            thetas = np.insert(thetas, n_phi, thetas[:, first_row], axis=phi_index)
            phis = np.insert(phis, n_phi, phis[:, first_row], axis=phi_index)

        # Compute location of vertices.
        x, y, z = (
            ssht.spherical_to_cart(f_normalised, thetas, phis)
            if parametric
            else ssht.s2_to_cart(thetas, phis)
        )
        return x, y, z, f_plot, vmin, vmax

    def _prepare_field(self, f: np.ndarray) -> np.ndarray:
        """Boosts and calculates the plot type before plotting

        Args:
            f (np.ndarray): the pixel values of the signal

        Returns:
            np.ndarray: boosted and i.e. 'real' part of signal
        """
        boosted_field = boost_field(f, self.L, self.resolution)
        return create_plot_type(boosted_field, self.plot_type)
