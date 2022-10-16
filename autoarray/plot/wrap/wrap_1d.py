from autoarray.plot.wrap.wrap_base import set_backend

set_backend()

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union

from autoarray.plot.wrap.wrap_base import AbstractMatWrap
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray import exc


class AbstractMatWrap1D(AbstractMatWrap):
    """
    An abstract base class for wrapping matplotlib plotting methods which take as input and plot data structures. For
    example, the `ArrayOverlay` object specifically plots `Array2D` data structures.

    As full description of the matplotlib wrapping can be found in `mat_base.AbstractMatWrap`.
    """

    @property
    def config_folder(self):
        return "mat_wrap_1d"


class YXPlot(AbstractMatWrap1D):
    def __init__(self, plot_axis_type=None, label=None, **kwargs):
        """
        Plots 1D data structures as a y vs x figure.

        This object wraps the following Matplotlib methods:

        - plt.plot: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html
        """

        super().__init__(**kwargs)

        self.plot_axis_type = plot_axis_type
        self.label = label

    def plot_y_vs_x(
        self,
        y: Union[np.ndarray, Array1D],
        x: Union[np.ndarray, Array1D],
        label: str = None,
        plot_axis_type=None,
        y_errors=None,
        x_errors=None,
    ):
        """
        Plots 1D y-data against 1D x-data using the matplotlib method `plt.plot`, `plt.semilogy`, `plt.loglog`,
        or `plt.scatter`.

        Parameters
        ----------
        y
            The ydata that is plotted.
        x
            The xdata that is plotted.
        plot_axis_type
            The method used to make the plot that defines the scale of the axes {"linear", "semilogy", "loglog",
            "scatter"}.
        label
            Optionally include a label on the plot for a `Legend` to display.
        """

        if self.label is not None:
            label = self.label

        if plot_axis_type == "linear" or plot_axis_type == "symlog":
            plt.plot(x, y, label=label, **self.config_dict)
        elif plot_axis_type == "semilogy":
            plt.semilogy(x, y, label=label, **self.config_dict)
        elif plot_axis_type == "loglog":
            plt.loglog(x, y, label=label, **self.config_dict)
        elif plot_axis_type == "scatter":
            plt.scatter(x, y, label=label, **self.config_dict)
        elif plot_axis_type == "errorbar":
            plt.errorbar(
                x,
                y,
                yerr=y_errors,
                xerr=x_errors,
                marker="o",
                fmt="o",
                **self.config_dict
            )
        else:
            raise exc.PlottingException(
                "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
                "{semilogy, loglog})"
            )


class YXScatter(AbstractMatWrap1D):
    def __init__(self, **kwargs):
        """
        Scatters a 1D set of points on a 1D plot. Unlike the `YXPlot` object these are scattered over an existing plot.

        This object wraps the following Matplotlib methods:

        - plt.scatter: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html
        """

        super().__init__(**kwargs)

    def scatter_yx(self, y: Union[np.ndarray, Grid1D], x: list):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        Parameters
        ----------
        grid
            The points that are
        errors
            The error on every point of the grid that is plotted.
        """

        config_dict = self.config_dict

        if len(config_dict["c"]) > 1:
            config_dict["c"] = config_dict["c"][0]

        plt.scatter(y=y, x=x, **config_dict)


class AXVLine(AbstractMatWrap1D):
    def __init__(self, no_label=False, **kwargs):
        """
        Plots vertical lines on 1D plot of y versus x using the method `plt.axvline`.

        This method is typically called after `plot_y_vs_x` to add vertical lines to the figure.

        This object wraps the following Matplotlib methods:

        - plt.avxline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html

        Parameters
        ----------
        vertical_line
            The vertical lines of data that are plotted on the figure.
        label
            Labels for each vertical line used by a `Legend`.
        """
        super().__init__(**kwargs)

        self.no_label = no_label

    def axvline_vertical_line(
        self,
        vertical_line: float,
        vertical_errors: Optional[List[float]] = None,
        label: Optional[str] = None,
    ):
        """
        Plot an input vertical line given by its x coordinate as a float using the method `plt.axvline`.

        Parameters
        ----------
        vertical_line
            The vertical lines of data that are plotted on the figure.
        label
            Labels for each vertical line used by a `Legend`.
        """

        if vertical_line is [] or vertical_line is None:
            return

        if self.no_label:
            label = None

        plt.axvline(x=vertical_line, label=label, **self.config_dict)

        if vertical_errors is not None:

            config_dict = self.config_dict

            if "linestyle" in config_dict:
                config_dict.pop("linestyle")

            plt.axvline(x=vertical_errors[0], linestyle="--", **config_dict)
            plt.axvline(x=vertical_errors[1], linestyle="--", **config_dict)


class FillBetween(AbstractMatWrap1D):
    def __init__(self, match_color_to_yx: bool = True, **kwargs):
        """
        Fills between two lines on a 1D plot of y versus x using the method `plt.fill_between`.

        This method is typically called after `plot_y_vs_x` to add a shaded region to the figure.

        This object wraps the following Matplotlib methods:

        - plt.fill_between: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.fill_between.html

        Parameters
        ----------
        match_color_to_yx
            If True, the color of the shaded region is automatically matched to that of the yx line that is plotted,
            irrespective of the user inputs.
        """
        super().__init__(**kwargs)
        self.match_color_to_yx = match_color_to_yx

    def fill_between_shaded_regions(
        self,
        x: Union[np.ndarray, Array1D, List],
        y1: Union[np.ndarray, Array1D, List],
        y2: Union[np.ndarray, Array1D, List],
    ):
        """
        Fill in between two lines `y1` and `y2` on a plot of y vs x.

        Parameters
        ----------
        x
            The xdata that is plotted.
        y1
            The first line of ydata that defines the region that is filled in.
        y1
            The second line of ydata that defines the region that is filled in.
        """

        config_dict = self.config_dict

        if self.match_color_to_yx:

            config_dict["color"] = plt.gca().lines[-1].get_color()

        plt.fill_between(x=x, y1=y1, y2=y2, **config_dict)
