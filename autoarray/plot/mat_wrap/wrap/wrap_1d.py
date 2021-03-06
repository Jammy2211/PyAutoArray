from autoarray.plot.mat_wrap.wrap import wrap_base

wrap_base.set_backend()

import matplotlib.pyplot as plt
import numpy as np
import typing

from autoarray.structures import lines
from autoarray import exc


class AbstractMatWrap1D(wrap_base.AbstractMatWrap):
    """
    An abstract base class for wrapping matplotlib plotting methods which take as input and plot data structures. For
    example, the `ArrayOverlay` object specifically plots `Array` data structures.

    As full description of the matplotlib wrapping can be found in `mat_base.AbstractMatWrap`.
    """

    @property
    def config_folder(self):
        return "mat_wrap_1d"


class LinePlot(AbstractMatWrap1D):
    """
    Plots a `Line` data structure as a y vs x figure.

    This object wraps the following Matplotlib methods:

    - plt.plot: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html
    - plt.avxline: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axvline.html
    """

    def plot_y_vs_x(
        self,
        y: typing.Union[np.ndarray, lines.Line],
        x: typing.Union[np.ndarray, lines.Line],
        plot_axis_type: str,
        label: str = None,
    ):
        """
        Plots 1D y-data against 1D x-data using the matplotlib method `plt.plot`, `plt.semilogy`, `plt.loglog`,
        or `plt.scatter`.

        Parameters
        ----------
        y : np.ndarray or lines.Line
            The ydata that is plotted.
        x : np.ndarray or lines.Line
            The xdata that is plotted.
        plot_axis_type : str
            The method used to make the plot that defines the scale of the axes {"linear", "semilogy", "loglog",
            "scatter"}.
        label : str
            Optionally include a label on the plot for a `Legend` to display.
        """

        if plot_axis_type == "linear":
            plt.plot(x, y, label=label, **self.config_dict)
        elif plot_axis_type == "semilogy":
            plt.semilogy(x, y, label=label, **self.config_dict)
        elif plot_axis_type == "loglog":
            plt.loglog(x, y, label=label, **self.config_dict)
        elif plot_axis_type == "scatter":
            plt.scatter(x, y, label=label, **self.config_dict)
        else:
            raise exc.PlottingException(
                "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
                "{semilogy, loglog})"
            )

    def plot_vertical_lines(
        self,
        vertical_lines: typing.List[np.ndarray],
        vertical_line_labels: typing.List[str] = None,
    ):
        """
        Plots vertical lines on 1D plot of y versus x using the method `plt.axvline`.

        This method is typically called after `plot_y_vs_x` to add vertical lines to the figure.

        Parameters
        ----------
        vertical_lines : [np.ndarray]
            The vertical lines of data that are plotted on the figure.
        vertical_line_labels : [str]
            Labels for each vertical line used by a `Legend`.
        """

        if vertical_lines is [] or vertical_lines is None:
            return

        if vertical_line_labels is None:
            vertical_line_labels = [None for i in range(len(vertical_lines))]

        for vertical_line, vertical_line_label in zip(
            vertical_lines, vertical_line_labels
        ):

            plt.axvline(x=vertical_line, label=vertical_line_label, **self.config_dict)
