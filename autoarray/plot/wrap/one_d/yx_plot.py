import matplotlib.pyplot as plt
import numpy as np
from typing import Union

from autoarray.plot.wrap.one_d.abstract import AbstractMatWrap1D
from autoarray.structures.arrays.uniform_1d import Array1D

from autoarray import exc


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
        y_extra=None,
        ls_errorbar="",
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
        elif plot_axis_type == "errorbar" or plot_axis_type == "errorbar_logy":
            plt.errorbar(
                x,
                y,
                yerr=y_errors,
                xerr=x_errors,
                #     marker="o",
                fmt="o",
                # ls=ls_errorbar,
                **self.config_dict
            )
            if plot_axis_type == "errorbar_logy":
                plt.yscale("log")
        else:
            raise exc.PlottingException(
                "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
                "{semilogy, loglog})"
            )

        if y_extra is not None:
            plt.plot(x, y_extra, c="r")
