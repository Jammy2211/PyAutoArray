from autoarray import exc

import matplotlib.pyplot as plt
import numpy as np

from autoarray.plotters import plotters


class LinePlotter(plotters.Plotter):
    def __init__(
        self,
        is_sub_plotter=False,
        use_scaled_units=None,
        unit_conversion_factor=None,
        include_legend=True,
        legend_fontsize=12,
        figsize=None,
        aspect=None,
        yticks=None,
        xticks=None,
        xyticksize=None,
        pointsize=5,
        labels=plotters.Labels(),
        output=plotters.Output()
    ):
        super(LinePlotter, self).__init__(
            is_sub_plotter=is_sub_plotter,
            use_scaled_units=use_scaled_units,
            unit_conversion_factor=unit_conversion_factor,
            figsize=figsize,
            aspect=aspect,
            yticks=yticks,
            xticks=xticks,
            xyticksize=xyticksize,
            labels=labels,
            output=output
        )

        self.pointsize = pointsize
        self.include_legend = include_legend
        self.legend_fontsize = legend_fontsize

    def plot_line(
        self,
        y,
        x,
        label=None,
        plot_axis_type="semilogy",
        vertical_lines=None,
        vertical_line_labels=None,
    ):

        if y is None:
            return

        self.setup_figure()
        self.labels.set_title()

        if x is None:
            x = np.arange(len(y))

        self.plot_y_vs_x(y=y, x=x, plot_axis_type=plot_axis_type, label=label)

        self.labels.set_yunits(include_brackets=False)
        self.labels.set_xunits(include_brackets=False)

        self.plot_vertical_lines(
            vertical_lines=vertical_lines, vertical_line_labels=vertical_line_labels
        )

        self.set_legend()

        self.set_xticks(extent=[np.min(x), np.max(x)])

        if self.output.format is not "fits":

            self.output_figure(array=None)

        self.close_figure()

    def plot_y_vs_x(self, y, x, plot_axis_type, label):

        if plot_axis_type is "linear":
            plt.plot(x, y, label=label)
        elif plot_axis_type is "semilogy":
            plt.semilogy(x, y, label=label)
        elif plot_axis_type is "loglog":
            plt.loglog(x, y, label=label)
        elif plot_axis_type is "scatter":
            plt.scatter(x, y, label=label, s=self.pointsize)
        else:
            raise exc.PlottingException(
                "The plot_axis_type supplied to the plotter is not a valid string (must be linear "
                "| semilogy | loglog)"
            )

    def set_xticks(self, extent):
        """Get the extent of the dimensions of the array in the unit_label of the figure (e.g. arc-seconds or kpc).

        This is used to set the extent of the array and thus the y / x axis limits.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        yticks_manual :  [] or None
            If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
        """

        plt.tick_params(labelsize=self.xyticksize)

        xticks = np.round(np.linspace(extent[0], extent[1], 5), 2)

        if self.xticks is not None:
            xtick_labels = np.asarray([self.xticks[0], self.xticks[3]])
        elif self.unit_conversion_factor is None:
            xtick_labels = np.round(np.linspace(extent[0], extent[1], 5), 2)
        elif self.unit_conversion_factor is not None:
            xtick_labels = (
                np.round(np.linspace(extent[0], extent[1], 5), 2)
                * self.unit_conversion_factor
            )
        else:
            raise exc.PlottingException(
                "The y and y ticks cannot be set using the input options."
            )

        plt.xticks(ticks=xticks, labels=xtick_labels)

    def plot_vertical_lines(self, vertical_lines, vertical_line_labels):

        if vertical_lines is [] or vertical_lines is None:
            return

        for vertical_line, vertical_line_label in zip(
            vertical_lines, vertical_line_labels
        ):

            if self.unit_conversion_factor is None:
                x_value_plot = vertical_line
            else:
                x_value_plot = vertical_line * self.unit_conversion_factor

            plt.axvline(x=x_value_plot, label=vertical_line_label, linestyle="--")

    def set_legend(self):
        if self.include_legend:
            plt.legend(fontsize=self.legend_fontsize)
