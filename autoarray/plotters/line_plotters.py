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
        pointsize=5,
        ticks=plotters.Ticks(),
        labels=plotters.Labels(),
        output=plotters.Output(),
    ):
        super(LinePlotter, self).__init__(
            is_sub_plotter=is_sub_plotter,
            use_scaled_units=use_scaled_units,
            unit_conversion_factor=unit_conversion_factor,
            figsize=figsize,
            aspect=aspect,
            ticks=ticks,
            labels=labels,
            output=output,
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

        self.ticks.set_xticks(
            array=None,
            extent=[np.min(x), np.max(x)],
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
        )

        self.output.to_figure(structure=None, is_sub_plotter=self.is_sub_plotter)

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
