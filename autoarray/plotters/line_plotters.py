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
        titlesize=None,
        xlabelsize=None,
        ylabelsize=None,
        xyticksize=None,
        pointsize=5,
        label_title=None,
        label_yunits=None,
        label_xunits=None,
        label_yticks=None,
        label_xticks=None,
        output_path=None,
        output_format="show",
        output_filename=None,
    ):
        super(LinePlotter, self).__init__(
            is_sub_plotter=is_sub_plotter,
            use_scaled_units=use_scaled_units,
            unit_conversion_factor=unit_conversion_factor,
            figsize=figsize,
            aspect=aspect,
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            label_title=label_title,
            label_yunits=label_yunits,
            label_xunits=label_xunits,
            label_yticks=label_yticks,
            label_xticks=label_xticks,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

        self.pointsize = pointsize
        self.include_legend = include_legend
        self.legend_fontsize = legend_fontsize

    def plotter_as_sub_plotter(self,):

        return LinePlotter(
            is_sub_plotter=True,
            include_legend=self.include_legend,
            legend_fontsize=self.legend_fontsize,
            pointsize=self.pointsize,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            figsize=self.figsize,
            aspect=self.aspect,
            titlesize=self.titlesize,
            xlabelsize=self.xlabelsize,
            ylabelsize=self.ylabelsize,
            xyticksize=self.xyticksize,
            label_title=self.label_title,
            label_yunits=self.label_yunits,
            label_xunits=self.label_xunits,
            label_yticks=self.label_yticks,
            label_xticks=self.label_xticks,
            output_path=self.output_path,
            output_format=self.output_format,
            output_filename=self.output_filename,
        )

    def plotter_with_new_labels_and_filename(
        self,
        label_title=None,
        label_yunits=None,
        label_xunits=None,
        output_filename=None,
        unit_conversion_factor=None,
    ):

        label_title = self.label_title if label_title is None else label_title
        label_yunits = self.label_yunits if label_yunits is None else label_yunits
        label_xunits = self.label_xunits if label_xunits is None else label_xunits
        output_filename = (
            self.output_filename if output_filename is None else output_filename
        )
        unit_conversion_factor = (
            self.unit_conversion_factor
            if unit_conversion_factor is None
            else unit_conversion_factor
        )

        return LinePlotter(
            is_sub_plotter=self.is_sub_plotter,
            use_scaled_units=self.use_scaled_units,
            include_legend=self.include_legend,
            legend_fontsize=self.legend_fontsize,
            unit_conversion_factor=unit_conversion_factor,
            pointsize=self.pointsize,
            figsize=self.figsize,
            aspect=self.aspect,
            titlesize=self.titlesize,
            xlabelsize=self.xlabelsize,
            ylabelsize=self.ylabelsize,
            xyticksize=self.xyticksize,
            label_title=label_title,
            label_yunits=label_yunits,
            label_xunits=label_xunits,
            label_yticks=self.label_yticks,
            label_xticks=self.label_xticks,
            output_path=self.output_path,
            output_format=self.output_format,
            output_filename=output_filename,
        )

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
        self.set_title()

        if x is None:
            x = np.arange(len(y))

        self.plot_y_vs_x(y=y, x=x, plot_axis_type=plot_axis_type, label=label)

        self.set_xy_labels_and_ticksize()

        self.plot_vertical_lines(
            vertical_lines=vertical_lines, vertical_line_labels=vertical_line_labels
        )

        self.set_legend()

        self.set_xticks(extent=[np.min(x), np.max(x)])

        if self.output_format is not "fits":

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

    def set_xy_labels_and_ticksize(self,):
        """Set the x and y labels of the figure, and set the fontsize of those labels.

        The x label is always the distance scale / radius, thus the x-label is either arc-seconds or kpc and depending \
        on the unit_label the figure is plotted in.

        The ylabel is the physical quantity being plotted and is passed as an input parameter.

        Parameters
        -----------
        label_xunits : str
            The unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        label_yunits : str
            The y-label of the figure, which is the physical quantity being plotted.
        xlabelsize : int
            The fontsize of the x axes label.
        ylabelsize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        """

        plt.ylabel(ylabel=self.label_yunits, fontsize=self.ylabelsize)
        plt.xlabel("x (" + self.label_xunits + ")", fontsize=self.xlabelsize)
        plt.tick_params(labelsize=self.xyticksize)

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

        xticks = np.round(np.linspace(extent[0], extent[1], 5), 2)

        if self.label_xticks is not None:
            xtick_labels = np.asarray([self.label_xticks[0], self.label_xticks[3]])
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
