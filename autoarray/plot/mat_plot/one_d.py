import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Optional, List, Union

from autoarray.plot.mat_plot.abstract import AbstractMatPlot
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.visuals.one_d import Visuals1D
from autoarray.plot.wrap import base as wb
from autoarray.plot.wrap import one_d as w1d
from autoarray.structures.arrays.uniform_1d import Array1D


class MatPlot1D(AbstractMatPlot):
    def __init__(
        self,
        units: Optional[wb.Units] = None,
        figure: Optional[wb.Figure] = None,
        axis: Optional[wb.Axis] = None,
        cmap: Optional[wb.Cmap] = None,
        colorbar: Optional[wb.Colorbar] = None,
        colorbar_tickparams: Optional[wb.ColorbarTickParams] = None,
        tickparams: Optional[wb.TickParams] = None,
        yticks: Optional[wb.YTicks] = None,
        xticks: Optional[wb.XTicks] = None,
        title: Optional[wb.Title] = None,
        ylabel: Optional[wb.YLabel] = None,
        xlabel: Optional[wb.XLabel] = None,
        text: Optional[Union[wb.Text, List[wb.Text]]] = None,
        annotate: Optional[Union[wb.Annotate, List[wb.Annotate]]] = None,
        legend: Optional[wb.Legend] = None,
        output: Optional[wb.Output] = None,
        yx_plot: Optional[w1d.YXPlot] = None,
        vertical_line_axvline: Optional[w1d.AXVLine] = None,
        yx_scatter: Optional[w1d.YXPlot] = None,
        fill_between: Optional[w1d.FillBetween] = None,
    ):
        """
        Visualizes 1D data structures (e.g a `Line`, etc.) using Matplotlib.

        The `Plotter` is passed objects from the `wrap_base` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.

        The following 1D data structures can be plotted using the following matplotlib functions:

        - `Line` using `plt.plot`.

        Parameters
        ----------
        units
            The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure
            Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
            via `plt.close`.
        axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
        cmap
            Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such
            as `colors.Normalize` and `colors.LogNorm`.
        colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams
            Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks
            Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.yticks`.
        xticks
            Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.xticks`.
        title
            Sets the figure title and customizes its appearance using `plt.title`.
        ylabel
            Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel
            Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        text
            Sets any text on the figure and customizes its appearance using `plt.text`.
        annotate
            Sets any annotations on the figure and customizes its appearance using `plt.annotate`.
        legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        yx_plot
            Sets how the y versus x plot appears, for example if it each axis is linear or log, using `plt.plot`.
        vertical_line_axvline
            Sets how a vertical line plotted on the figure using the `plt.axvline` method.
        """

        super().__init__(
            units=units,
            figure=figure,
            axis=axis,
            cmap=cmap,
            colorbar=colorbar,
            colorbar_tickparams=colorbar_tickparams,
            tickparams=tickparams,
            yticks=yticks,
            xticks=xticks,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            text=text,
            annotate=annotate,
            legend=legend,
            output=output,
        )

        self.yx_plot = yx_plot or w1d.YXPlot(is_default=True)
        self.vertical_line_axvline = vertical_line_axvline or w1d.AXVLine(
            is_default=True
        )
        self.yx_scatter = yx_scatter or w1d.YXScatter(is_default=True)
        self.fill_between = fill_between or w1d.FillBetween(is_default=True)

        self.is_for_multi_plot = False
        self.is_for_subplot = False

    def set_for_multi_plot(
        self, is_for_multi_plot: bool, color: str, xticks=None, yticks=None
    ):
        """
        Sets the `is_for_subplot` attribute for every `MatWrap` object in this `MatPlot` object by updating
        the `is_for_subplot`. By changing this tag:

            - The subplot: section of the config file of every `MatWrap` object is used instead of figure:.
            - Calls which output or close the matplotlib figure are over-ridden so that the subplot is not removed.

        Parameters
        ----------
        is_for_subplot
            The entry the `is_for_subplot` attribute of every `MatWrap` object is set too.
        """
        self.is_for_multi_plot = is_for_multi_plot
        self.output.bypass = is_for_multi_plot

        self.yx_plot.kwargs["c"] = color
        self.vertical_line_axvline.kwargs["c"] = color

        self.vertical_line_axvline.no_label = True

        if yticks is not None:
            self.yticks = yticks

        if xticks is not None:
            self.xticks = xticks

    def plot_yx(
        self,
        y: Union[Array1D],
        visuals_1d: Visuals1D,
        auto_labels: AutoLabels,
        x: Optional[Union[np.ndarray, Iterable, List, Array1D]] = None,
        plot_axis_type_override: Optional[str] = None,
        y_errors=None,
        x_errors=None,
        y_extra=None,
        ls_errorbar="",
        should_plot_grid=False,
        should_plot_zero=False,
        text_manual_dict=None,
        text_manual_dict_y=None,
        bypass: bool = False,
    ):
        if (y is None) or np.count_nonzero(y) == 0:
            return

        ax = None

        if not self.is_for_subplot:
            fig, ax = self.figure.open()
        else:
            if not bypass:
                ax = self.setup_subplot()

        self.title.set(auto_title=auto_labels.title)

        use_integers = False

        if x is None:
            x = np.arange(len(y))
            use_integers = True
            pixel_scales = (x[1] - x[0],)
            x = Array1D.no_mask(values=x, pixel_scales=pixel_scales)

        if self.yx_plot.plot_axis_type is None:
            plot_axis_type = "linear"
        else:
            plot_axis_type = self.yx_plot.plot_axis_type

        if plot_axis_type_override is not None:
            plot_axis_type = plot_axis_type_override

        label = self.legend.label or auto_labels.legend

        self.yx_plot.plot_y_vs_x(
            y=y,
            x=x,
            label=label,
            plot_axis_type=plot_axis_type,
            y_errors=y_errors,
            x_errors=x_errors,
            y_extra=y_extra,
            ls_errorbar=ls_errorbar,
        )

        if should_plot_zero:
            plt.plot(x, 1.0e-6 * np.ones(shape=y.shape), c="b", ls="--")

        if should_plot_grid:
            plt.grid(True)

        if visuals_1d.shaded_region is not None:
            self.fill_between.fill_between_shaded_regions(
                x=x, y1=visuals_1d.shaded_region[0], y2=visuals_1d.shaded_region[1]
            )

        if "extent" in self.axis.config_dict:
            self.axis.set()

        self.tickparams.set()

        if plot_axis_type == "symlog":
            plt.yscale("symlog")

        if x_errors is not None:
            min_value_x = np.nanmin(x - x_errors)
            max_value_x = np.nanmax(x + x_errors)
        else:
            min_value_x = np.nanmin(x)
            max_value_x = np.nanmax(x)

        if y_errors is not None:
            min_value_y = np.nanmin(y - y_errors)
            max_value_y = np.nanmax(y + y_errors)
        else:
            min_value_y = np.nanmin(y)
            max_value_y = np.nanmax(y)

        if should_plot_zero:
            if min_value_y > 0:
                min_value_y = 0

        self.xticks.set(
            min_value=min_value_x,
            max_value=max_value_x,
            pixels=len(x),
            units=self.units,
            use_integers=use_integers,
            is_for_1d_plot=True,
        )

        self.yticks.set(
            min_value=min_value_y,
            max_value=max_value_y,
            pixels=len(y),
            units=self.units,
            yunit=auto_labels.yunit,
            is_for_1d_plot=True,
            is_log10="logy" in plot_axis_type,
        )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set(auto_label=auto_labels.ylabel)
        self.xlabel.set(auto_label=auto_labels.xlabel)

        if not isinstance(self.text, list):
            self.text.set()
        else:
            [text.set() for text in self.text]

        # This is a horrific hack to get CTI plots to work, refactor one day.

        from autoarray.plot.wrap.base.text import Text

        if text_manual_dict is not None and ax is not None:
            y = text_manual_dict_y
            text_manual_list = []

            for key, value in text_manual_dict.items():
                text_manual_list.append(
                    Text(
                        x=0.95,
                        y=y,
                        s=f"{key} : {value}",
                        c="b",
                        transform=ax.transAxes,
                        horizontalalignment="right",
                        fontsize=12,
                    )
                )
                y = y - 0.05

            [text.set() for text in text_manual_list]

        if not isinstance(self.annotate, list):
            self.annotate.set()
        else:
            [annotate.set() for annotate in self.annotate]

        visuals_1d.plot_via_plotter(plotter=self)

        if label is not None:
            self.legend.set()

        if (not self.is_for_subplot) and (not self.is_for_multi_plot):
            self.output.to_figure(structure=None, auto_filename=auto_labels.filename)
            self.figure.close()
