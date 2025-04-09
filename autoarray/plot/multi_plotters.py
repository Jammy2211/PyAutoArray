import os
from pathlib import Path
from typing import List, Optional, Tuple

from autoarray.plot.wrap.base.ticks import YTicks
from autoarray.plot.wrap.base.ticks import XTicks


class MultiFigurePlotter:
    def __init__(
        self,
        plotter_list,
        subplot_shape: Tuple[int, int] = None,
        subplot_title: Optional[str] = None,
    ):
        """
        Plots multiple figures of plotter objects on the same subplot.

        For example, suppose you have multiple `ImagingPlotter` objects corresponding to different `Imaging` objects.
        You may want to plot the `data`, `noise_map` and `psf` of each imaging dataset on the same subplot, so that
        their data, noise-map and psf can be easily compared.

        The `MultiFigurePlotter` object allows you to do this, receiving a list of `Plotter` objects and calling
        their `figure` methods to plot each on the same subplot.

        This requires careful inputs to the plotting functions in order to ensure that the correct plotting
        functions are called for each plotter object.

        Parameters
        ----------
        plotter_list
            The list of plotter objects that are plotted on the same subplot.
        subplot_shape
            Optionally input the shape of the subplot (e.g. 2, 2) which is used to determine the shape of the figures
            on the subplot. If not input, the subplot shape is determined automatically via config files.
        subplot_title
            Optionally input a title for the subplot.
        """

        self.plotter_list = plotter_list
        self.subplot_shape = subplot_shape
        self.subplot_title = subplot_title

    def setup_subplot_via_mat_plot(
        self, plotter, number_subplots: int, subplot_index: int
    ):
        """
        Sets the `MatPlot` internal attributes  which track if the plot is being made on a subplot and what index
        the subplot is in the figure.

        Outside of the `MultiFigurePlotter` class when subplots are made from a single `Plotter` object, these
        attributes are updated after each subplot figure is made.

        This class plots multiple plotter objects on the same subplot, so these attributes tracked and updated
        separately for each plotter object by this class.

        Parameters
        ----------
        plotter
            The plotter which is used to plot the next figure on the subplot and therefore requires its mat-plot
            subplot attributes to be updated.
        number_subplots
            The number of subplots that are being made on the figure.
        subplot_index
            The index of the subplot that is next being made on the figure uing the input plotter object.
        """

        try:
            plotter.mat_plot_2d.set_for_subplot(is_for_subplot=True)
            plotter.mat_plot_2d.number_subplots = number_subplots
            plotter.mat_plot_2d.subplot_shape = self.subplot_shape
            plotter.mat_plot_2d.subplot_index = subplot_index
        except AttributeError:
            plotter.mat_plot_1d.set_for_subplot(is_for_subplot=True)
            plotter.mat_plot_1d.number_subplots = number_subplots
            plotter.mat_plot_1d.subplot_shape = self.subplot_shape
            plotter.mat_plot_1d.subplot_index = subplot_index

    def plot_via_func(self, plotter, figure_name: str, func_name: str, kwargs):
        """
        Plots a figure on the subplot using an input plotter object, figure name and function name.

        For example, if you have an `ImagingPlotter` object and you want to plot the `data` on the subplot, you would
        input `plotter=imaging_plotter`, `figure_name='data'` and `func_name='figures_2d'`.

        The code then knows to call the `figures_2d` function of the `ImagingPlotter` object and plot the `data`.

        This function is called repeatedly for each plotter object in the `plotter_list` to plot each figure
        on the subplot.

        Parameters
        ----------
        plotter
            The plotter object that is used to plot the figure on the subplot.
        figure_name
            The name of the figure that is plotted on the subplot.
        func_name
            The name of the function that is called to plot the figure on the subplot.
        kwargs
            Any additional keyword arguments that are passed to the function that plots the figure on the subplot.
        """
        func = getattr(plotter, func_name)

        if figure_name is None:
            func(**{**{}, **kwargs})
        else:
            func(**{**{figure_name: True}, **kwargs})

    def subplot_of_figure(
        self, func_name: str, figure_name: str, filename_suffix: str = "", **kwargs
    ):
        """
        Outputs a subplot of figures of the plotter objects in the `plotter_list`, where only a single function name
        and figure name is input.

        For example, if you have multiple `ImagingPlotter` objects and you want to plot the `data` of each on the same
        subplot, you would input `func_name='figures_2d'` and `figure_name='data'`.

        This function cannot plot different attributes of the plotter objects on the same subplot, for example the
        `data` and `noise_map` of the `ImagingPlotter` objects. For this, use the `subplot_of_figures_multi` function.

        Parameters
        ----------
        func_name
            The name of the function that is called to plot the figure on the subplot.
        figure_name
            The name of the figure that is plotted on the subplot.
        filename_suffix
            The suffix of the filename that the subplot is output to.
        kwargs
            Any additional keyword arguments that are passed to the function that plots the figure on the subplot.
        """
        number_subplots = len(self.plotter_list)

        self.plotter_list[0].open_subplot_figure(
            number_subplots=number_subplots, subplot_shape=self.subplot_shape
        )

        for i, plotter in enumerate(self.plotter_list):
            self.setup_subplot_via_mat_plot(
                plotter=plotter, number_subplots=number_subplots, subplot_index=i + 1
            )

            self.plot_via_func(
                plotter=plotter,
                figure_name=figure_name,
                func_name=func_name,
                kwargs=kwargs,
            )

        self.output_subplot(filename_suffix=f"{figure_name}{filename_suffix}")

    def subplot_of_figures_multi(
        self,
        func_name_list: List[str],
        figure_name_list: List[str],
        filename_suffix: str = "",
        subplot_index_offset: int = 0,
        number_subplots: Optional[int] = None,
        open_subplot: bool = True,
        close_subplot: bool = True,
        **kwargs,
    ):
        """
        Outputs a subplot of figures of the plotter objects in the `plotter_list`, where multiple function names and
        figure names are input.

        For example, if you have multiple `ImagingPlotter` objects and you want to plot the `data` and `noise_map` of
        each on the same subplot, you would input `func_name_list=['figures_2d', 'figures_2d']` and
        `figure_name_list=['data', 'noise_map']`.

        Parameters
        ----------
        func_name_list
            The list of function names that are called to plot the figures on the subplot.
        figure_name_list
            The list of figure names that are plotted on the subplot.
        filename_suffix
            The suffix of the filename that the subplot is output to.
        kwargs
            Any additional keyword arguments that are passed to the function that plots the figure on the subplot.
        """
        if number_subplots is None:
            number_subplots = len(self.plotter_list) * len(func_name_list)

        if open_subplot:
            self.plotter_list[0].open_subplot_figure(
                number_subplots=number_subplots, subplot_shape=self.subplot_shape
            )

        for i, plotter in enumerate(self.plotter_list):
            for j, (func_name, figure_name) in enumerate(
                zip(func_name_list, figure_name_list)
            ):
                subplot_shape = self.plotter_list[0].mat_plot_2d.subplot_shape

                subplot_index = subplot_index_offset + (i * subplot_shape[1]) + j + 1

                self.setup_subplot_via_mat_plot(
                    plotter=plotter,
                    number_subplots=number_subplots,
                    subplot_index=subplot_index,
                )

                self.plot_via_func(
                    plotter=plotter,
                    figure_name=figure_name,
                    func_name=func_name,
                    kwargs=kwargs,
                )

        if close_subplot:
            self.output_subplot(filename_suffix=filename_suffix)

    def subplot_of_multi_yx_1d(self, filename_suffix="", **kwargs):
        number_subplots = len(self.plotter_list)

        self.plotter_list[0].plotter_list[0].open_subplot_figure(
            number_subplots=number_subplots,
            subplot_shape=self.subplot_shape,
            subplot_title=self.subplot_title,
        )

        for i, plotter in enumerate(self.plotter_list):
            for plott in plotter.plotter_list:
                plott.mat_plot_1d.set_for_subplot(is_for_subplot=True)
                plott.mat_plot_1d.number_subplots = number_subplots
                plott.mat_plot_1d.subplot_shape = self.subplot_shape
                plott.mat_plot_1d.subplot_index = i + 1

            func = getattr(plotter, "figure_1d")
            func(
                **{
                    **{
                        "func_name": "figure_1d",
                        "figure_name": None,
                        "is_for_subplot": True,
                    },
                    **kwargs,
                }
            )

        self.plotter_list[0].plotter_list[0].mat_plot_1d.output.subplot_to_figure(
            auto_filename=f"subplot_{filename_suffix}"
        )
        self.plotter_list[0].plotter_list[0].close_subplot_figure()

    def output_subplot(self, filename_suffix: str = ""):
        """
        Outplot the subplot to a file after all figures have been plotted on the subplot.

        The multi-plotter requires its own output function to ensure that the subplot is output to a file, which
        this provides.

        Parameters
        ----------
        filename_suffix
            The suffix of the filename that the subplot is output to.
        """

        plotter = self.plotter_list[0]

        if plotter.mat_plot_1d is not None:
            plotter.mat_plot_1d.output.subplot_to_figure(
                auto_filename=f"subplot_{filename_suffix}"
            )
        if plotter.mat_plot_2d is not None:
            plotter.mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_{filename_suffix}"
            )
        plotter.close_subplot_figure()


class MultiYX1DPlotter:
    def __init__(
        self,
        plotter_list,
        color_list=None,
        legend_labels=None,
        y_manual_min_max_value=None,
        x_manual_min_max_value=None,
    ):
        self.plotter_list = plotter_list

        if color_list is None:
            color_list = 10 * ["k", "r", "b", "g", "c", "m", "y"]

        self.color_list = color_list
        self.legend_labels = legend_labels

        self.y_manual_min_max_value = y_manual_min_max_value
        self.x_manual_min_max_value = x_manual_min_max_value

    def figure_1d(self, func_name, figure_name, is_for_subplot=False, **kwargs):
        if not is_for_subplot:
            self.plotter_list[0].mat_plot_1d.figure.open()

        for i, plotter in enumerate(self.plotter_list):
            plotter.set_mat_plot_1d_for_multi_plot(
                is_for_multi_plot=True,
                color=self.color_list[i],
                yticks=self.yticks,
                xticks=self.xticks,
            )

            if self.legend_labels is not None:
                plotter.mat_plot_1d.yx_plot.label = self.legend_labels[i]

            func = getattr(plotter, func_name)

            if figure_name is None:
                func(**{**{}, **kwargs})
            else:
                func(**{**{figure_name: True}, **kwargs})

            plotter.set_mat_plot_1d_for_multi_plot(is_for_multi_plot=False, color=None)

        if not is_for_subplot:
            self.plotter_list[0].mat_plot_1d.output.subplot_to_figure(
                auto_filename=f"multi_{figure_name}"
            )
            self.plotter_list[0].mat_plot_1d.figure.close()

    @property
    def yticks(self):
        # TODO: Need to make this work for all plotters, rather than just y x, for example
        # TODO : GalaxyPlotters where y and x are computed inside the function called via
        # TODO : func(**{**{figure_name: True}, **kwargs})

        if self.y_manual_min_max_value is not None:
            return YTicks(manual_min_max_value=self.y_manual_min_max_value)

        try:
            min_value = min([min(plotter.y) for plotter in self.plotter_list])
            max_value = max([max(plotter.y) for plotter in self.plotter_list])
        except AttributeError:
            return

        return YTicks(manual_min_max_value=(min_value, max_value))

    @property
    def xticks(self):
        if self.x_manual_min_max_value is not None:
            return XTicks(manual_min_max_value=self.x_manual_min_max_value)

        try:
            min_value = min([min(plotter.x) for plotter in self.plotter_list])
            max_value = max([max(plotter.x) for plotter in self.plotter_list])
        except AttributeError:
            return

        return XTicks(manual_min_max_value=(min_value, max_value))
