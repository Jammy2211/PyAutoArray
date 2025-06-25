from autoconf import conf

from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

from typing import Optional, Tuple

from autoarray.plot.visuals.one_d import Visuals1D
from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.plot.include.one_d import Include1D
from autoarray.plot.include.two_d import Include2D
from autoarray.plot.mat_plot.one_d import MatPlot1D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.get_visuals.one_d import GetVisuals1D
from autoarray.plot.get_visuals.two_d import GetVisuals2D


class AbstractPlotter:
    def __init__(
        self,
        mat_plot_1d: MatPlot1D = None,
        visuals_1d: Visuals1D = None,
        include_1d: Include1D = None,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
        include_2d: Include2D = None,
    ):
        self.visuals_1d = visuals_1d or Visuals1D()
        self.include_1d = include_1d or Include1D()
        self.mat_plot_1d = mat_plot_1d or MatPlot1D()

        self.visuals_2d = visuals_2d or Visuals2D()
        self.include_2d = include_2d or Include2D()
        self.mat_plot_2d = mat_plot_2d or MatPlot2D()

        self.subplot_figsize = None

    def set_title(self, label):
        if self.mat_plot_1d is not None:
            self.mat_plot_1d.title.manual_label = label

        if self.mat_plot_2d is not None:
            self.mat_plot_2d.title.manual_label = label

    def set_filename(self, filename):
        if self.mat_plot_1d is not None:
            self.mat_plot_1d.output.filename = filename

        if self.mat_plot_2d is not None:
            self.mat_plot_2d.output.filename = filename

    def set_format(self, format):
        if self.mat_plot_1d is not None:
            self.mat_plot_1d.output._format = format

        if self.mat_plot_2d is not None:
            self.mat_plot_2d.output._format = format

    def set_mat_plot_1d_for_multi_plot(
        self, is_for_multi_plot, color: str, xticks=None, yticks=None
    ):
        self.mat_plot_1d.set_for_multi_plot(
            is_for_multi_plot=is_for_multi_plot,
            color=color,
            xticks=xticks,
            yticks=yticks,
        )

    def set_mat_plots_for_subplot(
        self, is_for_subplot, number_subplots=None, subplot_shape=None
    ):
        if self.mat_plot_1d is not None:
            self.mat_plot_1d.set_for_subplot(is_for_subplot=is_for_subplot)
            self.mat_plot_1d.number_subplots = number_subplots
            self.mat_plot_1d.subplot_shape = subplot_shape
            self.mat_plot_1d.subplot_index = 1
        if self.mat_plot_2d is not None:
            self.mat_plot_2d.set_for_subplot(is_for_subplot=is_for_subplot)
            self.mat_plot_2d.number_subplots = number_subplots
            self.mat_plot_2d.subplot_shape = subplot_shape
            self.mat_plot_2d.subplot_index = 1

    @property
    def is_for_subplot(self):
        if self.mat_plot_1d is not None:
            if self.mat_plot_1d.is_for_subplot:
                return True

        if self.mat_plot_2d is not None:
            if self.mat_plot_2d.is_for_subplot:
                return True

        return False

    def open_subplot_figure(
        self,
        number_subplots: int,
        subplot_shape: Optional[Tuple[int, int]] = None,
        subplot_figsize: Optional[Tuple[int, int]] = None,
        subplot_title: Optional[str] = None,
    ):
        """
        Setup a figure for plotting an image.

        Parameters
        ----------
        figsize
            The size of the figure in (total_y_pixels, total_x_pixels).
        as_subplot
            If the figure is a subplot, the setup_figure function is omitted to ensure that each subplot does not create a \
            new figure and so that it can be output using the *output.output_figure(structure=None)* function.
        """
        import matplotlib.pyplot as plt

        self.set_mat_plots_for_subplot(
            is_for_subplot=True,
            number_subplots=number_subplots,
            subplot_shape=subplot_shape,
        )

        self.subplot_figsize = subplot_figsize

        figsize = self.get_subplot_figsize(number_subplots=number_subplots)
        plt.figure(figsize=figsize)
        plt.suptitle(subplot_title, fontsize=40, y=0.93)

    def close_subplot_figure(self):
        try:
            self.mat_plot_2d.figure.close()
        except AttributeError:
            self.mat_plot_1d.figure.close()
        self.set_mat_plots_for_subplot(is_for_subplot=False)
        self.subplot_figsize = None

    def get_subplot_figsize(self, number_subplots):
        """
        Get the size of a sub plotter in (total_y_pixels, total_x_pixels), based on the number of subplots that are going to be plotted.

        Parameters
        ----------
        number_subplots
            The number of subplots that are to be plotted in the figure.
        """

        if self.subplot_figsize is not None:
            return self.subplot_figsize

        if self.mat_plot_1d is not None:
            if self.mat_plot_1d.figure.config_dict["figsize"] is not None:
                return self.mat_plot_1d.figure.config_dict["figsize"]

        if self.mat_plot_2d is not None:
            if self.mat_plot_2d.figure.config_dict["figsize"] is not None:
                return self.mat_plot_2d.figure.config_dict["figsize"]

        try:
            subplot_shape = self.mat_plot_1d.get_subplot_shape(
                number_subplots=number_subplots
            )
        except AttributeError:
            subplot_shape = self.mat_plot_2d.get_subplot_shape(
                number_subplots=number_subplots
            )

        subplot_shape_to_figsize_factor = conf.instance["visualize"]["general"][
            "subplot_shape_to_figsize_factor"
        ]
        subplot_shape_to_figsize_factor = tuple(
            map(int, subplot_shape_to_figsize_factor[1:-1].split(","))
        )

        return (
            subplot_shape[1] * subplot_shape_to_figsize_factor[1],
            subplot_shape[0] * subplot_shape_to_figsize_factor[0],
        )

    def _subplot_custom_plot(self, **kwargs):
        figures_dict = dict(
            (key, value) for key, value in kwargs.items() if value is True
        )

        self.open_subplot_figure(number_subplots=len(figures_dict))

        for index, (key, value) in enumerate(figures_dict.items()):
            if value:
                try:
                    self.figures_2d(**{key: True})
                except AttributeError:
                    self.figures_1d(**{key: True})

            try:
                self.mat_plot_2d.subplot_index = max(
                    self.mat_plot_1d.subplot_index, self.mat_plot_2d.subplot_index
                )
                self.mat_plot_1d.subplot_index = max(
                    self.mat_plot_1d.subplot_index, self.mat_plot_2d.subplot_index
                )
            except AttributeError:
                pass

        try:
            self.mat_plot_2d.output.subplot_to_figure(
                auto_filename=kwargs["auto_labels"].filename
            )
        except AttributeError:
            self.mat_plot_1d.output.subplot_to_figure(
                auto_filename=kwargs["auto_labels"].filename
            )

        self.close_subplot_figure()

    def subplot_of_plotters_figure(self, plotter_list, name):
        self.open_subplot_figure(number_subplots=len(plotter_list))

        for i, plotter in enumerate(plotter_list):
            plotter.figures_2d(**{name: True})

        self.mat_plot_2d.output.subplot_to_figure(auto_filename=f"subplot_{name}")

        self.close_subplot_figure()


class Plotter(AbstractPlotter):
    @property
    def get_1d(self):
        return GetVisuals1D(visuals=self.visuals_1d, include=self.include_1d)

    @property
    def get_2d(self):
        return GetVisuals2D(visuals=self.visuals_2d, include=self.include_2d)
