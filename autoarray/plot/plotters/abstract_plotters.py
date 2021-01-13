from autoarray.plot.mat_wrap.wrap import wrap_base

wrap_base.set_backend()

import matplotlib.pyplot as plt

from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot

from functools import wraps


def for_subplot(func):
    @wraps(func)
    def wrapper(plotter, *args, **kwargs):

        plotter.set_mat_plots_for_subplot(for_subplot=True)

        func(plotter, *args, **kwargs)

        plotter.set_mat_plots_for_subplot(for_subplot=False)

    return wrapper


def for_subplot_with_index(func):
    @wraps(func)
    def wrapper(plotter, *args, **kwargs):

        plotter.set_mat_plots_for_subplot(for_subplot=True)

        func(plotter, *args, **kwargs)

        plotter.set_mat_plots_for_subplot(for_subplot=False)

    return wrapper


class AbstractPlotter:
    def __init__(
        self,
        mat_plot_1d: mat_plot.MatPlot1D = None,
        visuals_1d: vis.Visuals1D = None,
        include_1d: inc.Include1D = None,
        mat_plot_2d: mat_plot.MatPlot2D = None,
        visuals_2d: vis.Visuals2D = None,
        include_2d: inc.Include2D = None,
    ):

        self.visuals_1d = visuals_1d
        self.include_1d = include_1d
        self.mat_plot_1d = mat_plot_1d
        self.visuals_2d = visuals_2d
        self.include_2d = include_2d
        self.mat_plot_2d = mat_plot_2d

    def figures(self, **kwargs):
        raise NotImplementedError

    @property
    def plot_line(self):
        return self.mat_plot_1d.plot_line

    @property
    def plot_array(self):
        return self.mat_plot_2d.plot_array

    @property
    def plot_frame(self):
        return self.mat_plot_2d.plot_frame

    @property
    def plot_grid(self):
        return self.mat_plot_2d.plot_grid

    @property
    def plot_mapper(self):
        return self.mat_plot_2d.plot_mapper

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

    def set_mat_plots_for_subplot(self, for_subplot):
        if self.mat_plot_1d is not None:
            self.mat_plot_1d.set_for_subplot(for_subplot=for_subplot)
        if self.mat_plot_2d is not None:
            self.mat_plot_2d.set_for_subplot(for_subplot=for_subplot)

    def _subplot_custom_plot(self, **kwargs):

        self.set_mat_plots_for_subplot(for_subplot=True)

        figures_dict = dict(
            (key, value) for key, value in kwargs.items() if value is True
        )

        number_subplots = len(figures_dict)

        self.open_subplot_figure(number_subplots=number_subplots)

        for index, (key, value) in enumerate(figures_dict.items()):
            if value:
                self.setup_subplot(
                    number_subplots=number_subplots, subplot_index=index + 1
                )
                self.figures(**{key: True})

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=kwargs["auto_labels"].filename
        )
        self.mat_plot_2d.figure.close()

        self.set_mat_plots_for_subplot(for_subplot=False)

    @property
    def for_subplot(self):

        if self.mat_plot_1d is not None:
            if self.mat_plot_1d.for_subplot:
                return True

        if self.mat_plot_2d is not None:
            if self.mat_plot_2d.for_subplot:
                return True

        return False

    def open_subplot_figure(self, number_subplots):
        """Setup a figure for plotting an image.

        Parameters
        -----------
        figsize : (int, int)
            The size of the figure in (total_y_pixels, total_x_pixels).
        as_subplot : bool
            If the figure is a subplot, the setup_figure function is omitted to ensure that each subplot does not create a \
            new figure and so that it can be output using the *output.output_figure(structure=None)* function.
        """
        figsize = self.get_subplot_figsize(number_subplots=number_subplots)
        plt.figure(figsize=figsize)

    def get_subplot_rows_columns(self, number_subplots):
        """Get the size of a sub plotter in (total_y_pixels, total_x_pixels), based on the number of subplots that are going to be plotted.

        Parameters
        -----------
        number_subplots : int
            The number of subplots that are to be plotted in the figure.
        """
        if number_subplots <= 2:
            return 1, 2
        elif number_subplots <= 4:
            return 2, 2
        elif number_subplots <= 6:
            return 2, 3
        elif number_subplots <= 9:
            return 3, 3
        elif number_subplots <= 12:
            return 3, 4
        elif number_subplots <= 16:
            return 4, 4
        elif number_subplots <= 20:
            return 4, 5
        else:
            return 6, 6

    def get_subplot_figsize(self, number_subplots):
        """Get the size of a sub plotter in (total_y_pixels, total_x_pixels), based on the number of subplots that are going to be plotted.

        Parameters
        -----------
        number_subplots : int
            The number of subplots that are to be plotted in the figure.
        """

        if self.mat_plot_1d is not None:
            if self.mat_plot_1d.figure.config_dict["figsize"] is not None:
                return self.mat_plot_1d.figure.config_dict["figsize"]

        if self.mat_plot_2d is not None:
            if self.mat_plot_2d.figure.config_dict["figsize"] is not None:
                return self.mat_plot_2d.figure.config_dict["figsize"]

        if number_subplots <= 2:
            return (18, 8)
        elif number_subplots <= 4:
            return (13, 10)
        elif number_subplots <= 6:
            return (18, 12)
        elif number_subplots <= 9:
            return (25, 20)
        elif number_subplots <= 12:
            return (25, 20)
        elif number_subplots <= 16:
            return (25, 20)
        elif number_subplots <= 20:
            return (25, 20)
        else:
            return (25, 20)

    def setup_subplot(
        self, number_subplots, subplot_index, aspect=None, subplot_rows_columns=None
    ):

        if subplot_rows_columns is None:
            rows, columns = self.get_subplot_rows_columns(
                number_subplots=number_subplots
            )
        else:
            rows = subplot_rows_columns[0]
            columns = subplot_rows_columns[1]
        if aspect is None:
            plt.subplot(rows, columns, subplot_index)
        else:
            plt.subplot(rows, columns, subplot_index, aspect=float(aspect))

    def extract_2d(self, name, value, include_name=None):
        """
        Extracts an attribute for plotting in a `Visuals2D` object based on the following criteria:

        1) If `visuals_2d` already has a value for the attribute this is returned, over-riding the input `value` of
          that attribute.

        2) If `visuals_2d` do not contain the attribute, the input `value` is returned provided its corresponding
          entry in the `Include2D` class is `True`.

        3) If the `Include2D` entry is `False` a None is returned and the attribute is therefore plotted.

        Parameters
        ----------
        name : str
            The name of the attribute which is to be extracted.
        value :
            The `value` of the attribute, which is used when criteria 2) above is met.

        Returns
        -------
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        if include_name is None:
            include_name = name

        if getattr(self.visuals_2d, name) is not None:
            return getattr(self.visuals_2d, name)
        else:
            if getattr(self.include_2d, include_name):
                return value
