from autoarray.plot.mat_wrap.wrap import wrap_base

wrap_base.set_backend()

import matplotlib.pyplot as plt

from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot

import copy
import inspect
import typing
from functools import wraps
from typing import Callable


def title_from_func(func: Callable, index=None) -> str:
    """If a title is not manually specified use the name of the function plotting the image to set the title.

    Parameters
    ----------
    func : func
       The function plotting the image.
    """
    if index is None:
        return func.__name__.replace("figure_", "").capitalize()
    return f"{func.__name__.replace('figure_', '').capitalize()}_{index}"


def units_from_func(func: Callable, for_ylabel=True) -> typing.Optional["Units"]:
    """
    If the x label is not manually specified use the function plotting the image to x label, assuming that it
    represents spatial units.

    Parameters
    ----------
    func : func
       The function plotting the image.
    """

    args = inspect.getfullargspec(func).args
    defaults = inspect.getfullargspec(func).defaults

    if defaults is not None:
        non_default_args = len(args) - len(defaults)
    else:
        non_default_args = 0

    if (not for_ylabel) and "label_xunits" in args:
        return defaults[args.index("label_xunits") - non_default_args]
    elif for_ylabel and "label_yunits" in args:
        return defaults[args.index("label_yunits") - non_default_args]
    else:
        return None


def filename_from_func(func: Callable, index=None) -> str:
    """If a filename is not manually specified use the name of the function plotting the image to set it.

    Parameters
    ----------
    func : func
       The function plotting the image.
    """
    funcname = func.__name__
    filename = funcname.replace("figure_", "")

    if index is None:
        return filename
    return f"{filename}_{index}"


def update_mat_plot_for_figure_of_subplot(mat_plot, func: Callable, kwargs, index=None):

    if mat_plot is None:
        return None

    mat_plot.title.kwargs["label"] = title_from_func(func=func, index=index)
    mat_plot.ylabel._units = units_from_func(func=func, for_ylabel=True)
    mat_plot.xlabel._units = units_from_func(func=func, for_ylabel=False)
    kpc_per_scaled = kpc_per_scaled_of_object_from_kwargs(kwargs=kwargs)

    mat_plot.units.conversion_factor = kpc_per_scaled

    return mat_plot


def update_mat_plot(mat_plot, func: Callable, kwargs, index=None, for_subplot=False):

    if mat_plot is None:
        return None

    if mat_plot.title.kwargs["label"] is None:
        mat_plot.title.kwargs["label"] = title_from_func(func=func, index=index)

    if mat_plot.ylabel._units is None:
        mat_plot.ylabel._units = units_from_func(func=func, for_ylabel=True)

    if mat_plot.xlabel._units is None:
        mat_plot.xlabel._units = units_from_func(func=func, for_ylabel=False)

    kpc_per_scaled = kpc_per_scaled_of_object_from_kwargs(kwargs=kwargs)

    mat_plot.units.conversion_factor = kpc_per_scaled

    if mat_plot.output.filename is None:
        mat_plot.output.filename = filename_from_func(func=func, index=index)

    if for_subplot:

        mat_plot.for_subplot = True
        mat_plot.output.bypass = True

        for attr, value in mat_plot.__dict__.items():
            if hasattr(value, "for_subplot"):
                value.for_subplot = True

    return mat_plot


def index_from_inputs(args, kwargs):

    try:
        return args[1]
    except IndexError:
        key = next(iter(kwargs))
        return kwargs[key]


def for_figure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        if args[0].for_subplot:

            args[0].mat_plot_2d = update_mat_plot_for_figure_of_subplot(
                mat_plot=args[0].mat_plot_2d, func=func, kwargs=kwargs
            )
            args[0].mat_plot_1d = update_mat_plot_for_figure_of_subplot(
                mat_plot=args[0].mat_plot_1d, func=func, kwargs=kwargs
            )

            return func(*args, **kwargs)

        plotter = args[0].new_with_updated_mat_plots(
            func=func, kwargs=kwargs, for_subplot=False
        )

        args = (plotter,) + tuple(arg for arg in args[1:])

        return func(*args, **kwargs)

    return wrapper


def for_figure_with_index(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        index = index_from_inputs(args=args, kwargs=kwargs)

        if args[0].for_subplot:

            args[0].mat_plot_2d = update_mat_plot_for_figure_of_subplot(
                mat_plot=args[0].mat_plot_2d, func=func, index=index, kwargs=kwargs
            )
            args[0].mat_plot_1d = update_mat_plot_for_figure_of_subplot(
                mat_plot=args[0].mat_plot_1d, func=func, index=index, kwargs=kwargs
            )

            return func(*args, **kwargs)

        plotter = args[0].new_with_updated_mat_plots(
            func=func, kwargs=kwargs, index=index, for_subplot=False
        )

        args = (plotter,) + tuple(arg for arg in args[1:])

        return func(*args, **kwargs)

    return wrapper


def for_subplot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        plotter = args[0].new_with_updated_mat_plots(
            func=func, kwargs=kwargs, for_subplot=True
        )

        args = (plotter,) + tuple(arg for arg in args[1:])

        return func(*args, **kwargs)

    return wrapper


def for_subplot_with_index(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        index = index_from_inputs(args=args, kwargs=kwargs)

        plotter = args[0].new_with_updated_mat_plots(
            func=func, kwargs=kwargs, index=index, for_subplot=True
        )

        args = (plotter,) + tuple(arg for arg in args[1:])

        return func(*args, **kwargs)

    return wrapper


def kpc_per_scaled_of_object_from_kwargs(kwargs):

    kpc_per_scaled = None

    for key, value in kwargs.items():
        if hasattr(value, "kpc_per_scaled"):
            return value.kpc_per_scaled

    return kpc_per_scaled


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

    def new_with_updated_mat_plots(self, func, kwargs, index=None, for_subplot=False):

        plotter = copy.deepcopy(self)

        plotter.mat_plot_1d = update_mat_plot(
            mat_plot=plotter.mat_plot_1d,
            func=func,
            kwargs=kwargs,
            index=index,
            for_subplot=for_subplot,
        )

        plotter.mat_plot_2d = update_mat_plot(
            mat_plot=plotter.mat_plot_2d,
            func=func,
            kwargs=kwargs,
            index=index,
            for_subplot=for_subplot,
        )

        return plotter

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
            if self.mat_plot_1d.figure.config_dict_figure["figsize"] is not None:
                return self.mat_plot_1d.figure.config_dict_figure["figsize"]

        if self.mat_plot_2d is not None:
            if self.mat_plot_2d.figure.config_dict_figure["figsize"] is not None:
                return self.mat_plot_2d.figure.config_dict_figure["figsize"]

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
