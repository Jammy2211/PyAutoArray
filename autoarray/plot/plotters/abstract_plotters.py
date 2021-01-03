from autoarray.plot.mat_wrap.wrap import wrap_base

wrap_base.set_backend()

import matplotlib.pyplot as plt

from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
from functools import wraps

from autoarray.structures import abstract_structure, grids

import copy
import inspect
import typing
from typing import Callable


def title_from_func(func: Callable) -> str:
    """If a title is not manually specified use the name of the function plotting the image to set the title.

    Parameters
    ----------
    func : func
       The function plotting the image.
    """
    return func.__name__.replace("figure_", "").capitalize()


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


def filename_from_func(func: Callable) -> str:
    """If a filename is not manually specified use the name of the function plotting the image to set it.

    Parameters
    ----------
    func : func
       The function plotting the image.
    """
    funcname = func.__name__
    return funcname.replace("figure_", "")


def update_mat_plot(mat_plot, func: Callable, kwargs, for_subplot=False):

    if mat_plot.title.kwargs["label"] is None:
        mat_plot.title.kwargs["label"] = title_from_func(func=func)

    if mat_plot.ylabel._units is None:
        mat_plot.ylabel._units = units_from_func(func=func, for_ylabel=True)

    if mat_plot.xlabel._units is None:
        mat_plot.xlabel._units = units_from_func(func=func, for_ylabel=False)

    if mat_plot.output.filename is None:
        mat_plot.output.filename = filename_from_func(func=func)

    kpc_per_scaled = kpc_per_scaled_of_object_from_kwargs(kwargs=kwargs)

    mat_plot.units.conversion_factor = kpc_per_scaled

    if for_subplot:

        mat_plot.for_subplot = True
        mat_plot.output.bypass = True

        for attr, value in mat_plot.__dict__.items():
            if hasattr(value, "for_subplot"):
                value.for_subplot = True

    return mat_plot


def for_figure(func):
    """
    Decorate a profile method that accepts a coordinate grid and returns a data_type grid.

    If an interpolator attribute is associated with the input grid then that interpolator is used to down sample the
    coordinate grid prior to calling the function and up sample the result of the function.

    If no interpolator attribute is associated with the input grid then the function is called as hyper.

    Parameters
    ----------
    func
        Some method that accepts a grid

    Returns
    -------
    decorated_function
        The function with optional interpolation
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        if args[0].for_subplot:
            return func(*args, **kwargs)

        plotter = copy.deepcopy(args[0])

        if plotter.mat_plot_1d is not None:

            plotter.mat_plot_1d = update_mat_plot(
                mat_plot=plotter.mat_plot_1d, func=func, kwargs=kwargs
            )

        if plotter.mat_plot_2d is not None:

            plotter.mat_plot_2d = update_mat_plot(
                mat_plot=plotter.mat_plot_2d, func=func, kwargs=kwargs
            )

        args = (plotter,) + tuple(arg for arg in args[1:])

        return func(*args, **kwargs)

    return wrapper


def for_subplot(func):
    """
    Decorate a profile method that accepts a coordinate grid and returns a data_type grid.

    If an interpolator attribute is associated with the input grid then that interpolator is used to down sample the
    coordinate grid prior to calling the function and up sample the result of the function.

    If no interpolator attribute is associated with the input grid then the function is called as hyper.

    Parameters
    ----------
    func
        Some method that accepts a grid

    Returns
    -------
    decorated_function
        The function with optional interpolation
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        plotter = copy.deepcopy(args[0])

        if plotter.mat_plot_1d is not None:
            plotter.mat_plot_1d = update_mat_plot(
                mat_plot=plotter.mat_plot_1d, func=func, kwargs=kwargs, for_subplot=True
            )

        if plotter.mat_plot_2d is not None:
            plotter.mat_plot_2d = update_mat_plot(
                mat_plot=plotter.mat_plot_2d, func=func, kwargs=kwargs, for_subplot=True
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

    def visuals_from_structure(
        self, structure: abstract_structure.AbstractStructure
    ) -> "vis.Visuals2D":
        """
        Extracts from a `Structure` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From an `AbstractStructure` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Parameters
        ----------
        structure : abstract_structure.AbstractStructure
            The structure whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        origin = (
            grids.GridIrregular(grid=[structure.origin])
            if self.include_2d.origin
            else None
        )

        mask = structure.mask if self.include_2d.mask else None

        border = (
            structure.mask.geometry.border_grid_sub_1.in_1d_binned
            if self.include_2d.border
            else None
        )

        return vis.Visuals2D(origin=origin, mask=mask, border=border) + self.visuals_2d
