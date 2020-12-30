import numpy as np
from functools import wraps
import copy

from autoarray import exc
from autoarray.plot.mat_wrap import plotter as p, include as inc, visuals as vis


def key_for_cls_from_kwargs(cls, kwargs):

    for key, value in kwargs.items():
        if isinstance(value, cls):
            return key


def set_subplot_filename(func):
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

        plotter_key = key_for_cls_from_kwargs(cls=p.AbstractPlotter, kwargs=kwargs)
        plotter = kwargs[plotter_key]

        if not isinstance(plotter, p.AbstractPlotter):
            raise exc.PlottingException(
                "The decorator set_subplot_title was applied to a function without a Plotter class"
            )

        filename = plotter.output.filename_from_func(func=func)

        plotter = plotter.plotter_with_new_output(filename=filename)

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def set_labels(func):
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

        plotter_key = key_for_cls_from_kwargs(cls=p.AbstractPlotter, kwargs=kwargs)
        plotter = kwargs[plotter_key]

        title = plotter.title.title_from_func(func=func)
        yunits = plotter.ylabel.units_from_func(func=func, for_ylabel=True)
        xunits = plotter.xlabel.units_from_func(func=func, for_ylabel=False)

        plotter = plotter.plotter_with_new_labels(
            title_label=title, ylabel_units=yunits, xlabel_units=xunits
        )

        filename = plotter.output.filename_from_func(func=func)

        plotter = plotter.plotter_with_new_output(filename=filename)

        kpc_per_scaled = kpc_per_scaled_of_object_from_kwargs(kwargs=kwargs)

        plotter = plotter.plotter_with_new_units(conversion_factor=kpc_per_scaled)

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def kpc_per_scaled_of_object_from_kwargs(kwargs):

    kpc_per_scaled = None

    for key, value in kwargs.items():
        if hasattr(value, "kpc_per_scaled"):
            return value.kpc_per_scaled

    return kpc_per_scaled
