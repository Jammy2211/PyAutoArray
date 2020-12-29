import numpy as np
from functools import wraps
import copy

from autoarray import exc
from autoarray.plot.mat_wrap import plotter as p, include as inc, visuals as vis


def key_for_cls_from_kwargs(cls, kwargs):

    key = None

    for key, value in kwargs.items():
        if isinstance(value, cls):
            return key


def set_include_1d(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_1d_key = key_for_cls_from_kwargs(cls=inc.Include1D, kwargs=kwargs)

        if include_1d_key is not None:
            include_1d = kwargs[include_1d_key]
        else:
            include_1d = inc.Include1D()
            include_1d_key = "include_1d"

        kwargs[include_1d_key] = include_1d

        return func(*args, **kwargs)

    return wrapper


def set_include_2d(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_2d_key = key_for_cls_from_kwargs(cls=inc.Include2D, kwargs=kwargs)

        if include_2d_key is not None:
            include_2d = kwargs[include_2d_key]
        else:
            include_2d = inc.Include2D()
            include_2d_key = "include_2d"

        kwargs[include_2d_key] = include_2d

        return func(*args, **kwargs)

    return wrapper


def set_plotter_1d_for_subplot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        plotter_1d_key = key_for_cls_from_kwargs(cls=p.Plotter1D, kwargs=kwargs)

        if plotter_1d_key is not None:
            plotter_1d = kwargs[plotter_1d_key]
        else:
            plotter_1d = p.Plotter1D()
            plotter_1d_key = "plotter_1d"

        plotter_1d.set_for_subplot(for_subplot=True)

        kwargs[plotter_1d_key] = plotter_1d

        return func(*args, **kwargs)

    return wrapper


def set_plotter_2d_for_subplot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        plotter_2d_key = key_for_cls_from_kwargs(cls=p.Plotter2D, kwargs=kwargs)

        if plotter_2d_key is not None:
            plotter_2d = kwargs[plotter_2d_key]
        else:
            plotter_2d = p.Plotter2D()
            plotter_2d_key = "plotter_2d"

        plotter_2d.set_for_subplot(for_subplot=True)

        kwargs[plotter_2d_key] = plotter_2d

        return func(*args, **kwargs)

    return wrapper


def set_plot_defaults_1d(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        visuals_1d_key = key_for_cls_from_kwargs(cls=vis.Visuals1D, kwargs=kwargs)

        if visuals_1d_key is not None:
            visuals_1d = kwargs[visuals_1d_key]
        else:
            visuals_1d = vis.Visuals1D()
            visuals_1d_key = "visuals_1d"

        kwargs[visuals_1d_key] = visuals_1d

        include_1d_key = key_for_cls_from_kwargs(cls=inc.Include1D, kwargs=kwargs)

        if include_1d_key is not None:
            include_1d = kwargs[include_1d_key]
        else:
            include_1d = visuals_1d.include
            include_1d_key = "include_1d"

        kwargs[include_1d_key] = include_1d

        plotter_1d_key = key_for_cls_from_kwargs(cls=p.Plotter1D, kwargs=kwargs)

        if plotter_1d_key is not None:
            plotter_1d = kwargs[plotter_1d_key]
        else:
            plotter_1d = visuals_1d.plotter
            plotter_1d_key = "plotter_1d"

        kwargs[plotter_1d_key] = plotter_1d

        return func(*args, **kwargs)

    return wrapper


def set_plot_defaults_2d(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        visuals_2d_key = key_for_cls_from_kwargs(cls=vis.Visuals2D, kwargs=kwargs)

        if visuals_2d_key is not None:
            visuals_2d = kwargs[visuals_2d_key]
        else:
            visuals_2d = vis.Visuals2D()
            visuals_2d_key = "visuals_2d"

        kwargs[visuals_2d_key] = visuals_2d

        include_2d_key = key_for_cls_from_kwargs(cls=inc.Include2D, kwargs=kwargs)

        if include_2d_key is not None:
            include_2d = kwargs[include_2d_key]
        else:
            include_2d = visuals_2d.include
            include_2d_key = "include_2d"

        kwargs[include_2d_key] = include_2d

        plotter_2d_key = key_for_cls_from_kwargs(cls=p.Plotter2D, kwargs=kwargs)

        if plotter_2d_key is not None:
            plotter_2d = kwargs[plotter_2d_key]
        else:
            plotter_2d = visuals_2d.plotter
            plotter_2d_key = "plotter_2d"

        kwargs[plotter_2d_key] = plotter_2d

        return func(*args, **kwargs)

    return wrapper


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
