from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
from functools import wraps

from autoarray.structures import abstract_structure, arrays, frames, grids, lines
from autoarray.dataset import abstract_dataset, interferometer as inter
from autoarray.fit import fit as f
from autoarray.inversion import mappers


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

        for arg in args:
            if isinstance(arg, AbstractPlotter):

                if arg._mat_plot_1d is not None:
                    mat_plot = arg._mat_plot_1d

                    title = mat_plot.title.title_from_func(func=func)
                    yunits = mat_plot.ylabel.units_from_func(func=func, for_ylabel=True)
                    xunits = mat_plot.xlabel.units_from_func(
                        func=func, for_ylabel=False
                    )
                    filename = mat_plot.output.filename_from_func(func=func)
                    kpc_per_scaled = kpc_per_scaled_of_object_from_kwargs(kwargs=kwargs)

                    mat_plot = mat_plot.mat_plot_with_new_labels(
                        title_label=title, ylabel_units=yunits, xlabel_units=xunits
                    )

                    mat_plot = mat_plot.mat_plot_with_new_output(filename=filename)
                    mat_plot = mat_plot.mat_plot_with_new_units(
                        conversion_factor=kpc_per_scaled
                    )

                    arg.mat_plot_1d = mat_plot

                if arg._mat_plot_2d is not None:

                    mat_plot = arg._mat_plot_2d

                    title = mat_plot.title.title_from_func(func=func)
                    yunits = mat_plot.ylabel.units_from_func(func=func, for_ylabel=True)
                    xunits = mat_plot.xlabel.units_from_func(
                        func=func, for_ylabel=False
                    )
                    filename = mat_plot.output.filename_from_func(func=func)
                    kpc_per_scaled = kpc_per_scaled_of_object_from_kwargs(kwargs=kwargs)

                    mat_plot = mat_plot.mat_plot_with_new_labels(
                        title_label=title, ylabel_units=yunits, xlabel_units=xunits
                    )

                    mat_plot = mat_plot.mat_plot_with_new_output(filename=filename)
                    mat_plot = mat_plot.mat_plot_with_new_units(
                        conversion_factor=kpc_per_scaled
                    )

                    arg.mat_plot_2d = mat_plot

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
        self._mat_plot_1d = mat_plot_1d
        self.visuals_2d = visuals_2d
        self.include_2d = include_2d
        self.mat_plot_2d = mat_plot_2d
        self._mat_plot_2d = mat_plot_2d

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
