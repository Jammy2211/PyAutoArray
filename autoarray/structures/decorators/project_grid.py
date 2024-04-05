from functools import wraps

from typing import Union

from autoarray import exc
from autoarray.structures.arrays.irregular import ArrayIrregular
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D


def project_grid(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of (y,x) coordinates that return the results
    as a NumPy array.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of (y,x) coordinates.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[Grid1D, Grid2D, Grid2DIrregular],
        *args,
        **kwargs,
    ) -> Union[Array1D, ArrayIrregular, Grid2DIrregular]:
        """
        This decorator homogenizes the input of a "grid_like" 2D structure (`Grid2D`, `Grid2DIrregular` or `Grid1D`)
        into a function. It allows these classes to be
        interchangeably input into a function, such that the grid is used to evaluate the function at every (y,x)
        coordinates of the grid using specific functionality of the input grid.

        If the `Grid2DLike` objects `Grid2D` and `Grid2DIrregular` are input into the function as a slimmed 2D NumPy
        array of shape [total_coordinates, 2] they are projected into 1D and evaluated on this 1D grid. If the
        decorator is wrapping an object with a `centre` or `angle`, the projected give aligns the angle at a 90
        degree offset, which for an ellipse is its major-axis.

        The outputs of the function are converted from a 1D ndarray to an `Array1D` or `ArrayIrregular`,
        whichever is applicable as follows:

        - If an object where the coordinates are on a uniformly spaced grid is input (e.g. `Grid1D`, the radially
        projected grid computed from a `Grid2D`) the returns values using an `Array1D` object which assumes
        uniform spacing.

        - If an object where the coordinates are on an irregular grid is input (e.g. `Grid2DIrregular`)`the function
        returns a `ArrayIrregular` object which is also irregular.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid2D or Grid2DIrregular
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        centre = (0.0, 0.0)

        if hasattr(obj, "centre"):
            if obj.centre is not None:
                centre = obj.centre

        angle = 0.0

        if hasattr(obj, "angle"):
            if obj.angle is not None:
                angle = obj.angle + 90.0

        if isinstance(grid, Grid2D):
            grid_2d_projected = grid.grid_2d_radial_projected_from(
                centre=centre, angle=angle
            )
            result = func(obj, grid_2d_projected, *args, **kwargs)
            return Array1D.no_mask(values=result, pixel_scales=grid.pixel_scale)

        elif isinstance(grid, Grid2DIrregular):
            result = func(obj, grid, *args, **kwargs)
            if len(result.shape) == 1:
                return ArrayIrregular(values=result)
            elif len(result.shape) == 2:
                return Grid2DIrregular(values=result)
        elif isinstance(grid, Grid1D):
            grid_2d_radial = grid.grid_2d_radial_projected_from(angle=angle)
            result = func(obj, grid_2d_radial, *args, **kwargs)
            return Array1D.no_mask(values=result, pixel_scales=grid.pixel_scale)

        raise exc.GridException(
            "You cannot input a NumPy array to a `quantity_1d_from` method."
        )

    return wrapper
