from functools import wraps

from typing import Union

from autoarray import exc
from autoarray.structures.arrays.irregular import ArrayIrregular
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D


def to_projected(func):
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
        obj,
        grid: Union[Grid1D, Grid2D, Grid2DIrregular],
        *args,
        **kwargs,
    ) -> Union[Array1D, ArrayIrregular]:
        """
        This decorator homogenizes the output of functions which compute a 1D result, by inspecting the output
        and converting the result to an `Array1D` object if it is uniformly spaced and a `ArrayIrregular` object if
        it is irregular. "grid_like" 2D structure (`Grid2D`),

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        result = func(obj, grid, *args, **kwargs)

        if isinstance(grid, Grid2D) or isinstance(grid, Grid1D):
            return Array1D.no_mask(values=result, pixel_scales=grid.pixel_scale)
        elif isinstance(grid, Grid2DIrregular):
            return ArrayIrregular(values=result)

        raise exc.GridException(
            "You cannot input a NumPy array to a `quantity_1d_from` method."
        )

    return wrapper
