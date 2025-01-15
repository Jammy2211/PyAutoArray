import numpy as np
from functools import wraps


from typing import List, Union

from autoarray.structures.arrays.irregular import ArrayIrregular
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


def over_sample(func):
    """
    Homogenize the inputs and outputs of functions that take 1D or 2D grids of coordinates and return a 1D ndarray
    which is converted to an `Array2D`, `ArrayIrregular` or `Array1D` object.

    Parameters
    ----------
    func
        A function which computes a set of values from a 1D or 2D grid of coordinates.

    Returns
    -------
        A function that has its outputs homogenized to `Array2D`, `ArrayIrregular` or `Array1D` objects.
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Array1D, Array2D, ArrayIrregular, List]:
        """

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid
            A grid_like object of coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        if isinstance(grid, Grid2DIrregular) or isinstance(grid, Grid1D):
            return func(obj=obj, grid=grid, *args, **kwargs)

        if obj is not None:
            values = func(obj, grid.over_sampled, *args, **kwargs)
        else:
            values = func(grid.over_sampled, *args, **kwargs)

        return grid.over_sampler.binned_array_2d_from(array=values)

    return wrapper
