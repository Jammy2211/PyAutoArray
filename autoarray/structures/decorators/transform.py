import numpy as np
from functools import wraps

from typing import Union

from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


def transform(func):
    """
    Checks whether the input Grid2D of (y,x) coordinates have previously been transformed. If they have not
    been transformed then they are transformed.

    Parameters
    ----------
    func
        A function where the input grid is the grid whose coordinates are transformed.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Grid2D, Grid2DIrregular]:
        """
        This decorator checks whether the input grid has been transformed to the reference frame of the class
        that owns the function. If it has not been transformed, it is transformed.

        A function call which uses this decorator often has many subsequent function calls which also use the
        decorator. To ensure the grid is only transformed once, the `is_transformed` keyword is used to track
        whether the grid has been transformed.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid
            The (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
            A grid_like object whose coordinates may be transformed.
        """

        if not kwargs.get("is_transformed"):

            kwargs = {"is_transformed": True}

            transformed_grid = obj.transformed_to_reference_frame_grid_from(
                grid, **kwargs
            )

            result = func(obj, transformed_grid, *args, **kwargs)

        else:
            result = func(obj, grid, *args, **kwargs)

        return result

    return wrapper
