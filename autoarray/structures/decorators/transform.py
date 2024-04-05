import numpy as np
from functools import wraps

from typing import Union

from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


def transform(func):
    """
    Checks whether the input Grid2D of (y,x) coordinates have previously been transformed. If they have not \
    been transformed then they are transformed.

    Parameters
    ----------
    func : (profile, grid *args, **kwargs) -> Object
        A function where the input grid is the grid whose coordinates are transformed.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        cls,
        grid: Union[
            np.ndarray,
            Grid2D,
            Grid2DIrregular,
        ],
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Grid2D, Grid2DIrregular]:
        """

        Parameters
        ----------
        cls : Profile
            The class that owns the function.
        grid
            The (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
            A grid_like object whose coordinates may be transformed.
        """

        if not kwargs.get("is_transformed"):
            kwargs = {"is_transformed": True}

            transformed_grid = cls.transformed_to_reference_frame_grid_from(
                grid, **kwargs
            )

            result = func(cls, transformed_grid, *args, **kwargs)

        else:
            result = func(cls, grid, *args, **kwargs)

        return result

    return wrapper
