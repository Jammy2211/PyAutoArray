from functools import wraps
import numpy as np
from typing import Union

from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


def transform(func=None, *, rotate_back=False):
    """
    Checks whether the input Grid2D of (y,x) coordinates have previously been transformed. If they have not
    been transformed then they are transformed.

    Can be used with or without arguments::

        @transform
        def convergence_2d_from(self, grid, xp=np, **kwargs): ...

        @transform(rotate_back=True)
        def deflections_yx_2d_from(self, grid, xp=np, **kwargs): ...

    When ``rotate_back=True``, after the decorated function returns its result the decorator automatically
    rotates the output vector back from the profile's reference frame to the original observer frame.
    This eliminates the need for deflection methods to manually call
    ``self.rotated_grid_from_reference_frame_from``.

    Parameters
    ----------
    func
        A function where the input grid is the grid whose coordinates are transformed.
    rotate_back
        If ``True``, the result is rotated back from the profile's reference frame after evaluation.
        Use this for functions that return vector quantities (e.g. deflection angles) computed in the
        profile's rotated frame.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    def decorator(func):
        @wraps(func)
        def wrapper(
            obj: object,
            grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
            xp=np,
            *args,
            **kwargs,
        ) -> Union[np.ndarray, Grid2D, Grid2DIrregular]:
            """
            This decorator checks whether the input grid has been transformed to the reference frame of the class
            that owns the function. If it has not been transformed, it is transformed.

            The transform state is tracked via the ``is_transformed`` property on the grid object itself.
            When a decorated function calls another decorated function with the same (already-transformed)
            grid, the flag prevents the grid from being transformed a second time.

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

            if not getattr(grid, "is_transformed", False):
                transformed_grid = obj.transformed_to_reference_frame_grid_from(
                    grid, xp, **kwargs
                )
                transformed_grid.is_transformed = True

                result = func(obj, transformed_grid, xp, *args, **kwargs)

            else:
                result = func(obj, grid, xp, *args, **kwargs)

            if rotate_back:
                result = obj.rotated_grid_from_reference_frame_from(
                    grid=result, xp=xp
                )

            return result

        return wrapper

    if func is not None:
        # Called without arguments: @transform
        return decorator(func)

    # Called with arguments: @transform(rotate_back=True)
    return decorator
