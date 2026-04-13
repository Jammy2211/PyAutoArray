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

    **Frame conventions and rotate_back**

    This decorator transforms the input grid into the profile's reference frame (centred on the
    profile centre and rotated by its position angle) before calling the decorated function.

    For **scalar** quantities (convergence, potential), the returned value is frame-invariant — no
    back-rotation is needed, so use ``@transform`` without ``rotate_back``.

    For **vector** quantities (e.g. deflection angles), whether back-rotation is needed depends on
    which frame the returned components are expressed in:

    - If the function computes vector components using the rotated grid coordinates (i.e. the
      components are expressed in the profile's frame), they must be rotated back to the observer
      frame before use in ray-tracing. Set ``rotate_back=True`` for this case.

    - If the function reconstructs observer-frame components from scalar quantities (e.g. computing
      a radial deflection magnitude and converting to Cartesian using observer-frame geometry), the
      result is already in the observer frame and ``rotate_back`` should remain ``False``.

    When ``rotate_back=True``, the decorator calls ``obj.rotated_grid_from_reference_frame_from``
    on the result after evaluation, applying the inverse rotation by the profile's position angle.

    For **spin-2** quantities (shear), the transformation law uses twice the profile angle. This
    is not handled by ``rotate_back`` — shear methods must apply the 2-theta rotation manually.

    Parameters
    ----------
    func
        A function where the input grid is the grid whose coordinates are transformed.
    rotate_back
        If ``True``, the result is rotated back from the profile's reference frame after
        evaluation. Use this when the decorated function returns vector components that were
        computed in the profile's rotated coordinate basis and need to be expressed in the
        original observer frame.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    def decorator(func):
        @wraps(func)
        def wrapper(
            obj: object,
            grid: Union[np.ndarray, Grid2D, Grid2DIrregular],
            xp=np,
            *args,
            **kwargs,
        ) -> Union[np.ndarray, Grid2D, Grid2DIrregular]:
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
