from functools import wraps
import numpy as np
from typing import List, Union

from autoarray.structures.decorators.abstract import AbstractMaker
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


class GridMaker(AbstractMaker):
    def via_grid_2d(self, result) -> Union[Grid2D, List[Grid2D]]:
        if not isinstance(result, list):
            return Grid2D(
                values=result,
                mask=self.mask,
                over_sample_size=self.over_sample_size,
                over_sampled=getattr(result, "over_sampled", None),
                over_sampler=getattr(result, "over_sampler", None),
            )

        return [
            Grid2D(
                values=res,
                mask=self.mask,
                over_sample_size=self.over_sample_size,
                over_sampled=getattr(res, "over_sampled", None),
                over_sampler=getattr(res, "over_sampler", None),
            )
            for res in result
        ]

    def via_grid_2d_irr(self, result) -> Union[Grid2DIrregular, List[Grid2DIrregular]]:
        if not isinstance(result, list):
            return Grid2DIrregular(values=result)
        return [Grid2DIrregular(values=res) for res in result]

    def via_grid_1d(self, result) -> Union[Grid2D, List[Grid2D]]:
        if not isinstance(result, list):
            return Grid2D(values=result, mask=self.mask.derive_mask.to_mask_2d)
        return [
            Grid2D(values=res, mask=self.mask.derive_mask.to_mask_2d) for res in result
        ]


def to_grid(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of coordinates and return a 2D ndarray
    which is converted to a `Grid2D` or `Grid2DIrregular` object.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of coordinates.

    Returns
    -------
        A function that has its outputs homogenized to `Grid2D` or `Grid2DIrregular` objects.
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
        xp=np,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Grid2D, Grid2DIrregular, List]:
        return GridMaker(func=func, obj=obj, grid=grid, xp=xp, *args, **kwargs).result

    return wrapper
