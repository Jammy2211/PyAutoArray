from functools import wraps
import numpy as np
from typing import List, Union

from autoarray.structures.decorators.abstract import AbstractMaker
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.vectors.irregular import VectorYX2DIrregular
from autoarray.structures.vectors.uniform import VectorYX2D


class VectorYXMaker(AbstractMaker):
    def via_grid_2d(self, result) -> Union[VectorYX2D, List[VectorYX2D]]:
        if not isinstance(result, list):
            return VectorYX2D(values=result, grid=self.grid, mask=self.grid.mask)
        return [
            VectorYX2D(values=res, grid=self.grid, mask=self.grid.mask)
            for res in result
        ]

    def via_grid_2d_irr(
        self, result
    ) -> Union[VectorYX2DIrregular, List[VectorYX2DIrregular]]:
        if not isinstance(result, list):
            return VectorYX2DIrregular(values=result, grid=self.grid)
        return [VectorYX2DIrregular(values=res, grid=self.grid) for res in result]


def to_vector_yx(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of coordinates and return a 2D ndarray
    which is converted to a `VectorYX2D` or `VectorYX2DIrregular` object.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of coordinates.

    Returns
    -------
        A function that has its outputs homogenized to `VectorYX2D` or `VectorYX2DIrregular` objects.
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular],
        xp=np,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, VectorYX2D, VectorYX2DIrregular, List]:
        return VectorYXMaker(
            func=func, obj=obj, grid=grid, xp=xp, *args, **kwargs
        ).result

    return wrapper
