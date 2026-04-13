import numpy as np
from functools import wraps
from typing import List, Union

from autoarray.structures.arrays.irregular import ArrayIrregular
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.decorators.abstract import AbstractMaker
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


class ArrayMaker(AbstractMaker):
    def via_grid_2d(self, result) -> Union[Array2D, List[Array2D]]:
        if not isinstance(result, list):
            return Array2D(values=result, mask=self.mask)
        return [Array2D(values=res, mask=self.mask) for res in result]

    def via_grid_2d_irr(self, result) -> Union[ArrayIrregular, List[ArrayIrregular]]:
        if not isinstance(result, list):
            return ArrayIrregular(values=result)
        return [ArrayIrregular(values=res) for res in result]


def to_array(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of coordinates and return a 1D ndarray
    which is converted to an `Array2D` or `ArrayIrregular` object.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of coordinates.

    Returns
    -------
        A function that has its outputs homogenized to `Array2D` or `ArrayIrregular` objects.
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular],
        xp=np,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Array2D, ArrayIrregular, List]:
        return ArrayMaker(func=func, obj=obj, grid=grid, xp=xp, *args, **kwargs).result

    return wrapper
