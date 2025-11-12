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
        """
        Convert the result of a decorated function which receives as input a `Grid2D` object to a `VectorYX2D` object.

        If the result returns a list, a list of `VectorYX2D` objects is returned.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to an VectorYX2D or list of VectorYX2D
            objects.
        """
        if not isinstance(result, list):
            return VectorYX2D(values=result, grid=self.grid, mask=self.grid.mask)
        return [
            VectorYX2D(values=res, grid=self.grid, mask=self.grid.mask)
            for res in result
        ]

    def via_grid_2d_irr(
        self, result
    ) -> Union[VectorYX2DIrregular, List[VectorYX2DIrregular]]:
        """
        Convert the result of a decorated function which receives as input a `VectorYX2DIrregular` object to
        an `VectorYX2DIrregular` object.

        If the result returns a list, a list of `VectorYX2DIrregular` objects is returned.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to an VectorYX2DIrregular or list of
            VectorYX2DIrregular objects.
        """
        if not isinstance(result, list):
            return VectorYX2DIrregular(values=result, grid=self.grid)
        return [VectorYX2DIrregular(values=res, grid=self.grid) for res in result]


def to_vector_yx(func):
    """
    Homogenize the inputs and outputs of functions that take 1D or 2D grids of coordinates and return a 1D ndarray
    which is converted to an `VectorYX2D` or `VectorYX2DIrregular` object.

    Parameters
    ----------
    func
        A function which computes a set of values from a 1D or 2D grid of coordinates.

    Returns
    -------
        A function that has its outputs homogenized to `VectorYX2D` or `VectorYX2DIrregular` objects.
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
        xp=np,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, VectorYX2D, VectorYX2DIrregular, List]:
        """
        This decorator homogenizes the input of a "grid_like" 2D structure (`Grid2D`, `Grid2DIrregular` or `Grid1D`)
        into a function which outputs a vector-like structure (`VectorYX2D` or `VectorYX2DIrregular`).

        It allows these classes to be interchangeably input into a function, such that the grid is used to evaluate
        the function at every (y,x) coordinates of the grid using specific functionality of the input grid.

        The grid_like objects `Grid2D` and `Grid2DIrregular` are input into the function as a slimmed 2D ndarray array
        of shape [total_coordinates, 2] where the second dimension stores the (y,x)

        There are three types of consistent data structures and therefore decorated function mappings:

        - Uniform (`Grid2D` -> `VectorYX2D`): 2D structures defined on a uniform grid of data points. Both structures are
        defined according to a `Mask2D`, which the maker object ensures is passed through self consistently.

        - Irregular (`Grid2DIrregular` -> `VectorYX2DIrregular`: 2D structures defined on an irregular grid of data points,
        Neither structure is defined according to a mask and the maker sures the lack of a mask does not prevent the
        function from being evaluated.

        - 1D (`Grid1D` -> `Grid2D`): 1D structures defined on a 1D grid of data points. These are not applicable
        for vector-like structures and are not supported by this decorator.

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

        return VectorYXMaker(
            func=func, obj=obj, grid=grid, xp=xp, *args, **kwargs
        ).result

    return wrapper
