import numpy as np
from functools import wraps


from typing import List, Union

from autoarray.structures.arrays.irregular import ArrayIrregular
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.decorators.abstract import AbstractMaker
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


class ArrayMaker(AbstractMaker):
    def via_grid_2d(self, result) -> Union[Array2D, List[Array2D]]:
        """
        Convert the result of a decorated function which receives as input a `Grid2D` object to an `Array2D` object.

        If the result returns a list, a list of `Array2D` objects is returned.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to an Array2D or list of Array2D objects.
        """

        if not isinstance(result, list):
            return Array2D(values=result, mask=self.mask)
        return [Array2D(values=res, mask=self.mask) for res in result]

    def via_grid_2d_irr(self, result) -> Union[ArrayIrregular, List[ArrayIrregular]]:
        """
        Convert the result of a decorated function which receives as input a `Grid2DIrregular` object to an `ArrayIrregular`
        object.

        If the result returns a list, a list of `ArrayIrregular` objects is returned.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to an ArrayIrregular or list of
            ArrayIrregular objects.
        """
        if not isinstance(result, list):
            return ArrayIrregular(values=result)
        return [ArrayIrregular(values=res) for res in result]

    def via_grid_1d(self, result) -> Union[Array1D, List[Array1D]]:
        """
        Convert the result of a decorated function which receives as input a `Grid1D` object to an `Array1D` object.

        If the result returns a list, a list of `Array1D` objects is returned.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to an Array1D or list of Array1D objects.
        """
        if not isinstance(result, list):
            return Array1D(values=result, mask=self.mask)
        return [Array1D(values=res, mask=self.mask) for res in result]


def to_array(func):
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
        xp=np,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Array1D, Array2D, ArrayIrregular, List]:
        """
        This decorator homogenizes the input of a "grid_like" 2D structure (`Grid2D`, `Grid2DIrregular` or `Grid1D`)
        into a function which outputs an array-like structure (`Array2D`, `ArrayIrregular` or `Array1D`).

        It allows these classes to be interchangeably input into a function, such that the grid is used to evaluate
        the function at every (y,x) coordinates of the grid using specific functionality of the input grid.

        The grid_like objects `Grid2D` and `Grid2DIrregular` are input into the function as a slimmed 2D ndarray array
        of shape [total_coordinates, 2] where the second dimension stores the (y,x)

        There are three types of consistent data structures and therefore decorated function mappings:

        - Uniform (`Grid2D` -> `Array`): 2D structures defined on a uniform grid of data points. Both structures are
        defined according to a `Mask2D`, which the maker object ensures is passed through self consistently.

        - Irregular (`Grid2DIrregular` -> `ArrayIrregular`: 2D structures defined on an irregular grid of data points,
        Neither structure is defined according to a mask and the maker sures the lack of a mask does not prevent the
        function from being evaluated.

        - 1D (`Grid1D` -> `Array1D`): 1D structures defined on a 1D grid of data points. These project the 1D grid
        to a 2D grid to ensure the function can be evaluated, and then deproject the 2D grid back to a 1D grid to
        ensure the output data structure is consistent with the input grid.

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
        return ArrayMaker(func=func, obj=obj, grid=grid, xp=xp, *args, **kwargs).result

    return wrapper
