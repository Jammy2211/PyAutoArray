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
from autoarray.structures.vectors.irregular import VectorYX2DIrregular
from autoarray.structures.vectors.uniform import VectorYX2D
from autoarray.structures.decorators import util


class ArrayMaker(AbstractMaker):

    def via_grid_2d(self, result) -> Union[Array2D, List[Array2D]]:
        """
        Convert a result from an ndarray to an aa.Array2D or aa.Grid2D structure, where the conversion depends on
        type(result) as follows:

        - 1D np.ndarray   -> aa.Array2D
        - 2D np.ndarray   -> aa.Grid2D

        This function is used by the grid_2d_to_structure decorator to convert the output result of a function
        to an autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        
        if not isinstance(result, list):
            return Array2D(values=result, mask=self.mask)
        return [Array2D(values=res, mask=self.mask) for res in result]

    def via_grid_2d_irr(self, result) -> Union[ArrayIrregular, List[ArrayIrregular]]:
        """
        Convert a result from a non autoarray structure to an aa.ArrayIrregular or aa.Grid2DIrregular structure, where
        the conversion depends on type(result) as follows:

        - 1D np.ndarray   -> aa.ArrayIrregular
        - 2D np.ndarray   -> aa.Grid2DIrregular
        - [1D np.ndarray] -> [aa.ArrayIrregular]
        - [2D np.ndarray] -> [aa.Grid2DIrregular]

        This function is used by the grid_2d_to_structure decorator to convert the output result of a function
        to an autoarray structure when a `Grid2DIrregular` instance is passed to the decorated function.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        if not isinstance(result, list):
            return ArrayIrregular(values=result)
        return [ArrayIrregular(values=res) for res in result]

    def via_grid_1d(self, result) -> Union[Array1D, List[Array1D]]:
        """
        Convert a result from an ndarray to an aa.Array2D or aa.Grid2D structure, where the conversion depends on
        type(result) as follows:

        .. code-block:: bash

            - 1D np.ndarray   -> aa.Array2D
            - 2D np.ndarray   -> aa.Grid2D

        This function is used by the grid_2d_to_structure decorator to convert the output result of a function
        to an autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        if not isinstance(result, list):
            return Array1D(values=result, mask=self.mask)
        return [Array1D(values=res, mask=self.mask) for res in result]


def grid_2d_to_array(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of (y,x) coordinates that return the results
    as a NumPy array.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of (y,x) coordinates.

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
    ) -> Union[np.ndarray, Array2D, ArrayIrregular, Grid2D, Grid2DIrregular]:
        """
        This decorator homogenizes the input of a "grid_like" 2D structure (`Grid2D`, `Grid2DIrregular` or `Grid1D`)
        into a function.

        It allows these classes to be interchangeably input into a function, such that the grid is used to evaluate
        the function at every (y,x) coordinates of the grid using specific functionality of the input grid.

        The grid_like objects `Grid2D` and `Grid2DIrregular` are input into the function as a slimmed 2D NumPy array
        of shape [total_coordinates, 2] where the second dimension stores the (y,x)  For a `Grid2D`, the
        function is evaluated using its `OverSample` object.

        The outputs of the function are converted from a 1D or 2D NumPy Array2D to an `Array2D`, `Grid2D`,
        `ArrayIrregular` or `Grid2DIrregular` objects, whichever is applicable as follows:

        - If the function returns (y,x) coordinates at every input point, the returned results are a `Grid2D`
        or `Grid2DIrregular` structure, the same structure as the input.

        - If the function returns scalar values at every input point and a `Grid2D` is input, the returned results are
        an `Array2D` structure which uses the same dimensions and mask as the `Grid2D`.

        - If the function returns scalar values at every input point and `Grid2DIrregular` are input, the returned
        results are a `ArrayIrregular` object with structure resembling that of the `Grid2DIrregular`.

        If the input array is not a `Grid2D` structure (e.g. it is a 2D NumPy array) the output is a NumPy array.

        This decorator serves the same purpose as the `grid_2d_to_structure` decorator, but it deals with functions
        whose output is a list of results as opposed to a single NumPy array. It simply iterates over these lists to
        perform the same conversions as `grid_2d_to_structure`.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid2D or Grid2DIrregular
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """
        
        return ArrayMaker(func=func, obj=obj, grid=grid, *args, **kwargs).result

    return wrapper