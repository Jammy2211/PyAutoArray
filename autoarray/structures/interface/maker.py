import numpy as np

from typing import List, Union

from autoarray.structures.arrays.irregular import ArrayIrregular
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.vectors.irregular import VectorYX2DIrregular
from autoarray.structures.vectors.uniform import VectorYX2D


class StructureMaker:
    def __init__(self, func, obj, grid, *args, **kwargs):
        self.func = func
        self.obj = obj
        self.grid = grid
        self.args = args
        self.kwargs = kwargs

    @property
    def mask(self):
        return self.grid.mask

    @property
    def over_sample(self):
        return self.grid.over_sample

    def via_grid_2d(self, result) -> Union[Array2D, "Grid2D"]:
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
        raise NotImplementedError

    def via_grid_2d_irr(self, result) -> Union[ArrayIrregular, Grid2DIrregular, List]:
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
        raise NotImplementedError

    def via_grid_1d(self, result) -> Union[Array1D, Grid2D]:
        raise NotImplementedError


class ArrayMaker(StructureMaker):
    @property
    def structure(self):
        grid = self.grid

        if isinstance(self.grid, Grid2D):
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        elif isinstance(self.grid, Grid2DIrregular):
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        elif isinstance(self.grid, Grid1D):
            grid = self.grid.grid_2d_radial_projected_from()
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        else:
            result = self.func(self.obj, grid, *self.args, **self.kwargs)

        return self.result_from(result=result)

    def result_from(self, result):
        if isinstance(self.grid, Grid2D):
            result_func = self.via_grid_2d
        elif isinstance(self.grid, Grid2DIrregular):
            result_func = self.via_grid_2d_irr
        elif isinstance(self.grid, Grid1D):
            result_func = self.via_grid_1d

        if not isinstance(result, list):
            return result_func(result)
        return [result_func(res) for res in result]

    def via_grid_2d(self, result) -> Union[Array2D, "Grid2D"]:
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
        return Array2D(values=result, mask=self.mask)

    def via_grid_2d_irr(self, result) -> Union[ArrayIrregular, Grid2DIrregular, List]:
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
        return ArrayIrregular(values=result)

    def via_grid_1d(self, result) -> Union[Array1D, Grid2D]:
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
        return Array1D(values=result, mask=self.mask)


class GridMaker(StructureMaker):
    @property
    def structure(self):
        grid = self.grid

        if isinstance(self.grid, Grid2D):
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        elif isinstance(self.grid, Grid2DIrregular):
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        elif isinstance(self.grid, Grid1D):
            grid = self.grid.grid_2d_radial_projected_from()
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        else:
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        #  raise exc.GridException("Invalid type")

        if isinstance(self.grid, Grid2D):
            result_func = self.via_grid_2d
        elif isinstance(self.grid, Grid2DIrregular):
            result_func = self.via_grid_2d_irr
        elif isinstance(self.grid, Grid1D):
            result_func = self.via_grid_1d
        else:
            return result

        if not isinstance(result, list):
            return result_func(result)
        return [result_func(res) for res in result]

    def via_grid_2d(self, result) -> Union[Array2D, "Grid2D"]:
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
        return Grid2D(values=result, mask=self.mask, over_sample=self.over_sample)

    def via_grid_2d_irr(self, result) -> Union[ArrayIrregular, Grid2DIrregular, List]:
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
        return Grid2DIrregular(values=result)

    def via_grid_1d(self, result) -> Union[Array1D, Grid2D]:
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
        return Grid2D(values=result, mask=self.mask.derive_mask.to_mask_2d)


class VectorYXMaker(StructureMaker):
    """
    This decorator homogenizes the input of a "grid_like" 2D vector_yx (`Grid2D`, `Grid2DIrregular` or `Grid1D`)
    into a function.

    It allows these classes to be interchangeably input into a function, such that the grid is used to evaluate
    the function at every (y,x) coordinates of the grid using specific functionality of the input grid.

    The grid_like objects `Grid2D` and `Grid2DIrregular` are input into the function as a slimmed 2D NumPy array
    of shape [total_coordinates, 2] where the second dimension stores the (y,x)  For a `Grid2D`, the
    function is evaluated using its `OverSample` object.

    The outputs of the function are converted from a 1D or 2D NumPy Array2D to an `Array2D`, `Grid2D`,
    `ArrayIrregular` or `Grid2DIrregular` objects, whichever is applicable as follows:

    - If the function returns (y,x) coordinates at every input point, the returned results are a `Grid2D`
    or `Grid2DIrregular` vector_yx, the same vector_yx as the input.

    - If the function returns scalar values at every input point and a `Grid2D` is input, the returned results are
    an `Array2D` vector_yx which uses the same dimensions and mask as the `Grid2D`.

    - If the function returns scalar values at every input point and `Grid2DIrregular` are input, the returned
    results are a `ArrayIrregular` object with vector_yx resembling that of the `Grid2DIrregular`.

    If the input array is not a `Grid2D` vector_yx (e.g. it is a 2D NumPy array) the output is a NumPy array.

    Parameters
    ----------
    obj
        An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
    grid : Grid2D or Grid2DIrregular
        A grid_like object of (y,x) coordinates on which the function values are evaluated.

    Returns
    -------
        The function values evaluated on the grid with the same vector_yx as the input grid_like object.
    """

    @property
    def structure(self):
        grid = self.grid

        if isinstance(self.grid, Grid2D):
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        elif isinstance(self.grid, Grid2DIrregular):
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        elif isinstance(self.grid, Grid1D):
            grid = self.grid.grid_2d_radial_projected_from()
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        else:
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        # raise exc.GridException("Invalid type")

        if result is not None:
            if isinstance(self.grid, Grid2D):
                result_func = self.via_grid_2d
            elif isinstance(self.grid, Grid2DIrregular):
                result_func = self.via_grid_2d_irr
            elif isinstance(self.grid, Grid1D):
                result_func = self.via_grid_1d

            if not isinstance(result, list):
                return result_func(result)
            return [result_func(res) for res in result]

    def via_grid_2d(self, result) -> Union[Array2D, "Grid2D"]:
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
        return VectorYX2D(values=result, grid=self.grid, mask=self.grid.mask)

    def via_grid_2d_irr(self, result) -> Union[ArrayIrregular, Grid2DIrregular, List]:
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
        return VectorYX2DIrregular(values=result, grid=self.grid)
