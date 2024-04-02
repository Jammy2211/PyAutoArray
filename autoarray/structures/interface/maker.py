import numpy as np

from typing import List, Union

from autoarray.abstract_ndarray import AbstractNDArray
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.irregular import ArrayIrregular
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.irregular_2d import Grid2DIrregularTransformed
from autoarray.structures.grids.transformed_2d import Grid2DTransformed
from autoarray.structures.grids.transformed_2d import Grid2DTransformedNumpy
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.vectors.irregular import VectorYX2DIrregular
from autoarray.structures.vectors.uniform import VectorYX2D
from autoconf import conf


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

    @property
    def result_basic(self):

        grid = np.array([[1.0, 1.0]])

        return self.func(self.obj, grid, *self.args, **self.kwargs)

    @property
    def result_type(self):

        result_basic = self.result_basic

        if isinstance(result_basic, list):
            result_basic = result_basic[0]

        if len(result_basic.shape) == 1:
            return "array"
        elif len(result_basic.shape) == 2:
            return "grid"

    @property
    def structure(self):

        result = None

        grid = self.grid

        if isinstance(self.grid, Grid2D):
            if self.result_type == "array":
                result = grid.over_sample_func.array_via_func_from(func=self.func, cls=self.obj)
            else:
                result = self.func(self.obj, grid, *self.args, **self.kwargs)
        elif isinstance(self.grid, Grid2DIrregular):
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        elif isinstance(self.grid, Grid1D):
            grid = self.grid.grid_2d_radial_projected_from()
            result = self.func(self.obj, grid, *self.args, **self.kwargs)

        if result is not None:

            if not isinstance(result, list):
                if isinstance(self.grid, Grid2D):
                    return self.via_grid_2d(result=result)
                elif isinstance(self.grid, Grid2DIrregular):
                    return self.via_grid_2d_irr(result=result)
                elif isinstance(self.grid, Grid1D):
                    return self.via_grid_1d(result=result)
            else:
                if isinstance(self.grid, Grid2D):
                    return self.via_grid_2d_list(result=result)
                elif isinstance(self.grid, Grid2DIrregular):
                    return self.via_grid_2d_irr_list(result=result)
                elif isinstance(self.grid, Grid1D):
                    return self.via_grid_1d_list(result=result)

        return self.func(self.obj, self.grid, *self.args, **self.kwargs)

    def via_grid_2d(
        self, result
    ) -> Union[Array2D, "Grid2D"]:
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
        if len(result.shape) == 1:
            return Array2D(values=result, mask=self.mask)
        else:
            if isinstance(result, Grid2DTransformedNumpy):
                return Grid2DTransformed(
                    values=result, mask=self.mask, over_sample=self.over_sample
                )
            return Grid2D(
                values=result, mask=self.mask, over_sample=self.over_sample
            )

    def via_grid_2d_list(
        self, result
    ) -> List[Union[Array2D, "Grid2D"]]:
        """
        Convert a result from a list of ndarrays to a list of aa.Array2D or aa.Grid2D structure, where the conversion
        depends on type(result) as follows:

        - [1D np.ndarray] -> [aa.Array2D]
        - [2D np.ndarray] -> [aa.Grid2D]

        This function is used by the grid_like_list_to_structure-list decorator to convert the output result of a
        function to a list of autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result_list or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        result_list = []

        for result in result:
            result_list.append(self.via_grid_2d(result=result))

        return result_list

    def grid_irr_from(
        self, grid_slim: np.ndarray
    ) -> Union["Grid2DIrregular", "Grid2DIrregularTransformed"]:
        """
        Create a `Grid2DIrregular` object from a 2D NumPy array of values of shape [total_coordinates, 2],
        which are structured following this *Grid2DIrregular* instance.
        """

        from autoarray.structures.grids.transformed_2d import Grid2DTransformedNumpy

        if isinstance(grid_slim, Grid2DTransformedNumpy):
            return Grid2DIrregularTransformed(values=grid_slim)
        return Grid2DIrregular(values=grid_slim)

    def via_grid_2d_irr(
        self, result
    ) -> Union[ArrayIrregular, Grid2DIrregular, Grid2DIrregularTransformed, List]:
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

        if isinstance(result, (np.ndarray, AbstractNDArray)):
            if len(result.shape) == 1:
                return ArrayIrregular(values=result)
            elif len(result.shape) == 2:
                return self.grid_irr_from(grid_slim=result)
        elif isinstance(result, list):
            if len(result[0].shape) == 1:
                return [ArrayIrregular(values=value) for value in result]
            elif len(result[0].shape) == 2:
                return [self.grid_irr_from(grid_slim=value) for value in result]

    def via_grid_2d_irr_list(
        self, result
    ) -> List[Union[ArrayIrregular, Grid2DIrregular, Grid2DIrregularTransformed]]:
        """
        Convert a result from a list of non autoarray structures to a list of aa.ArrayIrregular or aa.Grid2DIrregular
        structures, where the conversion depends on type(result) as follows:

        ::

            - [1D np.ndarray] -> [aa.ArrayIrregular]
            - [2D np.ndarray] -> [aa.Grid2DIrregular]

        This function is used by the grid_like_list_to_structure_list decorator to convert the output result of a
        function to a list of autoarray structure when a `Grid2DIrregular` instance is passed to the decorated function.

        Parameters
        ----------
        result_list
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        if len(result[0].shape) == 1:
            return [ArrayIrregular(values=value) for value in result]
        elif len(result[0].shape) == 2:
            return [self.grid_irr_from(grid_slim=value) for value in result]

    def via_grid_1d(
        self, result
    ) -> Union[Array1D, Grid2D, Grid2DTransformed, Grid2DTransformedNumpy]:
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

        if len(result.shape) == 1:
            return Array1D(values=result, mask=self.mask)

        if isinstance(result, Grid2DTransformedNumpy):
            return Grid2DTransformed(values=result, mask=self.mask)
        return Grid2D(values=result, mask=self.mask.derive_mask.to_mask_2d)

    def via_grid_1d_list(
        self, result
    ) -> List[Union[Array1D, Grid2D, Grid2DTransformed, Grid2DTransformedNumpy]]:
        """
        Convert a result from a list of ndarrays to a list of aa.Array2D or aa.Grid2D structure, where the conversion
        depends on type(result) as follows:

        .. code-block:: bash

            - [1D np.ndarray] -> [aa.Array2D]
            - [2D np.ndarray] -> [aa.Grid2D]

        This function is used by the grid_like_list_to_structure-list decorator to convert the output result of a
        function to a list of autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result_list
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        result_list = []

        for result in result:
            result_list.append(self.via_grid_1d(result=result))

        return result_list
