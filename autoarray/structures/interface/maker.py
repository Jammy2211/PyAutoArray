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
    def __init__(self, func, obj, grid, is_vector = False, *args, **kwargs):

        self.func = func
        self.obj = obj
        self.grid = grid
        self.is_vector = is_vector
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
                result = grid.over_sample_func.array_via_func_from(func=self.func, cls=self.obj, *self.args, **self.kwargs)
            else:
                result = self.func(self.obj, grid, *self.args, **self.kwargs)
        elif isinstance(self.grid, Grid2DIrregular):
            result = self.func(self.obj, grid, *self.args, **self.kwargs)
        elif isinstance(self.grid, Grid1D):
            grid = self.grid.grid_2d_radial_projected_from()
            result = self.func(self.obj, grid, *self.args, **self.kwargs)

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
        if len(result.shape) == 1:
            return ArrayIrregular(values=result)
        elif len(result.shape) == 2:
            if isinstance(result, Grid2DTransformedNumpy):
                return Grid2DIrregularTransformed(values=result)
            return Grid2DIrregular(values=result)

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


