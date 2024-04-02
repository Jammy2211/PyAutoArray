import numpy as np
from functools import wraps

from typing import List, Union

from autoconf.exc import ConfigException

from autoarray import exc
from autoarray.abstract_ndarray import AbstractNDArray
from autoarray.structures.arrays.irregular import ArrayIrregular
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.irregular_2d import Grid2DIrregularTransformed
from autoarray.operators.over_sample.uniform import OverSampleUniform
from autoarray.operators.over_sample.iterate import OverSampleIterate
from autoarray.structures.grids.transformed_2d import Grid2DTransformed
from autoarray.structures.grids.transformed_2d import Grid2DTransformedNumpy
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.vectors.irregular import VectorYX2DIrregular
from autoarray.structures.vectors.uniform import VectorYX2D
from autoconf import conf


class StructureMaker:
    def __init__(self, grid, result):
        self.grid = grid
        self.result = result

    @property
    def mask(self):
        return self.grid.mask

    @property
    def over_sample(self):
        return self.grid.over_sample

    @property
    def structure(self):
        if not isinstance(self.result, list):
            if isinstance(self.grid, Grid2D):
                return self.via_grid_2d
            elif isinstance(self.grid, Grid2DIrregular):
                return self.via_grid_2d_irr

        else:
            if isinstance(self.grid, Grid2D):
                return self.via_grid_2d_list
            elif isinstance(self.grid, Grid2DIrregular):
                return self.via_grid_2d_irr_list

    @property
    def via_grid_2d(
        self,
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
        from autoarray.structures.grids.transformed_2d import Grid2DTransformed
        from autoarray.structures.grids.transformed_2d import Grid2DTransformedNumpy

        if len(self.result.shape) == 1:
            return Array2D(values=self.result, mask=self.mask)
        else:
            if isinstance(self.result, Grid2DTransformedNumpy):
                return Grid2DTransformed(
                    values=self.result, mask=self.mask, over_sample=self.over_sample
                )
            return Grid2D(
                values=self.result, mask=self.mask, over_sample=self.over_sample
            )

    @property
    def via_grid_2d_list(
        self,
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

        for result in self.result:
            maker = StructureMaker(grid=self.grid, result=result)
            result_list.append(maker.via_grid_2d)

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

    @property
    def via_grid_2d_irr(
        self,
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

        if isinstance(self.result, (np.ndarray, AbstractNDArray)):
            if len(self.result.shape) == 1:
                return ArrayIrregular(values=self.result)
            elif len(self.result.shape) == 2:
                return self.grid_irr_from(grid_slim=self.result)
        elif isinstance(self.result, list):
            if len(self.result[0].shape) == 1:
                return [ArrayIrregular(values=value) for value in self.result]
            elif len(self.result[0].shape) == 2:
                return [self.grid_irr_from(grid_slim=value) for value in self.result]

    @property
    def via_grid_2d_irr_list(
        self,
    ) -> List[Union[ArrayIrregular, Grid2DIrregular, Grid2DIrregularTransformed]]:
        """
        Convert a self.result from a list of non autoarray structures to a list of aa.ArrayIrregular or aa.Grid2DIrregular
        structures, where the conversion depends on type(self.result) as follows:

        ::

            - [1D np.ndarray] -> [aa.ArrayIrregular]
            - [2D np.ndarray] -> [aa.Grid2DIrregular]

        This function is used by the grid_like_list_to_structure_list decorator to convert the output self.result of a
        function to a list of autoarray structure when a `Grid2DIrregular` instance is passed to the decorated function.

        Parameters
        ----------
        self.result_list
            The input self.result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        if len(self.result[0].shape) == 1:
            return [ArrayIrregular(values=value) for value in self.result]
        elif len(self.result[0].shape) == 2:
            return [self.grid_irr_from(grid_slim=value) for value in self.result]
