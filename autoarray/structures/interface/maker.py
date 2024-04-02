import numpy as np
from functools import wraps

from typing import List, Union

from autoconf.exc import ConfigException

from autoarray import exc
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

        if isinstance(self.grid, Grid2D):
            return self.via_grid_2d

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
            return Grid2D(values=self.result, mask=self.mask, over_sample=self.over_sample)
        
