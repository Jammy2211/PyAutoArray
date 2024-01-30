import numpy as np

from autoarray.abstract_ndarray import AbstractNDArray
from autoarray.structures.abstract_structure import Structure
from autoarray.structures.grids.uniform_2d import Grid2D


class Grid2DTransformed(Grid2D):
    pass


class Grid2DTransformedNumpy(AbstractNDArray):
    @property
    def native(self) -> Structure:
        return self.array

    def __init__(self, values):
        super().__init__(array=values)
