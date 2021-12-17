import numpy as np

from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.type import Grid2DLike


class LinearObject:
    def mapping_matrix_from(self, grid: Grid2DLike) -> np.ndarray:
        raise NotImplementedError
