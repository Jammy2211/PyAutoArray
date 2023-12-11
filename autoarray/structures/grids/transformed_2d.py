import numpy as np

from autoarray.structures.grids.uniform_2d import Grid2D


class Grid2DTransformed(Grid2D):
    pass


class Grid2DTransformedNumpy(np.ndarray):
    def __new__(cls, values, *args, **kwargs):
        return values.view(cls)
