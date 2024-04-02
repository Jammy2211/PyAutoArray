from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D

class ResultMaker:

    def __init__(self, func, obj, grid, *args, **kwargs):

        self.func = func
        self.obj = obj
        self.grid = grid
        self.args = args
        self.kwargs = kwargs

    @property
    def result_basic(self):

        grid = Grid2D(values=[[1.0, 1.0]], mask=Mask2D(mask=[[False]], pixel_scales=1.0))

        return self.func(self.obj, grid, *self.args, **self.kwargs)

    @property
    def result_type(self):

        if len(self.result_basic) == 1:
            return Array2D