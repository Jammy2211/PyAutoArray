from typing import List, Union

from autoarray.structures.arrays.irregular import ArrayIrregular
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


class AbstractMaker:
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

    def via_grid_2d(self, result) -> Union[Array2D, List[Array2D]]:
        raise NotImplementedError

    def via_grid_2d_irr(self, result) -> Union[ArrayIrregular, List[ArrayIrregular]]:
        raise NotImplementedError

    def via_grid_1d(self, result) -> Union[Array1D, List[Array1D]]:
        raise NotImplementedError

    @property
    def evaluate_func(self):
        if isinstance(self.grid, Grid1D):
            grid = self.grid.grid_2d_radial_projected_from()
            return self.func(self.obj, grid, *self.args, **self.kwargs)

        return self.func(self.obj, self.grid, *self.args, **self.kwargs)

    @property
    def result(self):
        if isinstance(self.grid, Grid2D):
            return self.via_grid_2d(self.evaluate_func)
        elif isinstance(self.grid, Grid2DIrregular):
            return self.via_grid_2d_irr(self.evaluate_func)
        elif isinstance(self.grid, Grid1D):
            return self.via_grid_1d(self.evaluate_func)

        return self.evaluate_func
