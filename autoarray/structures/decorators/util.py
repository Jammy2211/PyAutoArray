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
from autoarray.structures.decorators import util

def result_via_func_from(grid, func, obj, *args, **kwargs):

    if isinstance(grid, Grid1D):
        grid = grid.grid_2d_radial_projected_from()
        return func(obj, grid, *args, **kwargs)

    return func(obj, grid, *args, **kwargs)


def result_via_maker_from(grid, maker):

    if isinstance(grid, Grid2D):
        return maker.via_grid_2d
    elif isinstance(grid, Grid2DIrregular):
        return maker.via_grid_2d_irr
    elif isinstance(grid, Grid1D):
        return maker.via_grid_1d

    return maker.result
