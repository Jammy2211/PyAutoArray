import numpy as np
from typing import Optional, Tuple

from autoarray.structures.abstract_structure import Structure
from autoarray.structures.grids.uniform_2d import Grid2D


class Abstract2DMesh:

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid_over_sampled,
        preloads=None,
        _xp=np,
    ):
        self.mesh = mesh
        self.mesh_grid = mesh_grid
        self.data_grid_over_sampled = data_grid_over_sampled
        self.preloads = preloads
        self._xp = _xp