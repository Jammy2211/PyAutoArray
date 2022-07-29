import numpy as np
from typing import Dict, Optional

from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.sparse_2d import Grid2DSparse
from autoarray.structures.mesh.abstract_2d import Abstract2DMesh
from autoarray.inversion.pixelization.settings import SettingsPixelization

#


class MapperGrids:
    def __init__(
        self,
        source_grid_slim: Grid2D,
        source_mesh_grid: Abstract2DMesh = None,
        data_mesh_grid: Grid2DSparse = None,
        hyper_data: np.ndarray = None,
        settings: SettingsPixelization = SettingsPixelization(),
        preloads: "Preloads" = None,
        profiling_dict: Optional[Dict] = None,
    ):

        from autoarray.preloads import Preloads

        self.source_grid_slim = source_grid_slim
        self.source_mesh_grid = source_mesh_grid
        self.data_mesh_grid = data_mesh_grid
        self.hyper_data = hyper_data
        self.settings = settings
        self.preloads = preloads or Preloads()
        self.profiling_dict = profiling_dict
