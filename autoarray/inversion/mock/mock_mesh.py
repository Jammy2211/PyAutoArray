import numpy as np
from typing import Dict, Optional

from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh
from autoarray.inversion.pixelization.settings import SettingsPixelization
from autoarray.inversion.pixelization.mappers.mapper_grids import MapperGrids
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.sparse_2d import Grid2DSparse
from autoarray.preloads import Preloads


class MockMesh(AbstractMesh):
    def __init__(self, image_plane_mesh_grid=None):
        super().__init__()

        self.image_plane_mesh_grid = image_plane_mesh_grid

    def mapper_grids_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DSparse = None,
        image_plane_mesh_grid: Grid2DSparse = None,
        hyper_data: np.ndarray = None,
        settings=SettingsPixelization(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ) -> MapperGrids:
        return MapperGrids(
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=source_plane_mesh_grid,
            image_plane_mesh_grid=self.image_plane_mesh_grid,
            hyper_data=hyper_data,
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    def image_plane_mesh_grid_from(
        self, image_plane_data_grid, hyper_data, settings=None
    ):
        if hyper_data is not None and self.image_plane_mesh_grid is not None:
            return hyper_data * self.image_plane_mesh_grid

        return self.image_plane_mesh_grid
