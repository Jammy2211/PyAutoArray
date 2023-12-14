import numpy as np
from typing import Dict, Optional

from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh
from autoarray.inversion.pixelization.settings import SettingsPixelization
from autoarray.structures.mesh.abstract_2d import Abstract2DMesh
from autoarray.inversion.pixelization.mappers.mapper_grids import MapperGrids
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.preloads import Preloads


class MockMesh(AbstractMesh):
    def __init__(self, image_plane_mesh_grid=None):
        super().__init__()

        self.image_plane_mesh_grid = image_plane_mesh_grid

    def mapper_grids_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Optional[Abstract2DMesh] = None,
        image_plane_mesh_grid: Optional[Grid2DIrregular] = None,
        adapt_data: Optional[np.ndarray] = None,
        settings: SettingsPixelization = SettingsPixelization(),
        preloads: Optional[Preloads] = None,
        run_time_dict: Optional[Dict] = None,
    ) -> MapperGrids:
        return MapperGrids(
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=source_plane_mesh_grid,
            image_plane_mesh_grid=self.image_plane_mesh_grid,
            adapt_data=adapt_data,
            settings=settings,
            preloads=preloads,
            run_time_dict=run_time_dict,
        )

    def image_plane_mesh_grid_from(
        self,
        image_plane_data_grid,
        adapt_data,
        settings=None,
        noise_map: np.ndarray = None,
    ):
        if adapt_data is not None and self.image_plane_mesh_grid is not None:
            return adapt_data * self.image_plane_mesh_grid

        return self.image_plane_mesh_grid

    @property
    def requires_image_mesh(self):
        return False
