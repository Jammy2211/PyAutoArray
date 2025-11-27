import numpy as np
from typing import Optional

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.inversion.pixelization.image_mesh.abstract import AbstractImageMesh


class MockImageMesh(AbstractImageMesh):
    def __init__(self, image_plane_mesh_grid=None):
        super().__init__()

        self.image_plane_mesh_grid = image_plane_mesh_grid

    def image_plane_mesh_grid_from(
        self, mask: Mask2D, adapt_data: Optional[np.ndarray],
    ) -> Grid2DIrregular:
        if adapt_data is not None and self.image_plane_mesh_grid is not None:
            return adapt_data * self.image_plane_mesh_grid

        return self.image_plane_mesh_grid

    @property
    def uses_adapt_images(self) -> bool:
        return False
