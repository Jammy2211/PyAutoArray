import numpy as np
from typing import Optional, Tuple

from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.inversion.pixelization.interpolator.rectangular import InterpolatorRectangular
from autoarray.inversion.pixelization.mesh.rectangular_adapt_density import (
    RectangularAdaptDensity,
)
from autoarray.settings import Settings
from autoarray.inversion.pixelization.border_relocator import BorderRelocator
from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh
from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray import exc


class RectangularAdaptImage(RectangularAdaptDensity):

    def __init__(
        self,
        shape: Tuple[int, int] = (3, 3),
        weight_power: float = 1.0,
        weight_floor: float = 0.0,
    ):
        """
        A uniform mesh of rectangular pixels, which without interpolation are paired with a 2D grid of (y,x)
        coordinates.

        For a full description of how a mesh is paired with another grid,
        see the :meth:`Pixelization API documentation <autoarray.inversion.pixelization.pixelization.Pixelization>`.

        The rectangular grid is uniform, has dimensions (total_y_pixels, total_x_pixels) and has indexing beginning
        in the top-left corner and going rightwards and downwards.

        A ``Pixelization`` using a ``RectangularAdaptDensity`` mesh has three grids associated with it:

        - ``image_plane_data_grid``: The observed data grid in the image-plane (which is paired with the mesh in
          the source-plane).
        - ``source_plane_data_grid``: The observed data grid mapped to the source-plane after gravitational lensing.
        - ``source_plane_mesh_grid``: The centres of each rectangular pixel.

        It does not have a ``image_plane_mesh_grid`` because a rectangular pixelization is constructed by overlaying
        a grid of rectangular over the `source_plane_data_grid`.

        Each (y,x) coordinate in the `source_plane_data_grid` is associated with the rectangular pixelization pixel
        it falls within. No interpolation is performed when making these associations.
        Parameters
        ----------
        shape
            The 2D dimensions of the rectangular grid of pixels (total_y_pixels, total_x_pixel).
        """

        super().__init__(shape=shape)

        self.weight_power = weight_power
        self.weight_floor = weight_floor

    def mesh_weight_map_from(self, adapt_data, xp=np) -> np.ndarray:
        """
        The weight map of a rectangular pixelization is None, because magnificaiton adaption uses
        the distribution and density of traced (y,x) coordinates in the source plane and
        not weights or the adapt data.

        Parameters
        ----------
        xp
            The array library to use.
        """
        mesh_weight_map = adapt_data.array
        mesh_weight_map = xp.clip(mesh_weight_map, 1e-12, None)
        mesh_weight_map = mesh_weight_map**self.weight_power

        # Apply floor using xp.where (safe for NumPy and JAX)
        mesh_weight_map = xp.where(
            mesh_weight_map < self.weight_floor,
            self.weight_floor,
            mesh_weight_map,
        )

        # Normalize
        mesh_weight_map = mesh_weight_map / xp.sum(mesh_weight_map)

        return mesh_weight_map
