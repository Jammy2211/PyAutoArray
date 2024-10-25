from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.mesh.abstract_2d import Abstract2DMesh

from autoarray.structures.grids import grid_2d_util


class MapperGrids:
    def __init__(
        self,
        mask: Mask2D,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Optional[Abstract2DMesh] = None,
        image_plane_mesh_grid: Optional[Grid2DIrregular] = None,
        adapt_data: Optional[np.ndarray] = None,
        run_time_dict: Optional[Dict] = None,
    ):
        """
        Groups the different grids used by `Mesh` objects, the `mesh` package and the `pixelization` package, which
        create the following four grids:

        - `image_plane_data_grid`: the grid defining where data-points in frame of the data are.

        - `source_plane_data_grid`: the grid defining where the mapped coordinates of these data-points in the source-frame
         of the linear object are.

        - `image_plane_mesh_grid`: the grid defining where the linear object parameters (e.g. what are used as pixels of
        the mapper) are in the image-plane.

        - `source_plane_mesh_grid`: the grid defining where the mapped coordinates of the linear object parameters
        are in the source frame.

        Read the docstrings of the `mesh` package for more information is this is unclear.

         This grouped set of grids are input into  `Mapper` objects, in order to determine the mappings between the
         masked data grid's data points (`image_plane_data_grid` and `source_plane_data_grid`) and the mesh's pixels
         (`image_plane_mesh_grid` and `source_plane_mesh_grid`).

        Parameters
        ----------
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            The 2D grid of (y,x) centres of every pixelization pixel in the `source` frame.
        image_plane_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame. This has a
            transformation applied to it to create the `source_plane_mesh_grid`.
        adapt_data
            An image which is used to determine the `image_plane_mesh_grid` and therefore adapt the distribution of
            pixels of the Delaunay grid to the data it discretizes.
        run_time_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        self.mask = mask
        self.source_plane_data_grid = source_plane_data_grid
        self.source_plane_mesh_grid = source_plane_mesh_grid
        self.image_plane_mesh_grid = image_plane_mesh_grid
        self.adapt_data = adapt_data
        self.run_time_dict = run_time_dict

    @property
    def image_plane_data_grid(self):
        return self.mask.derive_grid.unmasked

    @property
    def mesh_pixels_per_image_pixels(self):
        mesh_pixels_per_image_pixels = grid_2d_util.grid_pixels_in_mask_pixels_from(
            grid=np.array(self.image_plane_mesh_grid),
            shape_native=self.mask.shape_native,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

        return Array2D(
            values=mesh_pixels_per_image_pixels,
            mask=self.mask,
        )
