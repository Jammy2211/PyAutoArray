import numpy as np
from typing import Dict, Optional

from autoarray.inversion.pixelization.mappers.mapper_grids import MapperGrids
from autoarray.inversion.pixelization.border_relocator import BorderRelocator
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.numba_util import profile_func


class AbstractMesh:
    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.__class__ is other.__class__

    @profile_func
    def relocated_grid_from(
        self,
        border_relocator: BorderRelocator,
        source_plane_data_grid: Grid2D,
    ) -> Grid2D:
        """
        Relocates all coordinates of the input `source_plane_data_grid` that are outside of a
        border (which is defined by a grid of (y,x) coordinates) to the edge of this border.

        The border is determined from the mask of the 2D data in the `data` frame before any transformations of the
        data's grid are performed. The border is all pixels in this mask that are pixels at its extreme edge. These
        pixel indexes are used to then determine a grid of (y,x) coordinates from the transformed `source_grid_grid` in
        the `source` reference frame, whereby points located outside of it are relocated to the border's edge.

        A full description of relocation is given in the method grid_2d.relocated_grid_from()`.

        This is used in the project PyAutoLens to relocate the coordinates that are ray-traced near the centre of mass
        of galaxies, which are heavily demagnified and may trace to outskirts of the source-plane well beyond the
        border.

        Parameters
        ----------
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        source_plane_data_grid
            A 2D (y,x) grid of coordinates, whose coordinates outside the border are relocated to its edge.
        """
        if border_relocator is not None:
            return border_relocator.relocated_grid_from(grid=source_plane_data_grid)
        return source_plane_data_grid

    @profile_func
    def relocated_mesh_grid_from(
        self,
        border_relocator: Optional[BorderRelocator],
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
    ):
        """
        Relocates all coordinates of the input `source_plane_mesh_grid` that are outside of a border (which
        is defined by a grid of (y,x) coordinates) to the edge of this border.

        The border is determined from the mask of the 2D data in the `data` frame before any transformations of the
        data's grid are performed. The border is all pixels in this mask that are pixels at its extreme edge. These
        pixel indexes are used to then determine a grid of (y,x) coordinates from the transformed `source_grid_grid` in
        the `source` reference frame, whereby points located outside of it are relocated to the border's edge.

        A full description of relocation is given in the method grid_2d.relocated_grid_from()`.

        This is used in the project `PyAutoLens` to relocate the coordinates that are ray-traced near the centre of mass
        of galaxies, which are heavily demagnified and may trace to outskirts of the source-plane well beyond the
        border.

        Parameters
        ----------
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            The centres of every Voronoi pixel in the `source` frame, which are initially derived by computing a sparse
            set of (y,x) coordinates computed from the unmasked data in the `data` frame and applying a transformation
            to this.
        """
        if border_relocator is not None:
            return border_relocator.relocated_mesh_grid_from(
                grid=source_plane_data_grid, mesh_grid=source_plane_mesh_grid
            )
        return source_plane_mesh_grid

    def mapper_grids_from(
        self,
        mask,
        source_plane_data_grid: Grid2D,
        border_relocator: Optional[BorderRelocator] = None,
        source_plane_mesh_grid: Optional[Grid2DIrregular] = None,
        image_plane_mesh_grid: Optional[Grid2DIrregular] = None,
        adapt_data: np.ndarray = None,
        run_time_dict: Optional[Dict] = None,
    ) -> MapperGrids:
        raise NotImplementedError

    def mesh_grid_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
    ):
        raise NotImplementedError

    @property
    def requires_image_mesh(self):
        return True

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])

    def __repr__(self):
        return "{}\n{}".format(self.__class__.__name__, str(self))
