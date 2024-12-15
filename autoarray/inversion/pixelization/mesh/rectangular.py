import numpy as np
from typing import Dict, Optional, Tuple


from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.mesh.rectangular_2d import Mesh2DRectangular
from autoarray.preloads import Preloads
from autoarray.inversion.pixelization.mappers.mapper_grids import MapperGrids
from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh
from autoarray.inversion.pixelization.border_relocator import BorderRelocator

from autoarray import exc
from autoarray.numba_util import profile_func


class Rectangular(AbstractMesh):
    def __init__(self, shape: Tuple[int, int] = (3, 3)):
        """
        A uniform mesh of rectangular pixels, which without interpolation are paired with a 2D grid of (y,x)
        coordinates.

        For a full description of how a mesh is paired with another grid,
        see the :meth:`Pixelization API documentation <autoarray.inversion.pixelization.pixelization.Pixelization>`.

        The rectangular grid is uniform, has dimensions (total_y_pixels, total_x_pixels) and has indexing beginning
        in the top-left corner and going rightwards and downwards.

        A ``Pixelization`` using a ``Rectangular`` mesh has three grids associated with it:

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

        if shape[0] <= 2 or shape[1] <= 2:
            raise exc.MeshException(
                "The rectangular pixelization must be at least dimensions 3x3"
            )

        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]
        super().__init__()

        self.run_time_dict = {}

    def mapper_grids_from(
        self,
        mask,
        source_plane_data_grid: Grid2D,
        border_relocator: Optional[BorderRelocator] = None,
        source_plane_mesh_grid: Grid2D = None,
        image_plane_mesh_grid: Grid2D = None,
        adapt_data: np.ndarray = None,
        preloads: Preloads = Preloads(),
        run_time_dict: Optional[Dict] = None,
    ) -> MapperGrids:
        """
        Mapper objects describe the mappings between pixels in the masked 2D data and the pixels in a pixelization,
        in both the `data` and `source` frames.

        This function returns a `MapperRectangular` as follows:

        1) If the bordr relocator is input, the border of the input `source_plane_data_grid` is used to relocate all of the
           grid's (y,x) coordinates beyond the border to the edge of the border.

        2) Determine the (y,x) coordinates of the pixelization's rectangular pixels, by laying this rectangular grid
           over the 2D grid of relocated (y,x) coordinates computed in step 1 (or the input `source_plane_data_grid` if step 1
           is bypassed).

        3) Return the `MapperRectangular`.

        Parameters
        ----------
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            Not used for a rectangular pixelization, because the pixelization grid in the `source` frame is computed
            by overlaying the `source_plane_data_grid` with the rectangular pixelization.
        image_plane_mesh_grid
            Not used for a rectangular pixelization.
        adapt_data
            Not used for a rectangular pixelization.
        preloads
            Object which may contain preloaded arrays of quantities computed in the pixelization, which are passed via
            this object speed up the calculation.
        run_time_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        self.run_time_dict = run_time_dict

        relocated_grid_over_sampled = self.relocated_grid_from(
            border_relocator=border_relocator,
            source_plane_data_grid=source_plane_data_grid.grid_over_sampled,
            preloads=preloads,
        )

        relocated_grid = Grid2D(
            values=source_plane_data_grid,
            mask=source_plane_data_grid.mask,
            over_sampling_size=source_plane_data_grid.over_sampling_size,
            grid_over_sampled=relocated_grid_over_sampled,
        )

        mesh_grid = self.mesh_grid_from(source_plane_data_grid=relocated_grid)

        return MapperGrids(
            mask=mask,
            source_plane_data_grid=relocated_grid,
            source_plane_mesh_grid=mesh_grid,
            image_plane_mesh_grid=image_plane_mesh_grid,
            adapt_data=adapt_data,
            preloads=preloads,
            run_time_dict=run_time_dict,
        )

    @profile_func
    def mesh_grid_from(
        self,
        source_plane_data_grid: Optional[Grid2D] = None,
        source_plane_mesh_grid: Optional[Grid2D] = None,
    ) -> Mesh2DRectangular:
        """
        Return the rectangular `source_plane_mesh_grid` as a `Mesh2DRectangular` object, which provides additional
        functionality for perform operatons that exploit the geometry of a rectangular pixelization.

        Parameters
        ----------
        source_plane_data_grid
            The (y,x) grid of coordinates over which the rectangular pixelization is overlaid, where this grid may have
            had exterior pixels relocated to its edge via the border.
        source_plane_mesh_grid
            Not used for a rectangular pixelization, because the pixelization grid in the `source` frame is computed
            by overlaying the `source_plane_data_grid` with the rectangular pixelization.
        """
        return Mesh2DRectangular.overlay_grid(
            shape_native=self.shape, grid=source_plane_data_grid.grid_over_sampled
        )

    @property
    def requires_image_mesh(self):
        return False
