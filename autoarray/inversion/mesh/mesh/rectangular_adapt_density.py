import numpy as np
from typing import Optional, Tuple

from autoarray.settings import Settings
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.inversion.mesh.mesh.abstract import AbstractMesh
from autoarray.inversion.mesh.border_relocator import BorderRelocator

from autoarray.structures.grids import grid_2d_util

from autoarray import exc


def overlay_grid_from(
    shape_native: Tuple[int, int],
    grid: np.ndarray,
    buffer: float = 1e-8,
    xp=np,
) -> np.ndarray:
    """
    Creates a `Grid2DRecntagular` by overlaying the rectangular pixelization over an input grid of (y,x)
    coordinates.

    This is performed by first computing the minimum and maximum y and x coordinates of the input grid. A
    rectangular pixelization with dimensions `shape_native` is then laid over the grid using these coordinates,
    such that the extreme edges of this rectangular pixelization overlap these maximum and minimum (y,x) coordinates.

    A a `buffer` can be included which increases the size of the rectangular pixelization, placing additional
    spacing beyond these maximum and minimum coordinates.

    Parameters
    ----------
    shape_native
        The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
    grid
        A grid of (y,x) coordinates which the rectangular pixelization is laid-over.
    buffer
        The size of the extra spacing placed between the edges of the rectangular pixelization and input grid.
    """
    grid = grid.array

    y_min = xp.min(grid[:, 0]) - buffer
    y_max = xp.max(grid[:, 0]) + buffer
    x_min = xp.min(grid[:, 1]) - buffer
    x_max = xp.max(grid[:, 1]) + buffer

    pixel_scales = xp.array(
        (
            (y_max - y_min) / shape_native[0],
            (x_max - x_min) / shape_native[1],
        )
    )
    origin = xp.array(((y_max + y_min) / 2.0, (x_max + x_min) / 2.0))

    grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_not_mask_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin, xp=xp
    )

    return grid_slim


class RectangularAdaptDensity(AbstractMesh):
    def __init__(self, shape: Tuple[int, int] = (3, 3)):
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

        if shape[0] <= 2 or shape[1] <= 2:
            raise exc.MeshException(
                "The rectangular pixelization must be at least dimensions 3x3"
            )

        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]
        super().__init__()

    @property
    def source_pixel_zeroed_indices(self):

        from autoarray.inversion.mesh.mesh_geometry.rectangular import rectangular_edge_pixel_list_from

        return rectangular_edge_pixel_list_from(
            shape_native=mesh_shape,
        )

    @property
    def source_pixel_zeroed_indices_to_keep(self):

        ids_zeros = np.array(self.source_pixel_zeroed_indices, dtype=int)

        values_to_solve = np.ones(np.max(mapper_indices) + 1, dtype=bool)
        values_to_solve[ids_zeros] = False

        return np.where(values_to_solve)[0]


    @property
    def interpolator_cls(self):
        from autoarray.inversion.mesh.interpolator.rectangular import (
            InterpolatorRectangular,
        )

        return InterpolatorRectangular

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
        return None

    def interpolator_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
        border_relocator: Optional[BorderRelocator] = None,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
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
        """
        relocated_grid = self.relocated_grid_from(
            border_relocator=border_relocator,
            source_plane_data_grid=source_plane_data_grid,
            xp=xp,
        )

        mesh_grid = overlay_grid_from(
            shape_native=self.shape,
            grid=Grid2DIrregular(relocated_grid.over_sampled),
            xp=xp,
        )

        mesh_weight_map = self.mesh_weight_map_from(adapt_data=adapt_data, xp=xp)

        return self.interpolator_cls(
            mesh=self,
            data_grid=relocated_grid,
            mesh_grid=Grid2DIrregular(mesh_grid),
            mesh_weight_map=mesh_weight_map,
            adapt_data=adapt_data,
            xp=xp,
        )
