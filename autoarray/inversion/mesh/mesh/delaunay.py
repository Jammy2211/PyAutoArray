import numpy as np
from typing import Optional

from autoarray.inversion.mesh.border_relocator import BorderRelocator
from autoarray.inversion.mesh.mesh.abstract import AbstractMesh
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


class Delaunay(AbstractMesh):
    def __init__(
        self, pixels: int, areas_factor: float = 0.5, zeroed_pixels: Optional[int] = 0
    ):
        """
        An irregular mesh of Delaunay triangle pixels, which using linear barycentric interpolation are paired with
        a 2D grid of (y,x) coordinates.

        For a full description of how a mesh is paired with another grid,
        see the :meth:`Pixelization API documentation <autoarray.inversion.pixelization.pixelization.Pixelization>`.

        The Delaunay mesh represents pixels as an irregular 2D grid of Delaunay triangles.

        - ``image_plane_data_grid``: The observed data grid in the image-plane (which is paired with the mesh in
          the source-plane).
        - ``image_plane_mesh_grid``: The (y,x) mesh coordinates in the image-plane (which are the corners of Delaunay
          triangles in the source-plane).
        - ``source_plane_data_grid``: The observed data grid mapped to the source-plane after gravitational lensing.
        - ``source_plane_mesh_grid``: The corner of each Delaunay triangle in the source-plane
          (the ``image_plane_mesh_grid`` maps to this after gravitational lensing).

        Each (y,x) coordinate in the ``source_plane_data_grid`` is paired with the three nearest Delaunay triangle
        corners, using a weighted interpolation scheme.

        Coordinates on the ``source_plane_data_grid`` are therefore given higher weights when paired with Delaunay
        triangle corners they are a closer distance to.
        """

        pixels = int(pixels) + zeroed_pixels

        super().__init__()
        self.pixels = pixels
        self.areas_factor = areas_factor
        self._zeroed_pixels = zeroed_pixels

    @property
    def zeroed_pixels(self):
        """
        Return the **positive** mesh-local pixel indices to zero for a Delaunay mesh.

        For Delaunay meshes, `self.zeroed_pixels` is interpreted as a *count* of pixels
        to be zeroed at the end of the pixel block. For example:
            self.pixels = 780, self.zeroed_pixels = 30
        returns indices 750..779.

        Returns
        -------
        np.ndarray
            1D array of positive pixel indices to zero.
        """
        if self._zeroed_pixels <= 0:
            return np.array([], dtype=int)

        start = self.pixels - self._zeroed_pixels
        return np.arange(start, self.pixels, dtype=int)

    @property
    def skip_areas(self):
        """
        Whether to skip barycentric  area calculations and split point computations during Delaunay triangulation.
        When True, the Delaunay interface returns only the minimal set of outputs (points, simplices, mappings)
        without computing split_points or splitted_mappings. This optimization is useful for regularization
        schemes like Matérn kernels that don't require area-based calculations. Default is False.
        """
        return False

    @property
    def interpolator_cls(self):

        from autoarray.inversion.mesh.interpolator.delaunay import (
            InterpolatorDelaunay,
        )

        return InterpolatorDelaunay

    def interpolator_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
        border_relocator: Optional[BorderRelocator] = None,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
        """
        Mapper objects describe the mappings between pixels in the masked 2D data and the pixels in a mesh,
        in both the `data` and `source` frames.

        This function returns a `MapperDelaunay` as follows:

        1) Before this routine is called, a sparse grid of (y,x) coordinates are computed from the 2D masked data,
           the `image_plane_mesh_grid`, which acts as the Delaunay triangle vertexes of the mesh and mapper.

        2) Before this routine is called, operations are performed on this `image_plane_mesh_grid` that transform it
           from a 2D grid which overlaps with the 2D mask of the data in the `data` frame to an irregular grid in
           the `source` frame, the `source_plane_mesh_grid`.

        3) If the border relocator is input, the border of the input `source_plane_data_grid` is used to relocate all of the
           grid's (y,x) coordinates beyond the border to the edge of the border.

        4) If the border relocatiro is input, the border of the input `source_plane_data_grid` is used to relocate all of the
           transformed `source_plane_mesh_grid`'s (y,x) coordinates beyond the border to the edge of the border.

        5) Use the transformed `source_plane_mesh_grid`'s (y,x) coordinates as the Vertex of the Delaunay mesh.

        Parameters
        ----------
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            The vertex of every Delaunay triangle pixel in the `source` frame, which are initially derived by
            computing a sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame and
            applying a transformation to this.
        image_plane_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame. This has a
            transformation applied to it to create the `source_plane_mesh_grid`.
        adapt_data
            Not used for a rectangular mesh.
        """
        relocated_grid = self.relocated_grid_from(
            border_relocator=border_relocator,
            source_plane_data_grid=source_plane_data_grid,
            xp=xp,
        )

        relocated_mesh_grid = self.relocated_mesh_grid_from(
            border_relocator=border_relocator,
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=source_plane_mesh_grid,
            xp=xp,
        )

        return self.interpolator_cls(
            mesh=self,
            data_grid=relocated_grid,
            mesh_grid=relocated_mesh_grid,
            adapt_data=adapt_data,
            xp=xp,
        )
