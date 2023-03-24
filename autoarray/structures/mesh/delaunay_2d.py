import numpy as np
from typing import List, Optional, Tuple

from autoconf import cached_property

from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.mesh.triangulation_2d import Abstract2DMeshTriangulation
from autoarray.inversion.pixelization.mesh import mesh_util


class Mesh2DDelaunay(Abstract2DMeshTriangulation):
    @cached_property
    def neighbors(self) -> Neighbors:
        """
        Returns a ndarray describing the neighbors of every pixel in a Delaunay triangulation, where a neighbor is
        defined as two Delaunay triangles which are directly connected to one another in the triangulation.

        see `Neighbors` for a complete description of the neighboring scheme.

        The neighbors of a Voronoi mesh are computed using the `ridge_points` attribute of the scipy `Voronoi`
        object, as described in the method `mesh_util.voronoi_neighbors_from`.
        """
        indptr, indices = self.delaunay.vertex_neighbor_vertices

        sizes = indptr[1:] - indptr[:-1]

        neighbors = -1 * np.ones(
            shape=(self.parameters, int(np.max(sizes))), dtype="int"
        )

        for k in range(self.parameters):
            neighbors[k][0 : sizes[k]] = indices[indptr[k] : indptr[k + 1]]

        return Neighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))

    def interpolated_array_from(
        self,
        values: np.ndarray,
        shape_native: Tuple[int, int] = (401, 401),
        extent: Optional[Tuple[float, float, float, float]] = None,
    ) -> Array2D:
        """
        The reconstruction of data on a `Delaunay` triangulation (e.g. the `reconstruction` output from an `Inversion`)
        is on  irregular pixelization.

        Analysing the reconstruction can therefore be difficult and require specific functionality tailored to the
        `Delaunay` triangulation.

        This function therefore interpolates the irregular reconstruction on to a regular grid of square pixels.
        The routine uses the Delaunay triangulation interpolation weights based on the area of each triangle to
        perform this interpolation.

        The output interpolated reconstruction cis by default returned on a grid of 401 x 401 square pixels. This
        can be customized by changing the `shape_native` input, and a rectangular grid with rectangular pixels can
        be returned by instead inputting the optional `shape_scaled` tuple.

        Parameters
        ----------
        values
            The value corresponding to the reconstructed value of Delaunay triangle vertex.
        shape_native
            The 2D shape in pixels of the interpolated reconstruction, which is always returned using square pixels.
        extent
            The (x0, x1, y0, y1) extent of the grid in scaled coordinates over which the grid is created if it
            is input.
        """
        interpolation_grid = self.interpolation_grid_from(
            shape_native=shape_native, extent=extent
        )

        interpolated_array = mesh_util.delaunay_interpolated_array_from(
            shape_native=shape_native,
            interpolation_grid_slim=interpolation_grid.slim,
            delaunay=self.delaunay,
            pixel_values=values,
        )

        return Array2D.no_mask(
            values=interpolated_array, pixel_scales=interpolation_grid.pixel_scales
        )
