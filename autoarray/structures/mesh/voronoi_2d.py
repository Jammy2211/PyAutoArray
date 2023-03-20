import numpy as np
from scipy.interpolate import griddata
from typing import Optional, Tuple

from autoconf import cached_property

from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.mesh.triangulation_2d import Abstract2DMeshTriangulation

from autoarray.inversion.pixelization.mesh import mesh_util


class Mesh2DVoronoi(Abstract2DMeshTriangulation):
    @cached_property
    def neighbors(self) -> Neighbors:
        """
        Returns a ndarray describing the neighbors of every pixel in a Voronoi mesh, where a neighbor is defined as
        two Voronoi cells which share an adjacent vertex.

        see `Neighbors` for a complete description of the neighboring scheme.

        The neighbors of a Voronoi mesh are computed using the `ridge_points` attribute of the scipy `Voronoi`
        object, as described in the method `mesh_util.voronoi_neighbors_from`.
        """
        neighbors, sizes = mesh_util.voronoi_neighbors_from(
            pixels=self.pixels, ridge_points=np.asarray(self.voronoi.ridge_points)
        )

        return Neighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))

    def interpolated_array_from(
        self,
        values: np.ndarray,
        shape_native: Tuple[int, int] = (401, 401),
        extent: Optional[Tuple[float, float, float, float]] = None,
        use_nn=False,
    ) -> Array2D:
        """
        The reconstruction of data on a `Voronoi` mesh (e.g. the `reconstruction` output from an `Inversion`)
        is on irregular pixelization.

        Analysing the reconstruction can therefore be difficult and require specific functionality tailored to the
        `Voronoi` mesh.

        This function therefore interpolates the irregular reconstruction on to a regular grid of square pixels.
        The routine uses the naturual neighbor Voronoi interpolation weights.

        The output interpolated reconstruction cis by default returned on a grid of 401 x 401 square pixels. This
        can be customized by changing the `shape_native` input, and a rectangular grid with rectangular pixels can
        be returned by instead inputting the optional `shape_scaled` tuple.

        Parameters
        ----------
        values
            The value corresponding to the reconstructed value of every Voronoi cell.
        shape_native
            The 2D shape in pixels of the interpolated reconstruction, which is always returned using square pixels.
        shape_scaled
            The 2D shape in scaled coordinates (e.g. arc-seconds in PyAutoGalaxy / PyAutoLens) that the interpolated
            reconstructed source is returned on.
        """
        interpolation_grid = self.interpolation_grid_from(
            shape_native=shape_native, extent=extent
        )

        if use_nn:

            interpolated_array = mesh_util.voronoi_nn_interpolated_array_from(
                shape_native=shape_native,
                interpolation_grid_slim=interpolation_grid.slim,
                pixel_values=values,
                voronoi=self.voronoi,
            )

        else:

            interpolated_array = griddata(
                points=self.voronoi.points, values=values, xi=interpolation_grid
            )

            interpolated_array = np.flipud(
                np.fliplr(interpolated_array.reshape(shape_native).T)
            )

        return Array2D.no_mask(
            values=interpolated_array, pixel_scales=interpolation_grid.pixel_scales
        )
