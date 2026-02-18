import numpy as np

from typing import List, Optional, Tuple

from autoarray import type as ty
from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.geometry.geometry_2d import Geometry2D
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray.inversion.pixelization.mesh_grid.abstract_2d import Abstract2DMesh

from autoarray.inversion.pixelization.mesh import mesh_util
from autoarray.structures.grids import grid_2d_util


class Mesh2DRectangular(Abstract2DMesh):

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid_over_sampled,
        preloads=None,
        _xp=np,
    ):
        """
        A grid of (y,x) coordinates which represent a uniform rectangular pixelization.

        A `Mesh2DRectangular` is ordered such pixels begin from the top-row and go rightwards and then downwards.
        It is an ndarray of shape [total_pixels, 2], where the first dimension of the ndarray corresponds to the
        pixelization's pixel index and second element whether it is a y or x arc-second coordinate.

        For example:

        - grid[3,0] = the y-coordinate of the 4th pixel in the rectangular pixelization.
        - grid[6,1] = the x-coordinate of the 7th pixel in the rectangular pixelization.

        This class is used in conjuction with the `inversion/pixelizations` package to create rectangular pixelizations
        and mappers that perform an `Inversion`.

        Parameters
        ----------
        values
            The grid of (y,x) coordinates corresponding to the centres of each pixel in the rectangular pixelization.
        shape_native
            The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) origin of the pixelization.
        """
        super().__init__(
            mesh=mesh,
            mesh_grid=mesh_grid,
            data_grid_over_sampled=data_grid_over_sampled,
            preloads=preloads,
            _xp=_xp,
        )

    @property
    def shape(self):
        """
        The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
        """
        return self.mesh.shape

    @property
    def geometry(self):

        xmin = np.min(self.mesh_grid[:, 1])
        xmax = np.max(self.mesh_grid[:, 1])
        ymin = np.min(self.mesh_grid[:, 0])
        ymax = np.max(self.mesh_grid[:, 0])

        pixel_scales = (ymax - ymin) / (self.shape[0] - 1), (xmax - xmin) / (self.shape[1] - 1)

        origin = ((ymax + ymin) / 2.0, (xmax + xmin) / 2.0)

        return Geometry2D(
            shape_native=self.shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @property
    def shape_native(self):
        """
        The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
        """
        return self.shape

    @property
    def pixel_scales(self) -> Tuple[float, float]:
        return self.geometry.pixel_scales

    @property
    def origin(self) -> Tuple[float, float]:
        return self.geometry.origin

    @property
    def neighbors(self) -> Neighbors:
        """
        A class packing the ndarrays describing the neighbors of every pixel in the rectangular pixelization (see
        `Neighbors` for a complete description of the neighboring scheme).

        The neighbors of a rectangular pixelization are computed by exploiting the uniform and symmetric nature of the
        rectangular grid, as described in the method `mesh_util.rectangular_neighbors_from`.
        """
        neighbors, sizes = mesh_util.rectangular_neighbors_from(
            shape_native=self.mesh.shape
        )

        return Neighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))
