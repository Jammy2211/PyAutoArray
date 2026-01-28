import numpy as np

from typing import List, Optional, Tuple

from autoarray import type as ty
from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray.structures.mesh.abstract_2d import Abstract2DMesh

from autoarray.inversion.pixelization.mesh import mesh_util
from autoarray.structures.grids import grid_2d_util


class Mesh2DRectangular(Abstract2DMesh):

    def __init__(
        self,
        values: np.ndarray,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
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

        mask = Mask2D.all_false(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        self.mask = mask

        super().__init__(array=values)

    @classmethod
    def overlay_grid(
        cls,
        shape_native: Tuple[int, int],
        grid: np.ndarray,
        buffer: float = 1e-8,
        xp=np,
    ) -> "Mesh2DRectangular":
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

        return cls(
            values=grid_slim,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @property
    def neighbors(self) -> Neighbors:
        """
        A class packing the ndarrays describing the neighbors of every pixel in the rectangular pixelization (see
        `Neighbors` for a complete description of the neighboring scheme).

        The neighbors of a rectangular pixelization are computed by exploiting the uniform and symmetric nature of the
        rectangular grid, as described in the method `mesh_util.rectangular_neighbors_from`.
        """
        neighbors, sizes = mesh_util.rectangular_neighbors_from(
            shape_native=self.shape_native
        )

        return Neighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))

    @property
    def pixels(self) -> int:
        """
        The total number of pixels in the rectangular pixelization.
        """
        return self.shape_native[0] * self.shape_native[1]
