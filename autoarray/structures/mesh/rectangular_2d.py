import jax.numpy as jnp
import numpy as np

from typing import List, Optional, Tuple

from autoconf import cached_property

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
        cls, shape_native: Tuple[int, int], grid: np.ndarray, buffer: float = 1e-8
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

        y_min = jnp.min(grid[:, 0]) - buffer
        y_max = jnp.max(grid[:, 0]) + buffer
        x_min = jnp.min(grid[:, 1]) - buffer
        x_max = jnp.max(grid[:, 1]) + buffer

        pixel_scales = jnp.array(
            (
                (y_max - y_min) / shape_native[0],
                (x_max - x_min) / shape_native[1],
            )
        )
        origin = jnp.array(((y_max + y_min) / 2.0, (x_max + x_min) / 2.0))

        grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_not_mask_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        return cls(
            values=grid_slim,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @cached_property
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

    @cached_property
    def edge_pixel_list(self) -> List:
        return mesh_util.rectangular_edge_pixel_list_from(
            shape_native=self.shape_native
        )

    @property
    def pixels(self) -> int:
        """
        The total number of pixels in the rectangular pixelization.
        """
        return self.shape_native[0] * self.shape_native[1]

    def interpolated_array_from(
        self,
        values: np.ndarray,
        shape_native: Tuple[int, int] = (401, 401),
        extent: Optional[Tuple[float, float, float, float]] = None,
    ) -> Array2D:
        """
        The reconstruction of data certain pixelizations, for example a `Delaunay` triangulation, requires that
        reconstructed data (e.g. the `reconstruction` output from an `Inversion`) is on an irregular pixelization.

        Analysing the reconstruction can therefore be difficult and require specific functionality tailored to the
        `Delaunay` triangulation.

        This function therefore interpolates the reconstruction on to a regular grid of square pixels.
        For a rectangular pixelization which is uniform, this is not stricly necessary as the native grid is
        easy to analyse. This interpolation function is included partly to mirror the API of other pixelizations.

        The output interpolated reconstruction cis by default returned on a grid of 401 x 401 square pixels. This
        can be customized by changing the `shape_native` input, and a rectangular grid with rectangular pixels can
        be returned by instead inputting the optional `shape_scaled` tuple.

        Parameters
        ----------
        values
            The value corresponding to the reconstructed value of every rectangular pixel on the rectangular grid.
        shape_native
            The 2D shape in pixels of the interpolated reconstruction, which is always returned using square pixels.
        extent
            The (x0, x1, y0, y1) extent of the grid in scaled coordinates over which the grid is created if it
            is input.
        """
        from scipy.interpolate import griddata

        interpolation_grid = self.interpolation_grid_from(
            shape_native=shape_native, extent=extent
        )

        interpolated_array = griddata(
            points=self.array, values=values, xi=interpolation_grid
        )

        interpolated_array = interpolated_array.reshape(shape_native)

        return Array2D.no_mask(
            values=interpolated_array, pixel_scales=interpolation_grid.pixel_scales
        )
