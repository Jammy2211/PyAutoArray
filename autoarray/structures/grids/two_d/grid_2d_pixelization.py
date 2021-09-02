import numpy as np
import scipy.spatial
import scipy.spatial.qhull as qhull
from typing import Tuple

from autoarray.structures.abstract_structure import AbstractStructure2D
from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.structures.grids.two_d import grid_2d_util
from autoarray.inversion import pixelization_util


class Grid2DRectangular(AbstractStructure2D):
    def __new__(
        cls, grid, shape_native, pixel_scales, origin=(0.0, 0.0), *args, **kwargs
    ):
        """
        A grid of (y,x) coordinates which reprsent a rectangular grid of pixels which are used to form the pixel centres of adaptive pixelizations in the \
        *pixelizations* module.

        A `Grid2DRectangular` is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_pix_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixelization pixel index and second element the y or x arc -econd coordinates. For example:

        - pix_grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - pix_grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Parameters
        -----------
        pix_grid
            The grid of (y,x) scaled coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        nearest_pixelization_index_for_slim_index
            A 1D array that maps every grid pixel to its nearest pixelization-grid pixel.
        """

        mask = Mask2D.unmasked(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
        )

        obj = grid.view(cls)
        obj.mask = mask

        (
            pixel_neighbors,
            pixel_neighbors_size,
        ) = pixelization_util.rectangular_neighbors_from(shape_native=shape_native)
        obj.pixel_neighbors = pixel_neighbors.astype("int")
        obj.pixel_neighbors_size = pixel_neighbors_size.astype("int")
        return obj

    @classmethod
    def overlay_grid(cls, shape_native, grid, buffer=1e-8):
        """The geometry of a rectangular grid.

        This is used to map grid of (y,x) scaled coordinates to the pixels on the rectangular grid.

        Parameters
        -----------
        shape_native
            The dimensions of the rectangular grid of pixels (y_pixels, x_pixel)
        pixel_scales
            The pixel conversion scale of a pixel in the y and x directions.
        origin
            The scaled origin of the rectangular pixelization's coordinate system.
        pixel_neighbors
            An array of length (y_pixels*x_pixels) which provides the index of all neighbors of every pixel in \
            the rectangular grid (entries of -1 correspond to no neighbor).
        pixel_neighbors_size
            An array of length (y_pixels*x_pixels) which gives the number of neighbors of every pixel in the \
            rectangular grid.
        """

        y_min = np.min(grid[:, 0]) - buffer
        y_max = np.max(grid[:, 0]) + buffer
        x_min = np.min(grid[:, 1]) - buffer
        x_max = np.max(grid[:, 1]) + buffer

        pixel_scales = (
            float((y_max - y_min) / shape_native[0]),
            float((x_max - x_min) / shape_native[1]),
        )

        origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)

        grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
        )

        return Grid2DRectangular(
            grid=grid_slim,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @property
    def pixels(self):
        return self.shape_native[0] * self.shape_native[1]

    @property
    def shape_native_scaled(self):
        return (
            (self.shape_native[0] * self.pixel_scales[0]),
            (self.shape_native[1] * self.pixel_scales[1]),
        )

    @property
    def scaled_maxima(self) -> Tuple[float, float]:
        """
        The maximum values of the grid in scaled coordinates returned as a tuple (y_max, x_max).
        """
        return (
            self.origin[0] + (self.shape_native_scaled[0] / 2.0),
            self.origin[1] + (self.shape_native_scaled[1] / 2.0),
        )

    @property
    def scaled_minima(self) -> Tuple[float, float]:
        """
        The minium values of the grid in scaled coordinates returned as a tuple (y_min, x_min).
        """
        return (
            (self.origin[0] - (self.shape_native_scaled[0] / 2.0)),
            (self.origin[1] - (self.shape_native_scaled[1] / 2.0)),
        )

    @property
    def extent(self) -> np.ndarray:
        """
        The extent of the grid in scaled units returned as an ndarray of the form [x_min, x_max, y_min, y_max].

        This follows the format of the extent input parameter in the matplotlib method imshow (and other methods) and
        is used for visualization in the plot module.
        """
        return np.asarray(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )


class Grid2DVoronoi(AbstractStructure2D):
    """
    Returns the geometry of the Voronoi pixelization, by alligning it with the outer-most coordinates on a \
    grid plus a small buffer.

    Parameters
    -----------
    grid
        The (y,x) grid of coordinates which determine the Voronoi pixelization's
    pixelization_grid
        The (y,x) centre of every Voronoi pixel in scaleds.
    origin
        The scaled origin of the Voronoi pixelization's coordinate system.
    pixel_neighbors
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in \
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the \
        Voronoi grid.
    """

    def __new__(
        cls, grid, nearest_pixelization_index_for_slim_index=None, *args, **kwargs
    ):
        """
        A pixelization-grid of (y,x) coordinates which are used to form the pixel centres of adaptive pixelizations in the \
        *pixelizations* module.

        A `Grid2DRectangular` is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_pix_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixelization pixel index and second element the y or x arc -econd coordinates. For example:

        - pix_grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - pix_grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Parameters
        -----------
        pix_grid
            The grid of (y,x) scaled coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        nearest_pixelization_index_for_slim_index
            A 1D array that maps every grid pixel to its nearest pixelization-grid pixel.
        """

        if type(grid) is list:
            grid = np.asarray(grid)

        obj = grid.view(cls)
        obj.nearest_pixelization_index_for_slim_index = (
            nearest_pixelization_index_for_slim_index
        )

        try:
            obj.voronoi = scipy.spatial.Voronoi(
                np.asarray([grid[:, 1], grid[:, 0]]).T, qhull_options="Qbb Qc Qx Qm"
            )
        except (ValueError, OverflowError, scipy.spatial.qhull.QhullError) as e:
            raise exc.PixelizationException() from e

        (
            pixel_neighbors,
            pixel_neighbors_size,
        ) = pixelization_util.voronoi_neighbors_from(
            pixels=obj.pixels, ridge_points=np.asarray(obj.voronoi.ridge_points)
        )

        obj.pixel_neighbors = pixel_neighbors.astype("int")
        obj.pixel_neighbors_size = pixel_neighbors_size.astype("int")
        obj.nearest_pixelization_index_for_slim_index = (
            nearest_pixelization_index_for_slim_index
        )

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "voronoi"):
            self.voronoi = obj.voronoi

        if hasattr(obj, "pixel_neighbors"):
            self.pixel_neighbors = obj.pixel_neighbors

        if hasattr(obj, "pixel_neighbors_size"):
            self.pixel_neighbors_size = obj.pixel_neighbors_size

        if hasattr(obj, "nearest_pixelization_index_for_slim_index"):
            self.nearest_pixelization_index_for_slim_index = (
                obj.nearest_pixelization_index_for_slim_index
            )

        if hasattr(obj, "_sub_border_flat_indexes"):
            self._sub_border_flat_indexes = obj.sub_border_flat_indexes

    @property
    def origin(self):
        return 0.0, 0.0

    @property
    def pixels(self):
        return self.shape[0]

    @property
    def sub_border_grid(self):
        """The (y,x) grid of all sub-pixels which are at the border of the mask.

        This is NOT all sub-pixels which are in mask pixels at the mask's border, but specifically the sub-pixels
        within these border pixels which are at the extreme edge of the border."""
        return self[self._sub_border_flat_indexes]

    @classmethod
    def manual_slim(cls, grid):
        return Grid2DVoronoi(grid=grid)

    @property
    def shape_native_scaled(self):
        return (
            np.amax(self[:, 0]).astype("float") - np.amin(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float") - np.amin(self[:, 1]).astype("float"),
        )

    @property
    def scaled_maxima(self):
        return (
            np.amax(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float"),
        )

    @property
    def scaled_minima(self):
        return (
            np.amin(self[:, 0]).astype("float"),
            np.amin(self[:, 1]).astype("float"),
        )

    @property
    def extent(self):
        return np.array(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )
