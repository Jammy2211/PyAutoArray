import numpy as np
import scipy.spatial

from autoarray import exc
from autoarray.mask import mask as msk
from autoarray.structures import grids
from autoarray.operators.inversion import mappers
from autoarray.util import grid_util, pixelization_util

class RectangularGrid(grids.Grid):

    def __new__(
            cls, grid_1d, shape_2d, pixel_scales, origin=(0.0, 0.0), *args,
            **kwargs
    ):
        """A pixelization-grid of (y,x) coordinates which are used to form the pixel centres of adaptive pixelizations in the \
        *pixelizations* module.

        A *PixGrid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_pix_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixelization pixel index and second element the y or x arc -econd coordinates. For example:

        - pix_grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - pix_grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Parameters
        -----------
        pix_grid : ndarray
            The grid of (y,x) arc-second coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        nearest_irregular_1d_index_for_mask_1d_index : ndarray
            A 1D array that maps every grid pixel to its nearest pixelization-grid pixel.
        """

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
        )

        obj = super(RectangularGrid, cls).__new__(cls=cls, grid_1d=grid_1d, mask=mask)
        pixel_neighbors, pixel_neighbors_size = pixelization_util.rectangular_neighbors_from_shape(shape=shape_2d)
        obj.pixel_neighbors = pixel_neighbors.astype("int")
        obj.pixel_neighbors_size = pixel_neighbors_size.astype("int")
        return obj

    def __array_finalize__(self, obj):
        pass

    @classmethod
    def from_shape_2d_and_grid(cls, shape_2d, grid, buffer=1e-8):
        """The geometry of a rectangular grid.

        This is used to map grid of (y,x) arc-second coordinates to the pixels on the rectangular grid.

        Parameters
        -----------
        shape_2d : (int, int)
            The dimensions of the rectangular grid of pixels (y_pixels, x_pixel)
        pixel_scales : (float, float)
            The pixel-to-arcsecond scale of a pixel in the y and x directions.
        origin : (float, float)
            The arc-second origin of the rectangular pixelization's coordinate system.
        pixel_neighbors : ndarray
            An array of length (y_pixels*x_pixels) which provides the index of all neighbors of every pixel in \
            the rectangular grid (entries of -1 correspond to no neighbor).
        pixel_neighbors_size : ndarrayy
            An array of length (y_pixels*x_pixels) which gives the number of neighbors of every pixel in the \
            rectangular grid.
        """

        y_min = np.min(grid[:, 0]) - buffer
        y_max = np.max(grid[:, 0]) + buffer
        x_min = np.min(grid[:, 1]) - buffer
        x_max = np.max(grid[:, 1]) + buffer

        pixel_scales = (
            float((y_max - y_min) / shape_2d[0]),
            float((x_max - x_min) / shape_2d[1]),
        )

        origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)

        grid_1d = grid_util.grid_1d_via_shape_2d(
                shape_2d=shape_2d,
                pixel_scales=pixel_scales,
                sub_size=1,
                origin=origin,
            )

        return RectangularGrid(grid_1d=grid_1d, shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin)

    @property
    def pixels(self):
        return self.shape_2d[0] * self.shape_2d[1]

    @property
    def shape_2d_arcsec(self):
        return (self.shape_2d[0] * self.pixel_scales[0], self.shape_2d[1] * self.pixel_scales[1])


class VoronoiGrid(grids.IrregularGrid):
    """Determine the geometry of the Voronoi pixelization, by alligning it with the outer-most coordinates on a \
    grid plus a small buffer.

    Parameters
    -----------
    grid : ndarray
        The (y,x) grid of coordinates which determine the Voronoi pixelization's
    pixelization_grid : ndarray
        The (y,x) centre of every Voronoi pixel in arc-seconds.
    origin : (float, float)
        The arc-second origin of the Voronoi pixelization's coordinate system.
    pixel_neighbors : ndarray
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in \
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : ndarrayy
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the \
        Voronoi grid.
    """
    def __new__(
            cls, grid_1d, nearest_irregular_1d_index_for_mask_1d_index=None, *args,
            **kwargs
    ):
        """A pixelization-grid of (y,x) coordinates which are used to form the pixel centres of adaptive pixelizations in the \
        *pixelizations* module.

        A *PixGrid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_pix_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixelization pixel index and second element the y or x arc -econd coordinates. For example:

        - pix_grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - pix_grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Parameters
        -----------
        pix_grid : ndarray
            The grid of (y,x) arc-second coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        nearest_irregular_1d_index_for_mask_1d_index : ndarray
            A 1D array that maps every grid pixel to its nearest pixelization-grid pixel.
        """

        obj = super(VoronoiGrid, cls).__new__(cls=cls, grid=grid_1d)

        obj.pixels = grid_1d.shape[0]

        try:
            obj.voronoi = scipy.spatial.Voronoi(
                np.asarray([grid_1d[:, 1], grid_1d[:, 0]]).T,
                qhull_options="Qbb Qc Qx Qm",
            )
        except OverflowError or scipy.spatial.qhull.QhullError:
            raise exc.PixelizationException()

        pixel_neighbors, pixel_neighbors_size = pixelization_util.voronoi_neighbors_from_pixels_and_ridge_points(
            pixels=obj.pixels, ridge_points=np.asarray(obj.voronoi.ridge_points)
        )

        obj.pixel_neighbors = pixel_neighbors.astype("int")
        obj.pixel_neighbors_size = pixel_neighbors_size.astype("int")
        obj.nearest_irregular_1d_index_for_mask_1d_index = nearest_irregular_1d_index_for_mask_1d_index

        return obj