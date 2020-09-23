import numpy as np
import scipy.spatial
import scipy.spatial.qhull as qhull

from autoarray import exc
from autoarray.structures import grids
from autoarray.mask import mask as msk
from autoarray.util import grid_util, pixelization_util


class GridRectangular(grids.Grid):
    def __new__(cls, grid, shape_2d, pixel_scales, origin=(0.0, 0.0), *args, **kwargs):
        """A grid of (y,x) coordinates which reprsent a rectangular grid of pixels which are used to form the pixel centres of adaptive pixelizations in the \
        *pixelizations* module.

        A *PixGrid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_pix_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixelization pixel index and second element the y or x arc -econd coordinates. For example:

        - pix_grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - pix_grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Parameters
        -----------
        pix_grid : ndarray
            The grid of (y,x) scaled coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        nearest_pixelization_1d_index_for_mask_1d_index : ndarray
            A 1D array that maps every grid pixel to its nearest pixelization-grid pixel.
        """

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=1, origin=origin
        )

        obj = super().__new__(cls=cls, grid=grid, mask=mask)

        pixel_neighbors, pixel_neighbors_size = pixelization_util.rectangular_neighbors_from(
            shape_2d=shape_2d
        )
        obj.pixel_neighbors = pixel_neighbors.astype("int")
        obj.pixel_neighbors_size = pixel_neighbors_size.astype("int")
        return obj

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super().__setstate__(state[0:-1])

    @classmethod
    def overlay_grid(cls, shape_2d, grid, buffer=1e-8):
        """The geometry of a rectangular grid.

        This is used to map grid of (y,x) scaled coordinates to the pixels on the rectangular grid.

        Parameters
        -----------
        shape_2d : (int, int)
            The dimensions of the rectangular grid of pixels (y_pixels, x_pixel)
        pixel_scales : (float, float)
            The pixel conversion scale of a pixel in the y and x directions.
        origin : (float, float)
            The scaled origin of the rectangular pixelization's coordinate system.
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

        grid_1d = grid_util.grid_1d_via_shape_2d_from(
            shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=1, origin=origin
        )

        return GridRectangular(
            grid=grid_1d, shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
        )

    @property
    def pixels(self):
        return self.shape_2d[0] * self.shape_2d[1]

    @property
    def shape_2d_scaled(self):
        return (
            (self.shape_2d[0] * self.pixel_scales[0]),
            (self.shape_2d[1] * self.pixel_scales[1]),
        )


class GridVoronoi(np.ndarray):
    """Determine the geometry of the Voronoi pixelization, by alligning it with the outer-most coordinates on a \
    grid plus a small buffer.

    Parameters
    -----------
    grid : ndarray
        The (y,x) grid of coordinates which determine the Voronoi pixelization's
    pixelization_grid : ndarray
        The (y,x) centre of every Voronoi pixel in scaleds.
    origin : (float, float)
        The scaled origin of the Voronoi pixelization's coordinate system.
    pixel_neighbors : ndarray
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in \
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : ndarrayy
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the \
        Voronoi grid.
    """

    def __new__(
        cls, grid, nearest_pixelization_1d_index_for_mask_1d_index=None, *args, **kwargs
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
            The grid of (y,x) scaled coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        nearest_pixelization_1d_index_for_mask_1d_index : ndarray
            A 1D array that maps every grid pixel to its nearest pixelization-grid pixel.
        """

        if type(grid) is list:
            grid = np.asarray(grid)

        obj = grid.view(cls)
        obj.nearest_pixelization_1d_index_for_mask_1d_index = (
            nearest_pixelization_1d_index_for_mask_1d_index
        )

        obj.pixels = grid.shape[0]

        try:
            obj.voronoi = scipy.spatial.Voronoi(
                np.asarray([grid[:, 1], grid[:, 0]]).T, qhull_options="Qbb Qc Qx Qm"
            )
        except ValueError or OverflowError or scipy.spatial.qhull.QhullError:
            raise exc.PixelizationException()

        pixel_neighbors, pixel_neighbors_size = pixelization_util.voronoi_neighbors_from(
            pixels=obj.pixels, ridge_points=np.asarray(obj.voronoi.ridge_points)
        )

        obj.pixel_neighbors = pixel_neighbors.astype("int")
        obj.pixel_neighbors_size = pixel_neighbors_size.astype("int")
        obj.nearest_pixelization_1d_index_for_mask_1d_index = (
            nearest_pixelization_1d_index_for_mask_1d_index
        )

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "pixel_neighbors"):
            self.pixel_neighbors = obj.pixel_neighbors

        if hasattr(obj, "pixel_neighbors_size"):
            self.pixel_neighbors_size = obj.pixel_neighbors_size

        if hasattr(obj, "nearest_pixelization_1d_index_for_mask_1d_index"):
            self.nearest_pixelization_1d_index_for_mask_1d_index = (
                obj.nearest_pixelization_1d_index_for_mask_1d_index
            )

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super().__setstate__(state[0:-1])

    @classmethod
    def manual_1d(cls, grid):
        return GridVoronoi(grid=grid)

    @classmethod
    def from_grid_and_unmasked_2d_grid_shape(cls, unmasked_sparse_shape, grid):

        sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=unmasked_sparse_shape, grid=grid
        )

        return GridVoronoi(
            grid=sparse_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_grid.sparse_1d_index_for_mask_1d_index,
        )

    @property
    def shape_2d_scaled(self):
        return (
            np.amax(self[:, 0]) - np.amin(self[:, 0]),
            np.amax(self[:, 1]) - np.amin(self[:, 1]),
        )

    @property
    def scaled_maxima(self):
        return (np.amax(self[:, 0]), np.amax(self[:, 1]))

    @property
    def scaled_minima(self):
        return (np.amin(self[:, 0]), np.amin(self[:, 1]))

    @property
    def extent(self):
        return np.asarray(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )
