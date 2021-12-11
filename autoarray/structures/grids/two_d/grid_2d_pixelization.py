import numpy as np
import scipy.spatial
import scipy.spatial.qhull as qhull
from typing import Optional, List, Union, Tuple

from autoconf import cached_property

from autoarray.structures.abstract_structure import AbstractStructure2D
from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.structures.grids.two_d import grid_2d_util
from autoarray.inversion.pixelizations import pixelization_util


class PixelNeighbors(np.ndarray):
    def __new__(cls, arr: np.ndarray, sizes: np.ndarray):

        obj = arr.view(cls)
        obj.sizes = sizes

        return obj


class Grid2DRectangular(AbstractStructure2D):
    def __new__(
        cls,
        grid: np.ndarray,
        shape_native: Tuple[int, int],
        pixel_scales: Union[Tuple[float, float], float],
        origin: Tuple[float, float] = (0.0, 0.0),
        *args,
        **kwargs
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

        return obj

    @classmethod
    def overlay_grid(
        cls, shape_native: Tuple[int, int], grid: np.ndarray, buffer: float = 1e-8
    ) -> "Grid2DRectangular":
        """
        The geometry of a rectangular grid.

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
        pixel_neighbors.sizes
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

    @cached_property
    def pixel_neighbors(self) -> PixelNeighbors:

        neighbors, sizes = pixelization_util.rectangular_neighbors_from(
            shape_native=self.shape_native
        )

        return PixelNeighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))

    @property
    def pixels(self) -> int:
        return self.shape_native[0] * self.shape_native[1]

    @property
    def shape_native_scaled(self) -> Tuple[float, float]:
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
    pixel_neighbors.sizes
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the \
        Voronoi grid.
    """

    def __new__(
        cls,
        grid: Union[np.ndarray, List],
        nearest_pixelization_index_for_slim_index: Optional[np.ndarray] = None,
        *args,
        **kwargs
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

        return obj

    def __array_finalize__(self, obj: object):

        if hasattr(obj, "nearest_pixelization_index_for_slim_index"):
            self.nearest_pixelization_index_for_slim_index = (
                obj.nearest_pixelization_index_for_slim_index
            )

    @cached_property
    def voronoi(self) -> scipy.spatial.Voronoi:
        try:
            return scipy.spatial.Voronoi(
                np.asarray([self[:, 1], self[:, 0]]).T, qhull_options="Qbb Qc Qx Qm"
            )
        except (ValueError, OverflowError, scipy.spatial.qhull.QhullError) as e:
            raise exc.PixelizationException() from e

    @cached_property
    def pixel_neighbors(self) -> PixelNeighbors:

        neighbors, sizes = pixelization_util.voronoi_neighbors_from(
            pixels=self.pixels, ridge_points=np.asarray(self.voronoi.ridge_points)
        )

        return PixelNeighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))

    @property
    def origin(self) -> Tuple[float, float]:
        return 0.0, 0.0

    @property
    def pixels(self) -> int:
        return self.shape[0]

    @property
    def sub_border_grid(self) -> np.ndarray:
        """
        The (y,x) grid of all sub-pixels which are at the border of the mask.

        This is NOT all sub-pixels which are in mask pixels at the mask's border, but specifically the sub-pixels
        within these border pixels which are at the extreme edge of the border.
        """
        return self[self.mask.sub_border_flat_indexes]

    @classmethod
    def manual_slim(cls, grid) -> "Grid2DVoronoi":
        return Grid2DVoronoi(grid=grid)

    @property
    def shape_native_scaled(self) -> Tuple[float, float]:
        return (
            np.amax(self[:, 0]).astype("float") - np.amin(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float") - np.amin(self[:, 1]).astype("float"),
        )

    @property
    def scaled_maxima(self) -> Tuple[float, float]:
        return (
            np.amax(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float"),
        )

    @property
    def scaled_minima(self) -> Tuple[float, float]:
        return (
            np.amin(self[:, 0]).astype("float"),
            np.amin(self[:, 1]).astype("float"),
        )

    @property
    def extent(self) -> np.ndarray:
        return np.array(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )


class Grid2DDelaunay(AbstractStructure2D):
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
    pixel_neighbors.sizes
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the \
        Voronoi grid.
    """

    def __new__(
        cls,
        grid: Union[np.ndarray, List],
        nearest_pixelization_index_for_slim_index: Optional[np.ndarray] = None,
        *args,
        **kwargs
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

        return obj

    def __array_finalize__(self, obj: object):

        if hasattr(obj, "nearest_pixelization_index_for_slim_index"):
            self.nearest_pixelization_index_for_slim_index = (
                obj.nearest_pixelization_index_for_slim_index
            )

    @cached_property
    def Delaunay(self) -> scipy.spatial.Delaunay:
        try:
            return scipy.spatial.Delaunay(
                np.asarray([self[:, 0], self[:, 1]]).T
            )
        except (ValueError, OverflowError, scipy.spatial.qhull.QhullError) as e:
            raise exc.PixelizationException() from e

    @cached_property
    def pixel_neighbors(self) -> PixelNeighbors:
    
        #neighbors, sizes = pixelization_util.voronoi_neighbors_from(
        #    pixels=self.pixels, ridge_points=np.asarray(self.voronoi.ridge_points)
        #)

        '''
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.vertex_neighbor_vertices.html#scipy.spatial.Delaunay.vertex_neighbor_vertices
        '''

        indptr, indices = self.Delaunay.vertex_neighbor_vertices

        #print(indptr)


        sizes = indptr[1:] - indptr[:-1]

        #print(sizes)

        neighbors = -1 * np.ones(shape=(self.pixels, int(np.max(sizes))), dtype='int')

        #print('Delaunay neighbors:')
        #print(neighbors)

        #print('Delaunay neighbor shape:')
        #print(np.shape(neighbors))

        for k in range(self.pixels):
            neighbors[k][0:sizes[k]] = indices[indptr[k]:indptr[k + 1]]


        return PixelNeighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))

    @property
    def origin(self) -> Tuple[float, float]:
        return 0.0, 0.0

    @property
    def pixels(self) -> int:
        return self.shape[0]

    @property
    def sub_border_grid(self) -> np.ndarray:
        """
        The (y,x) grid of all sub-pixels which are at the border of the mask.

        This is NOT all sub-pixels which are in mask pixels at the mask's border, but specifically the sub-pixels
        within these border pixels which are at the extreme edge of the border.
        """
        return self[self.mask.sub_border_flat_indexes]

    @classmethod
    def manual_slim(cls, grid) -> "Grid2DVoronoi":
        return Grid2DVoronoi(grid=grid)

    @property
    def shape_native_scaled(self) -> Tuple[float, float]:
        return (
            np.amax(self[:, 0]).astype("float") - np.amin(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float") - np.amin(self[:, 1]).astype("float"),
        )

    @property
    def scaled_maxima(self) -> Tuple[float, float]:
        return (
            np.amax(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float"),
        )

    @property
    def scaled_minima(self) -> Tuple[float, float]:
        return (
            np.amin(self[:, 0]).astype("float"),
            np.amin(self[:, 1]).astype("float"),
        )

    @property
    def extent(self) -> np.ndarray:
        return np.array(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )
