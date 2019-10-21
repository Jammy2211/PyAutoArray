import numpy as np
import scipy.spatial

from autoarray import exc
from autoarray.structures import grids
from autoarray.operators.inversion import mappers
from autoarray.util import grid_util, pixelization_util


class Pixelization(object):
    def __init__(self):
        """ Abstract base class for a pixelization, which discretizes grid of (y,x) coordinates into pixels.
        """

    def mapper_from_grid_and_pixelization_grid(self, grid, border):
        raise NotImplementedError(
            "pixelization_mapper_from_grids_and_borders should be overridden"
        )

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])

    def __repr__(self):
        return "{}\n{}".format(self.__class__.__name__, str(self))


class Rectangular(Pixelization):
    def __init__(self, shape=(3, 3)):
        """A rectangular pixelization, where pixels are defined on a Cartesian and uniform grid of shape \ 
        (rows, columns).

        Like structures, the indexing of the rectangular grid begins in the top-left corner and goes right and down.

        Parameters
        -----------
        shape : (int, int)
            The dimensions of the rectangular grid of pixels (y_pixels, x_pixel)
        """

        if shape[0] <= 2 or shape[1] <= 2:
            raise exc.PixelizationException(
                "The rectangular pixelization must be at least dimensions 3x3"
            )

        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]
        super(Rectangular, self).__init__()

    class Geometry(object):
        def __init__(
            self, shape_2d, pixel_scales, origin, pixel_neighbors, pixel_neighbors_size
        ):
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
            self.shape_2d = shape_2d
            self.shape_2d_arcsec = (shape_2d[0] * pixel_scales[0], shape_2d[1] * pixel_scales[1])
            self.pixel_scales = pixel_scales
            self.origin = origin
            self.pixel_neighbors = pixel_neighbors.astype("int")
            self.pixel_neighbors_size = pixel_neighbors_size.astype("int")

        @property
        def pixel_centres(self):
            """The centre of every pixel in the rectangular pixelization."""
            return grid_util.grid_1d_via_shape_2d(
            shape_2d=self.shape_2d, pixel_scales=self.pixel_scales, sub_size=1, origin=self.origin
        )

    def geometry_from_grid(self, grid, buffer=1e-8):
        """Determine the geometry of the rectangular grid, by overlaying it over a grid of coordinates such that its \
         outer-most pixels align with the grid's outer most coordinates plus a small buffer.

        Parameters
        -----------
        grid : ndarray
            The (y,x) grid of coordinates over which the rectangular pixelization is placed to determine its 
        buffer : float
            The size the pixelization is buffered relative to the grid.
        """
        y_min = np.min(grid[:, 0]) - buffer
        y_max = np.max(grid[:, 0]) + buffer
        x_min = np.min(grid[:, 1]) - buffer
        x_max = np.max(grid[:, 1]) + buffer
        pixel_scales = (
            float((y_max - y_min) / self.shape[0]),
            float((x_max - x_min) / self.shape[1]),
        )
        origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)
        pixel_neighbors, pixel_neighbors_size = self.pixel_neighbors
        return self.Geometry(
            shape_2d=self.shape,
            pixel_scales=pixel_scales,
            origin=origin,
            pixel_neighbors=pixel_neighbors,
            pixel_neighbors_size=pixel_neighbors_size,
        )

    @property
    def pixel_neighbors(self):
        return pixelization_util.rectangular_neighbors_from_shape(shape=self.shape)

    def mapper_from_grid_and_pixelization_grid(
        self,
        grid,
        pixelization_grid=None,
        inversion_uses_border=False,
        hyper_image=None,
    ):
        """Setup a rectangular mapper from a rectangular pixelization, as follows:

        1) If a border is supplied, relocate all of the grid's and sub grid pixels beyond the border.
        2) Determine the rectangular pixelization's geometry, by laying the pixelization over the sub-grid.
        3) Setup the rectangular mapper from the relocated grid and rectangular pixelization.

        Parameters
        ----------
        grid : aa.Grid
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        border : aa.GridBorder | None
            The border of the grid's grid.
        hyper_image : ndarray
            A pre-computed hyper_galaxies-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """

        if inversion_uses_border:
            relocated_grid = grid.relocated_grid_from_grid(grid=grid)
        else:
            relocated_grid = grid

        geometry = self.geometry_from_grid(grid=relocated_grid)

        return mappers.RectangularMapper(
            pixels=self.pixels,
            grid=relocated_grid,
            pixel_centres=geometry.pixel_centres,
            shape_2d=self.shape,
            geometry=geometry,
            hyper_image=hyper_image,
        )

    def pixelization_grid_from_grid(
        self, grid, hyper_image=None, seed=1
    ):
        return None


class Voronoi(Pixelization):
    def __init__(self):
        """Abstract base class for a Voronoi pixelization, which represents pixels as an irgrid of Voronoi \
         cells which can form any shape, size or tesselation.

         The grid's coordinates are paired to Voronoi pixels as the nearest-neighbors of the Voronoi \
        pixel-centers.
         """
        super(Voronoi, self).__init__()

    class Geometry(object):
        def __init__(
            self,
            shape_2d_arcsec,
            pixel_centres,
            origin,
            pixel_neighbors,
            pixel_neighbors_size,
        ):
            """The geometry of a Voronoi pixelization.

            Parameters
            -----------
            shape_2d_arcsec : (float, float)
                The dimensions of the Voronoi grid ni arc-second (y_arcseconds, x_arcseconds)
            pixel_centres : ndarray
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
            self.shape_2d_arcsec = shape_2d_arcsec
            self.pixel_centres = pixel_centres
            self.origin = origin
            self.pixel_neighbors = pixel_neighbors.astype("int")
            self.pixel_neighbors_size = pixel_neighbors_size.astype("int")

    def geometry_from_grid(
        self, grid, pixel_centres, pixel_neighbors, pixel_neighbors_size, buffer=1e-8
    ):
        """Determine the geometry of the Voronoi pixelization, by alligning it with the outer-most coordinates on a \
        grid plus a small buffer.

        Parameters
        -----------
        grid : ndarray
            The (y,x) grid of coordinates which determine the Voronoi pixelization's 
        pixel_centres : ndarray
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
        y_min = np.min(grid[:, 0]) - buffer
        y_max = np.max(grid[:, 0]) + buffer
        x_min = np.min(grid[:, 1]) - buffer
        x_max = np.max(grid[:, 1]) + buffer
        shape_2d_arcsec = (y_max - y_min, x_max - x_min)
        origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)
        return self.Geometry(
            shape_2d_arcsec=shape_2d_arcsec,
            pixel_centres=pixel_centres,
            origin=origin,
            pixel_neighbors=pixel_neighbors,
            pixel_neighbors_size=pixel_neighbors_size,
        )

    @staticmethod
    def voronoi_from_pixel_centers(pixel_centers):
        """Compute the Voronoi grid of the pixelization, using the pixel centers.

        Parameters
        ----------
        pixel_centers : ndarray
            The (y,x) centre of every Voronoi pixel.
        """
        try:
            return scipy.spatial.Voronoi(
                np.asarray([pixel_centers[:, 1], pixel_centers[:, 0]]).T,
                qhull_options="Qbb Qc Qx Qm",
            )
        except OverflowError or scipy.spatial.qhull.QhullError:
            raise exc.PixelizationException()

    def neighbors_from_pixels_and_ridge_points(self, pixels, ridge_points):
        """Compute the neighbors of every Voronoi pixel as an ndarray of the pixel index's each pixel shares a \
        vertex with.

        The ridge points of the Voronoi grid are used to derive this.

        Parameters
        ----------
        ridge_points : scipy.spatial.Voronoi.ridge_points
            Each Voronoi-ridge (two indexes representing a pixel mapping_matrix).
        """
        return pixelization_util.voronoi_neighbors_from_pixels_and_ridge_points(
            pixels=pixels, ridge_points=np.asarray(ridge_points)
        )

    def mapper_from_grid_and_pixelization_grid(
        self,
        grid,
        pixelization_grid=None,
        inversion_uses_border=False,
        hyper_image=None,
    ):
        """Setup a Voronoi mapper from an adaptive-magnification pixelization, as follows:

        1) (before this routine is called), setup the 'pix' grid as part of the grid, which corresponds to a \
           sparse set of pixels in the image-plane which are traced to form the pixel centres.
        2) If a border is supplied, relocate all of the grid's grid, sub and pix grid pixels beyond the border.
        3) Determine the adaptive-magnification pixelization's pixel centres, by extracting them from the relocated \
           pix grid.
        4) Use these pixelization centres to setup the Voronoi pixelization.
        5) Determine the neighbors of every Voronoi cell in the Voronoi pixelization.
        6) Setup the geometry of the pixelizatioon using the relocated sub-grid and Voronoi pixelization.
        7) Setup a Voronoi mapper from all of the above quantities.

        Parameters
        ----------
        grid : aa.Grid
            A collection of grid describing the observed image's pixel coordinates (includes an image and sub grid).
        border : aa.GridBorder
            The borders of the grid_stacks (defined by their image-plane masks).
        hyper_image : ndarray
            A pre-computed hyper_galaxies-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """

        if inversion_uses_border:
            relocated_grid = grid.relocated_grid_from_grid(grid=grid)
            relocated_pixelization_grid = grid.relocated_pixelization_grid_from_pixelization_grid(
                pixelization_grid=pixelization_grid
            )
        else:
            relocated_grid = grid
            relocated_pixelization_grid = pixelization_grid

        pixel_centres = relocated_pixelization_grid
        pixels = pixel_centres.shape[0]

        voronoi = self.voronoi_from_pixel_centers(pixel_centres)

        pixel_neighbors, pixel_neighbors_size = self.neighbors_from_pixels_and_ridge_points(
            pixels=pixels, ridge_points=voronoi.ridge_points
        )

        geometry = self.geometry_from_grid(
            grid=relocated_grid,
            pixel_centres=pixel_centres,
            pixel_neighbors=pixel_neighbors,
            pixel_neighbors_size=pixel_neighbors_size,
        )

        return mappers.VoronoiMapper(
            pixels=pixels,
            grid=relocated_grid,
            pixel_centres=pixel_centres,
            voronoi=voronoi,
            geometry=geometry,
            hyper_image=hyper_image,
        )


class VoronoiMagnification(Voronoi):
    def __init__(self, shape=(3, 3)):
        """A pixelization which adapts to the magnification pattern of a lens's mass model and uses a Voronoi \
        pixelization to discretize the grid into pixels.

        Parameters
        ----------
        shape : (int, int)
            The shape of the unmasked sparse-grid which is laid over the masked image, in order to derive the \
            adaptive-magnification pixelization (see *ImagePlanePixelization*)
        """
        super(VoronoiMagnification, self).__init__()
        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]

    def pixelization_grid_from_grid(
        self, grid, hyper_image=None, seed=1
    ):
        sparse_to_grid = grids.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            grid=grid, unmasked_sparse_shape=self.shape
        )

        return grids.PixelizationGrid(
            grid_1d=sparse_to_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_to_grid.sparse_1d_index_for_mask_1d_index,
        )


class VoronoiBrightnessImage(Voronoi):
    def __init__(self, pixels=10, weight_floor=0.0, weight_power=0.0):
        """A pixelization which adapts to the magnification pattern of a lens's mass model and uses a Voronoi \
        pixelization to discretize the grid into pixels.

        Parameters
        ----------

        """
        super(VoronoiBrightnessImage, self).__init__()
        self.pixels = int(pixels)
        self.weight_floor = weight_floor
        self.weight_power = weight_power

    def weight_map_from_hyper_image(self, hyper_image):
        weight_map = (hyper_image - np.min(hyper_image)) / (
            np.max(hyper_image) - np.min(hyper_image)
        ) + self.weight_floor * np.max(hyper_image)

        return np.power(weight_map, self.weight_power)

    def pixelization_grid_from_grid(self, grid, hyper_image, seed=0):
        weight_map = self.weight_map_from_hyper_image(
            hyper_image=hyper_image
        )

        sparse_to_grid = grids.SparseToGrid.from_total_pixels_grid_and_weight_map(
            total_pixels=self.pixels,
            grid=grid,
            weight_map=weight_map,
            seed=seed,
        )

        return grids.PixelizationGrid(
            grid_1d=sparse_to_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_to_grid.sparse_1d_index_for_mask_1d_index,
        )
