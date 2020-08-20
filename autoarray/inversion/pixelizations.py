import numpy as np
from autoconf import conf
from autoarray import exc
from autoarray.structures import grids
from autoarray.inversion import mappers

import copy


class SettingsPixelization:
    def __init__(
        self,
        use_border: bool = False,
        pixel_limit: int = None,
        is_stochastic: bool = False,
        kmeans_seed: int = 0,
        preload_sparse_grids_of_planes=None,
    ):

        self.use_border = use_border
        self.pixel_limit = pixel_limit
        self.is_stochastic = is_stochastic
        self.kmeans_seed = kmeans_seed
        self.preload_sparse_grids_of_planes = preload_sparse_grids_of_planes

    @property
    def tag(self):
        return self.use_border_tag + self.is_stochastic_tag

    @property
    def use_border_tag(self):
        """Generate a tag for whether a border is used by the inversion.

        This changes the setup folder as follows (the example tags below are the default config tags):

        inversion_use_border = False -> settings
        inversion_use_border = True -> settings___no_border
        """
        if self.use_border:

            tag = conf.instance.tag.get("pixelization", "use_border", str)

            if not tag:
                return str(tag)
            return "__" + tag
        elif not self.use_border:
            return "__" + conf.instance.tag.get("pixelization", "no_border", str)

    @property
    def is_stochastic_tag(self):
        """Generate a tag for whether the inversion is stochastic.

        This changes the setup folder as follows (the example tags below are the default config tags):

        inversion_stochastic = False -> settings
        inversion_stochastic = True -> settings___stochastic
        """
        if not self.is_stochastic:

            tag = conf.instance.tag.get("pixelization", "not_stochastic", str)
            if not tag:
                return tag
            return "__" + tag
        elif self.is_stochastic:
            return "__" + conf.instance.tag.get("pixelization", "stochastic", str)

    def settings_with_is_stochastic_true(self):
        settings = copy.copy(self)
        settings.is_stochastic = True
        return settings

    def modify_preload(self, preload_sparse_grids_of_planes):

        settings = copy.copy(self)
        settings.preload_sparse_grids_of_planes = preload_sparse_grids_of_planes
        return settings


class Pixelization:
    def __init__(self):
        """ Abstract base class for a pixelization, which discretizes grid of (y,x) coordinates into pixels.
        """

    def mapper_from_grid_and_sparse_grid(self, grid: grids.Grid, border: np.ndarray):
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

    def mapper_from_grid_and_sparse_grid(
        self,
        grid: grids.Grid,
        sparse_grid: grids.Grid = None,
        hyper_image: np.ndarray = None,
        settings=SettingsPixelization(),
    ):
        """Setup a rectangular mapper from a rectangular pixelization, as follows:

        1) If a border is supplied, relocate all of the grid's and sub grid pixels beyond the border.
        2) Determine the rectangular pixelization's geometry, by laying the pixelization over the sub-grid.
        3) Setup the rectangular mapper from the relocated grid and rectangular pixelization.

        Parameters
        ----------
        grid : aa.Grid
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        border : aa.GridBorder I None
            The border of the grid's grid.
        hyper_image : ndarray
            A pre-computed hyper-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """

        if settings.use_border:
            relocated_grid = grid.relocated_grid_from_grid(grid=grid)
        else:
            relocated_grid = grid

        pixelization_grid = grids.GridRectangular.overlay_grid(
            shape_2d=self.shape, grid=relocated_grid
        )

        return mappers.MapperRectangular(
            grid=relocated_grid,
            pixelization_grid=pixelization_grid,
            hyper_image=hyper_image,
        )

    def sparse_grid_from_grid(
        self,
        grid: grids.Grid,
        hyper_image: np.ndarray = None,
        settings=SettingsPixelization(),
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

    def mapper_from_grid_and_sparse_grid(
        self,
        grid: grids.Grid,
        sparse_grid: grids.Grid = None,
        hyper_image: np.ndarray = None,
        settings=SettingsPixelization(),
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
            A pre-computed hyper-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """

        if settings.use_border:
            relocated_grid = grid.relocated_grid_from_grid(grid=grid)
            relocated_pixelization_grid = grid.relocated_pixelization_grid_from_pixelization_grid(
                pixelization_grid=sparse_grid
            )
        else:
            relocated_grid = grid
            relocated_pixelization_grid = sparse_grid

        try:
            pixelization_grid = grids.GridVoronoi(
                grid=relocated_pixelization_grid,
                nearest_pixelization_1d_index_for_mask_1d_index=sparse_grid.nearest_pixelization_1d_index_for_mask_1d_index,
            )

            return mappers.MapperVoronoi(
                grid=relocated_grid,
                pixelization_grid=pixelization_grid,
                hyper_image=hyper_image,
            )
        except ValueError as e:
            raise e


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

    def sparse_grid_from_grid(
        self,
        grid: grids.Grid,
        hyper_image: np.ndarray = None,
        settings=SettingsPixelization(),
    ):
        sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
            grid=grid, unmasked_sparse_shape=self.shape
        )

        return grids.GridVoronoi(
            grid=sparse_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_grid.sparse_1d_index_for_mask_1d_index,
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

    def weight_map_from_hyper_image(self, hyper_image: np.ndarray):
        weight_map = (hyper_image - np.min(hyper_image)) / (
            np.max(hyper_image) - np.min(hyper_image)
        ) + self.weight_floor * np.max(hyper_image)

        return np.power(weight_map, self.weight_power)

    def sparse_grid_from_grid(
        self, grid: grids.Grid, hyper_image: np.ndarray, settings=SettingsPixelization()
    ):
        weight_map = self.weight_map_from_hyper_image(hyper_image=hyper_image)

        sparse_grid = grids.GridSparse.from_total_pixels_grid_and_weight_map(
            total_pixels=self.pixels,
            grid=grid,
            weight_map=weight_map,
            seed=settings.kmeans_seed,
            stochastic=settings.is_stochastic,
        )

        return grids.GridVoronoi(
            grid=sparse_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_grid.sparse_1d_index_for_mask_1d_index,
        )
