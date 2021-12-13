import copy
import numpy as np
from typing import Dict, Optional

from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d import Grid2DSparse
from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DDelaunay
from autoarray.preloads import Preloads
from autoarray.inversion.pixelizations.abstract import AbstractPixelization
from autoarray.inversion.pixelizations.settings import SettingsPixelization
from autoarray.inversion.mappers.delaunay import MapperDelaunay

from autoarray.numba_util import profile_func


class Delaunay(AbstractPixelization):
    def __init__(self):
        """Abstract base class for a Voronoi pixelization, which represents pixels as an irgrid of Voronoi \
         cells which can form any shape, size or tesselation.

         The grid's coordinates are paired to Voronoi pixels as the nearest-neighbors of the Voronoi \
        pixel-centers.
         """
        super().__init__()

    def mapper_from(
        self,
        grid: Grid2D,
        sparse_grid: Grid2DSparse = None,
        sparse_image_plane_grid: Grid2DSparse = None,
        hyper_image: np.ndarray = None,
        settings=SettingsPixelization(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
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
        grid : aa.Grid2D
            A collection of grid describing the observed image's pixel coordinates (includes an image and sub grid).
        border : aa.GridBorder
            The borders of the grid_stacks (defined by their image-plane masks).
        hyper_image
            A pre-computed hyper-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """

        self.profiling_dict = profiling_dict

        relocated_grid = self.relocate_grid_via_border(
            source_grid_slim=grid, settings=settings, preloads=preloads
        )
        relocated_pixelization_grid = self.relocate_pixelization_grid_via_border(
            grid=grid, pixelization_grid=sparse_grid, settings=settings
        )

        try:

            pixelization_grid = self.make_pixelization_grid(
                relocated_grid=relocated_grid,
                relocated_pixelization_grid=relocated_pixelization_grid,
                sparse_index_for_slim_index=sparse_grid.sparse_index_for_slim_index,
            )

            return MapperDelaunay(
                source_grid_slim=relocated_grid,
                source_pixelization_grid=pixelization_grid,
                data_pixelization_grid=sparse_image_plane_grid,
                hyper_image=hyper_image,
                profiling_dict=profiling_dict,
            )

        except ValueError as e:
            raise e

    @profile_func
    def relocate_pixelization_grid_via_border(
        self,
        grid: Grid2D,
        pixelization_grid: Grid2DSparse,
        settings: SettingsPixelization = SettingsPixelization(),
    ):
        """
        Return all coordinates of the pixeliztion itself that are outside the pixelization border to the edge of the
        border. The pixelization border is defined as the border of pixels in the original data's mask.

        This is used in the project PyAutoLens because the coordinates that are ray-traced near the centre of mass
        of galaxies are heavily demagnified and may trace to outskirts of the source-plane.
        """
        if settings.use_border:
            return grid.relocated_pixelization_grid_from(
                pixelization_grid=pixelization_grid
            )
        return pixelization_grid

    @profile_func
    def make_pixelization_grid(
        self,
        relocated_grid=None,
        relocated_pixelization_grid=None,
        sparse_index_for_slim_index=None,
    ):
        """
        The relocated pixelization grid is now used to create the pixelization's Voronoi grid using
        the scipy.spatial library.

        The array `sparse_index_for_slim_index` encodes the closest source pixel of every pixel on the
        (full resolution) sub image-plane grid. This is used for efficiently pairing every image-plane pixel to its
        corresponding source-plane pixel.
        """

        return Grid2DDelaunay(
            grid=relocated_pixelization_grid,
            #nearest_pixelization_index_for_slim_index=sparse_index_for_slim_index,
        )


class DelaunayMagnification(Delaunay):
    def __init__(self, shape=(3, 3)):
        """A pixelization which adapts to the magnification pattern of a lens's mass model and uses a Delaunay \
        pixelization to discretize the grid into pixels.

        Parameters
        ----------
        shape
            The shape of the unmasked sparse-grid which is laid over the masked image, in order to derive the \
            adaptive-magnification pixelization (see *ImagePlanePixelization*)
        """
        super(DelaunayMagnification, self).__init__()
        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]

    def sparse_grid_from(
        self,
        grid: Grid2D,
        hyper_image: np.ndarray = None,
        settings=SettingsPixelization(),
    ):

        return Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
            grid=grid, unmasked_sparse_shape=self.shape
        )


class DelaunayBrightnessImage(Delaunay):
    def __init__(self, pixels=10, weight_floor=0.0, weight_power=0.0):
        """A pixelization which adapts to the magnification pattern of a lens's mass model and uses a Delaunay \
        pixelization to discretize the grid into pixels.

        Parameters
        ----------

        """
        super().__init__()

        self.pixels = int(pixels)
        self.weight_floor = weight_floor
        self.weight_power = weight_power

    def weight_map_from(self, hyper_image: np.ndarray):

        weight_map = (hyper_image - np.min(hyper_image)) / (
            np.max(hyper_image) - np.min(hyper_image)
        ) + self.weight_floor * np.max(hyper_image)

        return np.power(weight_map, self.weight_power)

    def sparse_grid_from(
        self, grid: Grid2D, hyper_image: np.ndarray, settings=SettingsPixelization()
    ):

        weight_map = self.weight_map_from(hyper_image=hyper_image)

        return Grid2DSparse.from_total_pixels_grid_and_weight_map(
            total_pixels=self.pixels,
            grid=grid,
            weight_map=weight_map,
            seed=settings.kmeans_seed,
            stochastic=settings.is_stochastic,
        )
