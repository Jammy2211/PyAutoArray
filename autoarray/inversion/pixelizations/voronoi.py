import copy
import numpy as np
from typing import Dict, Optional

from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d import Grid2DSparse
from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DVoronoi
from autoarray.preloads import Preloads
from autoarray.inversion.pixelizations.abstract import AbstractPixelization
from autoarray.inversion.pixelizations.settings import SettingsPixelization
from autoarray.inversion.mappers.voronoi import MapperVoronoi

from autoarray.numba_util import profile_func


class Voronoi(AbstractPixelization):
    def __init__(self):
        """
        Abstract base class for a Voronoi pixelization, which represents pixels as an irregular grid of Voronoi
        cells which can form any shape, size or tesselation. The grid's coordinates are paired to Voronoi pixels as the
        nearest-neighbors of the Voronoi pixel-centers.

        The Voronoi pixelization grid, and the grids its used to discretize, have coordinates in both of the following
        two reference frames:

        - `data`: the original reference from of the masked data.

        - `source`: a reference frame where the grids in the `data` reference frame are transformed to create new grids
        of (y,x) coordinates. The transformation does not change the indexing, such that one can easily pair
        coordinates in the `source` frame to the `data` frame.

        The pixelization itself has its own (y,x) grid of coordinates, titled the `pixelization_grid`, which is
        typically much sparser than the grid associated with the original masked data. A Voronoi `pixelization_grid` is
        defined in both the `data` and `source` frame, where the initial grid is calculated in the `data` frame, which
        has the same transformation applied to it that is applied to the grid it discretizes.

        For example, in the project PyAutoLens, we have a 2D image which is typically masked with a circular mask.
        Its `data_grid_slim` is a 2D grid aligned with this circle, where each (y,x) coordinate is aligned with the
        centre of an image pixel. A "lensing transformation" is performed which maps this circular grid of (y,x)
        coordinates to a new grid of coordinates in the `source` frame, where the Voronoi pixelization is applied.
        """
        super().__init__()

    def mapper_from(
        self,
        source_grid_slim: Grid2D,
        source_pixelization_grid: Grid2DSparse = None,
        data_pixelization_grid: Grid2DSparse = None,
        hyper_image: np.ndarray = None,
        settings=SettingsPixelization(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Mapper objects describe the mappings between pixels in the untransformed masked 2D data and the pixels in a
        pixelization.

        This function returns a `MapperVoronoi` as follows:

        1) Before this routine is called, a sparse grid of (y,x) coordinates are computed from the 2D masked data,
        which act as the Voronoi pixel centres of the pixelization and mapper.

        2) Before this routine is called, operations are performed on this sparse grid that transform it from a 2D grid
        which overlaps with the 2D mask of the data to an irregular grid.

        3) If `settings.use_border=True`, the border of the input `grid` is used to relocate all of the grid's (y,x)
        coordinates beyond the border to the edge of the border.

        4) If `settings.use_border=True`, the border of the input `grid` is used to relocate all of the transformed
        sparse grid's (y,x) coordinates beyond the border to the edge of the border.

        5) Use the transformed sparse grid's (y,x) coordinates as the centres of the Voronoi pixelization.

        3) Return the `MapperVoronoi`.

        Parameters
        ----------
        source_grid_slim : aa.Grid2D
            A collection of grid describing the observed image's pixel coordinates (includes an image and sub grid).
        border : aa.GridBorder
            The borders of the grid_stacks (defined by their image-plane masks).
        hyper_image
            A pre-computed hyper-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """

        self.profiling_dict = profiling_dict

        relocated_source_grid_slim = self.relocate_grid_via_border(
            source_grid_slim=source_grid_slim, settings=settings, preloads=preloads
        )
        relocated_source_pixelization_grid = self.relocate_pixelization_grid_via_border_from(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            settings=settings,
        )

        try:

            source_pixelization_grid = self.make_pixelization_grid_from(
                source_grid_slim=relocated_source_grid_slim,
                source_pixelization_grid=relocated_source_pixelization_grid,
                sparse_index_for_slim_index=source_pixelization_grid.sparse_index_for_slim_index,
            )

            return MapperVoronoi(
                source_grid_slim=relocated_source_grid_slim,
                source_pixelization_grid=source_pixelization_grid,
                data_pixelization_grid=data_pixelization_grid,
                hyper_image=hyper_image,
                profiling_dict=profiling_dict,
            )

        except ValueError as e:
            raise e

    @profile_func
    def relocate_pixelization_grid_via_border_from(
        self,
        source_grid_slim: Grid2D,
        source_pixelization_grid: Grid2DSparse,
        settings: SettingsPixelization = SettingsPixelization(),
    ):
        """
        Return all coordinates of the pixeliztion itself that are outside the pixelization border to the edge of the
        border. The pixelization border is defined as the border of pixels in the original data's mask.

        This is used in the project PyAutoLens because the coordinates that are ray-traced near the centre of mass
        of galaxies are heavily demagnified and may trace to outskirts of the source-plane.
        """
        if settings.use_border:
            return source_grid_slim.relocated_pixelization_grid_from(
                pixelization_grid=source_pixelization_grid
            )
        return source_pixelization_grid

    @profile_func
    def make_pixelization_grid_from(
        self,
        source_grid_slim=None,
        source_pixelization_grid=None,
        sparse_index_for_slim_index=None,
    ):
        """
        The relocated pixelization grid is now used to create the pixelization's Voronoi grid using
        the scipy.spatial library.

        The array `sparse_index_for_slim_index` encodes the closest source pixel of every pixel on the
        (full resolution) sub image-plane grid. This is used for efficiently pairing every image-plane pixel to its
        corresponding source-plane pixel.
        """

        return Grid2DVoronoi(
            grid=source_pixelization_grid,
            nearest_pixelization_index_for_slim_index=sparse_index_for_slim_index,
        )


class VoronoiMagnification(Voronoi):
    def __init__(self, shape=(3, 3)):
        """
        A Voronoi pixelization, which represents pixels as an irregular grid of Voronoi cells which can form any
        shape, size or tesselation. The grid's coordinates are paired to Voronoi pixels as the nearest-neighbors of
        the Voronoi pixel-centers.

        The Voronoi pixelization grid, and the grids its used to discretize, have coordinates in both of the following
        two reference frames:

        - `data`: the original reference from of the masked data.

        - `source`: a reference frame where the grids in the `data` reference frame are transformed to create new grids
        of (y,x) coordinates. The transformation does not change the indexing, such that one can easily pair
        coordinates in the `source` frame to the `data` frame.

        The pixelization itself has its own (y,x) grid of coordinates, titled the `pixelization_grid`, which is
        typically much sparser than the grid associated with the original masked data. A Voronoi `pixelization_grid` is
        defined in both the `data` and `source` frame, where the initial grid is calculated in the `data` frame, which
        has the same transformation applied to it that is applied to the grid it discretizes.

        For the `VoronoiMagnification` pixelization the centres of the Voronoi grid are derived in the `data` frame,
        by overlaying a uniform grid with the input `shape` over the masked data's grid. All coordinates in this
        uniform grid which are contained within the mask are kept, have the same transformation applied to them as the
        masked data's grid and form the pixelization's Voronoi pixels.

        For example, in the project PyAutoLens, we have a 2D image which is typically masked with a circular mask.
        Its `data_grid_slim` is a 2D grid aligned with this circle, where each (y,x) coordinate is aligned with the
        centre of an image pixel. A "lensing transformation" is performed which maps this circular grid of (y,x)
        coordinates to a new grid of coordinates in the `source` frame, where the Voronoi pixelization is applied.

        Parameters
        ----------
        shape
            The shape of the unmasked sparse-grid which is laid over the masked image, in order to derive the \
            adaptive-magnification pixelization (see *ImagePlanePixelization*)
        """
        super(VoronoiMagnification, self).__init__()
        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]

    def data_pixelization_grid_from(
        self,
        data_grid_slim: Grid2D,
        hyper_image: np.ndarray = None,
        settings=SettingsPixelization(),
    ):

        return Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
            grid=data_grid_slim, unmasked_sparse_shape=self.shape
        )


class VoronoiBrightnessImage(Voronoi):
    def __init__(
        self, pixels: int = 10, weight_floor: float = 0.0, weight_power: float = 0.0
    ):
        """
        A Voronoi pixelization, which represents pixels as an irregular grid of Voronoi cells which can form any
        shape, size or tesselation. The grid's coordinates are paired to Voronoi pixels as the nearest-neighbors of
        the Voronoi pixel-centers.

        The Voronoi pixelization grid, and the grids its used to discretize, have coordinates in both of the following
        two reference frames:

        - `data`: the original reference from of the masked data.

        - `source`: a reference frame where the grids in the `data` reference frame are transformed to create new grids
        of (y,x) coordinates. The transformation does not change the indexing, such that one can easily pair
        coordinates in the `source` frame to the `data` frame.

        The pixelization itself has its own (y,x) grid of coordinates, titled the `pixelization_grid`, which is
        typically much sparser than the grid associated with the original masked data. A Voronoi `pixelization_grid` is
        defined in both the `data` and `source` frame, where the initial grid is calculated in the `data` frame, which
        has the same transformation applied to it that is applied to the grid it discretizes.

        For the `VoronoiBrightnessImage` pixelization the centres of the Voronoi grid are derived in the `data` frame,
        by applying a KMeans clustering algorithm to the masked data's values. These values are use compute `pixels`
        number of pixels, where the `weight_floor` and `weight_power` allow the KMeans algorithm to adapt the derived
        pixel centre locations to the data's brighest or faintest values.

        All coordinates have the same transformation applied to them as the masked data's grid and form the
        pixelization's Voronoi pixels.

        For example, in the project PyAutoLens, we have a 2D image which is typically masked with a circular mask.
        Its `data_grid_slim` is a 2D grid aligned with this circle, where each (y,x) coordinate is aligned with the
        centre of an image pixel. A "lensing transformation" is performed which maps this circular grid of (y,x)
        coordinates to a new grid of coordinates in the `source` frame, where the Voronoi pixelization is applied.

        Parameters
        ----------
        pixels
            The total number of pixels in the Voronoi pixelization, which is therefore also the number of (y,x)
            coordinates computed via the KMeans clustering algorithm in data frame.
        weight_floor
            A parameter which reweights the data values the KMeans algorithm is applied too; as the floor increases
            more weight is applied to values with lower values thus allowing Voronoi pixels to be placed in these
            regions of the data.
        weight_power
            A parameter which reweights the data values the KMeans algorithm is applied too; as the power increases
            more weight is applied to values with higher values thus allowing Voronoi pixels to be placed in these
            regions of the data.
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

    def data_pixelization_grid_from(
        self,
        data_grid_slim: Grid2D,
        hyper_image: np.ndarray,
        settings=SettingsPixelization(),
    ):

        weight_map = self.weight_map_from(hyper_image=hyper_image)

        return Grid2DSparse.from_total_pixels_grid_and_weight_map(
            total_pixels=self.pixels,
            grid=data_grid_slim,
            weight_map=weight_map,
            seed=settings.kmeans_seed,
            stochastic=settings.is_stochastic,
        )
