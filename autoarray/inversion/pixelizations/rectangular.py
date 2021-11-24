import numpy as np
from typing import Dict, Optional, Tuple


from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DRectangular
from autoarray.preloads import Preloads
from autoarray.inversion.pixelizations.abstract import AbstractPixelization
from autoarray.inversion.pixelizations.settings import SettingsPixelization
from autoarray.inversion.mappers.rectangular import MapperRectangular

from autoarray import exc
from autoarray.numba_util import profile_func


class Rectangular(AbstractPixelization):
    def __init__(self, shape: Tuple[int, int] = (3, 3)):
        """
        A rectangular pixelization, where pixels are on a uniform rectangular grid of dimensions
        (total_y_pixels, total_x_pixels). The pixels of this grid are therefore rectangles.

        The indexing of a rectangular grid begins in the top-left corner and goes rightwards and downwards.

        The rectangular pixelization grid, and the grids it is used to discretize, have coordinates in one or both of
        the following two reference frames:

        - `data`: the original reference from of the masked data.

        - `source`: a reference frame where the grids in the `data` reference frame are transformed to create new grids
        of (y,x) coordinates. The transformation does not change the indexing, such that one can easily pair
        coordinates in the `source` frame to the `data` frame.

        The pixelization itself has its own (y,x) grid of coordinates, titled the `pixelization_grid`, which is
        typically much sparser than the grid associated with the original masked data. A `Rectangular` pixelization
        does not have a grid in the `data` frame but only the `source` frame. The grid of coordinates it is applied too
        has coordinates in both frames.

        For example, in the project PyAutoLens, we have a 2D image which is typically masked with a circular mask.
        Its `data_grid_slim` is a 2D grid aligned with this circle, where each (y,x) coordinate is aligned with the
        centre of an image pixel. A "lensing transformation" is performed which maps this circular grid of (y,x)
        coordinates to a new grid of coordinates in the `source` frame, where the rectangular pixelization may be
        overlaid.

        Parameters
        -----------
        shape
            The 2D dimensions of the rectangular grid of pixels (total_y_pixels, total_x_pixel).
        """

        if shape[0] <= 2 or shape[1] <= 2:
            raise exc.PixelizationException(
                "The rectangular pixelization must be at least dimensions 3x3"
            )

        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]
        super().__init__()

    def mapper_from(
        self,
        source_grid_slim: Grid2D,
        source_pixelization_grid: Grid2D = None,
        data_pixelization_grid: Grid2D = None,
        hyper_image: np.ndarray = None,
        settings: SettingsPixelization = SettingsPixelization(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ) -> MapperRectangular:
        """
        Mapper objects describe the mappings between pixels in the untransformed masked 2D data and the pixels in a
        pixelization.

        This function returns a `MapperRectangular` as follows:

        1) If `settings.use_border=True`, the border of the input `grid` is used to relocate all of the grid's (y,x)
        coordinates beyond the border to the edge of the border.

        2) Determine the (y,x) coordinates of the pixelization's rectangular pixels, by laying this rectangular grid
        over the 2D grid of relocated (y,x) coordinates computed in step 1).

        3) Return the `MapperRectangular`.

        Parameters
        ----------
        source_grid_slim
            A 2D grid of (y,x) coordinates associated with the centres of every pixel in the 2D `data`. This could be a
            uniform grid which overlaps the data's pixels, but it may also have had transformation operations performed
            on it.
        source_pixelization_grid
            Not used for a rectangular pixelization.
        data_pixelization_grid
            Not used for a rectangular pixelization.
        hyper_image
            Not used for a rectangular pixelization.
        settings
            Settings controlling the pixelization for example if a border is used to relocate its exterior coordinates.
        preloads
            Object which may contain preloaded arrays of quantities computed in the pixelization, which are passed via
            this object speed up the calculation.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        self.profiling_dict = profiling_dict

        relocated_grid = self.relocate_grid_via_border(
            source_grid_slim=source_grid_slim, settings=settings, preloads=preloads
        )
        pixelization_grid = self.make_pixelization_grid_from(
            source_grid_slim=relocated_grid
        )

        return MapperRectangular(
            source_grid_slim=relocated_grid,
            source_pixelization_grid=pixelization_grid,
            hyper_image=hyper_image,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def make_pixelization_grid_from(
        self,
        source_grid_slim: Optional[Grid2D] = None,
        source_pixelization_grid: Optional[Grid2D] = None,
        sparse_index_for_slim_index: Optional[np.ndarray] = None,
    ) -> Grid2DRectangular:
        """
        Returns the rectangular pixelization grid.

        A specific function is used to do this so that it can be profiled via the `@profile_func` decorator, whereby
        the time this function takes to run is stored in the `profiling_dict`.

        Parameters
        ----------
        source_grid_slim
            The (y,x) grid of coordinates over which the rectangular pixelization is overlaid, where this grid may have
            had exterior pixels relocated to its edge via the border.
        source_pixelization_grid
            Not used for a rectangular pixelization.
        sparse_index_for_slim_index
            Not used for a rectangular pixelization.
        Returns
        -------

        """
        return Grid2DRectangular.overlay_grid(
            shape_native=self.shape, grid=source_grid_slim
        )

    def data_pixelization_grid_from(
        self,
        data_grid_slim: Grid2D,
        hyper_image: np.ndarray = None,
        settings=SettingsPixelization(),
    ):
        """
        Not used for rectangular pixelization.
        """
        return None
