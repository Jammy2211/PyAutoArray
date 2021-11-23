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
        A rectangular pixelization, where pixels are defined on a Cartesian and uniform grid of shape
        (total_y_pixels, total_x_pixels).

        The indexing of a rectangular grid begins in the top-left corner and goes rightwards and downwards.

        Parameters
        -----------
        shape
            The 2D dimensions of the rectangular grid of pixels (y_pixels, x_pixel).
        """

        if shape[0] <= 2 or shape[1] <= 2:
            raise exc.PixelizationException(
                "The rectangular pixelization must be at least dimensions 3x3"
            )

        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]
        super(Rectangular, self).__init__()

    def mapper_from(
        self,
        grid: Grid2D,
        sparse_grid: Grid2D = None,
        sparse_image_plane_grid: Grid2D = None,
        hyper_image: np.ndarray = None,
        settings: SettingsPixelization = SettingsPixelization(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Mapper objects describe the mappings between pixels in masked 2D data and 2D pixels in a pixelization.

        This function returns a rectangular mapper from a rectangular pixelization, as follows:

        1) If `settings.use_border=True`, the border of the input `grid` is used to relocate all of the grid (y,x)
        coordintes beyond the border to the edge of the border.

        2) Determine the rectangular pixelization's (y,x) pixel coordinates, by laying it over this 2D grid of
        relocated (y,x) coordinates.

        3) Setup the rectangular mapper from the relocated grid and rectangular pixelization.

        Parameters
        ----------
        grid
            A 2D grid of (y,x) coordinates associated with the centres of every pixel in the 2D `data`. This could be a
            uniform grid which overlaps the data's pixels, but it may also have had transformation operations performed
            on it.
        sparse_grid
            Not used for a rectangular pixelization.
        sparse_image_plane_grid
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
            grid=grid, settings=settings, preloads=preloads
        )
        pixelization_grid = self.make_pixelization_grid(relocated_grid=relocated_grid)

        return MapperRectangular(
            source_grid_slim=relocated_grid,
            source_pixelization_grid=pixelization_grid,
            hyper_image=hyper_image,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def make_pixelization_grid(
        self,
        relocated_grid: Optional[Grid2D] = None,
        relocated_pixelization_grid: Optional[Grid2D] = None,
        sparse_index_for_slim_index: Optional[np.ndarray] = None,
    ) -> Grid2DRectangular:
        """
        Returns the rectangular pixelization grid.

        A specific function is used to do this so that it can be profiled via the `@profile_func` decorator, whereby
        the time this function takes to run is stored in the `profiling_dict`.

        Parameters
        ----------
        relocated_grid
            The (y,x) grid of coordinates over which the rectangular pixelization is overlaid, where this grid may have
            had exterior pixels relocated to its edge via the border.
        relocated_pixelization_grid
            Not used for a rectangular pixelization.
        sparse_index_for_slim_index
            Not used for a rectangular pixelization.
        Returns
        -------

        """
        return Grid2DRectangular.overlay_grid(
            shape_native=self.shape, grid=relocated_grid
        )

    def sparse_grid_from(
        self,
        grid: Grid2D,
        hyper_image: np.ndarray = None,
        settings=SettingsPixelization(),
    ):
        """
        Not used for rectangular pixelization.
        """
        return None
