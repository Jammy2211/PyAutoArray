import copy
import numpy as np
from typing import Dict, Optional


from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DRectangular
from autoarray.preloads import Preloads
from autoarray.inversion.pixelizations.abstract import AbstractPixelization
from autoarray.inversion.pixelizations.settings import SettingsPixelization
from autoarray.inversion.mappers.rectangular import MapperRectangular

from autoarray import exc
from autoarray.numba_util import profile_func


class Rectangular(AbstractPixelization):
    def __init__(self, shape=(3, 3)):
        """A rectangular pixelization, where pixels are defined on a Cartesian and uniform grid of shape \
        (total_y_pixels, total_x_pixels).

        Like structures, the indexing of the rectangular grid begins in the top-left corner and goes right and down.

        Parameters
        -----------
        shape
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
        grid: Grid2D,
        sparse_grid: Grid2D = None,
        sparse_image_plane_grid: Grid2D = None,
        hyper_image: np.ndarray = None,
        settings: SettingsPixelization = SettingsPixelization(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Setup a rectangular mapper from a rectangular pixelization, as follows:

        1) If a border is supplied, relocate all of the grid's and sub grid pixels beyond the border.
        2) Determine the rectangular pixelization's geometry, by laying the pixelization over the sub-grid.
        3) Setup the rectangular mapper from the relocated grid and rectangular pixelization.

        Parameters
        ----------
        grid : aa.Grid2D
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        border : aa.GridBorder I None
            The border of the grid's grid.
        hyper_image
            A pre-computed hyper-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
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
        relocated_grid=None,
        relocated_pixelization_grid=None,
        sparse_index_for_slim_index=None,
    ):

        return Grid2DRectangular.overlay_grid(
            shape_native=self.shape, grid=relocated_grid
        )

    def sparse_grid_from_grid(
        self,
        grid: Grid2D,
        hyper_image: np.ndarray = None,
        settings=SettingsPixelization(),
    ):
        return None
