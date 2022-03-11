import numpy as np
from typing import Dict, Optional, Tuple


from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.grid_2d_pixelization import Grid2DRectangular
from autoarray.preloads import Preloads
from autoarray.inversion.pixelizations.abstract import AbstractPixelization
from autoarray.inversion.pixelizations.settings import SettingsPixelization
from autoarray.inversion.mappers.rectangular import MapperRectangularNoInterp

from autoarray import exc
from autoarray.numba_util import profile_func


class Rectangular(AbstractPixelization):
    def __init__(self, shape: Tuple[int, int] = (3, 3)):
        """
        A pixelization associates a 2D grid of (y,x) coordinates (which are expected to be aligned with a masked
        dataset) with a 2D grid of pixels.

        The rectangular pixelization represents pixels using a uniform rectangular grid.

        Both of these grids (e.g. the masked dataset's 2D grid and the grid of the Voronoi pixelization's pixels)
        have (y,x) coordinates in in two reference frames:

        - `data`: the original reference frame of the masked data.

        - `source`: a reference frame where grids in the `data` reference frame are transformed to a new reference
        frame (e.g. their (y,x) coordinates may be shifted, stretched or have a more complicated operation performed
        on them).

        The grid associated with the masked dataset and Voronoi pixelization have the following variable names:

        - `grid_slim`: the (y,x) grid of coordinates of the original masked data (which can be in the data frame and
        given the variable name `data_grid_slim` or in the transformed source frame with the variable
        name `source_grid_slim`).

        - `pixelization_grid`: the (y,x) grid of Voronoi pixels which are associated with the `grid_slim` (y,x)
        coordinates (association is always performed in the `source` reference frame).

        A rectangular pixelization has three grids associated with it: `data_grid_slim`, `source_grid_slim`,
        and `source_pixelization_grid`. It does not have a `data_pixelization_grid because a rectangular pixelization
        is constructed by overlaying a grid of rectangular over the `source_grid_slim` (it is therefore entirely
        constructed in the `source` frame).

        If a transformation of coordinates is not applied, the `data` frame and `source` frames are identical.

        The (y,x) coordinates of the `source_pixelization_grid` represent the centres of each rectangular pixel.

        Each (y,x) coordinate in the `source_grid_slim` is associated with the rectangular pixelization pixel it falls
        within. No interpolation is performed when making these associations.

        The rectangular grid is uniform, has dimensions (total_y_pixels, total_x_pixels) and has indexing beginning
        in the top-left corner and going rightwards and downwards.

        In the project `PyAutoLens`, one's data is a masked 2D image. Its `data_grid_slim` is a 2D grid where every
        (y,x) coordinate is aligned with the centre of every unmasked image pixel. A "lensing operation" transforms
        this grid of (y,x) coordinates from the `data` frame to a new grid of (y,x) coordinates in the `source` frame.
        The pixelization is then applied in the source frame.. In lensing terminology, the `data` frame is
        the `image-plane` and `source` frame the `source-plane`.

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

        self.profiling_dict = {}

    @property
    def uses_interpolation(self):
        return False

    def mapper_from(
        self,
        source_grid_slim: Grid2D,
        source_pixelization_grid: Grid2D = None,
        data_pixelization_grid: Grid2D = None,
        hyper_image: np.ndarray = None,
        settings: SettingsPixelization = SettingsPixelization(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ) -> MapperRectangularNoInterp:
        """
        Mapper objects describe the mappings between pixels in the masked 2D data and the pixels in a pixelization,
        in both the `data` and `source` frames.

        This function returns a `MapperRectangularNoInterp` as follows:

        1) If `settings.use_border=True`, the border of the input `source_grid_slim` is used to relocate all of the
        grid's (y,x) coordinates beyond the border to the edge of the border.

        2) Determine the (y,x) coordinates of the pixelization's rectangular pixels, by laying this rectangular grid
        over the 2D grid of relocated (y,x) coordinates computed in step 1 (or the input `source_grid_slim` if step 1
        is bypassed).

        3) Return the `MapperRectangularNoInterp`.

        Parameters
        ----------
        source_grid_slim
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_pixelization_grid
            Not used for a rectangular pixelization, because the pixelization grid in the `source` frame is computed
            by overlaying the `source_grid_slim` with the rectangular pixelization.
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

        relocated_grid = self.relocated_grid_from(
            source_grid_slim=source_grid_slim, settings=settings, preloads=preloads
        )
        pixelization_grid = self.pixelization_grid_from(source_grid_slim=relocated_grid)

        return MapperRectangularNoInterp(
            source_grid_slim=relocated_grid,
            source_pixelization_grid=pixelization_grid,
            hyper_image=hyper_image,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def pixelization_grid_from(
        self,
        source_grid_slim: Optional[Grid2D] = None,
        source_pixelization_grid: Optional[Grid2D] = None,
        sparse_index_for_slim_index: Optional[np.ndarray] = None,
    ) -> Grid2DRectangular:
        """
        Return the rectangular `source_pixelization_grid` as a `Grid2DRectangular` object, which provides additional
        functionality for perform operatons that exploit the geometry of a rectangular pixelization.

        Parameters
        ----------
        source_grid_slim
            The (y,x) grid of coordinates over which the rectangular pixelization is overlaid, where this grid may have
            had exterior pixels relocated to its edge via the border.
        source_pixelization_grid
            Not used for a rectangular pixelization, because the pixelization grid in the `source` frame is computed
            by overlaying the `source_grid_slim` with the rectangular pixelization.
        sparse_index_for_slim_index
            Not used for a rectangular pixelization.
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
