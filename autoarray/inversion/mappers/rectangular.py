import itertools
import numpy as np
from typing import Dict, Optional

from autoconf import cached_property

from autoarray.inversion.mappers.abstract import AbstractMapper
from autoarray.structures.arrays.two_d.array_2d import Array2D

from autoarray.numba_util import profile_func
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray.structures.grids.two_d import grid_2d_util


class MapperRectangular(AbstractMapper):
    def __init__(
        self,
        source_grid_slim,
        source_pixelization_grid,
        data_pixelization_grid=None,
        hyper_image=None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Class representing a rectangular mapper, which maps unmasked pixels on a masked 2D array in the form of
        a grid, see the *hyper_galaxies.array.grid* module to pixels discretized on a rectangular grid.

        The and uniform geometry of the rectangular grid is used to perform efficient pixel pairings.

        Parameters
        ----------
        pixels
            The number of pixels in the rectangular pixelization (y_pixels*x_pixels).
        source_grid_slim : gridStack
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        shape_native
            The dimensions of the rectangular grid of pixels (y_pixels, x_pixel)
        geometry
            The geometry (e.g. y / x edge locations, pixel-scales) of the rectangular pixelization.
        """
        super().__init__(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_image,
            profiling_dict=profiling_dict,
        )

    @property
    def shape_native(self):
        return self.source_pixelization_grid.shape_native

    @cached_property
    @profile_func
    def pixelization_index_for_sub_slim_index(self):
        """
        The `Mapper` contains:

         1) The traced grid of (y,x) source pixel coordinate centres.
         2) The traced grid of (y,x) image pixel coordinates.

        The function below pairs every image-pixel coordinate to every source-pixel centre.

        In the API, the `pixelization_index` refers to the source pixel index (e.g. source pixel 0, 1, 2 etc.) whereas
        the sub_slim index refers to the index of a sub-gridded image pixel (e.g. sub pixel 0, 1, 2 etc.).
        """
        return grid_2d_util.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=self.source_grid_slim,
            shape_native=self.source_pixelization_grid.shape_native,
            pixel_scales=self.source_pixelization_grid.pixel_scales,
            origin=self.source_pixelization_grid.origin,
        ).astype("int")

    def reconstruction_from(self, solution_vector):
        """
        Given the solution vector of an inversion (see *inversions.Inversion*), determine the reconstructed
        pixelization of the rectangular pixelization by using the mapper.
        """
        recon = array_2d_util.array_2d_native_from(
            array_2d_slim=solution_vector,
            mask_2d=np.full(
                fill_value=False, shape=self.source_pixelization_grid.shape_native
            ),
            sub_size=1,
        )
        return Array2D.manual(
            array=recon,
            sub_size=1,
            pixel_scales=self.source_pixelization_grid.pixel_scales,
            origin=self.source_pixelization_grid.origin,
        )
