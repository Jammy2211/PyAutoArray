from __future__ import annotations
import logging
import copy
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from autoarray.structures.arrays.uniform_2d import Array2D
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoconf import cached_property

from autoarray.mask.abstract_mask import Mask

from autoarray import exc
from autoarray import type as ty
from autoarray.structures.arrays import array_2d_util
from autoarray.geometry import geometry_util
from autoarray.structures.grids import grid_2d_util
from autoarray.mask import mask_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Geometry2D:
    def __init__(
        self,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
    ):

        self.shape_native = shape_native
        self.pixel_scales = pixel_scales
        self.origin = origin

    @property
    def central_pixel_coordinates(self) -> Tuple[float, float]:
        return geometry_util.central_pixel_coordinates_2d_from(
            shape_native=self.shape_native
        )

    @property
    def central_scaled_coordinates(self) -> Tuple[float, float]:

        return geometry_util.central_scaled_coordinate_2d_from(
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    def pixel_coordinates_2d_from(
        self, scaled_coordinates_2d: Tuple[float, float]
    ) -> Tuple[float, float]:

        return geometry_util.pixel_coordinates_2d_from(
            scaled_coordinates_2d=scaled_coordinates_2d,
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origins=self.origin,
        )

    def scaled_coordinates_2d_from(
        self, pixel_coordinates_2d: Tuple[float, float]
    ) -> Tuple[float, float]:

        return geometry_util.scaled_coordinates_2d_from(
            pixel_coordinates_2d=pixel_coordinates_2d,
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origins=self.origin,
        )

    def grid_pixels_from(self, grid_scaled_1d: Grid2D) -> Grid2D:
        """
        Convert a grid of (y,x) scaled coordinates to a grid of (y,x) pixel values. Pixel coordinates are
        returned as floats such that they include the decimal offset from each pixel's top-left corner.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
        highest y scaled coordinate value and lowest x scaled coordinate.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_1d: np.ndarray
            A grid of (y,x) coordinates in scaled units.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_pixels_1d = geometry_util.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_1d,
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )
        return Grid2D(grid=grid_pixels_1d, mask=grid_scaled_1d.mask)
