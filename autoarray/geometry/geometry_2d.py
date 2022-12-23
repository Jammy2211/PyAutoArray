from __future__ import annotations
import logging
import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from autoarray.structures.arrays.uniform_2d import Array2D
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.geometry.abstract_2d import AbstractGeometry2D

from autoarray import type as ty
from autoarray.geometry import geometry_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Geometry2D(AbstractGeometry2D):
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
    def shape_native_scaled(self) -> Tuple[float, float]:
        """
        The (y,x) 2D shape of the mask in scaled units, computed from the 2D `shape` (units pixels) and
        the `pixel_scales` (units scaled/pixels) conversion factor.
        """
        return (
            float(self.pixel_scales[0] * self.shape_native[0]),
            float(self.pixel_scales[1] * self.shape_native[1]),
        )

    @property
    def scaled_maxima(self) -> Tuple[float, float]:
        return (
            (self.shape_native_scaled[0] / 2.0) + self.origin[0],
            (self.shape_native_scaled[1] / 2.0) + self.origin[1],
        )

    @property
    def scaled_minima(self) -> Tuple[float, float]:
        return (
            (-(self.shape_native_scaled[0] / 2.0)) + self.origin[0],
            (-(self.shape_native_scaled[1] / 2.0)) + self.origin[1],
        )

    @property
    def extent(self) -> np.ndarray:
        """
        The extent of the grid in scaled units returned as an ndarray of the form [x_min, x_max, y_min, y_max].

        This follows the format of the extent input parameter in the matplotlib method imshow (and other methods) and
        is used for visualization in the plot module, which is why the x and y coordinates are swapped compared to
        the normal PyAutoArray convention.
        """
        return np.array(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )

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

    def grid_pixels_2d_from(self, grid_scaled_2d: Grid2D) -> Grid2D:
        """
        Convert a grid of 2D (y,x) scaled coordinates to a grid of 2D (y,x) pixel values, which are returned as floats
        and include the decimal offset from each pixel's top-left corner.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
        highest y scaled coordinate value and lowest x scaled coordinate.

        The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_scaled_2d
            A grid of (y,x) coordinates in scaled units.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_pixels_2d = geometry_util.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_2d,
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )
        return Grid2D(grid=grid_pixels_2d, mask=grid_scaled_2d.mask)
