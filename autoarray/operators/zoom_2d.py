from __future__ import annotations
import numpy as np
from typing import List, Tuple, Union

from autoarray.mask.mask_2d import Mask2D

from autoarray.structures.grids import grid_2d_util


class Zoom2D:

    def __init__(self, mask: Union[np.ndarray, List]):

        self.mask = mask

    @property
    def zoom_centre(self) -> Tuple[float, float]:
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=np.array(self),
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

        grid = Grid2D(values=grid, mask=self)

        extraction_grid_1d = self.geometry.grid_pixels_2d_from(grid_scaled_2d=grid)
        y_pixels_max = np.max(extraction_grid_1d[:, 0])
        y_pixels_min = np.min(extraction_grid_1d[:, 0])
        x_pixels_max = np.max(extraction_grid_1d[:, 1])
        x_pixels_min = np.min(extraction_grid_1d[:, 1])

        return (
            ((y_pixels_max + y_pixels_min - 1.0) / 2.0),
            ((x_pixels_max + x_pixels_min - 1.0) / 2.0),
        )

    @property
    def zoom_offset_pixels(self) -> Tuple[float, float]:
        if self.pixel_scales is None:
            return self.geometry.central_pixel_coordinates

        return (
            self.zoom_centre[0] - self.geometry.central_pixel_coordinates[0],
            self.zoom_centre[1] - self.geometry.central_pixel_coordinates[1],
        )

    @property
    def zoom_offset_scaled(self) -> Tuple[float, float]:
        return (
            -self.pixel_scales[0] * self.zoom_offset_pixels[0],
            self.pixel_scales[1] * self.zoom_offset_pixels[1],
        )

    @property
    def zoom_region(self) -> List[int]:
        """
        The zoomed rectangular region corresponding to the square encompassing all unmasked values. This zoomed
        extraction region is a squuare, even if the mask is rectangular.

        This is used to zoom in on the region of an image that is used in an analysis for visualization.
        """

        where = np.array(np.where(np.invert(self.astype("bool"))))
        y0, x0 = np.amin(where, axis=1)
        y1, x1 = np.amax(where, axis=1)

        # Have to convert mask to bool for invert function to work.

        ylength = y1 - y0
        xlength = x1 - x0

        if ylength > xlength:
            length_difference = ylength - xlength
            x1 += int(length_difference / 2.0)
            x0 -= int(length_difference / 2.0)
        elif xlength > ylength:
            length_difference = xlength - ylength
            y1 += int(length_difference / 2.0)
            y0 -= int(length_difference / 2.0)

        return [y0, y1 + 1, x0, x1 + 1]

    @property
    def zoom_shape_native(self) -> Tuple[int, int]:
        region = self.zoom_region
        return (region[1] - region[0], region[3] - region[2])

    @property
    def zoom_mask_unmasked(self) -> "Mask2D":
        """
        The scaled-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x
        value y value in scaled units.
        """

        return Mask2D.all_false(
            shape_native=self.zoom_shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.zoom_offset_scaled,
        )