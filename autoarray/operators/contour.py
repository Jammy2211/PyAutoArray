from __future__ import annotations
import numpy as np
from skimage import measure
from typing import TYPE_CHECKING

from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.geometry import geometry_util


class Grid2DContour:
    def __init__(self, grid, pixel_scales=None, shape_native=None):
        """
        Returns the contours surrounding grids of points as a 2D grid of (y,x) coordinates.

        For a grid of (y,x) coordinates, this function computes contours which encapsulate the points in regions of
        the grid. These contours are returned as a list of 2D grids of (y,x) coordinates.

        The calculation is performed as follows:

        1) Overlay a uniform grid of pixels over the grid of (y,x) coordinates (which can be irregular), where this


        Parameters
        ----------
        grid_2d

        Returns
        -------

        """
        self.grid = grid
        self._pixel_scales = pixel_scales
        self._shape_native = shape_native

    @property
    def pixel_scales(self):
        if self._pixel_scales is not None:
            return self._pixel_scales

    @property
    def shape_native(self):
        if self._shape_native is not None:
            return self._shape_native

        shape_y = (
            int(
                (np.amax(self.grid[:, 0]) - np.amin(self.grid[:, 0]))
                / self.pixel_scales[0]
            )
            + 1
        )
        shape_x = (
            int(
                (np.amax(self.grid[:, 1]) - np.amin(self.grid[:, 1]))
                / self.pixel_scales[1]
            )
            + 1
        )

        return (shape_y, shape_x)

    @property
    def contour_array(self):
        pixel_centres = geometry_util.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=np.array(self.grid),
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
        ).astype("int")

        arr = np.zeros(self.shape_native)

        for cen in pixel_centres:
            arr[cen[0], cen[1]] = 1

        # arr = np.zeros(self.shape_native)
        # arr[tuple(np.array(pixel_centres).T)] = 1

        return arr

    @property
    def contour_list(self):
        contour_indices_list = measure.find_contours(self.contour_array, 0)

        contour_list = []

        for contour_indices in contour_indices_list:
            grid_scaled_1d = geometry_util.grid_scaled_2d_slim_from(
                grid_pixels_2d_slim=contour_indices,
                shape_native=self.shape_native,
                pixel_scales=self.pixel_scales,
            )

            grid_scaled_1d[:, 0] -= self.pixel_scales[0] / 2.0
            grid_scaled_1d[:, 1] += self.pixel_scales[1] / 2.0

            contour_list.append(Grid2DIrregular(values=grid_scaled_1d))

        return contour_list
