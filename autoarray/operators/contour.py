from __future__ import annotations
import numpy as np
from skimage import measure
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.geometry import geometry_util


class Grid2DContour:
    def __init__(self, grid, pixel_scales, shape_native, contour_array=None):
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
        self.pixel_scales = pixel_scales
        self.shape_native = shape_native
        self._contour_array = contour_array

    @property
    def contour_array(self):
        if self._contour_array is not None:
            return self._contour_array

        pixel_centres = geometry_util.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=np.array(self.grid),
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
        ).astype("int")

        arr = np.zeros(self.shape_native)
        arr[tuple(np.array(pixel_centres).T)] = 1

        return arr

    @property
    def contour_list(self):
        contour_indices_list = measure.find_contours(np.array(self.contour_array), 0)

        if len(contour_indices_list) == 0:
            return []

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

    @property
    def hull(
        self,
    ):
        if self.grid.shape[0] < 3:
            return None

        grid_convex = np.zeros((len(self.grid), 2))

        grid_convex[:, 0] = self.grid[:, 1]
        grid_convex[:, 1] = self.grid[:, 0]

        try:
            hull = ConvexHull(grid_convex)
        except QhullError:
            return None

        hull_vertices = hull.vertices

        hull_x = grid_convex[hull_vertices, 0]
        hull_y = grid_convex[hull_vertices, 1]

        grid_hull = np.zeros((len(hull_vertices), 2))

        grid_hull[:, 1] = hull_x
        grid_hull[:, 0] = hull_y

        return grid_hull
