from autoarray.structures import abstract_structure
from autoarray.structures.grids.one_d import grid_1d
from autoarray.structures.grids.one_d import grid_1d_util
from autoarray.structures.grids.two_d import grid_2d_irregular

import numpy as np


class AbstractGrid1D(abstract_structure.AbstractStructure1D):
    @property
    def slim(self):
        """
        Convenience method to access the grid's 1D representation, which is a Grid2D stored as an ndarray of shape
        [total_unmasked_pixels*(sub_size**2), 2].

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.
        """
        return self

    @property
    def native(self):
        """
        Convenience method to access the grid's 2D representation, which is a Grid2D stored as an ndarray of shape
        [sub_size*total_y_pixels, sub_size*total_x_pixels, 2] where all masked values are given values (0.0, 0.0).

        If the grid is stored in 2D it is return as is. If it is stored in 1D, it must first be mapped from 1D to 2D.
        """

        grid = grid_1d_util.grid_1d_native_from(
            grid_1d_slim=self, mask_1d=self.mask, sub_size=self.mask.sub_size
        )

        return grid_1d.Grid1D(grid=grid, mask=self.mask)

    @property
    def slim_binned(self):
        """
        Convenience method to access the binned-up grid in its 1D representation, which is a Grid2D stored as an
        ndarray of shape [total_unmasked_pixels, 2].

        The binning up process converts a grid from (y,x) values where each value is a coordinate on the sub-grid to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a grid with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.
        """

        binned_grid_1d_slim = np.multiply(
            self.mask.sub_fraction, self.reshape(-1, self.mask.sub_length).sum(axis=1)
        )

        return grid_1d.Grid1D(grid=binned_grid_1d_slim, mask=self.mask.mask_sub_1)

    @property
    def native_binned(self):
        """
        Convenience method to access the binned-up grid in its 2D representation, which is a Grid2D stored as an
        ndarray of shape [total_y_pixels, total_x_pixels, 2].

        The binning up process conerts a grid from (y,x) values where each value is a coordinate on the sub-grid to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a grid with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the grid is stored in 2D it is return as is. If it is stored in 1D, it must first be mapped from 1D to 2D.
        """

        binned_grid_1d_slim = np.multiply(
            self.mask.sub_fraction, self.reshape(-1, self.mask.sub_length).sum(axis=1)
        )

        binned_grid_1d = grid_1d_util.grid_1d_native_from(
            grid_1d_slim=binned_grid_1d_slim, mask_1d=self.mask, sub_size=1
        )

        return grid_1d.Grid1D(
            grid=binned_grid_1d, mask=self.mask.mask_sub_1, store_slim=False
        )

    def project_to_radial_grid_2d(self, angle=0.0):

        grid = np.zeros((self.sub_shape_native[0], 2))

        grid[:, 1] = self

        return grid_2d_irregular.Grid2DIrregular(grid=grid)
