import numpy as np
from typing import Union

from autoarray.structures.abstract_structure import Structure1D
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular

from autoarray.geometry import geometry_util
from autoarray.structures.grids.one_d import grid_1d_util


class AbstractGrid1D(Structure1D):
    @property
    def slim(self) -> Union["AbstractGrid1D", "Grid1D"]:
        """
        Return a `Grid1D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels * sub_size, 2].

        If it is already stored in its `slim` representation  the `Grid1D` is returned as it is. If not, it is
        mapped from  `native` to `slim` and returned as a new `Grid1D`.
        """
        from autoarray.structures.grids.one_d.grid_1d import Grid1D

        if self.shape[0] != self.mask.sub_shape_native[0]:
            return self

        grid = grid_1d_util.grid_1d_slim_from(
            grid_1d_native=self, mask_1d=self.mask, sub_size=self.mask.sub_size
        )

        return Grid1D(grid=grid, mask=self.mask)

    @property
    def native(self) -> Union["AbstractGrid1D", "Grid1D"]:
        """
        Return a `Grid1D` where the data is stored in its `native` representation, which is an ndarray of shape
        [sub_size*total_x_pixels, 2].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Grid1D`.
        """
        from autoarray.structures.grids.one_d.grid_1d import Grid1D

        if self.shape[0] == self.mask.sub_shape_native[0]:
            return self

        grid = grid_1d_util.grid_1d_native_from(
            grid_1d_slim=self, mask_1d=self.mask, sub_size=self.mask.sub_size
        )

        return Grid1D(grid=grid, mask=self.mask)

    @property
    def binned(self) -> "Grid1D":
        """
        Convenience method to access the binned-up grid in its 1D representation, which is a Grid2D stored as an
        ndarray of shape [total_unmasked_pixels, 2].

        The binning up process converts a grid from (y,x) values where each value is a coordinate on the sub-grid to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a grid with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.
        """
        from autoarray.structures.grids.one_d.grid_1d import Grid1D

        grid_1d_slim = self.slim

        binned_grid_1d_slim = np.multiply(
            self.mask.sub_fraction,
            grid_1d_slim.reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        return Grid1D(grid=binned_grid_1d_slim, mask=self.mask.mask_sub_1)

    def project_to_radial_grid_2d(self, angle: float = 0.0) -> Grid2DIrregular:
        """
        Project the 1D grid of (y,x) coordinates to an irregular 2d grid of (y,x) coordinates. The projection works
        as follows:

        1) Map the 1D (x) coordinates to 2D along the x-axis, such that the x value of every 2D coordinate is the
        corresponding (x) value in the 1D grid, and every y value is 0.0.

        2) Rotate this projected 2D grid clockwise by the input angle.

        Parameters
        ----------
        angle
            The angle with which the project 2D grid of coordinates is rotated clockwise.

        Returns
        -------
        Grid2DIrregular
            The projected and rotated 2D grid of (y,x) coordinates.
        """
        grid = np.zeros((self.sub_shape_slim, 2))

        grid[:, 1] = self.slim

        grid = geometry_util.transform_grid_2d_to_reference_frame(
            grid_2d=grid, centre=(0.0, 0.0), angle=angle
        )

        return Grid2DIrregular(grid=grid)
