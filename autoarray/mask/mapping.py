import numpy as np

from autoarray import exc
from autoarray.mask import mask as msk
from autoarray.structures import grids, arrays
from autoarray.util import binning_util, array_util, grid_util, mask_util


class Mapping:
    def __init__(self, mask):

        self.mask = mask

    def array_stored_1d_from_array_1d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original 2D array.

        Values which were masked in the util to the 1D array are returned as zeros.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array which is mapped to its masked 2D array.
        """

        if array_1d.shape[0] != self.mask.pixels_in_mask:
            raise exc.MappingException(
                "The number of pixels in array_1d is not equal to the number of pixels in"
                "the mask."
            )

        return arrays.Array(array=array_1d, mask=self.mask_sub_1, store_in_1d=True)

    def array_stored_1d_from_array_2d(self, array_2d):
        """For a 2D array (e.g. an image, noise_map, etc.) map it to a masked 1D array of valuees using this mask.

        Parameters
        ----------
        array_2d : ndarray | None | float
            The 2D array to be mapped to a masked 1D array.
        """

        if array_2d.shape != self.mask.shape_2d:
            raise exc.MappingException(
                "The number of pixels in array_1d is not equal to the number of pixels in"
                "the mask."
            )

        array_1d = array_util.sub_array_1d_from(
            mask=self.mask, sub_array_2d=array_2d, sub_size=1
        )
        return self.array_stored_1d_from_array_1d(array_1d=array_1d)

    def array_stored_1d_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return arrays.Array(array=sub_array_1d, mask=self.mask, store_in_1d=True)

    def array_stored_1d_from_sub_array_2d(self, sub_array_2d):
        """ Map a 2D sub-array to its masked 1D sub-array.

        Values which are masked in the util to the 1D array are returned as zeros.

        Parameters
        -----------
        su_array_2d : ndarray
            The 2D sub-array which is mapped to its masked 1D sub-array.
        """
        sub_array_1d = array_util.sub_array_1d_from(
            sub_array_2d=sub_array_2d, mask=self.mask, sub_size=self.mask.sub_size
        )
        return self.array_stored_1d_from_sub_array_1d(sub_array_1d=sub_array_1d)

    def array_stored_1d_binned_from_sub_array_1d(self, sub_array_1d):
        """For an input 1D sub-array, map its values to a 1D array of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_array_1d : ndarray
            A 1D sub-array of values (e.g. image, convergence, potential) which is mapped to
            a 1d array.
        """
        binned_array_1d = np.multiply(
            self.mask.sub_fraction,
            sub_array_1d.reshape(-1, self.mask.sub_length).sum(axis=1),
        )
        return arrays.Array(
            array=binned_array_1d, mask=self.mask_sub_1, store_in_1d=True
        )

    def array_stored_2d_from_array_1d(self, array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """

        if array_1d.shape[0] != self.mask.pixels_in_mask:
            raise exc.MappingException(
                "The number of pixels in array_1d is not equal to the number of pixels in"
                "the mask."
            )

        array_2d = array_util.sub_array_2d_from(
            sub_array_1d=array_1d, mask=self.mask, sub_size=1
        )
        return arrays.Array(array=array_2d, mask=self.mask, store_in_1d=False)

    def array_stored_2d_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        sub_array_2d = array_util.sub_array_2d_from(
            sub_array_1d=sub_array_1d, mask=self.mask, sub_size=self.mask.sub_size
        )
        return self.array_stored_2d_from_sub_array_2d(sub_array_2d=sub_array_2d)

    def array_stored_2d_from_sub_array_2d(self, sub_array_2d):
        """ Map a 2D sub-array to its masked 1D sub-array.

        Values which are masked in the util to the 1D array are returned as zeros.

        Parameters
        -----------
        su_array_2d : ndarray
            The 2D sub-array which is mapped to its masked 1D sub-array.
        """
        return arrays.Array(array=sub_array_2d, mask=self.mask, store_in_1d=False)

    def array_stored_2d_binned_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid to its original masked 2D sub-array and return it as
        a hyper array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub-array of which is mapped to a 2D hyper sub-array the dimensions.
        """
        binned_array_1d = np.multiply(
            self.mask.sub_fraction,
            sub_array_1d.reshape(-1, self.mask.sub_length).sum(axis=1),
        )
        binned_array_2d = array_util.sub_array_2d_from(
            sub_array_1d=binned_array_1d, mask=self.mask, sub_size=1
        )
        return arrays.Array(
            array=binned_array_2d, mask=self.mask_sub_1, store_in_1d=False
        )

    def grid_stored_1d_from_grid_1d(self, grid_1d):
        """ Map a 1D grid the same dimension as the grid to its original 2D grid.

        Values which were masked in the util to the 1D grid are returned as zeros.

        Parameters
        -----------
        grid_1d : ndgrid
            The 1D grid which is mapped to its masked 2D grid.
        """
        return grids.Grid(grid=grid_1d, mask=self.mask_sub_1, store_in_1d=True)

    def grid_stored_1d_from_grid_2d(self, grid_2d):
        """For a 2D grid (e.g. an image, noise_map, etc.) map it to a masked 1D grid of valuees using this mask.

        Parameters
        ----------
        grid_2d : ndgrid | None | float
            The 2D grid to be mapped to a masked 1D grid.
        """
        grid_1d = grid_util.sub_grid_1d_from(
            mask=self.mask, sub_grid_2d=grid_2d, sub_size=1
        )
        return self.grid_stored_1d_from_grid_1d(grid_1d=grid_1d)

    def grid_stored_1d_from_sub_grid_1d(self, sub_grid_1d, is_transformed=False):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        if not is_transformed:
            return grids.Grid(grid=sub_grid_1d, mask=self.mask, store_in_1d=True)
        else:
            return grids.GridTransformed(
                grid=sub_grid_1d, mask=self.mask, store_in_1d=True
            )

    def grid_stored_1d_from_sub_grid_2d(self, sub_grid_2d):
        """ Map a 2D sub-grid to its masked 1D sub-grid.

        Values which are masked in the util to the 1D grid are returned as zeros.

        Parameters
        -----------
        su_grid_2d : ndgrid
            The 2D sub-grid which is mapped to its masked 1D sub-grid.
        """
        sub_grid_1d = grid_util.sub_grid_1d_from(
            sub_grid_2d=sub_grid_2d, mask=self.mask, sub_size=self.mask.sub_size
        )
        return self.grid_stored_1d_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

    def grid_stored_1d_binned_from_sub_grid_1d(self, sub_grid_1d):
        """For an input 1D sub-grid, map its values to a 1D grid of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            A 1D sub-grid of values (e.g. image, convergence, potential) which is mapped to
            a 1d grid.
        """

        grid_1d_y = self.array_stored_1d_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 0]
        )

        grid_1d_x = self.array_stored_1d_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 1]
        )

        return grids.Grid(
            grid=np.stack((grid_1d_y, grid_1d_x), axis=-1),
            mask=self.mask_sub_1,
            store_in_1d=True,
        )

    def grid_stored_2d_from_grid_1d(self, grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        grid_2d = grid_util.sub_grid_2d_from(
            sub_grid_1d=grid_1d, mask=self.mask, sub_size=1
        )
        return grids.Grid(grid=grid_2d, mask=self.mask_sub_1, store_in_1d=False)

    def grid_stored_2d_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        sub_grid_2d = grid_util.sub_grid_2d_from(
            sub_grid_1d=sub_grid_1d, mask=self.mask, sub_size=self.mask.sub_size
        )
        return grids.Grid(grid=sub_grid_2d, mask=self.mask, store_in_1d=False)

    def grid_stored_2d_binned_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid to its original masked 2D sub-grid and return it as
        a hyper grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub-grid of which is mapped to a 2D hyper sub-grid the dimensions.
        """

        grid_1d_y = np.multiply(
            self.mask.sub_fraction,
            sub_grid_1d[:, 0].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        grid_1d_x = np.multiply(
            self.mask.sub_fraction,
            sub_grid_1d[:, 1].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        binned_grid_1d = np.stack((grid_1d_y, grid_1d_x), axis=-1)

        binned_grid_2d = grid_util.sub_grid_2d_from(
            sub_grid_1d=binned_grid_1d, mask=self.mask, sub_size=1
        )
        return grids.Grid(grid=binned_grid_2d, mask=self.mask_sub_1, store_in_1d=False)
