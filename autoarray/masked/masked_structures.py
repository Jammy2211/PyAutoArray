import logging

import numpy as np

from autoarray import exc
from autoarray.structures.arrays import AbstractArray
from autoarray.structures.grids import AbstractGrid
from autoarray.util import array_util, grid_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class MaskedArray(AbstractArray):
    @classmethod
    def manual_1d(cls, array, mask):

        if type(array) is list:
            array = np.asarray(array)

        if array.shape[0] != mask.sub_pixels_in_mask:
            raise exc.ArrayException(
                "The input 1D array does not have the same number of entries as sub-pixels in"
                "the mask."
            )

        return mask.mapping.array_from_sub_array_1d(sub_array_1d=array)

    @classmethod
    def manual_2d(cls, array, mask):

        if type(array) is list:
            array = np.asarray(array)

        if array.shape != mask.sub_shape_2d:
            raise exc.ArrayException(
                "The input array is 2D but not the same dimensions as the sub-mask "
                "(e.g. the mask 2D shape multipled by its sub size."
            )

        return mask.mapping.array_from_sub_array_2d(sub_array_2d=array)

    @classmethod
    def full(cls, fill_value, mask):
        return cls.manual_2d(
            array=np.full(fill_value=fill_value, shape=mask.sub_shape_2d), mask=mask
        )

    @classmethod
    def ones(cls, mask):
        return cls.full(fill_value=1.0, mask=mask)

    @classmethod
    def zeros(cls, mask):
        return cls.full(fill_value=0.0, mask=mask)

    @classmethod
    def from_fits(cls, file_path, hdu, mask):
        array_2d = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)
        return cls.manual_2d(array=array_2d, mask=mask)


class MaskedGrid(AbstractGrid):
    @classmethod
    def manual_1d(cls, grid, mask):

        if type(grid) is list:
            grid = np.asarray(grid)

        if grid.shape[0] != mask.sub_pixels_in_mask:
            raise exc.GridException(
                "The input 1D grid does not have the same number of entries as sub-pixels in"
                "the mask."
            )

        return mask.mapping.grid_from_sub_grid_1d(sub_grid_1d=grid)

    @classmethod
    def manual_2d(cls, grid, mask):

        if type(grid) is list:
            grid = np.asarray(grid)

        if (grid.shape[0], grid.shape[1]) != mask.sub_shape_2d:
            raise exc.GridException(
                "The input grid is 2D but not the same dimensions as the sub-mask "
                "(e.g. the mask 2D shape multipled by its sub size."
            )

        return mask.mapping.grid_from_sub_grid_2d(sub_grid_2d=grid)

    @classmethod
    def from_mask(cls, mask):
        """Setup a sub-grid of the unmasked pixels, using a mask and a specified sub-grid size. The center of \
        every unmasked pixel's sub-pixels give the grid's (y,x) arc-second coordinates.

        Parameters
        -----------
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        """

        sub_grid_1d = grid_util.grid_1d_via_mask_2d(
            mask_2d=mask,
            pixel_scales=mask.pixel_scales,
            sub_size=mask.sub_size,
            origin=mask.origin,
        )

        return mask.mapping.grid_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)
