import numpy as np

from autoarray import exc
from autoarray.mask import mask as msk
from autoarray.structures import grids, arrays
from autoarray.util import binning_util, array_util, grid_util, mask_util


class Mapping:
    def __init__(self, mask):

        self.mask = mask

    def resized_mask_from_new_shape(self, new_shape):
        """resized the array to a new shape and at a new origin.

        Parameters
        -----------
        new_shape : (int, int)
            The new two-dimensional shape of the array.
        """

        resized_mask_2d = array_util.resized_array_2d_from_array_2d(
            array_2d=self.mask, resized_shape=new_shape
        ).astype("bool")

        return msk.Mask(
            mask_2d=resized_mask_2d,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
            origin=self.mask.origin,
        )

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

        array_1d = array_util.sub_array_1d_from_sub_array_2d(
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
        sub_array_1d = array_util.sub_array_1d_from_sub_array_2d(
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

        array_2d = array_util.sub_array_2d_from_sub_array_1d(
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
        sub_array_2d = array_util.sub_array_2d_from_sub_array_1d(
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
        binned_array_2d = array_util.sub_array_2d_from_sub_array_1d(
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
        grid_1d = grid_util.sub_grid_1d_from_sub_grid_2d(
            mask_2d=self.mask, sub_grid_2d=grid_2d, sub_size=1
        )
        return self.grid_stored_1d_from_grid_1d(grid_1d=grid_1d)

    def grid_stored_1d_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        return grids.Grid(grid=sub_grid_1d, mask=self.mask, store_in_1d=True)

    def grid_stored_1d_from_sub_grid_2d(self, sub_grid_2d):
        """ Map a 2D sub-grid to its masked 1D sub-grid.

        Values which are masked in the util to the 1D grid are returned as zeros.

        Parameters
        -----------
        su_grid_2d : ndgrid
            The 2D sub-grid which is mapped to its masked 1D sub-grid.
        """
        sub_grid_1d = grid_util.sub_grid_1d_from_sub_grid_2d(
            sub_grid_2d=sub_grid_2d, mask_2d=self.mask, sub_size=self.mask.sub_size
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
        grid_2d = grid_util.sub_grid_2d_from_sub_grid_1d(
            sub_grid_1d=grid_1d, mask_2d=self.mask, sub_size=1
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
        sub_grid_2d = grid_util.sub_grid_2d_from_sub_grid_1d(
            sub_grid_1d=sub_grid_1d, mask_2d=self.mask, sub_size=self.mask.sub_size
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

        binned_grid_2d = grid_util.sub_grid_2d_from_sub_grid_1d(
            sub_grid_1d=binned_grid_1d, mask_2d=self.mask, sub_size=1
        )
        return grids.Grid(grid=binned_grid_2d, mask=self.mask_sub_1, store_in_1d=False)

    def trimmed_array_from_padded_array_and_image_shape(
        self, padded_array, image_shape
    ):
        """ Map a padded 1D array of values to its original 2D array, trimming all edge values.

        Parameters
        -----------
        padded_array : ndarray
            A 1D array of values which were computed using a padded grid
        """

        pad_size_0 = self.mask.shape[0] - image_shape[0]
        pad_size_1 = self.mask.shape[1] - image_shape[1]
        trimmed_array = padded_array.in_2d_binned[
            pad_size_0 // 2 : self.mask.shape[0] - pad_size_0 // 2,
            pad_size_1 // 2 : self.mask.shape[1] - pad_size_1 // 2,
        ]
        return arrays.Array.manual_2d(
            array=trimmed_array,
            pixel_scales=self.mask.pixel_scales,
            sub_size=1,
            origin=self.mask.origin,
        )

    def convolve_padded_array_1d_with_psf(self, padded_array_1d, psf):
        """Convolve a 1d padded array of values (e.g. image before PSF blurring) with a PSF, and then trim \
        the convolved array to its original 2D shape.

        Parameters
        -----------
        padded_array_1d: ndarray
            A 1D array of values which were computed using the a padded grid.
        psf : ndarray
            An array describing the PSF kernel of the image.
        """

        padded_array_2d = array_util.sub_array_2d_from_sub_array_1d(
            sub_array_1d=padded_array_1d,
            mask=np.full(fill_value=False, shape=self.mask.shape),
            sub_size=1,
        )

        # noinspection PyUnresolvedReferences
        blurred_padded_array_2d = psf.convolved_array_from_array(array=padded_array_2d)

        return array_util.sub_array_1d_from_sub_array_2d(
            sub_array_2d=blurred_padded_array_2d,
            mask=np.full(self.mask.shape, False),
            sub_size=1,
        )

    def unmasked_blurred_array_from_padded_array_psf_and_image_shape(
        self, padded_array, psf, image_shape
    ):
        """For a padded grid and psf, compute an unmasked blurred image from an unmasked unblurred image.

        This relies on using the lens dataset's padded-grid, which is a grid of (y,x) coordinates which extends over the \
        entire image as opposed to just the masked region.

        Parameters
        ----------
        psf : aa.Kernel
            The PSF of the image used for convolution.
        unmasked_image_1d : ndarray
            The 1D unmasked image which is blurred.
        """

        blurred_image = psf.convolved_array_from_array(array=padded_array)

        return self.trimmed_array_from_padded_array_and_image_shape(
            padded_array=blurred_image, image_shape=image_shape
        )

    def rescaled_mask_from_rescale_factor(self, rescale_factor):
        rescaled_mask = mask_util.rescaled_mask_2d_from_mask_2d_and_rescale_factor(
            mask_2d=self.mask, rescale_factor=rescale_factor
        )
        return msk.Mask(
            mask_2d=rescaled_mask,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
            origin=self.mask.origin,
        )

    @property
    def edge_buffed_mask(self):
        edge_buffed_mask = mask_util.buffed_mask_2d_from_mask_2d(
            mask_2d=self.mask
        ).astype("bool")
        return msk.Mask(
            mask_2d=edge_buffed_mask,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
            origin=self.mask.origin,
        )

    @property
    def mask_sub_1(self):
        return msk.Mask(
            mask_2d=self.mask,
            sub_size=1,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    def mask_new_sub_size_from_mask(self, mask, sub_size=1):
        return msk.Mask(
            mask_2d=mask,
            sub_size=sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    def binned_pixel_scales_from_bin_up_factor(self, bin_up_factor):
        if self.mask.pixel_scales is not None:
            return (
                self.mask.pixel_scales[0] * bin_up_factor,
                self.mask.pixel_scales[1] * bin_up_factor,
            )
        else:
            return None

    def binned_mask_from_bin_up_factor(self, bin_up_factor):

        binned_up_mask = binning_util.bin_mask_2d(
            mask_2d=self.mask, bin_up_factor=bin_up_factor
        )

        return msk.Mask(
            mask_2d=binned_up_mask,
            pixel_scales=self.binned_pixel_scales_from_bin_up_factor(
                bin_up_factor=bin_up_factor
            ),
            sub_size=self.mask.sub_size,
            origin=self.mask.origin,
        )
