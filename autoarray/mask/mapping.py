import logging

import numpy as np

from autoarray import exc
from autoarray.mask import mask, regions, geometry
from autoarray.structures import arrays, grids
from autoarray.util import array_util, grid_util, binning_util


class Mapping(object):

    def __init__(self, mask_2d):
        self.mask_2d = mask_2d

    @property
    def regions(self):
        return regions.Regions(mapping=self)

    @property
    def geometry(self):
        return geometry.Geometry(mapping=self)

    @property
    def mask(self):
        return mask.Mask(array_2d=self.mask_2d)

    def mask_from_array_2d(self, array_2d):
        return mask.Mask(array_2d=array_2d)

    def mask_no_sub_from_array_2d(self, array_2d):
        return mask.Mask(array_2d=array_2d)

    def resized_mask_from_new_shape(
        self, new_shape,
    ):
        """resized the array to a new shape and at a new origin.

        Parameters
        -----------
        new_shape : (int, int)
            The new two-dimensional shape of the array.
        """

        resized_mask_2d = array_util.resized_array_2d_from_array_2d_and_resized_shape(
            array_2d=self.mask_2d, resized_shape=new_shape,
        ).astype("bool")

        return self.mask_from_array_2d(
            array_2d=resized_mask_2d,
        )

    def binned_mask_from_bin_up_factor(self, bin_up_factor):

        binned_up_mask = binning_util.binned_up_mask_from_mask_2d_and_bin_up_factor(
            mask_2d=self.mask_2d, bin_up_factor=bin_up_factor
        )

        return mask.Mask(
            array_2d=binned_up_mask,
        )

    def array_2d_from_array_1d(self, array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return array_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=array_1d, mask=self.mask_2d, sub_size=1
        )

    def array_from_array_1d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original 2D array.

        Values which were masked in the util to the 1D array are returned as zeros.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array which is mapped to its masked 2D array.
        """
        return arrays.Array(array_1d=array_1d, mask=self.mask)

    def array_from_array_2d(self, array_2d):
        """For a 2D array (e.g. an image, noise_map, etc.) map it to a masked 1D array of valuees using this mask.

        Parameters
        ----------
        array_2d : ndarray | None | float
            The 2D array to be mapped to a masked 1D array.
        """
        array_1d = array_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=self.mask_2d, sub_array_2d=array_2d, sub_size=1
        )
        return self.array_from_array_1d(array_1d=array_1d)

    def trimmed_array_2d_from_padded_array_1d_and_image_shape(
        self, padded_array_1d, image_shape
    ):
        """ Map a padded 1D array of values to its original 2D array, trimming all edge values.

        Parameters
        -----------
        padded_array_1d : ndarray
            A 1D array of values which were computed using a padded grid
        """

        padded_array_2d = array_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(sub_array_1d=padded_array_1d, mask=self.mask_2d, sub_size=1)

        pad_size_0 = self.mask_2d.shape[0] - image_shape[0]
        pad_size_1 = self.mask_2d.shape[1] - image_shape[1]
        return padded_array_2d[
            pad_size_0 // 2 : self.mask_2d.shape[0] - pad_size_0 // 2,
            pad_size_1 // 2 : self.mask_2d.shape[1] - pad_size_1 // 2,
        ]

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

        padded_array_2d = array_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=padded_array_1d,
            mask=np.full(fill_value=False, shape=self.mask_2d.shape),
            sub_size=1,
        )

        # noinspection PyUnresolvedReferences
        blurred_padded_array_2d = psf.convolved_array_from_array(
            array=padded_array_2d
        )

        return array_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=blurred_padded_array_2d,
            mask=np.full(self.mask_2d.shape, False),
            sub_size=1,
        )

    def unmasked_blurred_array_2d_from_padded_array_1d_psf_and_image_shape(
        self, padded_array_1d, psf, image_shape
    ):
        """For a padded grid and psf, compute an unmasked blurred image from an unmasked unblurred image.

        This relies on using the lens data's padded-grid, which is a grid of (y,x) coordinates which extends over the \
        entire image as opposed to just the masked region.

        Parameters
        ----------
        psf : abstract_data.PSF
            The PSF of the image used for convolution.
        unmasked_image_1d : ndarray
            The 1D unmasked image which is blurred.
        """

        padded_array_2d = array_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(sub_array_1d=padded_array_1d, mask=self.mask_2d, sub_size=1)

        blurred_image_2d = psf.convolved_array_from_array(
            array=padded_array_2d.in_2d
        )

        blurred_image_1d = array_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(sub_array_1d=blurred_image_2d, mask=self.mask_2d, sub_size=1)

        return self.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=blurred_image_1d, image_shape=image_shape
        )


class ScaledMapping(Mapping):

    def __init__(self, mask_2d, pixel_scales, origin=(0.0, 0.0)):

        super(ScaledMapping, self).__init__(mask_2d=mask_2d)

        if pixel_scales[0] <= 0.0 or pixel_scales[1] <= 0:
            raise exc.MaskException(
                "A pixel scale supplied to a ScaledMask (and therefore the Image) "
                "is zero or negative"
            )

        self.pixel_scales = pixel_scales
        self.origin = origin

    @property
    def geometry(self):
        return geometry.ScaledGeometry(mapping=self)

    @property
    def pixel_scale(self):
        if self.pixel_scales[0] == self.pixel_scales[1]:
            return self.pixel_scales[0]
        else:
            raise exc.ScaledException(
                "Cannot return a pixel_scale for a a grid where each dimension has a "
                "different pixel scale (e.g. pixel_scales[0] != pixel_scales[1]"
            )

    @property
    def mask(self):
        return mask.ScaledMask(mask_2d=self.mask_2d, pixel_scales=self.pixel_scales, origin=self.origin)

    @property
    def mask_no_sub(self):
        return mask.ScaledMask(mask_2d=self.mask_2d, pixel_scales=self.pixel_scales, origin=self.origin)

    def mask_from_array_2d(self, array_2d):
        return mask.ScaledMask(mask_2d=array_2d, pixel_scales=self.pixel_scales, origin=self.origin)

    def mask_no_sub_from_array_2d(self, array_2d):
        return mask.ScaledMask(mask_2d=array_2d, pixel_scales=self.pixel_scales, origin=self.origin)

    def array_from_array_1d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original 2D array.

        Values which were masked in the util to the 1D array are returned as zeros.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array which is mapped to its masked 2D array.
        """
        return arrays.ScaledArray(array_1d=array_1d, mask=self.mask)

    def binned_mask_from_bin_up_factor(self, bin_up_factor):

        binned_up_mask = binning_util.binned_up_mask_from_mask_2d_and_bin_up_factor(
            mask_2d=self.mask_2d, bin_up_factor=bin_up_factor
        )

        return mask.ScaledMask(
            mask_2d=binned_up_mask,
            pixel_scales=(
                self.pixel_scales[0] * bin_up_factor,
                self.pixel_scales[1] * bin_up_factor,
            ),
            origin=self.geometry.origin,
        )

    def grid_from_grid_1d(self, grid_1d):
        """ Map a 1D grid the same dimension as the grid to its original 2D grid.

        Values which were masked in the util to the 1D grid are returned as zeros.

        Parameters
        -----------
        grid_1d : ndgrid
            The 1D grid which is mapped to its masked 2D grid.
        """
        return grids.ScaledGrid(grid_1d=grid_1d, mask=self.mask)

    def grid_from_grid_2d(self, grid_2d):
        """For a 2D grid (e.g. an image, noise_map, etc.) map it to a masked 1D grid of valuees using this mask.

        Parameters
        ----------
        grid_2d : ndgrid | None | float
            The 2D grid to be mapped to a masked 1D grid.
        """
        grid_1d = grid_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            mask=self.mask_2d, sub_grid_2d=grid_2d, sub_size=1
        )
        return self.grid_from_grid_1d(grid_1d=grid_1d)

    def grid_2d_from_grid_1d(self, grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        return grid_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=grid_1d, mask=self.mask_2d, sub_size=1
        )


class ScaledSubMapping(ScaledMapping):

    def __init__(self, mask_2d, sub_size, pixel_scales, origin=(0.0, 0.0)):

        super(ScaledSubMapping, self).__init__(mask_2d=mask_2d, pixel_scales=pixel_scales, origin=origin)
        self.sub_size=sub_size
        self.sub_length = int(self.sub_size ** 2.0)
        self.sub_fraction = 1.0 / self.sub_length

    @property
    def regions(self):
        return regions.SubRegions(mapping=self)

    @property
    def geometry(self):
        return geometry.ScaledSubGeometry(mapping=self)

    @property
    def mask(self):
        return mask.ScaledSubMask(mask_2d=self.mask_2d, sub_size=self.sub_size, pixel_scales=self.pixel_scales, origin=self.origin)

    def mask_from_array_2d(self, array_2d):
        return mask.ScaledSubMask(mask_2d=array_2d, sub_size=self.sub_size, pixel_scales=self.pixel_scales, origin=self.origin)

    def binned_mask_from_bin_up_factor(self, bin_up_factor):

        binned_up_mask = binning_util.binned_up_mask_from_mask_2d_and_bin_up_factor(
            mask_2d=self.mask_2d, bin_up_factor=bin_up_factor
        )

        return mask.ScaledSubMask(
            mask_2d=binned_up_mask,
            pixel_scales=(
                self.pixel_scales[0] * bin_up_factor,
                self.pixel_scales[1] * bin_up_factor,
            ),
            sub_size=self.sub_size,
            origin=self.geometry.origin,
        )

    def array_from_array_1d(self, array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return arrays.ScaledSubArray(array_1d=array_1d, mask=self.mask)

    def sub_array_2d_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return array_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=sub_array_1d, mask=self.mask_2d, sub_size=self.sub_size
        )

    def array_2d_binned_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid to its original masked 2D sub-array and return it as
        a hyper array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub-array of which is mapped to a 2D hyper sub-array the dimensions.
        """
        binned_array_1d = np.multiply(
            self.sub_fraction, sub_array_1d.reshape(-1, self.sub_length).sum(axis=1)
        )
        return array_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=binned_array_1d, mask=self.mask_2d, sub_size=1
        )

    def sub_grid_2d_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        return grid_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=sub_grid_1d, mask=self.mask_2d, sub_size=self.sub_size
        )

    def grid_2d_binned_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid to its original masked 2D sub-grid and return it as
        a hyper grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub-grid of which is mapped to a 2D hyper sub-grid the dimensions.
        """

        grid_1d_y = np.multiply(
            self.sub_fraction, sub_grid_1d[:, 0].reshape(-1, self.sub_length).sum(axis=1)
        )

        grid_1d_x = np.multiply(
            self.sub_fraction, sub_grid_1d[:, 1].reshape(-1, self.sub_length).sum(axis=1)
        )

        binned_grid_1d = np.stack((grid_1d_y, grid_1d_x), axis=-1)

        return grid_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=binned_grid_1d, mask=self.mask_2d, sub_size=1
        )

    def array_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return arrays.ScaledSubArray(array_1d=sub_array_1d, mask=self.mask)

    def array_from_sub_array_2d(self, sub_array_2d):
        """ Map a 2D sub-array to its masked 1D sub-array.

        Values which are masked in the util to the 1D array are returned as zeros.

        Parameters
        -----------
        su_array_2d : ndarray
            The 2D sub-array which is mapped to its masked 1D sub-array.
        """
        sub_array_1d = array_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=sub_array_2d, mask=self.mask_2d, sub_size=self.sub_size
        )
        return self.array_from_sub_array_1d(sub_array_1d=sub_array_1d)

    def array_binned_from_sub_array_1d(self, sub_array_1d):
        """For an input 1D sub-array, map its values to a 1D array of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_array_1d : ndarray
            A 1D sub-array of values (e.g. image, convergence, potential) which is mapped to
            a 1d array.
        """
        binned_array_1d = np.multiply(
            self.sub_fraction, sub_array_1d.reshape(-1, self.sub_length).sum(axis=1)
        )
        return arrays.ScaledArray(array_1d=binned_array_1d, mask=self.mask_no_sub)

    def grid_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        return grids.ScaledSubGrid(grid_1d=sub_grid_1d, mask=self.mask)

    def grid_from_sub_grid_2d(self, sub_grid_2d):
        """ Map a 2D sub-grid to its masked 1D sub-grid.

        Values which are masked in the util to the 1D grid are returned as zeros.

        Parameters
        -----------
        su_grid_2d : ndgrid
            The 2D sub-grid which is mapped to its masked 1D sub-grid.
        """
        sub_grid_1d = grid_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=sub_grid_2d, mask=self.mask_2d, sub_size=self.sub_size
        )
        return self.grid_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

    def grid_binned_from_sub_grid_1d(self, sub_grid_1d):
        """For an input 1D sub-grid, map its values to a 1D grid of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            A 1D sub-grid of values (e.g. image, convergence, potential) which is mapped to
            a 1d grid.
        """

        grid_1d_y = self.array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 0]
        )

        grid_1d_x = self.array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 1]
        )

        return grids.ScaledGrid(
            grid_1d=np.stack((grid_1d_y, grid_1d_x), axis=-1), mask=self.mask_no_sub
        )