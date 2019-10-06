import logging

import numpy as np

from autoarray import exc
from autoarray.structures import arrays, grids
from autoarray.util import binning_util
from autoarray.util import array_util, grid_util, mask_util

class Mapping(object):
    
    def __init__(self, mask):
        
        self.mask = mask

    def sub_array_2d_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return array_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=sub_array_1d, mask=self.mask, sub_size=self.mask.sub_size
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
            self.mask.sub_fraction, sub_array_1d.reshape(-1, self.mask.sub_length).sum(axis=1)
        )
        return array_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=binned_array_1d, mask=self.mask, sub_size=1
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
            sub_grid_1d=sub_grid_1d, mask=self.mask, sub_size=self.mask.sub_size
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
            self.mask.sub_fraction, sub_grid_1d[:,0].reshape(-1, self.mask.sub_length).sum(axis=1)
        )

        grid_1d_x = np.multiply(
            self.mask.sub_fraction, sub_grid_1d[:,1].reshape(-1, self.mask.sub_length).sum(axis=1)
        )

        binned_grid_1d = np.stack((grid_1d_y, grid_1d_x), axis=-1)

        return grid_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=binned_grid_1d, mask=self.mask, sub_size=1
        )

    def trimmed_array_2d_from_padded_array_1d_and_image_shape(
        self, padded_array_1d, image_shape
    ):
        """ Map a padded 1D array of values to its original 2D array, trimming all edge values.

        Parameters
        -----------
        padded_array_1d : ndarray
            A 1D array of values which were computed using a padded grid
        """

        padded_array_2d = array_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(sub_array_1d=padded_array_1d, mask=self.mask, sub_size=1)

        pad_size_0 = self.mask.shape[0] - image_shape[0]
        pad_size_1 = self.mask.shape[1] - image_shape[1]
        return padded_array_2d[
            pad_size_0 // 2 : self.mask.shape[0] - pad_size_0 // 2,
            pad_size_1 // 2 : self.mask.shape[1] - pad_size_1 // 2,
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
            mask=np.full(fill_value=False, shape=self.mask.shape),
            sub_size=1,
        )

        # noinspection PyUnresolvedReferences
        blurred_padded_array_2d = psf.convolved_array_2d_from_array_2d(
            array_2d=padded_array_2d
        )

        return array_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=blurred_padded_array_2d,
            mask=np.full(self.mask.shape, False),
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

        padded_array_2d = array_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(sub_array_1d=padded_array_1d, mask=self.mask, sub_size=1)

        blurred_image_2d = psf.convolved_array_2d_from_array_2d(
            array_2d=padded_array_2d.in_2d
        )

        blurred_image_1d = array_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(sub_array_1d=blurred_image_2d, mask=self.mask, sub_size=1)

        return self.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=blurred_image_1d, image_shape=image_shape
        )


class ScaledMapping(Mapping):
    
    def __init__(self, mask):
        
        super(ScaledMapping, self).__init__(mask=mask)

    def scaled_array_from_array_1d(self, array_1d):
        """ Map a 1D array the same dimension as the grid to its original 2D array.

        Values which were masked in the util to the 1D array are returned as zeros.

        Parameters
        -----------
        array_1d : ndarray
            The 1D array which is mapped to its masked 2D array.
        """
        mask = self.mask.new_mask_with_new_sub_size(sub_size=1)
        return arrays.ScaledArray(sub_array_1d=array_1d, mask=mask)

    def scaled_array_from_array_2d(self, array_2d):
        """For a 2D array (e.g. an image, noise_map, etc.) map it to a masked 1D array of valuees using this mask.

        Parameters
        ----------
        array_2d : ndarray | None | float
            The 2D array to be mapped to a masked 1D array.
        """
        array_1d = array_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=self.mask, sub_array_2d=array_2d, sub_size=1
        )
        return self.scaled_array_from_array_1d(array_1d=array_1d)

    def scaled_array_from_sub_array_1d(self, sub_array_1d):
        """ Map a 1D sub-array the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub array.

        Parameters
        -----------
        sub_array_1d : ndarray
            The 1D sub_array which is mapped to its masked 2D sub-array.
        """
        return arrays.ScaledArray(sub_array_1d=sub_array_1d, mask=self.mask)

    def scaled_array_from_sub_array_2d(self, sub_array_2d):
        """ Map a 2D sub-array to its masked 1D sub-array.

        Values which are masked in the util to the 1D array are returned as zeros.

        Parameters
        -----------
        su_array_2d : ndarray
            The 2D sub-array which is mapped to its masked 1D sub-array.
        """
        sub_array_1d = array_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            sub_array_2d=sub_array_2d, mask=self.mask, sub_size=self.mask.sub_size
        )
        return self.scaled_array_from_sub_array_1d(sub_array_1d=sub_array_1d)

    def scaled_array_binned_from_sub_array_1d(self, sub_array_1d):
        """For an input 1D sub-array, map its values to a 1D array of values by summing each set \of sub-pixel \
        values and dividing by the total number of sub-pixels.

        Parameters
        -----------
        sub_array_1d : ndarray
            A 1D sub-array of values (e.g. image, convergence, potential) which is mapped to
            a 1d array.
        """
        binned_array_1d = np.multiply(
            self.mask.sub_fraction, sub_array_1d.reshape(-1, self.mask.sub_length).sum(axis=1)
        )

        mask = self.mask.new_mask_with_new_sub_size(sub_size=1)
        return arrays.ScaledArray(sub_array_1d=binned_array_1d, mask=mask)

    def grid_from_grid_1d(self, grid_1d):
        """ Map a 1D grid the same dimension as the grid to its original 2D grid.

        Values which were masked in the util to the 1D grid are returned as zeros.

        Parameters
        -----------
        grid_1d : ndgrid
            The 1D grid which is mapped to its masked 2D grid.
        """
        mask = self.mask.new_mask_with_new_sub_size(sub_size=1)
        return grids.Grid(sub_grid_1d=grid_1d, mask=mask)

    def grid_from_grid_2d(self, grid_2d):
        """For a 2D grid (e.g. an image, noise_map, etc.) map it to a masked 1D grid of valuees using this mask.

        Parameters
        ----------
        grid_2d : ndgrid | None | float
            The 2D grid to be mapped to a masked 1D grid.
        """
        grid_1d = grid_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            mask=self.mask, sub_grid_2d=grid_2d, sub_size=1
        )
        return self.grid_from_grid_1d(grid_1d=grid_1d)

    def grid_from_sub_grid_1d(self, sub_grid_1d):
        """ Map a 1D sub-grid the same dimension as the sub-grid (e.g. including sub-pixels) to its original masked
        2D sub grid.

        Parameters
        -----------
        sub_grid_1d : ndgrid
            The 1D sub_grid which is mapped to its masked 2D sub-grid.
        """
        return grids.Grid(sub_grid_1d=sub_grid_1d, mask=self.mask)

    def grid_from_sub_grid_2d(self, sub_grid_2d):
        """ Map a 2D sub-grid to its masked 1D sub-grid.

        Values which are masked in the util to the 1D grid are returned as zeros.

        Parameters
        -----------
        su_grid_2d : ndgrid
            The 2D sub-grid which is mapped to its masked 1D sub-grid.
        """
        sub_grid_1d = grid_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=sub_grid_2d, mask=self.mask, sub_size=self.mask.sub_size
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

        grid_1d_y = self.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 0]
        )

        grid_1d_x = self.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_grid_1d[:, 1]
        )

        mask = self.mask.new_mask_with_new_sub_size(sub_size=1)
        return grids.Grid(
            sub_grid_1d=np.stack((grid_1d_y, grid_1d_x), axis=-1), mask=mask
        )

    def grid_pixels_from_grid_arcsec(self, grid_arcsec_1d):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
        returned as floats such that they include the decimal offset from each pixel's top-left corner.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        highest y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_arcsec_1d: ndarray
            A grid of (y,x) coordinates in arc seconds.
        """
        grid_pixels_1d = grid_util.grid_pixels_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec_1d,
            shape=self.mask.shape,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )
        return grid_arcsec_1d.mask.mapping.grid_from_grid_1d(grid_1d=grid_pixels_1d)

    def grid_pixel_centres_from_grid_arcsec_1d(self, grid_arcsec_1d):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel values. Pixel coordinates are \
        returned as integers such that they map directly to the pixel they are contained within.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_arcsec_1d: ndarray
            The grid of (y,x) coordinates in arc seconds.
        """
        grid_pixel_centres_1d = grid_util.grid_pixel_centres_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec_1d,
            shape=self.mask.shape,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        ).astype(
            "int"
        )
        return grid_arcsec_1d.mask.mapping.grid_from_grid_1d(grid_1d=grid_pixel_centres_1d)

    def grid_pixel_indexes_from_grid_arcsec_1d(self, grid_arcsec_1d):
        """Convert a grid of (y,x) arc second coordinates to a grid of (y,x) pixel 1D indexes. Pixel coordinates are \
        returned as integers such that they are the pixel from the top-left of the 2D grid going rights and then \
        downwards.

        For example:

        The pixel at the top-left, whose 2D index is [0,0], corresponds to 1D index 0.
        The fifth pixel on the top row, whose 2D index is [0,5], corresponds to 1D index 4.
        The first pixel on the second row, whose 2D index is [0,1], has 1D index 10 if a row has 10 pixels.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_arcsec_1d: ndarray
            The grid of (y,x) coordinates in arc seconds.
        """
        grid_pixel_indexes_1d = grid_util.grid_pixel_indexes_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec_1d,
            shape=self.mask.shape,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        ).astype(
            "int"
        )
        return grid_arcsec_1d.mask.mapping.grid_from_grid_1d(grid_1d=grid_pixel_indexes_1d)

    def grid_arcsec_from_grid_pixels_1d(self, grid_pixels_1d):
        """Convert a grid of (y,x) pixel coordinates to a grid of (y,x) arc second values.

        The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to \
        higher y arc-second coordinate value and lowest x arc-second coordinate.

        The arc-second coordinate origin is defined by the class attribute origin, and coordinates are shifted to this \
        origin before computing their 1D grid pixel indexes.

        Parameters
        ----------
        grid_pixels_1d : ndarray
            The grid of (y,x) coordinates in pixels.
        """
        grid_arcsec_1d = grid_util.grid_arcsec_1d_from_grid_pixels_1d_shape_and_pixel_scales(
            grid_pixels_1d=grid_pixels_1d,
            shape=self.mask.shape,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )
        return grid_pixels_1d.mask.mapping.grid_from_grid_1d(grid_1d=grid_arcsec_1d)

    def grid_arcsec_from_grid_pixels_1d_for_marching_squares(
        self, grid_pixels_1d, shape
    ):

        grid_arcsec_1d = grid_util.grid_arcsec_1d_from_grid_pixels_1d_shape_and_pixel_scales(
            grid_pixels_1d=grid_pixels_1d,
            shape=shape,
            pixel_scales=(
                self.mask.pixel_scales[0] / self.sub_size,
                self.mask.pixel_scales[1] / self.sub_size,
            ),
            origin=self.mask.origin,
        )

        grid_arcsec_1d[:, 0] -= self.mask.pixel_scales[0] / (2.0 * self.sub_size)
        grid_arcsec_1d[:, 1] += self.mask.pixel_scales[1] / (2.0 * self.sub_size)

        return grid_arcsec_1d