import logging

from scipy.stats import norm

import scipy.signal
from astropy import units
from skimage.transform import resize, rescale

import numpy as np

from autoarray.structures import arrays
from autoarray import exc

class Kernel(arrays.Array):

    # noinspection PyUnusedLocal
    def __new__(cls, array_1d, mask, renormalize=False, *args, **kwargs):
        """ A hyper array with square-pixels.

        Parameters
        ----------
        array_1d: ndarray
            An array representing image (e.g. an image, noise-map, etc.)
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The arc-second origin of the hyper array's coordinate system.
        """

        #        obj = arrays.Scaled(array_1d=sub_array_1d, mask=mask)

        obj = super(Kernel, cls).__new__(cls=cls, array_1d=array_1d, mask=mask)

        if renormalize:
            obj[:] = np.divide(obj, np.sum(obj))

        return obj

    @classmethod
    def from_2d_and_pixel_scale(cls, array_2d, pixel_scale, origin=(0.0, 0.0), renormalize=False):

        scaled_array = arrays.Array.from_array_2d_and_pixel_scales(array_2d=array_2d, pixel_scales=(pixel_scale, pixel_scale), origin=origin)

        return cls(array_1d=scaled_array, mask=scaled_array.mask, renormalize=renormalize)

    @classmethod
    def from_array_2d_and_pixel_scales(
        cls, array_2d, pixel_scales, origin=(0.0, 0.0), renormalize=False
    ):

        scaled_array = arrays.Array.from_array_2d_and_pixel_scales(array_2d=array_2d, pixel_scales=pixel_scales, origin=origin)
        return cls(array_1d=scaled_array, mask=scaled_array.mask, renormalize=renormalize)

    @classmethod
    def from_no_blurring_kernel(cls, pixel_scale):

        array = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        return cls.from_2d_and_pixel_scale(array_2d=array, pixel_scale=pixel_scale, renormalize=False)
    #
    # @classmethod
    # def from_gaussian(
    #     cls, shape, pixel_scale, sigma, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0
    # ):
    #     """Simulate the Kernel as an elliptical Gaussian profile."""
    #     from autolens.model.profiles.light_profiles import EllipticalGaussian
    #
    #     gaussian = EllipticalGaussian(
    #         centre=centre, axis_ratio=axis_ratio, phi=phi, intensity=1.0, sigma=sigma
    #     )
    #
    #     grid = arrays.SubGrid.from_shape_pixel_scale_and_sub_size(
    #         shape=shape, pixel_scale=pixel_scale, sub_size=1
    #     )
    #
    #     gaussian = gaussian.profile_image_from_grid(grid=grid)
    #
    #     return Kernel.from_2d_and_pixel_scale(
    #         array_2d=gaussian.in_2d, pixel_scale=pixel_scale, renormalize=True
    #     )
    #
    # @classmethod
    # def from_as_gaussian_via_alma_fits_header_parameters(
    #     cls, shape, pixel_scale, y_stddev, x_stddev, theta, centre=(0.0, 0.0)
    # ):
    #
    #     x_stddev = (
    #         x_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    #     )
    #     y_stddev = (
    #         y_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    #     )
    #
    #     axis_ratio = x_stddev / y_stddev
    #
    #     gaussian = EllipticalGaussian(
    #         centre=centre,
    #         axis_ratio=axis_ratio,
    #         phi=90.0 - theta,
    #         intensity=1.0,
    #         sigma=y_stddev,
    #     )
    #
    #     grid = arrays.SubGrid.from_shape_pixel_scale_and_sub_size(
    #         shape=shape, pixel_scale=pixel_scale, sub_size=1
    #     )
    #
    #     gaussian = gaussian.profile_image_from_grid(grid=grid)
    #
    #     return Kernel.from_2d_and_pixel_scale(
    #         array_2d=gaussian.in_2d, pixel_scale=pixel_scale, renormalize=True
    #     )

    @classmethod
    def from_fits_and_pixel_scale(cls, file_path, hdu, pixel_scale, renormalize=False):
        """
        Loads the Kernel from a .fits file.

        Parameters
        ----------
        pixel_scale
        file_path: String
            The path to the file containing the Kernel
        hdu : int
            The HDU the Kernel is stored in the .fits file.
        """
        return cls.from_2d_and_pixel_scale(
            array_2d=arrays.array_util.numpy_array_2d_from_fits(
                file_path=file_path, hdu=hdu
            ),
            pixel_scale=pixel_scale,
            renormalize=renormalize,
        )

    def rescaled_with_odd_dimensions_from_rescale_factor(
        self, rescale_factor, renormalize=False
    ):

        kernel_rescaled = rescale(
            self.in_2d,
            rescale_factor,
            anti_aliasing=False,
            mode="constant",
            multichannel=False,
        )

        if kernel_rescaled.shape[0] % 2 == 0 and kernel_rescaled.shape[1] % 2 == 0:
            kernel_rescaled = resize(
                kernel_rescaled,
                output_shape=(kernel_rescaled.shape[0] + 1, kernel_rescaled.shape[1] + 1),
                anti_aliasing=False,
                mode="constant",
            )
        elif kernel_rescaled.shape[0] % 2 == 0 and kernel_rescaled.shape[1] % 2 != 0:
            kernel_rescaled = resize(
                kernel_rescaled,
                output_shape=(kernel_rescaled.shape[0] + 1, kernel_rescaled.shape[1]),
                anti_aliasing=False,
                mode="constant",
            )
        elif kernel_rescaled.shape[0] % 2 != 0 and kernel_rescaled.shape[1] % 2 == 0:
            kernel_rescaled = resize(
                kernel_rescaled,
                output_shape=(kernel_rescaled.shape[0], kernel_rescaled.shape[1] + 1),
                anti_aliasing=False,
                mode="constant",
            )

        pixel_scale_factors = (
            self.mask.shape[0] / kernel_rescaled.shape[0],
            self.mask.shape[1] / kernel_rescaled.shape[1],
        )
        pixel_scales = (
            self.mask.geometry.pixel_scales[0] * pixel_scale_factors[0],
            self.mask.geometry.pixel_scales[1] * pixel_scale_factors[1],
        )

        return Kernel.from_array_2d_and_pixel_scales(
            array_2d=kernel_rescaled, pixel_scales=pixel_scales, renormalize=renormalize
        )

    @property
    def renormalized(self):
        """Renormalize the Kernel such that its data_vector values sum to unity."""
        return Kernel(array_1d=self, mask=self.mask, renormalize=True)

    def convolved_array_from_array(self, array):
        """
        Convolve an array with this Kernel

        Parameters
        ----------
        image : ndarray
            An array representing the image the Kernel is convolved with.

        Returns
        -------
        convolved_image : ndarray
            An array representing the image after convolution.

        Raises
        ------
        KernelException if either Kernel psf dimension is odd
        """
        if self.mask.shape[0] % 2 == 0 or self.mask.shape[1] % 2 == 0:
            raise exc.KernelException("Kernel Kernel must be odd")

        return array.mapping.array_from_array_2d(array_2d=scipy.signal.convolve2d(array.in_2d, self.in_2d, mode="same"))

    def convolved_array_from_array_2d_and_mask(self, array_2d, mask):
        """
        Convolve an array with this Kernel

        Parameters
        ----------
        image : ndarray
            An array representing the image the Kernel is convolved with.

        Returns
        -------
        convolved_image : ndarray
            An array representing the image after convolution.

        Raises
        ------
        KernelException if either Kernel psf dimension is odd
        """
        return mask.mapping.array_from_array_2d(
            array_2d=self.convolved_array_from_array(array=array_2d)
        )