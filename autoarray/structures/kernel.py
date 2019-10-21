
from astropy import units
import scipy.signal
from skimage.transform import resize, rescale

import numpy as np

from autoarray.structures import grids, arrays
from autoarray import exc

class Kernel(arrays.AbstractArray):

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
    def manual_1d(cls, array, shape_2d, pixel_scales=None, origin=(0.0, 0.0), renormalize=False):

        array = arrays.Array.manual_1d(
            array=array,
            shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin)

        return Kernel(array_1d=array, mask=array.mask, renormalize=renormalize)

    @classmethod
    def manual_2d(cls, array, pixel_scales=None, origin=(0.0, 0.0), renormalize=False):

        array = arrays.Array.manual_2d(
            array=array, pixel_scales=pixel_scales, origin=origin)

        return Kernel(array_1d=array, mask=array.mask, renormalize=renormalize)

    @classmethod
    def full(cls, fill_value, shape_2d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)):

        if sub_size is not None:
            shape_2d = (shape_2d[0] * sub_size, shape_2d[1] * sub_size)

        return cls.manual_2d(array=np.full(fill_value=fill_value, shape=shape_2d), pixel_scales=pixel_scales, origin=origin)

    @classmethod
    def ones(cls, shape_2d, pixel_scales=None, origin=(0.0, 0.0)):
        return cls.full(fill_value=1.0, shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin)

    @classmethod
    def zeros(cls, shape_2d, pixel_scales=None, origin=(0.0, 0.0)):
        return cls.full(fill_value=0.0, shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin)

    @classmethod
    def no_blur(cls, pixel_scales=None):

        array = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        return cls.manual_2d(array=array, pixel_scales=pixel_scales)

    @classmethod
    def from_gaussian(
        cls, shape_2d, pixel_scales, sigma, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0
    ):
        """Simulate the Kernel as an elliptical Gaussian profile."""

        grid = grids.Grid.uniform(shape_2d=shape_2d, pixel_scales=pixel_scales)
        grid_shifted = np.subtract(grid, centre)
        grid_radius = np.sqrt(np.sum(grid_shifted ** 2.0, 1))
        theta_coordinate_to_profile = (
            np.arctan2(grid_shifted[:, 0], grid_shifted[:, 1])
            - np.radians(phi)
        )
        grid_transformed = np.vstack(
            (
                grid_radius * np.sin(theta_coordinate_to_profile),
                grid_radius * np.cos(theta_coordinate_to_profile),
            )
        ).T

        grid_elliptical_radii = np.sqrt(
            np.add(
                np.square(grid_transformed[:, 1]), np.square(np.divide(grid_transformed[:, 0], axis_ratio))
            )
        )

        gaussian = np.multiply(
            np.divide(1.0, sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(grid_elliptical_radii, sigma))),
        )

        return Kernel.manual_1d(
            array=gaussian, shape_2d=shape_2d, pixel_scales=pixel_scales, renormalize=True
        )

    @classmethod
    def from_as_gaussian_via_alma_fits_header_parameters(
        cls, shape_2d, pixel_scales, y_stddev, x_stddev, theta, centre=(0.0, 0.0)
    ):

        x_stddev = (
            x_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        y_stddev = (
            y_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )

        axis_ratio = x_stddev / y_stddev

        return Kernel.from_gaussian(shape_2d=shape_2d, pixel_scales=pixel_scales, sigma=y_stddev, axis_ratio=axis_ratio, phi=90.0 - theta, centre=centre)

    @classmethod
    def from_fits(cls, file_path, hdu, pixel_scales=None, origin=(0.0, 0.0), renormalize=False):
        """
        Loads the Kernel from a .fits file.

        Parameters
        ----------
        pixel_scales
        file_path: String
            The path to the file containing the Kernel
        hdu : int
            The HDU the Kernel is stored in the .fits file.
        """

        array = arrays.Array.from_fits(
            file_path=file_path, hdu=hdu,
            pixel_scales=pixel_scales, origin=origin)

        return Kernel(array_1d=array, mask=array.mask, renormalize=renormalize)

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

        if self.pixel_scales is not None:

            pixel_scale_factors = (
                self.mask.shape[0] / kernel_rescaled.shape[0],
                self.mask.shape[1] / kernel_rescaled.shape[1],
            )

            pixel_scales = (
                self.pixel_scales[0] * pixel_scale_factors[0],
                self.pixel_scales[1] * pixel_scale_factors[1],
            )

        else:

            pixel_scales = None

        return Kernel.manual_2d(
            array=kernel_rescaled, pixel_scales=pixel_scales, renormalize=renormalize
        )

    @property
    def in_2d(self):
        return self.mask.mapping.sub_array_2d_from_sub_array_1d(sub_array_1d=self)

    @property
    def renormalized(self):
        """Renormalize the Kernel such that its data_vector values sum to unity."""
        return Kernel(array_1d=self, mask=self.mask, renormalize=False)

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

    def convolved_array_2d_from_array_2d(self, array_2d):
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

        return scipy.signal.convolve2d(array_2d, self.in_2d, mode="same")

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

        mask_sub_1 = mask.mapping.mask_sub_1

        return mask_sub_1.mapping.array_from_array_2d(
            array_2d=self.convolved_array_2d_from_array_2d(array_2d=array_2d)
        )