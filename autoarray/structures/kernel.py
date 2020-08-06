from astropy import units
import scipy.signal
from skimage.transform import resize, rescale

import numpy as np

from autoarray.structures import grids, arrays
from autoarray.util import array_util
from autoarray import exc


class Kernel(arrays.Array):

    # noinspection PyUnusedLocal
    def __new__(cls, array, mask, renormalize=False, *args, **kwargs):
        """An array of values, which are paired to a uniform 2D mask of pixels and sub-pixels. Each entry
        on the array corresponds to a value at the centre of a sub-pixel in an unmasked pixel. See the *Array* class
        for a full description of how Arrays work.
        
        The *Kernel* class is an *Array* but with additioonal methods that allow it to be convolved with data. 

        Parameters
        ----------
        array : np.ndarray
            The values of the array.
        mask : msk.Mask
            The 2D mask associated with the array, defining the pixels each array value is paired with and
            originates from.
        renormalize : bool
            If True, the Kernel's array values are renormalized such that they sum to 1.0.
        """
        obj = super(Kernel, cls).__new__(cls=cls, array=array, mask=mask)

        if renormalize:
            obj[:] = np.divide(obj, np.sum(obj))

        return obj

    @classmethod
    def manual_1d(
        cls, array, shape_2d, pixel_scales=None, origin=(0.0, 0.0), renormalize=False
    ):
        """Create a Kernel (see *Kernel.__new__*) by inputting the kernel values in 1D, for example:

        kernel=np.array([1.0, 2.0, 3.0, 4.0])

        kernel=[1.0, 2.0, 3.0, 4.0]

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked *Mask* of shape_2d.

        Parameters
        ----------
        array : np.ndarray or list
            The values of the array input as an ndarray of shape [total_unmasked_pixels*(sub_size**2)] or a list of
            lists.
        shape_2d : (float, float)
            The 2D shape of the mask the array is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        renormalize : bool
            If True, the Kernel's array values are renormalized such that they sum to 1.0.
        """
        array = arrays.Array.manual_1d(
            array=array, shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
        )

        return cls(array=array, mask=array.mask, renormalize=renormalize)

    @classmethod
    def manual_2d(cls, array, pixel_scales=None, origin=(0.0, 0.0), renormalize=False):
        """Create an Kernel (see *Kernel.__new__*) by inputting the kernel values in 2D, for example:

        kernel=np.ndarray([[1.0, 2.0],
                         [3.0, 4.0]])

        kernel=[[1.0, 2.0],
              [3.0, 4.0]]

        The 2D shape of the array and its mask are determined from the input array and the mask is setup as an
        unmasked *Mask* of shape_2d.

        Parameters
        ----------
        array : np.ndarray or list
            The values of the array input as an ndarray of shape [total_y_pixels*sub_size, total_x_pixel*sub_size] or a
             list of lists.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        renormalize : bool
            If True, the Kernel's array values are renormalized such that they sum to 1.0.
        """
        array = arrays.Array.manual_2d(
            array=array, pixel_scales=pixel_scales, origin=origin
        )

        return cls(array=array, mask=array.mask, renormalize=renormalize)

    @classmethod
    def manual(
        cls,
        array,
        shape_2d=None,
        pixel_scales=None,
        origin=(0.0, 0.0),
        renormalize=False,
    ):
        """Create a Kernel (see *Kernel.__new__*) by inputting the kernel values in 1D or 2D, automatically
        determining whether to use the 'manual_1d' or 'manual_2d' methods.

        See the manual_1d and manual_2d methods for examples.
        Parameters
        ----------
        array : np.ndarray or list
            The values of the array input as an ndarray of shape [total_unmasked_pixels*(sub_size**2)] or a list of
            lists.
        shape_2d : (float, float)
            The 2D shape of the mask the array is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        renormalize : bool
            If True, the Kernel's array values are renormalized such that they sum to 1.0.
        """
        if len(array.shape) == 1:
            return cls.manual_1d(
                array=array,
                shape_2d=shape_2d,
                pixel_scales=pixel_scales,
                origin=origin,
                renormalize=renormalize,
            )
        return cls.manual_2d(
            array=array,
            pixel_scales=pixel_scales,
            origin=origin,
            renormalize=renormalize,
        )

    @classmethod
    def full(
        cls,
        fill_value,
        shape_2d,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        renormalize=False,
    ):
        """Create a Kernel (see *Kernel.__new__*) where all values are filled with an input fill value, analogous to
         the method numpy ndarray.full.

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked *Mask* of shape_2d.

        Parameters
        ----------
        fill_value : float
            The value all array elements are filled with.
        shape_2d : (float, float)
            The 2D shape of the mask the array is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        renormalize : bool
            If True, the Kernel's array values are renormalized such that they sum to 1.0.
        """
        if sub_size is not None:
            shape_2d = (shape_2d[0] * sub_size, shape_2d[1] * sub_size)

        return cls.manual_2d(
            array=np.full(fill_value=fill_value, shape=shape_2d),
            pixel_scales=pixel_scales,
            origin=origin,
            renormalize=renormalize,
        )

    @classmethod
    def ones(cls, shape_2d, pixel_scales=None, origin=(0.0, 0.0), renormalize=False):
        """Create an Kernel (see *Kernel.__new__*) where all values are filled with ones, analogous to the method numpy
        ndarray.ones.

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked *Mask* of shape_2d.

        Parameters
        ----------
        shape_2d : (float, float)
            The 2D shape of the mask the array is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        renormalize : bool
            If True, the Kernel's array values are renormalized such that they sum to 1.0.
        """
        return cls.full(
            fill_value=1.0,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            origin=origin,
            renormalize=renormalize,
        )

    @classmethod
    def zeros(cls, shape_2d, pixel_scales=None, origin=(0.0, 0.0), renormalize=False):
        """Create an Kernel (see *Kernel.__new__*) where all values are filled with zeros, analogous to the method numpy
        ndarray.ones.

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked *Mask* of shape_2d.

        Parameters
        ----------
        shape_2d : (float, float)
            The 2D shape of the mask the array is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        renormalize : bool
            If True, the Kernel's array values are renormalized such that they sum to 1.0.
        """
        return cls.full(
            fill_value=0.0,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            origin=origin,
            renormalize=renormalize,
        )

    @classmethod
    def no_blur(cls, pixel_scales=None):
        """Setup the Kernel as a kernel which does not convolve any signal, which is simply an array of shape (1, 1)
        with value 1.

        Parameters
        ----------
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        """

        array = np.array([[1.0]])

        return cls.manual_2d(array=array, pixel_scales=pixel_scales)

    @classmethod
    def from_gaussian(
        cls,
        shape_2d,
        pixel_scales,
        sigma,
        centre=(0.0, 0.0),
        axis_ratio=1.0,
        phi=0.0,
        renormalize=False,
    ):
        """Setup the Kernel as a 2D symmetric elliptical Gaussian profile, according to the equation:

        (1.0 / (sigma * sqrt(2.0*pi))) * exp(-0.5 * (r/sigma)**2)


        Parameters
        ----------
        shape_2d : (float, float)
            The 2D shape of the mask the array is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sigma : float
            The value of sigma in the equation, describing the size and full-width half maximum of the Gaussian.
        centre : (float, float)
            The (y,x) central coordinates of the Gaussian.
        axis_ratio : float
            The axis-ratio of the elliptical Gaussian.
        phi : float
            The rotational angle of the Gaussian's ellipse defined counter clockwise from the positive x-axis.
        renormalize : bool
            If True, the Kernel's array values are renormalized such that they sum to 1.0.
        """

        grid = grids.Grid.uniform(shape_2d=shape_2d, pixel_scales=pixel_scales)
        grid_shifted = np.subtract(grid, centre)
        grid_radius = np.sqrt(np.sum(grid_shifted ** 2.0, 1))
        theta_coordinate_to_profile = np.arctan2(
            grid_shifted[:, 0], grid_shifted[:, 1]
        ) - np.radians(phi)
        grid_transformed = np.vstack(
            (
                grid_radius * np.sin(theta_coordinate_to_profile),
                grid_radius * np.cos(theta_coordinate_to_profile),
            )
        ).T

        grid_elliptical_radii = np.sqrt(
            np.add(
                np.square(grid_transformed[:, 1]),
                np.square(np.divide(grid_transformed[:, 0], axis_ratio)),
            )
        )

        gaussian = np.multiply(
            np.divide(1.0, sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(grid_elliptical_radii, sigma))),
        )

        return cls.manual_1d(
            array=gaussian,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            renormalize=renormalize,
        )

    @classmethod
    def from_as_gaussian_via_alma_fits_header_parameters(
        cls,
        shape_2d,
        pixel_scales,
        y_stddev,
        x_stddev,
        theta,
        centre=(0.0, 0.0),
        renormalize=False,
    ):

        x_stddev = (
            x_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        y_stddev = (
            y_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )

        axis_ratio = x_stddev / y_stddev

        return cls.from_gaussian(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sigma=y_stddev,
            axis_ratio=axis_ratio,
            phi=90.0 - theta,
            centre=centre,
            renormalize=renormalize,
        )

    @classmethod
    def from_fits(
        cls, file_path, hdu, pixel_scales=None, origin=(0.0, 0.0), renormalize=False
    ):
        """
        Loads the Kernel from a .fits file.

        Parameters
        ----------
        file_path : str
            The path the file is output to, including the filename and the '.fits' extension,
            e.g. '/path/to/filename.fits'
        hdu : int
            The Header-Data Unit of the .fits file the array data is loaded from.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        origin : (float, float)
            The origin of the array's mask.
        renormalize : bool
            If True, the Kernel's array values are renormalized such that they sum to 1.0.
        """

        array = arrays.Array.from_fits(
            file_path=file_path, hdu=hdu, pixel_scales=pixel_scales, origin=origin
        )

        return cls(array=array[:], mask=array.mask, renormalize=renormalize)

    def rescaled_with_odd_dimensions_from_rescale_factor(
        self, rescale_factor, renormalize=False
    ):
        """
        If the PSF kernel has one or two even-sized dimensions, return a PSF object where the kernel has odd-sized
        dimensions (odd-sized dimensions are required by a *Convolver*).

        The PSF can be scaled to larger / smaller sizes than the input size, if the rescale factor uses values that
        deviate furher from 1.0.

        Kernels are rescald using the scikit-image routine rescale, which performs rescaling via an interpolation
        routine. This may lead to loss of accuracy in the PSF kernel and it is advised that users, where possible,
        create their PSF on an odd-sized array using their data reduction pipelines that remove this approximation.

        Parameters
        ----------
        rescale_factor : float
            The factor by which the kernel is rescaled. If this has a value of 1.0, the kernel is rescaled to the
            closest odd-sized dimensions (e.g. 20 -> 19). Higher / lower values scale to higher / lower dimensions.
        renormalize : bool
            Whether the PSF should be renormalized after being rescaled.
        """

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
                output_shape=(
                    kernel_rescaled.shape[0] + 1,
                    kernel_rescaled.shape[1] + 1,
                ),
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

        return self.__class__.manual_2d(
            array=kernel_rescaled, pixel_scales=pixel_scales, renormalize=renormalize
        )

    @property
    def in_2d(self):
        """Convenience method to access the kerne;'s 2D representation, which is an ndarray of shape
        [sub_size*total_y_pixels, sub_size*total_x_pixels, 2] where all masked values are given values (0.0, 0.0).

        If the array is stored in 2D it is return as is. If it is stored in 1D, it must first be mapped from 1D to 2D."""
        return array_util.sub_array_2d_from(
            sub_array_1d=self, mask=self.mask, sub_size=self.mask.sub_size
        )

    @property
    def renormalized(self):
        """Renormalize the Kernel such that its data_vector values sum to unity."""
        return self.__class__(array=self, mask=self.mask, renormalize=True)

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

        array_binned_2d = array.in_2d_binned

        convolved_array_2d = scipy.signal.convolve2d(
            array_binned_2d, self.in_2d, mode="same"
        )

        convolved_array_1d = array_util.sub_array_1d_from(
            mask=array_binned_2d.mask, sub_array_2d=convolved_array_2d, sub_size=1
        )

        return arrays.Array(
            array=convolved_array_1d, mask=array_binned_2d.mask, store_in_1d=True
        )

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

        if self.mask.shape[0] % 2 == 0 or self.mask.shape[1] % 2 == 0:
            raise exc.KernelException("Kernel Kernel must be odd")

        convolved_array_2d = scipy.signal.convolve2d(array_2d, self.in_2d, mode="same")

        convolved_array_1d = array_util.sub_array_1d_from(
            mask=mask, sub_array_2d=convolved_array_2d, sub_size=1
        )

        return arrays.Array(
            array=convolved_array_1d, mask=mask.mask_sub_1, store_in_1d=True
        )
