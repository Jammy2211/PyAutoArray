from astropy import units
import numpy as np
import scipy.signal
from skimage.transform import resize, rescale
from typing import List, Tuple, Union

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import AbstractArray2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.header import Header

from autoarray import exc
from autoarray import type as ty
from autoarray.structures.arrays import array_2d_util


class Kernel2D(AbstractArray2D):
    def __new__(
        cls, array, mask, header=None, normalize: bool = False, *args, **kwargs
    ):
        """
        An array of values, which are paired to a uniform 2D mask of pixels and sub-pixels. Each entry
        on the array corresponds to a value at the centre of a sub-pixel in an unmasked pixel. See the *Array2D* class
        for a full description of how Arrays work.

        The *Kernel2D* class is an *Array2D* but with additioonal methods that allow it to be convolved with data.

        Parameters
        ----------
        array
            The values of the array.
        mask :Mask2D
            The 2D mask associated with the array, defining the pixels each array value is paired with and
            originates from.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """
        obj = super().__new__(cls=cls, array=array, mask=mask, header=header)

        if normalize:
            obj[:] = np.divide(obj, np.sum(obj))

        return obj

    @classmethod
    def manual_slim(
        cls,
        array: Union[np.ndarray, List],
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        normalize: bool = False,
    ):
        """
        Create a Kernel2D (see *Kernel2D.__new__*) by inputting the kernel values in 1D, for example:

        kernel=np.array([1.0, 2.0, 3.0, 4.0])

        kernel=[1.0, 2.0, 3.0, 4.0]

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        array or list
            The values of the array input as an ndarray of shape [total_unmasked_pixels*(sub_size**2)] or a list of
            lists.
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """
        array = Array2D.manual_slim(
            array=array,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        return Kernel2D(array=array, mask=array.mask, normalize=normalize)

    @classmethod
    def manual_native(
        cls,
        array: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        normalize: bool = False,
    ) -> "Kernel2D":
        """
        Create an Kernel2D (see *Kernel2D.__new__*) by inputting the kernel values in 2D, for example:

        kernel=np.ndarray([[1.0, 2.0], [3.0, 4.0]])

        kernel=[[1.0, 2.0],
              [3.0, 4.0]]

        The 2D shape of the array and its mask are determined from the input array and the mask is setup as an
        unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        array or list
            The values of the array input as an ndarray of shape [total_y_pixels*sub_size, total_x_pixel*sub_size] or a
             list of lists.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """
        array = Array2D.manual_native(
            array=array, pixel_scales=pixel_scales, origin=origin
        )

        return Kernel2D(array=array, mask=array.mask, normalize=normalize)

    @classmethod
    def manual(
        cls,
        array: Union[np.ndarray, List],
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        normalize: bool = False,
    ):
        """
        Create a Kernel2D (see *Kernel2D.__new__*) by inputting the kernel values in 1D or 2D, automatically
        determining whether to use the 'manual_slim' or 'manual_native' methods.

        See the manual_slim and manual_native methods for examples.
        Parameters
        ----------
        array or list
            The values of the array input as an ndarray of shape [total_unmasked_pixels*(sub_size**2)] or a list of
            lists.
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """
        if len(array.shape) == 1:
            return cls.manual_slim(
                array=array,
                shape_native=shape_native,
                pixel_scales=pixel_scales,
                origin=origin,
                normalize=normalize,
            )
        return cls.manual_native(
            array=array, pixel_scales=pixel_scales, origin=origin, normalize=normalize
        )

    @classmethod
    def full(
        cls,
        fill_value: float,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        normalize: bool = False,
    ) -> "Kernel2D":
        """
        Create a Kernel2D (see *Kernel2D.__new__*) where all values are filled with an input fill value, analogous to
        the method numpy ndarray.full.

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        fill_value
            The value all array elements are filled with.
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """
        return Kernel2D.manual_native(
            array=np.full(fill_value=fill_value, shape=shape_native),
            pixel_scales=pixel_scales,
            origin=origin,
            normalize=normalize,
        )

    @classmethod
    def ones(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        normalize: bool = False,
    ):
        """
        Create an Kernel2D (see *Kernel2D.__new__*) where all values are filled with ones, analogous to the method numpy
        ndarray.ones.

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """
        return cls.full(
            fill_value=1.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
            normalize=normalize,
        )

    @classmethod
    def zeros(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        normalize: bool = False,
    ) -> "Kernel2D":
        """
        Create an Kernel2D (see *Kernel2D.__new__*) where all values are filled with zeros, analogous to the method numpy
        ndarray.ones.

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """
        return cls.full(
            fill_value=0.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
            normalize=normalize,
        )

    @classmethod
    def no_blur(cls, pixel_scales):
        """
        Setup the Kernel2D as a kernel which does not convolve any signal, which is simply an array of shape (1, 1)
        with value 1.

        Parameters
        ----------
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        """

        array = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        return cls.manual_native(array=array, pixel_scales=pixel_scales)

    @classmethod
    def from_gaussian(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        sigma: float,
        centre: Tuple[float, float] = (0.0, 0.0),
        axis_ratio: float = 1.0,
        angle: float = 0.0,
        normalize: bool = False,
    ) -> "Kernel2D":
        """
        Setup the Kernel2D as a 2D symmetric elliptical Gaussian profile, according to the equation:

        (1.0 / (sigma * sqrt(2.0*pi))) * exp(-0.5 * (r/sigma)**2)


        Parameters
        ----------
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sigma
            The value of sigma in the equation, describing the size and full-width half maximum of the Gaussian.
        centre
            The (y,x) central coordinates of the Gaussian.
        axis_ratio
            The axis-ratio of the elliptical Gaussian.
        angle
            The rotational angle of the Gaussian's ellipse defined counter clockwise from the positive x-axis.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """

        grid = Grid2D.uniform(shape_native=shape_native, pixel_scales=pixel_scales)
        grid_shifted = np.subtract(grid, centre)
        grid_radius = np.sqrt(np.sum(grid_shifted**2.0, 1))
        theta_coordinate_to_profile = np.arctan2(
            grid_shifted[:, 0], grid_shifted[:, 1]
        ) - np.radians(angle)
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

        return cls.manual_slim(
            array=gaussian,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            normalize=normalize,
        )

    @classmethod
    def from_as_gaussian_via_alma_fits_header_parameters(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        y_stddev: float,
        x_stddev: float,
        theta: float,
        centre: Tuple[float, float] = (0.0, 0.0),
        normalize: bool = False,
    ) -> "Kernel2D":

        x_stddev = (
            x_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        y_stddev = (
            y_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )

        axis_ratio = x_stddev / y_stddev

        return Kernel2D.from_gaussian(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sigma=y_stddev,
            axis_ratio=axis_ratio,
            angle=90.0 - theta,
            centre=centre,
            normalize=normalize,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: str,
        hdu: int,
        pixel_scales,
        origin=(0.0, 0.0),
        normalize: bool = False,
    ) -> "Kernel2D":
        """
        Loads the Kernel2D from a .fits file.

        Parameters
        ----------
        file_path
            The path the file is loaded from, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        hdu
            The Header-Data Unit of the .fits file the array data is loaded from.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """

        array = Array2D.from_fits(
            file_path=file_path, hdu=hdu, pixel_scales=pixel_scales, origin=origin
        )

        header_sci_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=0)
        header_hdu_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=hdu)

        return Kernel2D(
            array=array[:],
            mask=array.mask,
            normalize=normalize,
            header=Header(header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj),
        )

    def rescaled_with_odd_dimensions_from(
        self, rescale_factor: float, normalize: bool = False
    ) -> "Kernel2D":
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
        rescale_factor
            The factor by which the kernel is rescaled. If this has a value of 1.0, the kernel is rescaled to the
            closest odd-sized dimensions (e.g. 20 -> 19). Higher / lower values scale to higher / lower dimensions.
        normalize
            Whether the PSF should be normalized after being rescaled.
        """

        kernel_rescaled = rescale(
            self.native,
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

        return Kernel2D.manual_native(
            array=kernel_rescaled, pixel_scales=pixel_scales, normalize=normalize
        )

    @property
    def normalized(self) -> "Kernel2D":
        """
        Normalize the Kernel2D such that its data_vector values sum to unity.
        """
        return Kernel2D(array=self, mask=self.mask, normalize=True)

    def convolved_array_from(self, array: Array2D) -> Array2D:
        """
        Convolve an array with this Kernel2D

        Parameters
        ----------
        image
            An array representing the image the Kernel2D is convolved with.

        Returns
        -------
        convolved_image
            An array representing the image after convolution.

        Raises
        ------
        KernelException if either Kernel2D psf dimension is odd
        """
        if self.mask.shape[0] % 2 == 0 or self.mask.shape[1] % 2 == 0:
            raise exc.KernelException("Kernel2D Kernel2D must be odd")

        array_binned_2d = array.binned.native

        convolved_array_2d = scipy.signal.convolve2d(
            array_binned_2d, self.native, mode="same"
        )

        convolved_array_1d = array_2d_util.array_2d_slim_from(
            mask_2d=array_binned_2d.mask, array_2d_native=convolved_array_2d, sub_size=1
        )

        return Array2D(array=convolved_array_1d, mask=array_binned_2d.mask)

    def convolved_array_with_mask_from(self, array: Array2D, mask: Mask2D) -> Array2D:
        """
        Convolve an array with this Kernel2D

        Parameters
        ----------
        image
            An array representing the image the Kernel2D is convolved with.

        Returns
        -------
        convolved_image
            An array representing the image after convolution.

        Raises
        ------
        KernelException if either Kernel2D psf dimension is odd
        """

        if self.mask.shape[0] % 2 == 0 or self.mask.shape[1] % 2 == 0:
            raise exc.KernelException("Kernel2D Kernel2D must be odd")

        convolved_array_2d = scipy.signal.convolve2d(array, self.native, mode="same")

        convolved_array_1d = array_2d_util.array_2d_slim_from(
            mask_2d=mask, array_2d_native=convolved_array_2d, sub_size=1
        )

        return Array2D(array=convolved_array_1d, mask=mask.mask_sub_1)
