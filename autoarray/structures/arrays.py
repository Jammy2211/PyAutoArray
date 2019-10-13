import logging

import numpy as np

from autoarray import exc
from autoarray.mask import mask as msk
from autoarray.util import array_util, binning_util

logging.basicConfig()
logger = logging.getLogger(__name__)


def array(array, shape_2d=None, pixel_scales=None, sub_size=None, origin=(0.0, 0.0)):

    array = np.asarray(array)

    if type(pixel_scales) is float:
        pixel_scales = (pixel_scales, pixel_scales)

    if len(array.shape) == 1 and shape_2d is None:
        raise exc.ArrayException('A 1D array cannot be used to set up an Array class without its 2D shape.')

    if shape_2d is not None and len(shape_2d) != 2:
        raise exc.ArrayException('The input shape_2d parameter is not a tuple of type (float, float)')

    if pixel_scales is None and sub_size is None:
        
        if len(array.shape) == 2:
            return Array.from_2d(array_2d=array)
        elif len(array.shape) == 1:
            return Array.from_1d_and_shape_2d(array_1d=array, shape_2d=shape_2d)

    elif pixel_scales is not None and sub_size is None:
        
        if len(array.shape) == 2:
            return ScaledArray.from_2d_and_pixel_scales(array_2d=array, pixel_scales=pixel_scales, origin=origin)
        elif len(array.shape) == 1:
            return ScaledArray.from_1d_shape_2d_and_pixel_scales(array_1d=array, shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin)

    elif pixel_scales is not None and sub_size is not None:

        if len(array.shape) == 2:
            return ScaledSubArray.from_2d_pixel_scales_and_sub_size(sub_array_2d=array, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin)
        elif len(array.shape) == 1:
            return ScaledSubArray.from_1d_shape_2d_pixel_scales_and_sub_size(sub_array_1d=array, shape_2d=shape_2d,
                                                                             pixel_scales=pixel_scales, sub_size=sub_size, origin=origin)


def full(fill_value, shape_2d, pixel_scales=None, sub_size=None, origin=(0.0, 0.0)):

    if sub_size is not None:
        shape_2d = (shape_2d[0] * sub_size, shape_2d[1] * sub_size)

    return array(array=np.full(fill_value=fill_value, shape=shape_2d), pixel_scales=pixel_scales, sub_size=sub_size, origin=origin)


class AbstractArray(np.ndarray):

    # noinspection PyUnusedLocal
    def __new__(cls, array_1d, mask, *args, **kwargs):
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

        obj = array_1d.view(cls)
        obj.mask = mask
        return obj

    @property
    def mapping(self):
        return self.mask.mapping

    @property
    def regions(self):
        return self.mask.regions

    @property
    def geometry(self):
        return self.mask.geometry

    def new_with_array(self, array):
        """
        Parameters
        ----------
        array: ndarray
            An ndarray

        Returns
        -------
        new_array: ScaledArray
            A new instance of this class that shares all of this instances attributes with a new ndarray.
        """
        arguments = vars(self)
        arguments.update({"array": array})
        return self.__class__(**arguments)

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(AbstractArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(AbstractArray, self).__setstate__(state[0:-1])

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_finalize__(self, obj):

        if hasattr(obj, 'mask'):
            self.mask = obj.mask

    def __eq__(self, other):
        super_result = super(AbstractArray, self).__eq__(other)
        try:
            return super_result.all()
        except AttributeError:
            return super_result

    @property
    def in_1d(self):
        return self

    @property
    def in_2d(self):
        return self.mask.mapping.array_2d_from_array_1d(array_1d=self)

    def map(self, func):
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                func(y, x)

    def zoomed_array_from_mask(self, mask, buffer=1):
        """Extract the 2D region of an array corresponding to the rectangle encompassing all unmasked values.

        This is used to extract and visualize only the region of an image that is used in an analysis.

        Parameters
        ----------
        mask : mask.Mask
            The mask around which the hyper array is extracted.
        buffer : int
            The buffer of pixels around the extraction.
        """

        extracted_array_2d = array_util.extracted_array_2d_from_array_2d_and_coordinates(
            array_2d=self.in_2d,
            y0=mask.geometry._zoom_region[0] - buffer,
            y1=mask.geometry._zoom_region[1] + buffer,
            x0=mask.geometry._zoom_region[2] - buffer,
            x1=mask.geometry._zoom_region[3] + buffer,
        )

        extracted_mask_2d = self.mask.mapping.resized_mask_from_new_shape(
            new_shape=extracted_array_2d.shape
        )

        return extracted_mask_2d.mapping.array_from_array_2d(array_2d=extracted_array_2d)

    def resized_array_from_new_shape(
        self, new_shape,
    ):
        """resized the array to a new shape and at a new origin.

        Parameters
        -----------
        new_shape : (int, int)
            The new two-dimensional shape of the array.
        """

        resized_array_2d = array_util.resized_array_2d_from_array_2d_and_resized_shape(
            array_2d=self.in_2d, resized_shape=new_shape,
        )

        resized_mask_2d = self.mask.mapping.resized_mask_from_new_shape(
            new_shape=new_shape,
        )

        return resized_mask_2d.mapping.array_from_array_2d(array_2d=resized_array_2d)

    def trimmed_array_from_kernel_shape(self, kernel_shape):
        psf_cut_y = np.int(np.ceil(kernel_shape[0] / 2)) - 1
        psf_cut_x = np.int(np.ceil(kernel_shape[1] / 2)) - 1
        array_y = np.int(self.mask.shape[0])
        array_x = np.int(self.mask.shape[1])
        trimmed_array_2d = self.in_2d[
            psf_cut_y : array_y - psf_cut_y, psf_cut_x : array_x - psf_cut_x
        ]

        resized_mask_2d = self.mask.mapping.resized_mask_from_new_shape(
            new_shape=trimmed_array_2d.shape,
        )

        return resized_mask_2d.mapping.array_from_array_2d(array_2d=trimmed_array_2d)

    def binned_array_from_bin_up_factor(self, bin_up_factor, method):

        binned_mask = self.mapping.binned_mask_from_bin_up_factor(bin_up_factor=bin_up_factor)

        if method is "mean":

            binned_array_2d = binning_util.binned_up_array_2d_using_mean_from_array_2d_and_bin_up_factor(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

            return binned_mask.mapping.array_from_array_2d(
                array_2d=binned_array_2d,
            )

        elif method is "quadrature":

            binned_array_2d = binning_util.binned_array_2d_using_quadrature_from_array_2d_and_bin_up_factor(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

            return binned_mask.mapping.array_from_array_2d(
                array_2d=binned_array_2d,
            )

        elif method is "sum":

            binned_array_2d = binning_util.binned_array_2d_using_sum_from_array_2d_and_bin_up_factor(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

            return binned_mask.mapping.array_from_array_2d(
                array_2d=binned_array_2d,
            )

        else:

            raise exc.ScaledException(
                "The method used in binned_up_array_from_array is not a valid method "
                "[mean | quadrature | sum]"
            )


class Array(AbstractArray):

    @classmethod
    def from_1d_and_shape_2d(cls, array_1d, shape_2d):

        mask = msk.Mask.unmasked_from_shape(
            shape=shape_2d,
        )

        return Array(array_1d=array_1d, mask=mask)

    @classmethod
    def from_2d(cls, array_2d):

        mask = msk.Mask.unmasked_from_shape(
            shape=array_2d.shape,
        )

        array_1d = array_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        return Array(array_1d=array_1d, mask=mask)

    @classmethod
    def from_single_value_and_shape_2d(
        cls, value, shape_2d,
    ):
        """
        Creates an instance of Array and fills it with a single value

        Parameters
        ----------
        value: float
            The value with which the array should be filled
        shape_2d: (int, int)
            The shape of the array
        pixel_scales: (float, float)
            The scale of a pixel in arc seconds

        Returns
        -------
        array: ScaledArray
            An array filled with a single value
        """
        array_2d = np.ones(shape_2d) * value
        return cls.from_2d(array_2d=array_2d)

    @classmethod
    def from_fits(cls, file_path, hdu):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        array_2d = array_util.numpy_array_2d_from_fits(
            file_path=file_path, hdu=hdu
        ).astype("float64")
        return cls.from_2d(array_2d=array_2d)


class ScaledArray(AbstractArray):

    @classmethod
    def from_1d_shape_2d_and_pixel_scales(cls, array_1d, shape_2d, pixel_scales, origin=(0.0, 0.0)):

        mask = msk.ScaledMask.unmasked_from_shape(
            shape=shape_2d, pixel_scales=pixel_scales, origin=origin
        )

        return ScaledArray(array_1d=array_1d, mask=mask)

    @classmethod
    def from_2d_and_pixel_scales(cls, array_2d, pixel_scales, origin=(0.0, 0.0)):

        mask = msk.ScaledMask.unmasked_from_shape(
            shape=array_2d.shape, pixel_scales=pixel_scales, origin=origin
        )

        array_1d = array_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        return cls(array_1d=array_1d, mask=mask)

    @classmethod
    def from_single_value_shape_2d_and_pixel_scales(
        cls, value, shape_2d, pixel_scales, origin=(0.0, 0.0),
    ):
        """
        Creates an instance of Array and fills it with a single value

        Parameters
        ----------
        value: float
            The value with which the array should be filled
        shape_2d: (int, int)
            The shape of the array
        pixel_scales: float
            The scale of a pixel in arc seconds

        Returns
        -------
        array: ScaledArray
            An array filled with a single value
        """
        array_2d = np.ones(shape_2d) * value
        return cls.from_2d_and_pixel_scales(array_2d=array_2d, pixel_scales=pixel_scales, origin=origin)

    @classmethod
    def from_fits_and_pixel_scales(cls, file_path, hdu, pixel_scales, origin=(0.0, 0.0)):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        array_2d = array_util.numpy_array_2d_from_fits(
            file_path=file_path, hdu=hdu
        ).astype("float64")
        return cls.from_2d_and_pixel_scales(array_2d=array_2d, pixel_scales=pixel_scales, origin=origin)


class ScaledSubArray(AbstractArray):
    """
    Class storing the grid_stacks for 2D pixel grid_stacks (e.g. image, PSF, signal_to_noise_ratio).
    """

    @classmethod
    def from_1d_shape_2d_pixel_scales_and_sub_size(cls, sub_array_1d, shape_2d, pixel_scales, sub_size, origin=(0.0, 0.0)):

        mask = msk.ScaledSubMask.unmasked_from_shape(
            shape=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return mask.mapping.array_from_sub_array_1d(sub_array_1d=sub_array_1d)

    @classmethod
    def from_2d_pixel_scales_and_sub_size(cls, sub_array_2d, pixel_scales, sub_size, origin=(0.0, 0.0)):

        shape = (sub_array_2d.shape[0] / sub_size, sub_array_2d.shape[1] / sub_size)

        mask = msk.ScaledSubMask.unmasked_from_shape(
            shape=shape, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

        return mask.mapping.array_from_sub_array_2d(sub_array_2d=sub_array_2d)

    @classmethod
    def from_sub_array_2d_and_mask(cls, sub_array_2d, mask):
        return mask.mapping.array_from_sub_array_2d(sub_array_2d=sub_array_2d)

    @classmethod
    def from_single_value_shape_2d_pixel_scales_and_sub_size(
        cls, value, shape_2d, pixel_scales, sub_size, origin=(0.0, 0.0)
    ):
        """
        Creates an instance of Array and fills it with a single value

        Parameters
        ----------
        value: float
            The value with which the array should be filled
        shape_2d: (int, int)
            The shape of the array
        pixel_scales: float
            The scale of a pixel in arc seconds

        Returns
        -------
        array: ScaledArray
            An array filled with a single value
        """
        sub_shape_2d = (shape_2d[0] * sub_size, shape_2d[1] * sub_size)
        sub_array_2d = np.ones(sub_shape_2d) * value
        return cls.from_2d_pixel_scales_and_sub_size(sub_array_2d=sub_array_2d, sub_size=sub_size, pixel_scales=pixel_scales, origin=origin)

    @classmethod
    def from_fits_pixel_scales_and_sub_size(cls, file_path, hdu, pixel_scales, sub_size, origin=(0.0, 0.0)):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales: float
            The arc-second to pixel conversion factor of each pixel.
        """
        sub_array_2d = array_util.numpy_array_2d_from_fits(
            file_path=file_path, hdu=hdu
        ).astype("float64")
        return cls.from_2d_pixel_scales_and_sub_size(sub_array_2d=sub_array_2d, sub_size=sub_size, pixel_scales=pixel_scales, origin=origin)

    @property
    def in_2d(self):
        return self.mask.mapping.sub_array_2d_from_sub_array_1d(sub_array_1d=self)

    @property
    def in_1d_binned(self):
        return self.mask.mapping.array_binned_from_sub_array_1d(sub_array_1d=self)

    @property
    def in_2d_binned(self):
        return self.mask.mapping.array_2d_binned_from_sub_array_1d(sub_array_1d=self)
