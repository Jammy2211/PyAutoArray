import logging

import numpy as np

from autoarray import exc
from autoarray.structures import abstract_structure
from autoarray.mask import mask as msk
from autoarray.util import array_util, binning_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractArray(abstract_structure.AbstractStructure):

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
        obj = super(AbstractArray, cls).__new__(cls=cls, structure_1d=array_1d, mask=mask)
        return obj

    def new_with_array(self, array):
        """
        Parameters
        ----------
        array: ndarray
            An ndarray

        Returns
        -------
        new_array: Array
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

    def __eq__(self, other):
        super_result = super(AbstractArray, self).__eq__(other)
        try:
            return super_result.all()
        except AttributeError:
            return super_result

    def map(self, func):
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                func(y, x)

    def zoomed_from_mask(self, mask, buffer=1):
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

    def resized_from_new_shape(
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

    def trimmed_from_kernel_shape(self, kernel_shape):
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

    def binned_from_bin_up_factor(self, bin_up_factor, method):

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

    @property
    def in_2d(self):
        return self.mask.mapping.sub_array_2d_from_sub_array_1d(sub_array_1d=self)

    @property
    def in_1d_binned(self):
        return self.mask.mapping.array_binned_from_sub_array_1d(sub_array_1d=self)

    @property
    def in_2d_binned(self):
        return self.mask.mapping.array_2d_binned_from_sub_array_1d(sub_array_1d=self)


class Array(AbstractArray):

    @classmethod
    def from_sub_array_1d_shape_2d_pixel_scales_and_sub_size(cls, sub_array_1d, shape_2d, pixel_scales, sub_size, origin=(0.0, 0.0)):

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return mask.mapping.array_from_sub_array_1d(sub_array_1d=sub_array_1d)

    @classmethod
    def from_sub_array_2d_pixel_scales_and_sub_size(cls, sub_array_2d, pixel_scales, sub_size, origin=(0.0, 0.0)):

        shape_2d = (int(sub_array_2d.shape[0] / sub_size), int(sub_array_2d.shape[1] / sub_size))

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

        return mask.mapping.array_from_sub_array_2d(sub_array_2d=sub_array_2d)

    @classmethod
    def manual_1d(cls, array, shape_2d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)):

        array = np.asarray(array)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        if shape_2d is not None and len(shape_2d) != 2:
            raise exc.ArrayException('The input shape_2d parameter is not a tuple of type (float, float)')

        if len(array.shape) == 2:
            return Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array, pixel_scales=pixel_scales,
                                                                     sub_size=sub_size, origin=origin)
        elif len(array.shape) == 1:
            return Array.from_sub_array_1d_shape_2d_pixel_scales_and_sub_size(sub_array_1d=array, shape_2d=shape_2d,
                                                                              pixel_scales=pixel_scales,
                                                                              sub_size=sub_size, origin=origin)

    @classmethod
    def manual_2d(cls, array, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)):

        array = np.asarray(array)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        return Array.from_sub_array_2d_pixel_scales_and_sub_size(
            sub_array_2d=array, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin)

    @classmethod
    def full(cls, fill_value, shape_2d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)):

        if sub_size is not None:
            shape_2d = (shape_2d[0] * sub_size, shape_2d[1] * sub_size)

        return cls.manual_2d(array=np.full(fill_value=fill_value, shape=shape_2d), pixel_scales=pixel_scales, sub_size=sub_size, origin=origin)

    @classmethod
    def ones(cls, shape_2d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)):
        return cls.full(fill_value=1.0, shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin)

    @classmethod
    def zeros(cls, shape_2d, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)):
        return cls.full(fill_value=0.0, shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin)

    @classmethod
    def from_fits(cls, file_path, hdu, pixel_scales=None, sub_size=1, origin=(0.0, 0.0)):
        array_2d = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)
        return cls.manual_2d(array=array_2d, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin)

    @classmethod
    def from_sub_array_2d_and_mask(cls, sub_array_2d, mask):
        return mask.mapping.array_from_sub_array_2d(sub_array_2d=sub_array_2d)


class ArrayMasked(AbstractArray):

    @classmethod
    def manual_1d(cls, array, mask):

        array = np.asarray(array)

        if array.shape[0] != mask.sub_pixels_in_mask:
            raise exc.ArrayException('The input 1D array does not have the same number of entries as sub-pixels in'
                                     'the mask.')

        return mask.mapping.array_from_sub_array_1d(sub_array_1d=array)

    @classmethod
    def manual_2d(cls, array, mask):

        array = np.asarray(array)

        if array.shape != mask.sub_shape:
            raise exc.ArrayException('The input array is 2D but not the same dimensions as the sub-mask '
                                     '(e.g. the mask 2D shape multipled by its sub size.')

        return mask.mapping.array_from_sub_array_2d(sub_array_2d=array)

    @classmethod
    def full(cls, fill_value, mask):
        return cls.manual_2d(array=np.full(fill_value=fill_value, shape=mask.sub_shape), mask=mask)

    @classmethod
    def ones(cls, mask):
        return cls.full(fill_value=1.0, mask=mask)

    @classmethod
    def zeros(cls, mask):
        return cls.full(fill_value=0.0, mask=mask)

    @classmethod
    def from_fits(cls, file_path, hdu, mask):
        array_2d = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)
        return cls.manual_2d(array=array_2d,  mask=mask)