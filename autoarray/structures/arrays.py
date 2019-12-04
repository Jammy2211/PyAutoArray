import logging

import numpy as np

from autoarray import exc
from autoarray.structures import abstract_structure
from autoarray.mask import mask as msk
from autoarray.util import binning_util, array_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractArray(abstract_structure.AbstractStructure):

    # noinspection PyUnusedLocal
    def __new__(cls, array, mask, store_in_1d=True, *args, **kwargs):
        """ A hyper array with square-pixels.

        Parameters
        ----------
        array: ndarray
            An array representing image (e.g. an image, noise-map, etc.)
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The arc-second origin of the hyper array's coordinate system.
        """

        if store_in_1d and len(array.shape) != 1:
            raise exc.ArrayException("Fill In")

        obj = super(AbstractArray, cls).__new__(
            cls=cls, structure=array, mask=mask, store_in_1d=store_in_1d
        )
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

    def zoomed_around_mask(self, buffer=1):
        """Extract the 2D region of an array corresponding to the rectangle encompassing all unmasked values.

        This is used to extract and visualize only the region of an image that is used in an analysis.

        Parameters
        ----------
        mask : mask.Mask
            The mask around which the hyper array is extracted.
        buffer : int
            The buffer of pixels around the extraction.
        """

        extracted_array_2d = array_util.extracted_array_2d_from_array_2d(
            array_2d=self.in_2d,
            y0=self.geometry._zoom_region[0] - buffer,
            y1=self.geometry._zoom_region[1] + buffer,
            x0=self.geometry._zoom_region[2] - buffer,
            x1=self.geometry._zoom_region[3] + buffer,
        )

        mask = msk.Mask.unmasked(
            shape_2d=extracted_array_2d.shape,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

        return mask.mapping.array_stored_1d_from_array_2d(array_2d=extracted_array_2d)

    def extent_of_zoomed_array(self, buffer=1):

        extracted_array_2d = array_util.extracted_array_2d_from_array_2d(
            array_2d=self.in_2d,
            y0=self.geometry._zoom_region[0] - buffer,
            y1=self.geometry._zoom_region[1] + buffer,
            x0=self.geometry._zoom_region[2] - buffer,
            x1=self.geometry._zoom_region[3] + buffer,
        )

        mask = msk.Mask.unmasked(
            shape_2d=extracted_array_2d.shape,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.mask.geometry.mask_centre,
        )

        return mask.geometry.extent

    def resized_from_new_shape(self, new_shape):
        """resized the array to a new shape and at a new origin.

        Parameters
        -----------
        new_shape : (int, int)
            The new two-dimensional shape of the array.
        """

        resized_array_2d = array_util.resized_array_2d_from_array_2d(
            array_2d=self.in_2d, resized_shape=new_shape
        )

        resized_mask_2d = self.mask.mapping.resized_mask_from_new_shape(
            new_shape=new_shape
        )

        return resized_mask_2d.mapping.array_stored_1d_from_array_2d(
            array_2d=resized_array_2d
        )

    def trimmed_from_kernel_shape(self, kernel_shape_2d):
        psf_cut_y = np.int(np.ceil(kernel_shape_2d[0] / 2)) - 1
        psf_cut_x = np.int(np.ceil(kernel_shape_2d[1] / 2)) - 1
        array_y = np.int(self.mask.shape[0])
        array_x = np.int(self.mask.shape[1])
        trimmed_array_2d = self.in_2d[
            psf_cut_y : array_y - psf_cut_y, psf_cut_x : array_x - psf_cut_x
        ]

        resized_mask_2d = self.mask.mapping.resized_mask_from_new_shape(
            new_shape=trimmed_array_2d.shape
        )

        return resized_mask_2d.mapping.array_stored_1d_from_array_2d(
            array_2d=trimmed_array_2d
        )

    def binned_from_bin_up_factor(self, bin_up_factor, method):

        binned_mask = self.mapping.binned_mask_from_bin_up_factor(
            bin_up_factor=bin_up_factor
        )

        if method is "mean":

            binned_array_2d = binning_util.bin_array_2d_via_mean(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

            return binned_mask.mapping.array_stored_1d_from_array_2d(
                array_2d=binned_array_2d
            )

        elif method is "quadrature":

            binned_array_2d = binning_util.bin_array_2d_via_quadrature(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

            return binned_mask.mapping.array_stored_1d_from_array_2d(
                array_2d=binned_array_2d
            )

        elif method is "sum":

            binned_array_2d = binning_util.bin_array_2d_via_sum(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

            return binned_mask.mapping.array_stored_1d_from_array_2d(
                array_2d=binned_array_2d
            )

        else:

            raise exc.ArrayException(
                "The method used in binned_up_array_from_array is not a valid method "
                "[mean | quadrature | sum]"
            )

    @property
    def in_1d(self):
        if self.store_in_1d:
            return self
        else:
            return self.mask.mapping.array_stored_1d_from_sub_array_2d(
                sub_array_2d=self
            )

    @property
    def in_2d(self):
        if self.store_in_1d:
            return self.mask.mapping.array_stored_2d_from_sub_array_1d(
                sub_array_1d=self
            )
        else:
            return self

    @property
    def in_1d_binned(self):
        if self.store_in_1d:
            return self.mask.mapping.array_stored_1d_binned_from_sub_array_1d(
                sub_array_1d=self
            )
        else:
            sub_array_1d = self.mask.mapping.array_stored_1d_from_sub_array_2d(
                sub_array_2d=self
            )
            return self.mask.mapping.array_stored_1d_binned_from_sub_array_1d(
                sub_array_1d=sub_array_1d
            )

    @property
    def in_2d_binned(self):
        if self.store_in_1d:
            return self.mask.mapping.array_stored_2d_binned_from_sub_array_1d(
                sub_array_1d=self
            )
        else:
            sub_array_1d = self.mask.mapping.array_stored_1d_from_sub_array_2d(
                sub_array_2d=self
            )
            return self.mask.mapping.array_stored_2d_binned_from_sub_array_1d(
                sub_array_1d=sub_array_1d
            )

    def output_to_fits(self, file_path, overwrite=False):

        array_util.numpy_array_2d_to_fits(
            array_2d=self.in_2d, file_path=file_path, overwrite=overwrite
        )


class Array(AbstractArray):
    @classmethod
    def manual_1d(
        cls,
        array,
        shape_2d,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=True,
    ):

        if type(array) is list:
            array = np.asarray(array)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        if shape_2d is not None and len(shape_2d) != 2:
            raise exc.ArrayException(
                "The input shape_2d parameter is not a tuple of type (float, float)"
            )

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        if store_in_1d:
            return mask.mapping.array_stored_1d_from_sub_array_1d(sub_array_1d=array)
        else:
            return mask.mapping.array_stored_2d_from_sub_array_1d(sub_array_1d=array)

    @classmethod
    def manual_2d(
        cls, array, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), store_in_1d=True
    ):

        if type(array) is list:
            array = np.asarray(array)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        shape_2d = (int(array.shape[0] / sub_size), int(array.shape[1] / sub_size))

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        if store_in_1d:
            return mask.mapping.array_stored_1d_from_sub_array_2d(sub_array_2d=array)
        else:
            return mask.mapping.array_stored_2d_from_sub_array_2d(sub_array_2d=array)

    @classmethod
    def full(
        cls,
        fill_value,
        shape_2d,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=True,
    ):

        if sub_size is not None:
            shape_2d = (shape_2d[0] * sub_size, shape_2d[1] * sub_size)

        return cls.manual_2d(
            array=np.full(fill_value=fill_value, shape=shape_2d),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def ones(
        cls,
        shape_2d,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=True,
    ):
        return cls.full(
            fill_value=1.0,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def zeros(
        cls,
        shape_2d,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=True,
    ):
        return cls.full(
            fill_value=0.0,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def from_fits(
        cls,
        file_path,
        hdu=0,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=True,
    ):
        array_2d = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)
        return cls.manual_2d(
            array=array_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_in_1d=store_in_1d,
        )
