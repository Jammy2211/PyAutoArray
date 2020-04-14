import ast
import logging

import numpy as np
import os

from autoarray import exc
from autoarray.structures import abstract_structure, grids
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
        """Resize the array around its centre to a new input shape.

        If a new_shape dimension is smaller than the current dimension, the data at the edges is trimmed and removed.
        If it is larger, the data is padded with zeros.

        If the array has even sized dimensions, the central pixel around which data is trimmed / padded is chosen as
        the top-left pixel of the central quadrant of pixels.

        Parameters
        -----------
        new_shape : (int, int)
            The new 2D shape of the array.
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

    def padded_from_kernel_shape(self, kernel_shape_2d):
        new_shape = (
            self.shape_2d[0] + (kernel_shape_2d[0] - 1),
            self.shape_2d[1] + (kernel_shape_2d[1] - 1),
        )
        return self.resized_from_new_shape(new_shape=new_shape)

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


class MaskedArray(AbstractArray):
    @classmethod
    def manual_1d(cls, array, mask, store_in_1d=True):

        if type(array) is list:
            array = np.asarray(array)

        if array.shape[0] != mask.sub_pixels_in_mask:
            raise exc.ArrayException(
                "The input 1D array does not have the same number of entries as sub-pixels in"
                "the mask."
            )

        if store_in_1d:
            return mask.mapping.array_stored_1d_from_sub_array_1d(sub_array_1d=array)
        else:
            return mask.mapping.array_stored_2d_from_sub_array_1d(sub_array_1d=array)

    @classmethod
    def manual_2d(cls, array, mask, store_in_1d=True):

        if type(array) is list:
            array = np.asarray(array)

        if array.shape != mask.sub_shape_2d:
            raise exc.ArrayException(
                "The input array is 2D but not the same dimensions as the sub-mask "
                "(e.g. the mask 2D shape multipled by its sub size."
            )

        if store_in_1d:
            return mask.mapping.array_stored_1d_from_sub_array_2d(sub_array_2d=array)
        else:
            masked_sub_array_1d = mask.mapping.array_stored_1d_from_sub_array_2d(
                sub_array_2d=array
            )
            return mask.mapping.array_stored_2d_from_sub_array_1d(
                sub_array_1d=masked_sub_array_1d
            )

    @classmethod
    def full(cls, fill_value, mask, store_in_1d=True):
        return cls.manual_2d(
            array=np.full(fill_value=fill_value, shape=mask.sub_shape_2d),
            mask=mask,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def ones(cls, mask, store_in_1d=True):
        return cls.full(fill_value=1.0, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def zeros(cls, mask, store_in_1d=True):
        return cls.full(fill_value=0.0, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def from_fits(cls, file_path, hdu, mask, store_in_1d=True):
        array_2d = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)
        return cls.manual_2d(array=array_2d, mask=mask, store_in_1d=store_in_1d)


class Values(np.ndarray):
    def __new__(cls, values, mask=None):
        """ A collection of values structured in a way defining groups of values which share a common origin (for
        example values may be grouped if they are from a specific region of a dataset).

        Grouping is structured as follows:

        [[(y0,x0), (y1,x1)], [(y0,x0), (y1,x1), (y2,x2)]]

        Here, we have two groups of values, where each group is associated.

        The values object does not store the values as a list of list of floats, but instead a 1D NumPy array
        of shape [total_values]. Index information is stored so that this array can be mapped to the list of
        list of float structure above. They are stored as a NumPy array so the values can be used efficiently for
        calculations.

        The values input to this function can have any of the following forms:

        [[x0, x1], [x0]]
        [[[x0, x1]], [x0]]

        In all cases, they will be converted to a list of list of floats followed by a 1D NumPy array.

        Print methods are overridden so a user always "sees" the values as the list structure.

        In contrast to a *Array* structure, *Values* do not lie on a uniform grid or correspond to values that
        originate from a uniform grid. Therefore, when handling irregular data-sets *Values* should be used.

        Parameters
        ----------
        values : [[float]] or equivalent
            A collection of values that are grouped if they correpsond to a shared origin.
        mask : aa.Mask
            The mask whose attributes are used to perform coordinate conversions.
        """

        if values == []:
            return []

        if isinstance(values[0], float):
            values = [values]

        upper_indexes = []

        a = 0

        for coords in values:
            a = a + len(coords)
            upper_indexes.append(a)

        values_arr = np.concatenate([np.array(i) for i in values])

        obj = values_arr.view(cls)
        obj.upper_indexes = upper_indexes
        obj.lower_indexes = [0] + upper_indexes[:-1]
        obj.mask = mask

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "lower_indexes"):
            self.lower_indexes = obj.lower_indexes

        if hasattr(obj, "upper_indexes"):
            self.upper_indexes = obj.upper_indexes

        if hasattr(obj, "mask"):
            self.mask = obj.mask

    @classmethod
    def from_pixels_and_mask(cls, pixels, mask):
        """Create *Coordinates* from a list of values in pixel units and a mask which allows these values to
        be converted to scaled units."""
        values = []
        for coordinate_set in pixels:
            values.append(
                [
                    mask.geometry.scaled_values_from_pixel_values(pixel_values=values)
                    for values in coordinate_set
                ]
            )
        return cls(values=values, mask=mask)

    @property
    def in_1d(self):
        return self

    @property
    def in_list(self):
        """Return the values on a structured list which groups values with a common origin."""
        return [list(self[i:j]) for i, j in zip(self.lower_indexes, self.upper_indexes)]

    def values_from_arr_1d(self, arr_1d):
        """Create a *Values* object from a 1D NumPy array of values of shape [total_values]. The
        *Values* are structured and grouped following this *Coordinate* instance."""
        values_1d = [
            list(arr_1d[i:j]) for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]
        return Values(values=values_1d, mask=self.mask)

    def coordinates_from_grid_1d(self, grid_1d):
        """Create a *Coordinates* object from a 2D NumPy array of values of shape [total_values, 2]. The
        *Coordinates* are structured and grouped following this *Coordinate* instance."""
        coordinates_1d = [
            list(map(tuple, grid_1d[i:j, :]))
            for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]

        return grids.Coordinates(coordinates=coordinates_1d, mask=self.mask)

    @classmethod
    def from_file(cls, file_path):
        """Create a *Coordinates* object from a file which stores the values as a list of list of tuples.

        Parameters
        ----------
        file_path : str
            The path to the values .dat file containing the values (e.g. '/path/to/values.dat')
        """
        with open(file_path) as f:
            values_string = f.readlines()

        values = []

        for line in values_string:
            values_list = ast.literal_eval(line)
            values.append(values_list)

        return Values(values=values)

    def output_to_file(self, file_path, overwrite=False):
        """Output this instance of the *Coordinates* object to a list of list of tuples.

        Parameters
        ----------
        file_path : str
            The path to the values .dat file containing the values (e.g. '/path/to/values.dat')
        overwrite : bool
            If there is as exsiting file it will be overwritten if this is *True*.
        """

        if overwrite and os.path.exists(file_path):
            os.remove(file_path)
        elif not overwrite and os.path.exists(file_path):
            raise FileExistsError(
                "The file ",
                file_path,
                " already exists. Set overwrite=True to overwrite this" "file",
            )

        with open(file_path, "w") as f:
            for value in self.in_list:
                f.write("%s\n" % value)

    @property
    def mapping(self):
        return self.mask.mapping

    @property
    def shape_2d_scaled(self):
        return (
            np.amax(self.in_1d[:, 0]) - np.amin(self.in_1d[:, 0]),
            np.amax(self.in_1d[:, 1]) - np.amin(self.in_1d[:, 1]),
        )

    @property
    def scaled_maxima(self):
        return (np.amax(self.in_1d[:, 0]), np.amax(self.in_1d[:, 1]))

    @property
    def scaled_minima(self):
        return (np.amin(self.in_1d[:, 0]), np.amin(self.in_1d[:, 1]))

    @property
    def extent(self):
        return np.asarray(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )
