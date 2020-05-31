import ast
import logging

import numpy as np
import os

from autoarray import exc
from autoarray.structures import abstract_structure, grids
from autoarray.mask import mask as msk
from autoarray.util import binning_util, array_util, grid_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Array(abstract_structure.AbstractStructure):

    # noinspection PyUnusedLocal
    def __new__(cls, array, mask, store_in_1d=True, *args, **kwargs):
        """An array of values, which are paired to a uniform 2D mask of pixels and sub-pixels. Each entry
        on the array corresponds to a value at the centre of a sub-pixel in an unmasked pixel.

        An *Array* is ordered such that pixels begin from the top-row of the corresponding mask and go right and down.
        The positive y-axis is upwards and positive x-axis to the right.

        The array can be stored in 1D or 2D, as detailed below.

        Case 1: [sub-size=1, store_in_1d = True]:
        -----------------------------------------

        The Array is an ndarray of shape [total_unmasked_pixels].

        The first element of the ndarray corresponds to the pixel index, for example:

        - array[3] = the 4th unmasked pixel's value.
        - array[6] = the 7th unmasked pixel's value.

        Below is a visual illustration of a array, where a total of 10 pixels are unmasked and are included in \
        the array.

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from the array)
        |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in the array)
        |x|x|x|o|o|o|o|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        The mask pixel index's will come out like this (and the direction of scaled values is highlighted
        around the mask.

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                        y      x
        |x|x|x|x|x|x|x|x|x|x|  ^   array[0] = 0
        |x|x|x|x|x|x|x|x|x|x|  |   array[1] = 1
        |x|x|x|x|x|x|x|x|x|x|  |   array[2] = 2
        |x|x|x|x|0|1|x|x|x|x| +ve  array[3] = 3
        |x|x|x|2|3|4|5|x|x|x|  y   array[4] = 4
        |x|x|x|6|7|8|9|x|x|x| -ve  array[5] = 5
        |x|x|x|x|x|x|x|x|x|x|  |   array[6] = 6
        |x|x|x|x|x|x|x|x|x|x|  |   array[7] = 7
        |x|x|x|x|x|x|x|x|x|x| \/   array[8] = 8
        |x|x|x|x|x|x|x|x|x|x|      array[9] = 9

        Case 2: [sub-size>1, store_in_1d=True]:
        ------------------

        If the masks's sub size is > 1, the array is defined as a sub-array where each entry corresponds to the values
        at the centre of each sub-pixel of an unmasked pixel.

        The sub-array indexes are ordered such that pixels begin from the first (top-left) sub-pixel in the first
        unmasked pixel. Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel.
        Therefore, the sub-array is an ndarray of shape [total_unmasked_pixels*(sub_array_shape)**2]. For example:

        - array[9] - using a 2x2 sub-array, gives the 3rd unmasked pixel's 2nd sub-pixel value.
        - array[9] - using a 3x3 sub-array, gives the 2nd unmasked pixel's 1st sub-pixel value.
        - array[27] - using a 3x3 sub-array, gives the 4th unmasked pixel's 1st sub-pixel value.

        Below is a visual illustration of a sub array. Indexing of each sub-pixel goes from the top-left corner. In
        contrast to the array above, our illustration below restricts the mask to just 2 pixels, to keep the
        illustration brief.

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|x|x|o|o|x|x|x|x|     o = False (Pixel is not masked and included in lens)
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        Our array with a sub-size looks like it did before:

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->

        |x|x|x|x|x|x|x|x|x|x|  ^
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x| +ve
        |x|x|x|0|1|x|x|x|x|x|  y
        |x|x|x|x|x|x|x|x|x|x| -ve
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x| \/
        |x|x|x|x|x|x|x|x|x|x|

        However, if the sub-size is 2,each unmasked pixel has a set of sub-pixels with values. For example, for pixel 0,
        if *sub_size=2*, it has 4 values on a 2x2 sub-array:

        Pixel 0 - (2x2):

               array[0] = value of first sub-pixel in pixel 0.
        |0|1|  array[1] = value of first sub-pixel in pixel 1.
        |2|3|  array[2] = value of first sub-pixel in pixel 2.
               array[3] = value of first sub-pixel in pixel 3.

        If we used a sub_size of 3, for the first pixel we we would create a 3x3 sub-array:


                 array[0] = value of first sub-pixel in pixel 0.
                 array[1] = value of first sub-pixel in pixel 1.
                 array[2] = value of first sub-pixel in pixel 2.
        |0|1|2|  array[3] = value of first sub-pixel in pixel 3.
        |3|4|5|  array[4] = value of first sub-pixel in pixel 4.
        |6|7|8|  array[5] = value of first sub-pixel in pixel 5.
                 array[6] = value of first sub-pixel in pixel 6.
                 array[7] = value of first sub-pixel in pixel 7.
                 array[8] = value of first sub-pixel in pixel 8.

        Case 3: [sub_size=1 store_in_1d=False]
        --------------------------------------

        The Array has the same properties as Case 1, but is stored as an an ndarray of shape
        [total_y_values, total_x_values].

        All masked entries on the array have values of 0.0.

        For the following example mask:

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from the array)
        |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in the array)
        |x|x|x|o|o|o|o|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        - array[0,0] = 0.0 (it is masked, thus zero)
        - array[0,0] = 0.0 (it is masked, thus zero)
        - array[3,3] = 0.0 (it is masked, thus zero)
        - array[3,3] = 0.0 (it is masked, thus zero)
        - array[3,4] = 0
        - array[3,4] = -1

        Case 4: [sub_size>1 store_in_1d=False]
        --------------------------------------

        The properties of this array can be derived by combining Case's 2 and 3 above, whereby the array is stored as
        an ndarray of shape [total_y_values*sub_size, total_x_values*sub_size].

        All sub-pixels in masked pixels have values 0.0.

        Parameters
        ----------
        array : np.ndarray
            The values of the array.
        mask : msk.Mask
            The 2D mask associated with the array, defining the pixels each array value is paired with and
            originates from.
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """

        if store_in_1d and len(array.shape) != 1:
            raise exc.ArrayException("Fill In")

        obj = super(Array, cls).__new__(
            cls=cls, structure=array, mask=mask, store_in_1d=store_in_1d
        )
        return obj

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
        """Create an Array (see *Array.__new__*) by inputting the array values in 1D, for example:

        array=np.array([1.0, 2.0, 3.0, 4.0])

        array=[1.0, 2.0, 3.0, 4.0]

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
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
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
            return Array(array=array, mask=mask, store_in_1d=store_in_1d)

        sub_array_2d = array_util.sub_array_2d_from(
            sub_array_1d=array, mask=mask, sub_size=mask.sub_size
        )

        return Array(array=sub_array_2d, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def manual_2d(
        cls, array, pixel_scales=None, sub_size=1, origin=(0.0, 0.0), store_in_1d=True
    ):
        """Create an Array (see *Array.__new__*) by inputting the array values in 2D, for example:

        array=np.ndarray([[1.0, 2.0],
                         [3.0, 4.0]])

        array=[[1.0, 2.0],
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
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
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

        if not store_in_1d:
            return Array(array=array, mask=mask, store_in_1d=store_in_1d)

        sub_array_1d = array_util.sub_array_1d_from(
            sub_array_2d=array, mask=mask, sub_size=mask.sub_size
        )

        return Array(array=sub_array_1d, mask=mask, store_in_1d=store_in_1d)

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
        """Create a Array (see *Array.__new__*) where all values are filled with an input fill value, analogous to
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
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
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
        """Create an Array (see *Array.__new__*) where all values are filled with ones, analogous to the method numpy
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
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
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
        """Create an Array (see *Array.__new__*) where all values are filled with zeros, analogous to the method numpy
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
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
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
        """Create an Array (see *Array.__new__*) by loaing the array values from a .fits file.

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
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        array_2d = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)
        return cls.manual_2d(
            array=array_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def manual_yx_and_values(
        cls, y, x, values, shape_2d, sub_size=1, pixel_scales=None
    ):
        """Create a Array (see *Array.__new__*) by inputting the y and x pixel values where the array is filled
        and the values to fill the array, for example:

        y = np.array([0, 0, 0, 1])
        x = np.array([0, 1, 2, 0])
        value = [1.0, 2.0, 3.0, 4.0]

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked *Mask* of shape_2d.

        Parameters
        ----------
        y : np.ndarray or list
            The y pixel indexes where value sare input, as an ndarray of shape [total_y_pixels*sub_size] or a list.
        x : np.ndarray or list
            The x pixel indexes where value sare input, as an ndarray of shape [total_y_pixels*sub_size] or a list.
        values : np.ndarray or list
            The values which are used too fill in the array.
        shape_2d : (float, float)
            The 2D shape of the mask the grid is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin : (float, float)
            The origin of the grid's mask.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        grid = grids.Grid.manual_yx_1d(
            y=y, x=x, shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=1
        )

        grid_pixels = grid_util.grid_pixel_indexes_1d_from(
            grid_scaled_1d=grid.in_1d, shape_2d=shape_2d, pixel_scales=pixel_scales
        )

        array_1d = np.zeros(shape=shape_2d[0] * shape_2d[1])

        for i in range(grid_pixels.shape[0]):
            array_1d[i] = values[int(grid_pixels[i])]

        return cls.manual_1d(
            array=array_1d,
            pixel_scales=pixel_scales,
            shape_2d=shape_2d,
            sub_size=sub_size,
        )

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(Array, self).__reduce__()
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
        super(Array, self).__setstate__(state[0:-1])

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __eq__(self, other):
        super_result = super(Array, self).__eq__(other)
        try:
            return super_result.all()
        except AttributeError:
            return super_result

    @property
    def in_1d(self):
        """Convenience method to access the array's 1D representation, which is an ndarray of shape
        [total_unmasked_pixels*(sub_size**2)].

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D."""
        if self.store_in_1d:
            return self

        sub_array_1d = array_util.sub_array_1d_from(
            sub_array_2d=self, mask=self.mask, sub_size=self.mask.sub_size
        )

        return Array(array=sub_array_1d, mask=self.mask, store_in_1d=True)

    @property
    def in_2d(self):
        """Convenience method to access the array's 2D representation, which is an ndarray of shape
        [sub_size*total_y_pixels, sub_size*total_x_pixels, 2] where all masked values are given values (0.0, 0.0).

        If the array is stored in 2D it is return as is. If it is stored in 1D, it must first be mapped from 1D to 2D."""
        if self.store_in_1d:
            sub_array_2d = array_util.sub_array_2d_from(
                sub_array_1d=self, mask=self.mask, sub_size=self.mask.sub_size
            )
            return Array(array=sub_array_2d, mask=self.mask, store_in_1d=False)

        return self

    @property
    def in_1d_binned(self):
        """Convenience method to access the binned-up array in its 1D representation, which is a Grid stored as an
        ndarray of shape [total_unmasked_pixels, 2].

        The binning up process converts a array from (y,x) values where each value is a coordinate on the sub-array to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a array with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the array is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D."""

        if not self.store_in_1d:

            sub_array_1d = array_util.sub_array_1d_from(
                sub_array_2d=self, mask=self.mask, sub_size=self.mask.sub_size
            )

        else:

            sub_array_1d = self

        binned_array_1d = np.multiply(
            self.mask.sub_fraction,
            sub_array_1d.reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        return Array(array=binned_array_1d, mask=self.mask.mask_sub_1, store_in_1d=True)

    @property
    def in_2d_binned(self):
        """Convenience method to access the binned-up array in its 2D representation, which is a Grid stored as an
        ndarray of shape [total_y_pixels, total_x_pixels, 2].

        The binning up process conerts a array from (y,x) values where each value is a coordinate on the sub-array to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a array with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the array is stored in 2D it is return as is. If it is stored in 1D, it must first be mapped from 1D to 2D."""
        if not self.store_in_1d:

            sub_array_1d = array_util.sub_array_1d_from(
                sub_array_2d=self, mask=self.mask, sub_size=self.mask.sub_size
            )

        else:

            sub_array_1d = self

        binned_array_1d = np.multiply(
            self.mask.sub_fraction,
            sub_array_1d[:].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        binned_array_2d = array_util.sub_array_2d_from(
            sub_array_1d=binned_array_1d, mask=self.mask, sub_size=1
        )

        return Array(
            array=binned_array_2d, mask=self.mask.mask_sub_1, store_in_1d=False
        )

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

    def map(self, func):
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                func(y, x)

    def zoomed_around_mask(self, buffer=1):
        """Extract the 2D region of an array corresponding to the rectangle encompassing all unmasked values.

        This is used to extract and visualize only the region of an image that is used in an analysis.

        Parameters
        ----------
        buffer : int
            The number pixels around the extracted array used as a buffer.
        """

        extracted_array_2d = array_util.extracted_array_2d_from(
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

        return MaskedArray.manual_2d(
            array=extracted_array_2d, mask=mask, store_in_1d=self.store_in_1d
        )

    def extent_of_zoomed_array(self, buffer=1):
        """For an extracted zoomed array computed from the method *zoomed_around_mask* compute its extent in scaled
        coordinates.

        The extent of the grid in scaled units returned as an ndarray of the form [x_min, x_max, y_min, y_max].

        This is used visualize zoomed and extracted arrays via the imshow() method.

        Parameters
        ----------
        buffer : int
            The number pixels around the extracted array used as a buffer.
        """
        extracted_array_2d = array_util.extracted_array_2d_from(
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

        resized_mask = self.mask.resized_mask_from_new_shape(new_shape=new_shape)

        return MaskedArray.manual_2d(
            array=resized_array_2d, mask=resized_mask, store_in_1d=self.store_in_1d
        )

    def padded_from_kernel_shape(self, kernel_shape_2d):
        """When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the array can be padded, where it is 'buffed' such that it includes all 
        pixels whose signal will be convolved into the unmasked pixels given the 2D kernel shape. The values of
        these pixels are zeros.

        Parameters
        ----------
        kernel_shape_2d : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        new_shape = (
            self.shape_2d[0] + (kernel_shape_2d[0] - 1),
            self.shape_2d[1] + (kernel_shape_2d[1] - 1),
        )
        return self.resized_from_new_shape(new_shape=new_shape)

    def trimmed_from_kernel_shape(self, kernel_shape_2d):
        """When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the array can be padded, a padded array can be computed via the method
        *padded_from_kernel_shape*. This function trims the array back to its original shape, after the padded array
        has been used for computationl.

        Parameters
        ----------
        kernel_shape_2d : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        psf_cut_y = np.int(np.ceil(kernel_shape_2d[0] / 2)) - 1
        psf_cut_x = np.int(np.ceil(kernel_shape_2d[1] / 2)) - 1
        array_y = np.int(self.mask.shape[0])
        array_x = np.int(self.mask.shape[1])
        trimmed_array_2d = self.in_2d[
            psf_cut_y : array_y - psf_cut_y, psf_cut_x : array_x - psf_cut_x
        ]

        resized_mask = self.mask.resized_mask_from_new_shape(
            new_shape=trimmed_array_2d.shape
        )

        return MaskedArray.manual_2d(
            array=trimmed_array_2d, mask=resized_mask, store_in_1d=self.store_in_1d
        )

    def binned_from_bin_up_factor(self, bin_up_factor, method):
        """Compute a binned version of the Array, where binning up occurs by coming all pixel values in a set of 
        (bin_up_factor x bin_up_factor) pixels.
        
        The pixels can be combined:
         
        - By taking the mean of their values, which one may use for binning up an image.
        - By adding them in quadranture, which one may use for binning up a noise map.
        - By summing them, which one may use for binning up an exposure time map.
        
        Parameters
        ----------
        bin_up_factor : int
            The factor which the array is binned up by (e.g. a value of 2 bins every 2 x 2 pixels into one pixel).
        method : str
            The method used to combine the set of values that are binned up.
        """

        binned_mask = self.mask.binned_mask_from_bin_up_factor(
            bin_up_factor=bin_up_factor
        )

        if method is "mean":

            binned_array_2d = binning_util.bin_array_2d_via_mean(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

        elif method is "quadrature":

            binned_array_2d = binning_util.bin_array_2d_via_quadrature(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

        elif method is "sum":

            binned_array_2d = binning_util.bin_array_2d_via_sum(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

        else:

            raise exc.ArrayException(
                "The method used in binned_up_array_from_array is not a valid method "
                "[mean | quadrature | sum]"
            )

        binned_array_1d = array_util.sub_array_1d_from(
            mask=binned_mask, sub_array_2d=binned_array_2d, sub_size=1
        )

        return Array(array=binned_array_1d, mask=binned_mask, store_in_1d=True)

    def output_to_fits(self, file_path, overwrite=False):
        """Output the array to a .fits file.

        Parameters
        ----------
        file_path : str
            The path the file is output to, including the filename and the '.fits' extension,
            e.g. '/path/to/filename.fits'
        overwrite : bool
            If a file already exists at the path, if overwrite=True it is overwritten else an error is raised."""
        array_util.numpy_array_2d_to_fits(
            array_2d=self.in_2d, file_path=file_path, overwrite=overwrite
        )


class MaskedArray(Array):
    @classmethod
    def manual_1d(cls, array, mask, store_in_1d=True):
        """Create a Array (see *Array.__new__*) by inputting the array values in 1D with its mask, for example:

        mask = Mask([[True, False, False, False])
        array=np.array([1.0, 2.0, 3.0])

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked *Mask* of shape_2d.

        Parameters
        ----------
        array : np.ndarray or list
            The values of the array input as an ndarray of shape [total_unmasked_pixels*(sub_size**2)] or a list of
            lists.
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        if type(array) is list:
            array = np.asarray(array)

        if array.shape[0] != mask.sub_pixels_in_mask:
            raise exc.ArrayException(
                "The input 1D array does not have the same number of entries as sub-pixels in"
                "the mask."
            )

        if store_in_1d:
            return Array(array=array, mask=mask, store_in_1d=store_in_1d)

        sub_array_2d = array_util.sub_array_2d_from(
            sub_array_1d=array, mask=mask, sub_size=mask.sub_size
        )

        return Array(array=sub_array_2d, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def manual_2d(cls, array, mask, store_in_1d=True):
        """Create an Array (see *Array.__new__*) by inputting the array values in 2D with its mask, for example:

        mask = Mask([[True, False, False, False])
        array=np.ndarray([[1.0, 2.0],
                         [3.0, 4.0]])

        array=[[1.0, 2.0],
              [3.0, 4.0]]

        Mask values are removed, such that the grid in 1D will be of length 3, omitting the values 1.0.

        Parameters
        ----------
        array : np.ndarray or list
            The values of the array input as an ndarray of shape [total_y_pixels*sub_size, total_x_pixel*sub_size] or a
             list of lists.
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        if type(array) is list:
            array = np.asarray(array)

        if array.shape != mask.sub_shape_2d:
            raise exc.ArrayException(
                "The input array is 2D but not the same dimensions as the sub-mask "
                "(e.g. the mask 2D shape multipled by its sub size."
            )

        sub_array_1d = array_util.sub_array_1d_from(
            sub_array_2d=array, mask=mask, sub_size=mask.sub_size
        )

        if store_in_1d:
            return Array(array=sub_array_1d, mask=mask, store_in_1d=store_in_1d)

        sub_array_2d = array_util.sub_array_2d_from(
            sub_array_1d=sub_array_1d, mask=mask, sub_size=mask.sub_size
        )

        return Array(array=sub_array_2d, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def full(cls, fill_value, mask, store_in_1d=True):
        """Create an Array (see *Array.__new__*) where all values are filled with an input fill value, with the
         corresponding mask input.

        Parameters
        ----------
        fill_value : float
            The value all array elements are filled with.
         mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        return cls.manual_2d(
            array=np.full(fill_value=fill_value, shape=mask.sub_shape_2d),
            mask=mask,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def ones(cls, mask, store_in_1d=True):
        """Create an Array (see *Array.__new__*) where all values are filled with ones,  with the
         corresponding mask input.

        Parameters
        ----------
        shape_2d : (float, float)
            The 2D shape of the mask the array is paired with.
         mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        return cls.full(fill_value=1.0, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def zeros(cls, mask, store_in_1d=True):
        """Create an Array (see *Array.__new__*) where all values are filled with zeros, with the
         corresponding mask input.

        Parameters
        ----------
        shape_2d : (float, float)
            The 2D shape of the mask the array is paired with.
          mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        return cls.full(fill_value=0.0, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def from_fits(cls, file_path, hdu, mask, store_in_1d=True):
        """Create an Array (see *Array.__new__*) by loaing the array values from a .fits file, with the
         corresponding mask input.

        Parameters
        ----------
        file_path : str
            The path the file is output to, including the filename and the '.fits' extension,
            e.g. '/path/to/filename.fits'
        hdu : int
            The Header-Data Unit of the .fits file the array data is loaded from.
          mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        array_2d = array_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)
        return cls.manual_2d(array=array_2d, mask=mask, store_in_1d=store_in_1d)


class Values(np.ndarray):
    def __new__(cls, values):
        """ A collection of values structured in a way defining groups of values which share a common origin (for
        example values may be grouped if they are from a specific region of a dataset).

        Grouping is structured as follows:

        [[value0, value1], [value0, value1, value2]]

        Here, we have two groups of values, where each group is associated with the other values.

        The values object does not store the values as a list of list of floats, but instead a 1D NumPy array
        of shape [total_values]. Index information is stored so that this array can be mapped to the list of
        list of float structure above. They are stored as a NumPy array so the values can be used efficiently for
        calculations.

        The values input to this function can have any of the following forms:

        [[value0, value1], [value0]]
        [[value0, value1]]

        In all cases, they will be converted to a list of list of floats followed by a 1D NumPy array.

        Print methods are overridden so a user always "sees" the values as the list structure.

        In contrast to a *Array* structure, *Values* do not lie on a uniform grid or correspond to values that
        originate from a uniform grid. Therefore, when handling irregular data-sets *Values* should be used.

        Parameters
        ----------
        values : [[float]] or equivalent
            A collection of values that are grouped according to whether they share a common origin.
        """

        if len(values) == 0:
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

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "lower_indexes"):
            self.lower_indexes = obj.lower_indexes

        if hasattr(obj, "upper_indexes"):
            self.upper_indexes = obj.upper_indexes

    @property
    def in_1d(self):
        """Convenience method to access the Values in their 1D representation, which is an ndarray of shape
        [total_values]."""
        return self

    @property
    def in_list(self):
        """Convenience method to access the Values in their list representation, whcih is a list of lists of floatss."""
        return [list(self[i:j]) for i, j in zip(self.lower_indexes, self.upper_indexes)]

    def values_from_arr_1d(self, arr_1d):
        """Create a *Values* object from a 1D ndarray of values of shape [total_values].

        The *Values* are structured and grouped following this *Values* instance.

        Parameters
        ----------
        arr_1d : ndarray
            The 1D array (shape [total_values]) of values that are mapped to a *Values* object."""
        values_1d = [
            list(arr_1d[i:j]) for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]
        return Values(values=values_1d)

    def coordinates_from_grid_1d(self, grid_1d):
        """Create a *GridCoordinates* object from a 2D ndarray array of values of shape [total_values, 2].

        The *GridCoordinates* are structured and grouped following this *Coordinate* instance.

        Parameters
        ----------
        grid_1d : ndarray
            The 2d array (shape [total_coordinates, 2]) of (y,x) coordinates that are mapped to a *GridCoordinates*
            object."""
        coordinates_1d = [
            list(map(tuple, grid_1d[i:j, :]))
            for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]

        return grids.GridCoordinates(coordinates=coordinates_1d)

    @classmethod
    def from_file(cls, file_path):
        """Create a *Values* object from a file which stores the values as a list of list of floats.

        Parameters
        ----------
        file_path : str
            The path to the values .dat file containing the values (e.g. '/path/to/values.dat')
        """
        with open(file_path) as f:
            values_lines = f.readlines()

        values = []

        for line in values_lines:
            values_list = ast.literal_eval(line)
            values.append(values_list)

        return Values(values=values)

    def output_to_file(self, file_path, overwrite=False):
        """Output this instance of the *Values* object to a list of list of floats.

        Parameters
        ----------
        file_path : str
            The path to the values .dat file containing the values (e.g. '/path/to/values.dat')
        overwrite : bool
            If there is as exsiting file it will be overwritten if this is *True*.
        """

        if os.path.exists(file_path):
            if overwrite:
                os.remove(file_path)
            else:
                raise FileExistsError(
                    f"The file {file_path} already exists. Set overwrite=True to overwrite this"
                    "file"
                )

        with open(file_path, "w+") as f:
            for value in self.in_list:
                f.write("%s\n" % value)
