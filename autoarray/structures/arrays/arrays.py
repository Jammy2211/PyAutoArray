import logging

import numpy as np

from autoarray import exc
from autoarray.structures.arrays import abstract_array
from autoarray.structures import abstract_structure, grids
from autoarray.mask import mask as msk
from autoarray.util import array_util, grid_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Array(abstract_array.AbstractArray):
    def __new__(
        cls, array, mask, exposure_info=None, store_in_1d=True, *args, **kwargs
    ):
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

        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     This is an example mask.Mask, where:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIoIoIxIxIxIxI     x = True (Pixel is masked and excluded from the array)
        IxIxIxIoIoIoIoIxIxIxI     o = False (Pixel is not masked and included in the array)
        IxIxIxIoIoIoIoIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI

        The mask pixel index's will come out like this (and the direction of scaled values is highlighted
        around the mask.

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                        y      x
        IxIxIxIxIxIxIxIxIxIxI  ^   array[0] = 0
        IxIxIxIxIxIxIxIxIxIxI  I   array[1] = 1
        IxIxIxIxIxIxIxIxIxIxI  I   array[2] = 2
        IxIxIxIxI0I1IxIxIxIxI +ve  array[3] = 3
        IxIxIxI2I3I4I5IxIxIxI  y   array[4] = 4
        IxIxIxI6I7I8I9IxIxIxI -ve  array[5] = 5
        IxIxIxIxIxIxIxIxIxIxI  I   array[6] = 6
        IxIxIxIxIxIxIxIxIxIxI  I   array[7] = 7
        IxIxIxIxIxIxIxIxIxIxI \/   array[8] = 8
        IxIxIxIxIxIxIxIxIxIxI      array[9] = 9

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

        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     This is an example mask.Mask, where:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     x = True (Pixel is masked and excluded from lens)
        IxIxIxIxIoIoIxIxIxIxI     o = False (Pixel is not masked and included in lens)
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI

        Our array with a sub-size looks like it did before:

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->

        IxIxIxIxIxIxIxIxIxIxI  ^
        IxIxIxIxIxIxIxIxIxIxI  I
        IxIxIxIxIxIxIxIxIxIxI  I
        IxIxIxIxIxIxIxIxIxIxI +ve
        IxIxIxI0I1IxIxIxIxIxI  y
        IxIxIxIxIxIxIxIxIxIxI -ve
        IxIxIxIxIxIxIxIxIxIxI  I
        IxIxIxIxIxIxIxIxIxIxI  I
        IxIxIxIxIxIxIxIxIxIxI \/
        IxIxIxIxIxIxIxIxIxIxI

        However, if the sub-size is 2,each unmasked pixel has a set of sub-pixels with values. For example, for pixel 0,
        if *sub_size=2*, it has 4 values on a 2x2 sub-array:

        Pixel 0 - (2x2):

               array[0] = value of first sub-pixel in pixel 0.
        I0I1I  array[1] = value of first sub-pixel in pixel 1.
        I2I3I  array[2] = value of first sub-pixel in pixel 2.
               array[3] = value of first sub-pixel in pixel 3.

        If we used a sub_size of 3, for the first pixel we we would create a 3x3 sub-array:


                 array[0] = value of first sub-pixel in pixel 0.
                 array[1] = value of first sub-pixel in pixel 1.
                 array[2] = value of first sub-pixel in pixel 2.
        I0I1I2I  array[3] = value of first sub-pixel in pixel 3.
        I3I4I5I  array[4] = value of first sub-pixel in pixel 4.
        I6I7I8I  array[5] = value of first sub-pixel in pixel 5.
                 array[6] = value of first sub-pixel in pixel 6.
                 array[7] = value of first sub-pixel in pixel 7.
                 array[8] = value of first sub-pixel in pixel 8.

        Case 3: [sub_size=1 store_in_1d=False]
        --------------------------------------

        The Array has the same properties as Case 1, but is stored as an an ndarray of shape
        [total_y_values, total_x_values].

        All masked entries on the array have values of 0.0.

        For the following example mask:

        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     This is an example mask.Mask, where:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIoIoIxIxIxIxI     x = True (Pixel is masked and excluded from the array)
        IxIxIxIoIoIoIoIxIxIxI     o = False (Pixel is not masked and included in the array)
        IxIxIxIoIoIoIoIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI

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

        obj = array.view(cls)
        obj.mask = mask
        obj.exposure_info = exposure_info
        obj.store_in_1d = store_in_1d

        abstract_array.check_array(array=obj)

        return obj

    @classmethod
    def manual_1d(
        cls,
        array,
        shape_2d,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        exposure_info=None,
        store_in_1d=True,
    ):
        """Create an Array (see *AbstractArray.__new__*) by inputting the array values in 1D, for example:

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

        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

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

        array = abstract_array.convert_manual_1d_array(
            array_1d=array, mask=mask, store_in_1d=store_in_1d
        )
        return cls(
            array=array, mask=mask, exposure_info=exposure_info, store_in_1d=store_in_1d
        )

    @classmethod
    def manual_2d(
        cls,
        array,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        exposure_info=None,
        store_in_1d=True,
    ):
        """Create an Array (see *AbstractArray.__new__*) by inputting the array values in 2D, for example:

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

        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

        array = abstract_array.convert_array(array=array)

        shape_2d = (int(array.shape[0] / sub_size), int(array.shape[1] / sub_size))

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        array = abstract_array.convert_manual_2d_array(
            array_2d=array, mask=mask, store_in_1d=store_in_1d
        )
        return cls(array=array, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def manual(
        cls,
        array,
        shape_2d=None,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        exposure_info=None,
        store_in_1d=True,
    ):
        """Create an Array (see *AbstractArray.__new__*) by inputting the array values in 1D or 2D, automatically
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
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        if len(array.shape) == 1:
            return cls.manual_1d(
                array=array,
                shape_2d=shape_2d,
                pixel_scales=pixel_scales,
                sub_size=sub_size,
                origin=origin,
                store_in_1d=store_in_1d,
            )
        return cls.manual_2d(
            array=array,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def manual_mask(cls, array, mask, exposure_info=None, store_in_1d=True):
        """Create a Array (see *AbstractArray.__new__*) by inputting the array values in 1D or 2D with its mask,
        for example:

        mask = Mask([[True, False, False, False])
        array=np.array([1.0, 2.0, 3.0])

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
        array = abstract_array.convert_manual_array(
            array=array, mask=mask, store_in_1d=store_in_1d
        )
        return cls(
            array=array, mask=mask, exposure_info=exposure_info, store_in_1d=store_in_1d
        )

    @classmethod
    def full(
        cls,
        fill_value,
        shape_2d,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        exposure_info=None,
        store_in_1d=True,
    ):
        """Create a Array (see *AbstractArray.__new__*) where all values are filled with an input fill value, analogous to
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
        exposure_info=None,
        store_in_1d=True,
    ):
        """Create an Array (see *AbstractArray.__new__*) where all values are filled with ones, analogous to the method numpy
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
        exposure_info=None,
        store_in_1d=True,
    ):
        """Create an Array (see *AbstractArray.__new__*) where all values are filled with zeros, analogous to the method numpy
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
        """Create an Array (see *AbstractArray.__new__*) by loaing the array values from a .fits file.

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
        cls, y, x, values, shape_2d, sub_size=1, pixel_scales=None, exposure_info=None
    ):
        """Create a Array (see *AbstractArray.__new__*) by inputting the y and x pixel values where the array is filled
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
        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

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
            exposure_info=exposure_info,
        )
