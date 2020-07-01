import ast
import logging

import numpy as np
import os

from autoarray import exc
from autoarray.structures.arrays import abstract_array
from autoarray.structures import grids
from autoarray.mask import mask as msk
from autoarray.util import array_util, grid_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Array(abstract_array.AbstractArray):
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
        cls, y, x, values, shape_2d, sub_size=1, pixel_scales=None
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


class MaskedArray(abstract_array.AbstractArray):
    @classmethod
    def manual_1d(cls, array, mask, store_in_1d=True):
        """Create a Array (see *AbstractArray.__new__*) by inputting the array values in 1D with its mask, for example:

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
        array = abstract_array.setup_manual_1d_array(
            array=array, mask=mask, store_in_1d=store_in_1d
        )
        return Array(array=array, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def manual_2d(cls, array, mask, store_in_1d=True):
        """Create an Array (see *AbstractArray.__new__*) by inputting the array values in 2D with its mask, for example:

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
        array = abstract_array.setup_manual_2d_array(
            array=array, mask=mask, store_in_1d=store_in_1d
        )
        return Array(array=array, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def full(cls, fill_value, mask, store_in_1d=True):
        """Create an Array (see *AbstractArray.__new__*) where all values are filled with an input fill value, with the
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
        """Create an Array (see *AbstractArray.__new__*) where all values are filled with ones,  with the
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
        """Create an Array (see *AbstractArray.__new__*) where all values are filled with zeros, with the
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
        """Create an Array (see *AbstractArray.__new__*) by loaing the array values from a .fits file, with the
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
