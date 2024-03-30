from astropy.io import fits
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

from autoconf import conf

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.abstract_structure import Structure
from autoarray.structures.header import Header
from autoarray.structures.arrays.uniform_1d import Array1D

from autoarray import exc
from autoarray import type as ty

from autoarray.structures.arrays import array_2d_util
from autoarray.geometry import geometry_util
from autoarray.layout import layout_util
from autoarray.numpy_wrapper import numpy as npw


logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractArray2D(Structure):
    def __init__(
        self,
        values: Union[np.ndarray, List, "AbstractArray2D"],
        mask: Mask2D,
        header: Header = None,
        store_native: bool = False,
        skip_mask: bool = False,
        *args,
        **kwargs,
    ):
        """
        A uniform 2D array of values, which are paired with a 2D mask of pixels which may be split into sub-pixels.

        The ``Array2D`, like all data structures (e.g. ``Grid2D``, ``VectorYX2D``) has in-built functionality which:

        - Applies a 2D mask (a ``Mask2D`` object) to the da_ta structure's values.

        - Maps the data structure between two data representations: `slim`` (all unmasked values in
          a 1D ``ndarray``) and ``native`` (all unmasked values in a 2D ``ndarray``).

        - Associates Cartesian ``Grid2D`` objects of (y,x) coordinates with the data structure (e.g.
          a (y,x) grid of all unmasked pixels).

        - Associates sub-grids with the data structure, which perform calculations higher resolutions which are then
          binned up.

        Each entry of an ``Array2D`` corresponds to a value at the centre of a sub-pixel in its
        corresponding ``Mask2D``.  It is ordered such that pixels begin from the top-row of the corresponding mask
        and go right and down. The positive y-axis is upwards and positive x-axis to the right.

        A detailed description of the data structure API is provided below.

        **SLIM DATA REPRESENTATION (sub-size=1)**

        Below is a visual illustration of an ``Array2D``'s 2D mask, where a total of 10 pixels are unmasked and are
        included in the array.

        ::

             x x x x x x x x x x
             x x x x x x x x x x     This is an example ``Mask2D``, where:
             x x x x x x x x x x
             x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the array)
             x x x O O O O x x x     O = `False` (Pixel is not masked and included in the array)
             x x x O O O O x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        The mask pixel index's are as follows (the positive / negative direction of the ``Grid2D`` objects associated
        with the array are also shown on the y and x axes).

        ::

            <--- -ve  x  +ve -->

             x x x x x x x x x x  ^   array_2d[0] = 10
             x x x x x x x x x x  I   array_2d[1] = 20
             x x x x x x x x x x  I   array_2d[2] = 30
             x x x x 0 1 x x x x +ve  array_2d[3] = 40
             x x x 2 3 4 5 x x x  y   array_2d[4] = 50
             x x x 6 7 8 9 x x x -ve  array_2d[5] = 60
             x x x x x x x x x x  I   array_2d[6] = 70
             x x x x x x x x x x  I   array_2d[7] = 80
             x x x x x x x x x x \/   array_2d[8] = 90
             x x x x x x x x x x      array_2d[9] = 100

        The ``Array2D`` in its ``slim`` data representation is an ``ndarray`` of shape [total_unmasked_pixels].

        For the ``Mask2D`` above the ``slim`` representation therefore contains 10 entries and two examples of these
        entries are:

        ::

            array[3] = the 4th unmasked pixel's value, given by value 40 above.
            array[6] = the 7th unmasked pixel's value, given by value 80 above.

        A Cartesian grid of (y,x) coordinates, corresponding to all ``slim`` values (e.g. unmasked pixels) is given
        by ``array_2d.derive_grid.masked.slim``.


        __native__

        The ``Array2D`` above, but represented as an an ``ndarray`` of shape [total_y_values, total_x_values], where
        all masked entries have values of 0.0.

        For the following mask:

        ::

             x x x x x x x x x x
             x x x x x x x x x x     This is an example ``Mask2D``, where:
             x x x x x x x x x x
             x x x x O O x x x x     x = `True` (Pixel is masked and excluded from the array)
             x x x O O O O x x x     O = `False` (Pixel is not masked and included in the array)
             x x x O O O O x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        Where the array has the following indexes (left figure) and values (right):

        ::

            <--- -ve  x  +ve -->

             x x x x x x x x x x  ^   array_2d[0] = 10
             x x x x x x x x x x  I   array_2d[1] = 20
             x x x x x x x x x x  I   array_2d[2] = 30
             x x x x 0 1 x x x x +ve  array_2d[3] = 40
             x x x 2 3 4 5 x x x  y   array_2d[4] = 50
             x x x 6 7 8 9 x x x -ve  array_2d[5] = 60
             x x x x x x x x x x  I   array_2d[6] = 70
             x x x x x x x x x x  I   array_2d[7] = 80
             x x x x x x x x x x \/   array_2d[8] = 90
             x x x x x x x x x x      array_2d[9] = 100

        In the above array:

        ::

            - array[0,0] = 0.0 (it is masked, thus zero)
            - array[0,0] = 0.0 (it is masked, thus zero)
            - array[3,3] = 0.0 (it is masked, thus zero)
            - array[3,3] = 0.0 (it is masked, thus zero)
            - array[3,4] = 10
            - array[3,5] = 20
            - array[4,5] = 50

        **SLIM TO NATIVE MAPPING**

        The ``Array2D`` has functionality which maps data between the ``slim`` and ``native`` data representations.

        For the example mask above, the 1D ``ndarray`` given by ``mask.derive_indexes.slim_to_native`` is:

        ::

            slim_to_native[0] = [3,4]
            slim_to_native[1] = [3,5]
            slim_to_native[2] = [4,3]
            slim_to_native[3] = [4,4]
            slim_to_native[4] = [4,5]
            slim_to_native[5] = [4,6]
            slim_to_native[6] = [5,3]
            slim_to_native[7] = [5,4]
            slim_to_native[8] = [5,5]
            slim_to_native[9] = [5,6]

        In **PyAutoCTI** all `Array2D` objects are used in their `native` representation. Significant memory can be
        saved by only store this format, thus the `native_only` config override can force this behaviour.
        It is recommended users do not use this option to avoid unexpected behaviour.

        Parameters
        ----------
        values
            The values of the array, which can be input in the ``slim`` or ``native`` format.
        mask
            The 2D mask associated with the array, defining the pixels each array value in its ``slim`` representation
            is paired with.
        store_native
            If True, the ndarray is stored in its native format [total_y_pixels, total_x_pixels]. This avoids
            mapping large data arrays to and from the slim / native formats, which can be a computational bottleneck.

        Examples
        --------

        This example uses the ``Array2D.no_mask`` method to create the ``Array2D``.

        Different methods using different inputs are available and documented throughout this webpage.

        .. code-block:: python

            import autoarray as aa

            array_2d = aa.Array2D.no_mask(
                values=np.array([1.0, 2.0, 3.0, 4.0]),
                shape_native=(2, 2),
                pixel_scales=1.0,
            )

        .. code-block:: python

            import autoarray as aa

            array_2d = aa.Array2D.no_mask(
                values=[1.0, 2.0, 3.0, 4.0],
                shape_native=(2, 1),
                pixel_scales=1.0,
            )

            mask = aa.Mask2D(
                mask=[[False, False], [True, True]],
                pixel_scales=2.0,
            )

            array_2d = array_2d.apply_mask(mask=mask)

            # Print certain array attributes.

            print(array_2d.slim) # masked 1D data representation.
            print(array_2d.native) # masked 2D data representation.

            # Output array to .fits file.

            array_2d.output_to_fits(file_path="/path/for/output")
        """

        try:
            values = values._array
        except AttributeError:
            pass
        if conf.instance["general"]["structures"]["native_binned_only"]:
            store_native = True

        values = array_2d_util.convert_array_2d(
            array_2d=values,
            mask_2d=mask,
            store_native=store_native,
            skip_mask=skip_mask,
        )

        super().__init__(values)
        self.mask = mask
        self.header = header

    @property
    def values(self):
        return self._array

    def __array_finalize__(self, obj):
        if hasattr(obj, "mask"):
            self.mask = obj.mask

        if hasattr(obj, "header"):
            self.header = obj.header
        else:
            self.header = None

    @property
    def store_native(self):
        return len(self.shape) != 1

    def apply_mask(self, mask: Mask2D) -> "Array2D":
        return Array2D(values=self.native, mask=mask, header=self.header)

    @property
    def slim(self) -> "Array2D":
        """
        Return an `Array2D` where the data is stored its `slim` representation, which is an ``ndarray`` of shape
        [total_unmasked_pixels].

        If it is already stored in its `slim` representation it is returned as it is. If not, it is mapped from
        `native` to `slim` and returned as a new `Array2D`.
        """
        return Array2D(values=self, mask=self.mask, header=self.header)

    @property
    def native(self) -> "Array2D":
        """
        Return a `Array2D` where the data is stored in its `native` representation, which is an ``ndarray`` of shape
        [total_y_pixels, total_x_pixels].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Array2D`.
        """
        return Array2D(
            values=self, mask=self.mask, header=self.header, store_native=True
        )

    @property
    def native_skip_mask(self) -> "Array2D":
        """
        Return a `Array2D` where the data is stored in its `native` representation, which is an ``ndarray`` of shape
        [total_y_pixels, total_x_pixels].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Array2D`.
        """
        return Array2D(
            values=self,
            mask=self.mask,
            header=self.header,
            store_native=True,
            skip_mask=True,
        )

    @property
    def in_counts(self) -> "Array2D":
        return self.header.array_eps_to_counts(array_eps=self)

    @property
    def in_counts_per_second(self) -> "Array2D":
        return self.header.array_counts_to_counts_per_second(
            array_counts=self.in_counts
        )

    @property
    def original_orientation(self) -> Union[np.ndarray, "Array2D"]:
        return layout_util.rotate_array_via_roe_corner_from(
            array=np.array(self), roe_corner=self.header.original_roe_corner
        )

    @property
    def readout_offsets(self) -> Tuple[int, int]:
        if self.header is not None:
            if self.header.readout_offsets is not None:
                return self.header.readout_offsets
        return (0, 0)

    @property
    def binned_across_rows(self) -> Array1D:
        """
        Bins the 2D array up to a 1D array, where each value is the mean of all unmasked values in each row.
        """
        binned_array = np.mean(self.native.array, axis=0, where=~self.mask)

        # binned_array = (self.native * np.invert(self.mask)).sum(axis=0) / np.invert(
        #     self.mask
        # ).sum(axis=0)
        return Array1D.no_mask(values=binned_array, pixel_scales=self.pixel_scale)

    @property
    def binned_across_columns(self) -> Array1D:
        """
        Bins the 2D array up to a 1D array, where each value is the mean of all unmasked values in each column.
        """
        binned_array = np.mean(self.native.array, axis=1, where=~self.mask)

        # binned_array = (self.native*np.invert(self.mask)).sum(axis=1)/np.invert(self.mask).sum(axis=1)
        return Array1D.no_mask(values=binned_array, pixel_scales=self.pixel_scale)

    def zoomed_around_mask(self, buffer: int = 1) -> "Array2D":
        """
        Extract the 2D region of an array corresponding to the rectangle encompassing all unmasked values.

        This is used to extract and visualize only the region of an image that is used in an analysis.

        Parameters
        ----------
        buffer
            The number pixels around the extracted array used as a buffer.
        """

        extracted_array_2d = array_2d_util.extracted_array_2d_from(
            array_2d=np.array(self.native),
            y0=self.mask.zoom_region[0] - buffer,
            y1=self.mask.zoom_region[1] + buffer,
            x0=self.mask.zoom_region[2] - buffer,
            x1=self.mask.zoom_region[3] + buffer,
        )

        mask = Mask2D.all_false(
            shape_native=extracted_array_2d.shape,
            pixel_scales=self.pixel_scales,
            origin=self.mask.mask_centre,
        )

        array = array_2d_util.convert_array_2d(
            array_2d=extracted_array_2d, mask_2d=mask
        )

        return Array2D(values=array, mask=mask, header=self.header)

    def extent_of_zoomed_array(self, buffer: int = 1) -> np.ndarray:
        """
        For an extracted zoomed array computed from the method *zoomed_around_mask* compute its extent in scaled
        coordinates.

        The extent of the grid in scaled units returned as an ``ndarray`` of the form [x_min, x_max, y_min, y_max].

        This is used visualize zoomed and extracted arrays via the imshow() method.

        Parameters
        ----------
        buffer
            The number pixels around the extracted array used as a buffer.
        """
        extracted_array_2d = array_2d_util.extracted_array_2d_from(
            array_2d=np.array(self.native),
            y0=self.mask.zoom_region[0] - buffer,
            y1=self.mask.zoom_region[1] + buffer,
            x0=self.mask.zoom_region[2] - buffer,
            x1=self.mask.zoom_region[3] + buffer,
        )

        mask = Mask2D.all_false(
            shape_native=extracted_array_2d.shape,
            pixel_scales=self.pixel_scales,
            origin=self.mask.mask_centre,
        )

        return mask.geometry.extent

    def resized_from(
        self, new_shape: Tuple[int, int], mask_pad_value: int = 0.0
    ) -> "Array2D":
        """
        Resize the array around its centre to a new input shape.

        If a new_shape dimension is smaller than the current dimension, the data at the edges is trimmed and removed.
        If it is larger, the data is padded with zeros.

        If the array has even sized dimensions, the central pixel around which data is trimmed / padded is chosen as
        the top-left pixel of the central quadrant of pixels.

        Parameters
        ----------
        new_shape
            The new 2D shape of the array.
        """

        resized_array_2d = array_2d_util.resized_array_2d_from(
            array_2d=np.array(self.native), resized_shape=new_shape
        )

        resized_mask = self.mask.derive_mask.resized_from(
            new_shape=new_shape, pad_value=mask_pad_value
        )

        array = array_2d_util.convert_array_2d(
            array_2d=resized_array_2d, mask_2d=resized_mask
        )

        return Array2D(
            values=array,
            mask=resized_mask,
            header=self.header,
            store_native=self.store_native,
        )

    def padded_before_convolution_from(
        self, kernel_shape: Tuple[int, int], mask_pad_value: int = 0.0
    ) -> "Array2D":
        """
        When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the array can be padded, where it is 'buffed' such that it includes all
        pixels whose signal will be convolved into the unmasked pixels given the 2D kernel shape. The values of
        these pixels are zeros.

        Parameters
        ----------
        kernel_shape
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        new_shape = (
            self.shape_native[0] + (kernel_shape[0] - 1),
            self.shape_native[1] + (kernel_shape[1] - 1),
        )
        return self.resized_from(new_shape=new_shape, mask_pad_value=mask_pad_value)

    def trimmed_after_convolution_from(
        self, kernel_shape: Tuple[int, int]
    ) -> "Array2D":
        """
        When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the array can be padded, a padded array can be computed via the method
        *padded_before_convolution_from*. This function trims the array back to its original shape, after the padded array
        has been used for computational.

        Parameters
        ----------
        kernel_shape
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        psf_cut_y = int(np.ceil(kernel_shape[0] / 2)) - 1
        psf_cut_x = int(np.ceil(kernel_shape[1] / 2)) - 1
        array_y = int(self.mask.shape[0])
        array_x = int(self.mask.shape[1])
        trimmed_array_2d = self.native[
            psf_cut_y : array_y - psf_cut_y, psf_cut_x : array_x - psf_cut_x
        ]

        resized_mask = self.mask.derive_mask.resized_from(
            new_shape=trimmed_array_2d.shape
        )

        array = array_2d_util.convert_array_2d(
            array_2d=trimmed_array_2d, mask_2d=resized_mask
        )

        return Array2D(
            values=array,
            mask=resized_mask,
            header=self.header,
            store_native=self.store_native,
        )

    @property
    def hdu_for_output(self) -> fits.PrimaryHDU:
        """
        The array as an HDU object, which can be output to a .fits file.

        The header of the HDU is used to store the `pixel_scale` of the array, which is used by the `Array2D.from_hdu`.

        This method is used in other projects (E.g. PyAutoGalaxy, PyAutoLens) to conveniently output the array to .fits
        files.

        Returns
        -------
        The HDU containing the data and its header which can then be written to .fits.
        """
        return array_2d_util.hdu_for_output_from(
            array_2d=np.array(self.native), header_dict=self.pixel_scale_header
        )

    def output_to_fits(self, file_path: Union[Path, str], overwrite: bool = False):
        """
        Output the array to a .fits file.

        The `pixel_scale` is stored in the header as `PIXSCALE`, which is used by the `Array2D.from_primary_hdu`
        method.

        Parameters
        ----------
        file_path
            The output path of the file, including the filename and the `.fits` extension e.g. '/path/to/filename.fits'
        overwrite
            If a file already exists at the path, if overwrite=True it is overwritten else an error is raised.
        """
        array_2d_util.numpy_array_2d_to_fits(
            array_2d=np.array(self.native),
            file_path=file_path,
            overwrite=overwrite,
            header_dict=self.pixel_scale_header,
        )


class Array2D(AbstractArray2D):
    @classmethod
    def no_mask(
        cls,
        values: Union[np.ndarray, List, AbstractArray2D],
        pixel_scales: ty.PixelScales,
        shape_native: Tuple[int, int] = None,
        origin: Tuple[float, float] = (0.0, 0.0),
        header: Optional[Header] = None,
    ) -> "Array2D":
        """
        Returns an ``Array2D`` from an array via inputs in its slim or native data representation.

        From a ``slim`` 1D input the method cannot determine the 2D shape of the array and its mask. The
        ``shape_native`` must therefore also be input into this method. The mask is setup as a unmasked `Mask2D` of
        ``shape_native``.

        For a full description of ``Array2D`` objects, including a description of the ``slim`` and ``native`` attribute
        used by the API, see
        the :meth:`Array2D class API documentation <autoarray.structures.arrays.uniform_2d.AbstractArray2D.__new__>`.

        Parameters
        ----------
        values
            The values of the array input with shape [total_unmasked_pixels] or
            shape [total_y_pixels, total_x_pixels].
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        shape_native
            The 2D shape of the array in its ``native`` format, and its 2D mask (only required if input shape is
            in ``slim`` format).
        origin
            The (y,x) scaled units origin of the mask's coordinate system.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            # Make Array2D from input list, native format
            # (This array has shape_native=(2,2)).

            array_2d = aa.Array2D.manual(
                array=np.array([[1.0, 2.0], [3.0, 4.0]]),
                pixel_scales=1.0.
            )
        """

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        values = array_2d_util.convert_array(array=values)

        if len(values.shape) == 1:
            if shape_native is None:
                raise exc.ArrayException(
                    f"""
                    The input array is not in its native shape (an ndarray / list of shape [total_y_pixels, total_x_pixels])
                    and the shape_native parameter has not been input the Array2D function.

                    Either change the input array to be its native shape or input its shape_native input the function.

                    The shape of the input array is {values.shape}
                    """
                )

            if shape_native and len(shape_native) != 2:
                raise exc.ArrayException(
                    """
                    The input shape_native parameter is not a tuple of type (int, int)
                    """
                )

        else:
            shape_native = (
                int(values.shape[0]),
                int(values.shape[1]),
            )

        mask = Mask2D.all_false(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

        return Array2D(values=values, mask=mask, header=header)

    @classmethod
    def full(
        cls,
        fill_value: float,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        header: Optional[Header] = None,
    ) -> "Array2D":
        """
        Returns an ``Array2D`` where all values are filled with an input fill value, analogous to ``np.full()``.

        For a full description of ``Array2D`` objects, including a description of the ``slim`` and ``native`` attribute
        used by the API, see
        the :meth:`Array2D class API documentation <autoarray.structures.arrays.uniform_2d.AbstractArray2D.__new__>`.

        From this input the method cannot determine the 2D shape of the array and its mask. The
        ``shape_native`` must therefore also be input into this method. The mask is setup as a unmasked `Mask2D` of
        ``shape_native``.

        Parameters
        ----------
        fill_value
            The value all array elements are filled with.
        shape_native
            The 2D shape of the array in its ``native`` format, and its 2D mask.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            array_2d = aa.Array2D.full(
                fill_value=2.0,
                shape_native=(2, 2),
                pixel_scales=1.0,
            )
        """
        return cls.no_mask(
            values=np.full(fill_value=fill_value, shape=shape_native),
            pixel_scales=pixel_scales,
            origin=origin,
            header=header,
        )

    @classmethod
    def ones(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        header: Header = None,
    ) -> "Array2D":
        """
        Returns an ``Array2D`` where all values are filled with ones, analogous to ``np.ones()``.

        For a full description of ``Array2D`` objects, including a description of the ``slim`` and ``native`` attribute
        used by the API, see
        the :meth:`Array2D class API documentation <autoarray.structures.arrays.uniform_2d.AbstractArray2D.__new__>`.

        From this input the method cannot determine the 2D shape of the array and its mask. The
        ``shape_native`` must therefore also be input into this method. The mask is setup as a unmasked `Mask2D` of
        ``shape_native``.

        Parameters
        ----------
        shape_native
            The 2D shape of the array in its ``native`` format, and its 2D mask.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            array_2d = aa.Array2D.ones(
                shape_native=(2, 2),
                pixel_scales=1.0,
            )
        """
        return cls.full(
            fill_value=1.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
            header=header,
        )

    @classmethod
    def zeros(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        header: Header = None,
    ) -> "Array2D":
        """
        Returns an ``Array2D`` where all values are filled with zeros, analogous to ``np.zeros()``.

        For a full description of ``Array2D`` objects, including a description of the ``slim`` and ``native`` attribute
        used by the API, see
        the :meth:`Array2D class API documentation <autoarray.structures.arrays.uniform_2d.AbstractArray2D.__new__>`.

        From this input the method cannot determine the 2D shape of the array and its mask. The
        ``shape_native`` must therefore also be input into this method. The mask is setup as a unmasked `Mask2D` of
        ``shape_native``.

        Parameters
        ----------
        shape_native
            The 2D shape of the array in its ``native`` format, and its 2D mask.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) scaled units origin of the mask's coordinate system.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            array_2d = aa.Array2D.zeros(
                shape_native=(2, 2),
                pixel_scales=1.0,
            )
        """
        return cls.full(
            fill_value=0.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
            header=header,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: Union[Path, str],
        pixel_scales: Optional[ty.PixelScales],
        hdu: int = 0,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "Array2D":
        """
        Returns an ``Array2D`` by loading the array values from a .fits file.

        For a full description of ``Array2D`` objects, including a description of the ``slim`` and ``native`` attribute
        used by the API, see
        the :meth:`Array2D class API documentation <autoarray.structures.arrays.uniform_2d.AbstractArray2D.__new__>`.

        Parameters
        ----------
        file_path
            The path the file is loaded from, including the filename and the `.fits` extension,
            e.g. '/path/to/filename.fits'
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        hdu
            The Header-Data Unit of the .fits file the array data is loaded from.
        origin
            The (y,x) scaled units origin of the coordinate system.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            array_2d = aa.Array2D.from_fits(
                file_path="path/to/file.fits",
                hdu=0,
                pixel_scales=1.0,
            )
        """
        array_2d = array_2d_util.numpy_array_2d_via_fits_from(
            file_path=file_path, hdu=hdu
        )

        header_sci_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=0)
        header_hdu_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=hdu)

        return cls.no_mask(
            values=array_2d,
            pixel_scales=pixel_scales,
            origin=origin,
            header=Header(header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj),
        )

    @classmethod
    def from_primary_hdu(
        cls,
        primary_hdu: fits.PrimaryHDU,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "Array2D":
        """
        Returns an ``Array2D`` by from a `PrimaryHDU` object which has been loaded via `astropy.fits`

        This assumes that the `header` of the `PrimaryHDU` contains an entry named `PIXSCALE` which gives the
        pixel-scale of the array.

        For a full description of ``Array2D`` objects, including a description of the ``slim`` and ``native`` attribute
        used by the API, see
        the :meth:`Array2D class API documentation <autoarray.structures.arrays.uniform_2d.AbstractArray2D.__new__>`.

        Parameters
        ----------
        primary_hdu
            The `PrimaryHDU` object which has already been loaded from a .fits file via `astropy.fits` and contains
            the array data and the pixel-scale in the header with an entry named `PIXSCALE`.
        origin
            The (y,x) scaled units origin of the coordinate system.

        Examples
        --------

        .. code-block:: python

            from astropy.io import fits
            import autoarray as aa

            primary_hdu = fits.open("path/to/file.fits")

            array_2d = aa.Array2D.from_primary_hdu(
                primary_hdu=primary_hdu,
            )
        """

        return cls.no_mask(
            values=cls.flip_hdu_for_ds9(primary_hdu.data.astype("float")),
            pixel_scales=primary_hdu.header["PIXSCALE"],
            origin=origin,
            header=Header(header_sci_obj=primary_hdu.header),
        )

    @classmethod
    def from_yx_and_values(
        cls,
        y: Union[np.ndarray, List],
        x: Union[np.ndarray, List],
        values: Union[np.ndarray, List],
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        header: Header = None,
    ) -> "Array2D":
        """
        Returns an ``Array2D`` by inputting the y and x pixel values where the array is filled and the values that
        fill it.

        For a full description of ``Array2D`` objects, including a description of the ``slim`` and ``native`` attribute
        used by the API, see
        the :meth:`Array2D class API documentation <autoarray.structures.arrays.uniform_2d.AbstractArray2D.__new__>`.

        Parameters
        ----------
        y
            The y pixel indexes where value are input, with shape [total_unmasked_pixels].
        x
            The x pixel indexes where value are input, with shape [total_unmasked_pixels].
        values or list
            The values which are used to fill in the array, with shape [total_unmasked_pixel].
        shape_native
            The 2D shape of the array in its ``native`` format, and its 2D mask.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The origin of the grid's mask.

        Examples
        --------
        .. code-block:: python

            import autoarray as aa

            array_2d = aa.Array2D.from_yx_and_values(
                y=np.array([0.5, 0.5, -0.5, -0.5]),
                x=np.array([-0.5, 0.5, -0.5, 0.5]),
                values=np.array([1.0, 2.0, 3.0, 4.0]),
                shape_native=(2, 2),
                pixel_scales=1.0,
            )
        """
        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        from autoarray.structures.grids.uniform_2d import Grid2D

        grid = Grid2D.from_yx_1d(
            y=y, x=x, shape_native=shape_native, pixel_scales=pixel_scales,
        )

        grid_pixels = geometry_util.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=np.array(grid.slim),
            shape_native=shape_native,
            pixel_scales=pixel_scales,
        )

        array_1d = np.array(
            [values[int(grid_pixels[i])] for i in range(grid_pixels.shape[0])]
        )

        return cls.no_mask(
            values=array_1d,
            pixel_scales=pixel_scales,
            shape_native=shape_native,
            header=header,
        )
