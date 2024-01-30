from astropy.io import fits
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, List

from autoarray.structures.header import Header

from autoarray.structures.abstract_structure import Structure
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.mask.mask_1d import Mask1D

from autoarray.structures.arrays import array_1d_util
from autoarray.structures.arrays import array_2d_util
from autoarray.geometry import geometry_util
from autoarray import type as ty


class Array1D(Structure):
    def structure_2d_list_from(self, result_list: list) -> List["Structure"]:
        raise NotImplementedError()

    def structure_2d_from(self, result: np.ndarray) -> "Structure":
        raise NotImplementedError()

    def trimmed_after_convolution_from(self, kernel_shape) -> "Structure":
        raise NotImplementedError()

    def __init__(
        self,
        values: Union[np.ndarray, List],
        mask: Mask1D,
        header: Optional[Header] = None,
        store_native: bool = False,
    ):
        values = array_1d_util.convert_array_1d(
            array_1d=values, mask_1d=mask, store_native=store_native,
        )

        self.mask = mask
        self.header = header

        super().__init__(values)

    @classmethod
    def no_mask(
        cls,
        values: Union[np.ndarray, Tuple[float], List[float]],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float] = (0.0,),
        header: Optional[Header] = None,
    ) -> "Array1D":
        """
        Create a Array1D (see `Array1D.__new__`) by inputting the array values in 1D

        Parameters
        ----------
        values
            The values of the array input as an ndarray of shape [total_unmasked_pixels*sub_size] or a list.
        pixel_scales
            The scaled units to pixel units conversion factor of the array data coordinates (e.g. the x-axis).
        sub_size
            The size of each unmasked pixels sub-grid.
        origin
            The origin of the 1D array's mask.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            # Make Array1D from input np.ndarray.

            array_1d = aa.Array1D.no_mask(values=np.array([1.0, 2.0, 3.0, 4.0]), pixel_scales=1.0)

            # Make Array2D from input list.

            array_1d = aa.Array1D.no_mask(values=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

            # Print array's slim (masked 1D data representation) and
            # native (masked 1D data representation)

            print(array_1d.slim)
            print(array_1d.native)
        """

        values = array_2d_util.convert_array(values)

        pixel_scales = geometry_util.convert_pixel_scales_1d(pixel_scales=pixel_scales)

        mask = Mask1D.all_false(
            shape_slim=values.shape[0] // sub_size,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return Array1D(values=values, mask=mask, header=header)

    @classmethod
    def full(
        cls,
        fill_value: float,
        shape_native: Union[int, Tuple[int]],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float] = (0.0,),
        header: Optional[Header] = None,
    ) -> "Array1D":
        """
        Create an `Array1D` (see `Array1D.__new__`) where all values are filled with an input fill value,
        analogous to the method np.full().

        From 1D input the method cannot determine the 1D shape of the array and its mask, thus the `shape_native` must
        be input into this method. The mask is setup as a unmasked `Mask1D` of size `shape_native`.

        Parameters
        ----------
        fill_value
            The value all array elements are filled with.
        shape_native : Tuple[int]
            The 1D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float,) structure.
        sub_size
            The size (sub_size) of each unmasked pixels sub-array.
        origin : (float,)
            The (x) scaled units origin of the mask's coordinate system.
        """
        shape_native = geometry_util.convert_shape_native_1d(shape_native=shape_native)

        if sub_size is not None:
            shape_native = (shape_native[0] * sub_size,)

        return cls.no_mask(
            values=np.full(fill_value=fill_value, shape=shape_native[0]),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            header=header,
        )

    @classmethod
    def zeros(
        cls,
        shape_native: Union[int, Tuple[int]],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float] = (0.0,),
        header: Optional[Header] = None,
    ) -> "Array1D":
        """
        Create an `Array1D` (see `Array1D.__new__`) where all values are filled with zeros, analogous to the
        method np.zeros().

        From 1D input the method cannot determine the 1D shape of the array and its mask, thus the `shape_native` must
        be input into this method. The mask is setup as a unmasked `Mask1D` of size `shape_native`.

        Parameters
        ----------
        shape_native : Tuple[int]
            The 1D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float,) structure.
        sub_size
            The size (sub_size) of each unmasked pixels sub-array.
        origin : (float,)
            The (x) scaled units origin of the mask's coordinate system.
        """
        return cls.full(
            fill_value=0.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            header=header,
        )

    @classmethod
    def ones(
        cls,
        shape_native: Union[int, Tuple[int]],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float] = (0.0,),
        header: Optional[Header] = None,
    ) -> "Array1D":
        """
        Create an `Array1D` (see `Array1D.__new__`) where all values are filled with ones, analogous to the
        method np.ones().

        From 1D input the method cannot determine the 1D shape of the array and its mask, thus the `shape_native` must
        be input into this method. The mask is setup as a unmasked `Mask1D` of size `shape_native`.

        Parameters
        ----------
        shape_native : Tuple[int]
            The 1D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float,) structure.
        sub_size
            The size (sub_size) of each unmasked pixels sub-array.
        origin : (float,)
            The (x) scaled units origin of the mask's coordinate system.
        """
        return cls.full(
            fill_value=1.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            header=header,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: Union[Path, str],
        pixel_scales: ty.PixelScales,
        hdu: int = 0,
        sub_size: int = 1,
        origin: Tuple[float] = (0.0, 0.0),
    ) -> "Array1D":
        """
        Create an Array1D (see `Array1D.__new__`) by loading the array values from a .fits file.

        Parameters
        ----------
        file_path
            The path the file is loaded from, including the filename and the `.fits` extension,
            e.g. '/path/to/filename.fits'
        hdu
            The Header-Data Unit of the .fits file the array data is loaded from.
        pixel_scales
            The (x,) scaled units to pixel units conversion factors of every pixel. If this is input as a float,
            it is converted to a (float,) structure.
        sub_size
            The sub-size of each unmasked pixels sub-array.
        origin
            The (x,) scaled units origin of the coordinate system.
        """
        array_1d = array_1d_util.numpy_array_1d_via_fits_from(
            file_path=file_path, hdu=hdu
        )

        header_sci_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=0)
        header_hdu_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=hdu)

        return cls.no_mask(
            values=array_1d.astype(
                "float64"
            ),  # Have to do this due to typing issues in 1D with astorpy fits.
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            header=Header(header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj),
        )

    @classmethod
    def from_primary_hdu(
        cls,
        primary_hdu: fits.PrimaryHDU,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "Array1D":
        """
        Returns an ``Array1D`` by from a `PrimaryHDU` object which has been loaded via `astropy.fits`

        This assumes that the `header` of the `PrimaryHDU` contains an entry named `PIXSCALE` which gives the
        pixel-scale of the array.

        For a full description of ``Array1D`` objects, including a description of the ``slim`` and ``native`` attribute
        used by the API, see
        the :meth:`Array1D class API documentation <autoarray.structures.arrays.uniform_1d.AbstractArray1D.__new__>`.

        Parameters
        ----------
        primary_hdu
            The `PrimaryHDU` object which has already been loaded from a .fits file via `astropy.fits` and contains
            the array data and the pixel-scale in the header with an entry named `PIXSCALE`.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin
            The (y,x) scaled units origin of the coordinate system.

        Examples
        --------

        .. code-block:: python

            from astropy.io import fits
            import autoarray as aa

            # Make Array1D with sub_size 1.

            primary_hdu = fits.open("path/to/file.fits")

            array_1d = aa.Array1D.from_primary_hdu(
                primary_hdu=primary_hdu,
                sub_size=1
            )

        .. code-block:: python

            import autoarray as aa

            # Make Array1D with sub_size 2.
            # (It is uncommon that a sub-gridded array would be loaded from
            # a .fits, but the API support its).

             primary_hdu = fits.open("path/to/file.fits")

            array_1d = aa.Array1D.from_primary_hdu(
                primary_hdu=primary_hdu,
                sub_size=2
            )
        """
        return cls.no_mask(
            values=primary_hdu.data.astype("float"),
            pixel_scales=primary_hdu.header["PIXSCALE"],
            sub_size=sub_size,
            origin=origin,
            header=Header(header_sci_obj=primary_hdu.header),
        )

    @property
    def slim(self) -> "Array1D":
        """
        Return an `Array1D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels * sub_size].

        If it is already stored in its `slim` representation  it is returned as it is. If not, it is  mapped from
        `native` to `slim` and returned as a new `Array1D`.
        """
        return Array1D(values=self, mask=self.mask)

    @property
    def native(self) -> "Array1D":
        """
        Return an `Array1D` where the data is stored in its `native` representation, which is an ndarray of shape
        [total_pixels * sub_size].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Array1D`.
        """
        return Array1D(values=self, mask=self.mask, store_native=True)

    @property
    def readout_offsets(self) -> Tuple[float]:
        if self.header is not None:
            if self.header.readout_offsets is not None:
                return self.header.readout_offsets
        return (0,)

    @property
    def grid_radial(self) -> Grid1D:
        return Grid1D.uniform_from_zero(
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
        )

    @property
    def hdu_for_output(self) -> fits.PrimaryHDU:
        """
        The array as an HDU object, which can be output to a .fits file.

        The header of the HDU is used to store the `pixel_scale` of the array, which is used by the `Array1D.from_hdu`.

        This method is used in other projects (E.g. PyAutoGalaxy, PyAutoLens) to conveniently output the array to .fits
        files.

        Returns
        -------
        The HDU containing the data and its header which can then be written to .fits.
        """
        return array_2d_util.hdu_for_output_from(
            array_2d=self.native, header_dict=self.pixel_scale_header
        )

    def output_to_fits(self, file_path: Union[Path, str], overwrite: bool = False):
        """
        Output the array to a .fits file.

        Parameters
        ----------
        file_path
            The output path of the file, including the filename and the `.fits` extension e.g. '/path/to/filename.fits'
        overwrite
            If a file already exists at the path, if overwrite=True it is overwritten else an error is raised.
        """
        array_1d_util.numpy_array_1d_to_fits(
            array_1d=self.native,
            file_path=file_path,
            overwrite=overwrite,
            header_dict=self.pixel_scale_header,
        )
