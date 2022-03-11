import numpy as np
from typing import Optional

from autoarray.structures.header import Header

from autoarray.structures.abstract_structure import Structure1D
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.mask.mask_1d import Mask1D

from autoarray.structures.arrays import array_1d_util
from autoarray.structures.arrays import array_2d_util
from autoarray.geometry import geometry_util

from typing import Union, Tuple, List


class Array1D(Structure1D):
    def __new__(
        cls,
        array: np.ndarray,
        mask: np.ndarray,
        header: Optional[Header] = None,
        *args,
        **kwargs
    ):

        obj = array.view(cls)
        obj.mask = mask
        obj.header = header

        return obj

    @classmethod
    def manual_slim(
        cls,
        array: Union[np.ndarray, Tuple[float], List[float]],
        pixel_scales: Union[float, Tuple[float]],
        sub_size: int = 1,
        origin: Tuple[float] = (0.0,),
        header: Optional[Header] = None,
    ) -> "Array1D":
        """
        Create a Line (see `Line.__new__`) by inputting the line values in 1D, for example:

        line=np.array([1.0, 2.0 3.0, 4.0])

        line=[1.0, 2.0, 3.0, 4.0]

        Parameters
        ----------
        array or list
            The values of the line input as an ndarray of shape [total_unmasked_pixels*sub_size] or a list.
        pixel_scales: float
            The scaled units to pixel units conversion factor of the line data coordinates (e.g. the x-axis).
        sub_size
            The size of each unmasked pixels sub-gridded line.
        origin : (float, )
            The origin of the line's mask.
        """

        array = array_2d_util.convert_array(array)

        pixel_scales = geometry_util.convert_pixel_scales_1d(pixel_scales=pixel_scales)

        mask = Mask1D.unmasked(
            shape_slim=array.shape[0] // sub_size,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return Array1D(array=array, mask=mask, header=header)

    @classmethod
    def manual_native(
        cls,
        array: Union[np.ndarray, Tuple[float], List[float]],
        pixel_scales: Union[float, Tuple[float]],
        sub_size: int = 1,
        origin: Tuple[float] = (0.0,),
        header: Optional[Header] = None,
    ) -> "Array1D":
        """
        Create a Line (see `Line.__new__`) by inputting the line values in 1D, for example:

        line=np.array([1.0, 2.0 3.0, 4.0])

        line=[1.0, 2.0, 3.0, 4.0]

        Parameters
        ----------
        array or list
            The values of the line input as an ndarray of shape [total_unmasked_pixels*sub_size] or a list.
        pixel_scales
            The scaled units to pixel units conversion factor of the line data coordinates (e.g. the x-axis).
        sub_size
            The size of each unmasked pixels sub-gridded line.
        origin : (float, )
            The origin of the line's mask.
        """
        return cls.manual_slim(
            array=array,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            header=header,
        )

    @classmethod
    def manual_mask(
        cls,
        array: Union[np.ndarray, Tuple[float], List[float]],
        mask: np.ndarray,
        header: Optional[Header] = None,
    ) -> "Array1D":
        """
        Create a Line (see `Line.__new__`) by inputting the native line values in 1D and including the mask that is
        applied to them, for example:

        mask=np.array([True, False, False, True, False, False])

        line=np.array([100.0, 1.0, 2.0, 100.0, 3.0 4.0])
        line=[100.0, 1.0, 2.0, 100.0, 3.0, 4.0]

        Parameters
        ----------
        array or list
            The values of the line input as an ndarray of shape [total_unmasked_pixels*sub_size] or a list.
        pixel_scales: float
            The scaled units to pixel units conversion factor of the line data coordinates (e.g. the x-axis).
        sub_size
            The size of each unmasked pixels sub-gridded line.
        origin
            The origin of the line's mask.
        """

        array = array_2d_util.convert_array(array)

        array = array_1d_util.array_1d_slim_from(
            array_1d_native=array, mask_1d=mask, sub_size=mask.sub_size
        )

        return Array1D(array=array, mask=mask, header=header)

    @classmethod
    def full(
        cls,
        fill_value: float,
        shape_native: Union[int, Tuple[int]],
        pixel_scales: Union[float, Tuple[float]],
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
        pixel_scales: (float, ) or float
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

        return cls.manual_native(
            array=np.full(fill_value=fill_value, shape=shape_native[0]),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            header=header,
        )

    @classmethod
    def zeros(
        cls,
        shape_native: Union[int, Tuple[int]],
        pixel_scales: Union[float, Tuple[float]],
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
        pixel_scales: (float, ) or float
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
        pixel_scales: Union[float, Tuple[float]],
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
        pixel_scales: (float, ) or float
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
        file_path: str,
        pixel_scales: Union[float, Tuple[float]],
        hdu: int = 0,
        sub_size: int = 1,
        origin: Tuple[float] = (0.0, 0.0),
    ) -> "Array1D":
        """
        Create an Array1D (see `Array1D.__new__`) by loading the array values from a .fits file.

        Parameters
        ----------
        file_path : str
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

        return cls.manual_native(
            array=array_1d.astype(
                "float64"
            ),  # Have to do this due to typing issues in 1D with astorpy fits.
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            header=Header(header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj),
        )

    @property
    def slim(self) -> Union["Array1D"]:
        """
        Return an `Array1D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels * sub_size].

        If it is already stored in its `slim` representation  it is returned as it is. If not, it is  mapped from
        `native` to `slim` and returned as a new `Array1D`.
        """

        if self.shape[0] != self.mask.sub_shape_native[0]:
            return self

        array = array_1d_util.array_1d_slim_from(
            array_1d_native=self, mask_1d=self.mask, sub_size=self.mask.sub_size
        )

        return Array1D(array=array, mask=self.mask)

    @property
    def native(self) -> Union["Array1D"]:
        """
        Return an `Array1D` where the data is stored in its `native` representation, which is an ndarray of shape
        [total_pixels * sub_size].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Array1D`.
        """
        from autoarray.structures.arrays.uniform_1d import Array1D

        if self.shape[0] == self.mask.sub_shape_native[0]:
            return self

        array = array_1d_util.array_1d_native_from(
            array_1d_slim=self, mask_1d=self.mask, sub_size=self.sub_size
        )

        return Array1D(array=array, mask=self.mask)

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

    def output_to_fits(self, file_path: str, overwrite: bool = False):
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
            array_1d=self.native, file_path=file_path, overwrite=overwrite
        )
