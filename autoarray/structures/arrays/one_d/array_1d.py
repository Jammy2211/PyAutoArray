import numpy as np
from autoarray.structures.arrays import abstract_array
from autoarray.structures.arrays.one_d import abstract_array_1d
from autoarray.mask import mask_1d as msk
from autoarray.structures.grids.one_d import grid_1d
from autoarray.structures.arrays.one_d import array_1d_util
from autoarray.geometry import geometry_util


class Array1D(abstract_array_1d.AbstractArray1D):
    def __new__(cls, array, mask, *args, **kwargs):

        obj = array.view(cls)
        obj.mask = mask

        return obj

    @classmethod
    def manual_slim(cls, array, pixel_scales, sub_size=1, origin=(0.0,)):
        """
        Create a Line (see `Line.__new__`) by inputting the line values in 1D, for example:

        line=np.array([1.0, 2.0 3.0, 4.0])

        line=[1.0, 2.0, 3.0, 4.0]

        Parameters
        ----------
        array : np.ndarray or list
            The values of the line input as an ndarray of shape [total_unmasked_pixels*sub_size] or a list.
        pixel_scales: float
            The scaled units to pixel units conversion factor of the line data coordinates (e.g. the x-axis).
        sub_size : int
            The size of each unmasked pixels sub-gridded line.
        origin : (float, )
            The origin of the line's mask.
        """

        array = abstract_array.convert_array(array)

        pixel_scales = geometry_util.convert_pixel_scales_1d(pixel_scales=pixel_scales)

        mask = msk.Mask1D.unmasked(
            shape_slim=array.shape[0],
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return Array1D(array=array, mask=mask)

    @classmethod
    def manual_native(cls, array, pixel_scales, sub_size=1, origin=(0.0,)):
        """
        Create a Line (see `Line.__new__`) by inputting the line values in 1D, for example:

        line=np.array([1.0, 2.0 3.0, 4.0])

        line=[1.0, 2.0, 3.0, 4.0]

        Parameters
        ----------
        array : np.ndarray or list
            The values of the line input as an ndarray of shape [total_unmasked_pixels*sub_size] or a list.
        pixel_scales: float
            The scaled units to pixel units conversion factor of the line data coordinates (e.g. the x-axis).
        sub_size : int
            The size of each unmasked pixels sub-gridded line.
        origin : (float, )
            The origin of the line's mask.
        """
        return cls.manual_slim(
            array=array, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

    @classmethod
    def manual_mask(cls, array, mask):
        """
        Create a Line (see `Line.__new__`) by inputting the native line values in 1D and including the mask that is
        applied to them, for example:

        mask=np.array([True, False, False, True, False, False])

        line=np.array([100.0, 1.0, 2.0, 100.0, 3.0 4.0])
        line=[100.0, 1.0, 2.0, 100.0, 3.0, 4.0]

        Parameters
        ----------
        array : np.ndarray or list
            The values of the line input as an ndarray of shape [total_unmasked_pixels*sub_size] or a list.
        pixel_scales: float
            The scaled units to pixel units conversion factor of the line data coordinates (e.g. the x-axis).
        sub_size : int
            The size of each unmasked pixels sub-gridded line.
        origin : (float, float)
            The origin of the line's mask.
        """

        array = abstract_array.convert_array(array)

        array = array_1d_util.array_1d_slim_from(
            array_1d_native=array, mask_1d=mask, sub_size=mask.sub_size
        )

        return Array1D(array=array, mask=mask)

    @property
    def grid_radial(self):
        return grid_1d.Grid1D.uniform_from_zero(
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
        )
