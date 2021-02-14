import numpy as np
from autoarray.structures import abstract_structure
from autoarray.mask import mask_1d as msk
from autoarray.structures.lines import array_1d_util
from autoarray.geometry import geometry_util


def convert_line(line):

    if type(line) is list:
        line = np.asarray(line)

    return line


class AbstractLine1D(abstract_structure.AbstractStructure1D):
    def __new__(cls, line, mask, *args, **kwargs):

        obj = line.view(cls)
        obj.mask = mask

        return obj

    @property
    def native(self):

        return array_1d_util.line_1d_native_from(
            line_1d_slim=self, mask_1d=self.mask, sub_size=self.sub_size
        )

    @property
    def slim(self):
        return self


class Line1D(AbstractLine1D):
    @classmethod
    def manual_slim(cls, line, pixel_scales, sub_size=1, origin=(0.0,)):
        """
        Create a Line (see `Line.__new__`) by inputting the line values in 1D, for example:

        line=np.array([1.0, 2.0 3.0, 4.0])

        line=[1.0, 2.0, 3.0, 4.0]

        Parameters
        ----------
        line : np.ndarray or list
            The values of the line input as an ndarray of shape [total_unmasked_pixels*sub_size] or a list.
        pixel_scales: float
            The scaled units to pixel units conversion factor of the line data coordinates (e.g. the x-axis).
        sub_size : int
            The size of each unmasked pixels sub-gridded line.
        origin : (float, float)
            The origin of the line's mask.
        """

        line = convert_line(line)

        pixel_scales = geometry_util.convert_pixel_scales_1d(pixel_scales=pixel_scales)

        mask = msk.Mask1D.unmasked(
            shape_slim=line.shape[0],
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return Line1D(line=line, mask=mask)

    @classmethod
    def manual_native(cls, line, pixel_scales, sub_size=1, origin=(0.0,)):
        """
        Create a Line (see `Line.__new__`) by inputting the line values in 1D, for example:

        line=np.array([1.0, 2.0 3.0, 4.0])

        line=[1.0, 2.0, 3.0, 4.0]

        Parameters
        ----------
        line : np.ndarray or list
            The values of the line input as an ndarray of shape [total_unmasked_pixels*sub_size] or a list.
        pixel_scales: float
            The scaled units to pixel units conversion factor of the line data coordinates (e.g. the x-axis).
        sub_size : int
            The size of each unmasked pixels sub-gridded line.
        origin : (float, float)
            The origin of the line's mask.
        """
        return cls.manual_slim(
            line=line, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

    @classmethod
    def manual_mask(cls, line, mask):
        """
        Create a Line (see `Line.__new__`) by inputting the native line values in 1D and including the mask that is
        applied to them, for example:

        mask=np.array([True, False, False, True, False, False])

        line=np.array([100.0, 1.0, 2.0, 100.0, 3.0 4.0])
        line=[100.0, 1.0, 2.0, 100.0, 3.0, 4.0]

        Parameters
        ----------
        line : np.ndarray or list
            The values of the line input as an ndarray of shape [total_unmasked_pixels*sub_size] or a list.
        pixel_scales: float
            The scaled units to pixel units conversion factor of the line data coordinates (e.g. the x-axis).
        sub_size : int
            The size of each unmasked pixels sub-gridded line.
        origin : (float, float)
            The origin of the line's mask.
        """

        line = convert_line(line)

        line = array_1d_util.line_1d_slim_from(
            line_1d_native=line, mask_1d=mask, sub_size=mask.sub_size
        )

        return Line1D(line=line, mask=mask)
