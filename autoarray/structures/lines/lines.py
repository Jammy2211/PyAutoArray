import numpy as np
from autoarray.structures import abstract_structure
from autoarray.mask import mask_1d as msk
from autoarray.util import geometry_util


def convert_line(line):

    if type(line) is list:
        line = np.asarray(line)

    return line


class AbstractLine(abstract_structure.AbstractStructure1D):
    def __new__(cls, line, mask, *args, **kwargs):

        obj = line.view(cls)
        obj.mask = mask

        return obj

    @property
    def in_1d(self):
        return self


class Line(AbstractLine):
    @classmethod
    def manual_1d(cls, line, pixel_scales, sub_size=1, origin=(0.0,)):
        """Create a Line (see `Line.__new__`) by inputting the line values in 1D, for example:

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
            shape_1d=line.shape[0],
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return Line(line=line, mask=mask)
