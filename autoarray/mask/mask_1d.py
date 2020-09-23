import logging

import numpy as np

from autoarray import exc
from autoarray.mask import abstract_mask
from autoarray.util import array_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractMask1d(abstract_mask.AbstractMask):
    def __new__(cls, mask, pixel_scales=None, sub_size=1, origin=0.0, *args, **kwargs):
        """ A mask, which is applied to data to extract a set of unmasked image pixels (i.e. mask entry \
        is *False* or 0) which are then fitted in an analysis.

        The mask retains the pixel scale of the array and has a centre and origin.

        Parameters
        ----------
        mask: ndarray
            An array of bools representing the mask.
        pixel_scales: (float, float)
            The arc-second to pixel conversion factor of each pixel.
        origin : (float, float)
            The (y,x) arc-second origin of the mask's coordinate system.
        centre : (float, float)
            The (y,x) arc-second centre of the mask provided it is a standard geometric shape (e.g. a circle).
        """
        # noinspection PyArgumentList

        return abstract_mask.AbstractMask.__new__(
            cls=cls,
            mask=mask,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    def __array_finalize__(self, obj):

        super().__array_finalize__(obj=obj)

        if isinstance(obj, Mask1D):
            pass
        else:
            self.origin = 0.0

    def output_to_fits(self, file_path, overwrite=False):
        array_util.numpy_array_1d_to_fits(
            array_2d=self.astype("float"), file_path=file_path, overwrite=overwrite
        )


class Mask1D(AbstractMask1d):
    @classmethod
    def manual(cls, mask, pixel_scales=None, sub_size=1, origin=0.0, invert=False):

        if type(mask) is list:
            mask = np.asarray(mask).astype("bool")

        if invert:
            mask = np.invert(mask)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales,)

        if len(mask.shape) != 1:
            raise exc.MaskException("The input mask is not a one dimensional array")

        return Mask1D(
            mask=mask, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

    @classmethod
    def unmasked(
        cls, shape_1d, pixel_scales=None, sub_size=1, origin=0.0, invert=False
    ):
        """Setup a mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : float or (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """
        return cls.manual(
            mask=np.full(shape=shape_1d, fill_value=False),
            pixel_scales=pixel_scales,
            origin=origin,
            sub_size=sub_size,
            invert=invert,
        )

    @classmethod
    def from_fits(cls, file_path, pixel_scales, sub_size=1, hdu=0, origin=0.0):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales : float or (float, float)
            The arc-second to pixel conversion factor of each pixel.
        """

        mask = cls(
            array_util.numpy_array_1d_from_fits(file_path=file_path, hdu=hdu),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return mask

    def output_to_fits(self, file_path, overwrite=False):

        array_util.numpy_array_1d_to_fits(
            array_1d=self.astype("float"), file_path=file_path, overwrite=overwrite
        )

    @property
    def pixels_in_mask(self):
        return int(np.size(self) - np.sum(self))

    @property
    def is_all_false(self):
        return self.pixels_in_mask == self.shape_1d

    @property
    def shape_1d(self):
        return self.shape[0]

    @property
    def shape_1d_scaled(self):
        return float(self.pixel_scales * self.shape_1d)

    @property
    def scaled_maxima(self):
        return (self.shape_1d_scaled / 2.0) + self.origin

    @property
    def scaled_minima(self):
        return -(self.shape_1d_scaled / 2.0) + self.origin

    @property
    def extent(self):
        return np.asarray([self.scaled_minima, self.scaled_maxima])
