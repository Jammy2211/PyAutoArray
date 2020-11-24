import logging

import numpy as np

from autoarray import exc
from autoarray.mask import abstract_mask
from autoarray.util import array_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractMask1d(abstract_mask.AbstractMask):
    def __new__(
        cls,
        mask: np.ndarray,
        pixel_scales: (float,),
        sub_size: int = 1,
        origin: (float,) = (0.0,),
        *args,
        **kwargs
    ):
        """A 1D mask, representing 1D data on a uniform line of pixels with equal spacing.

        When applied to 1D data it extracts or masks the unmasked image pixels corresponding to mask entries that are
        `False` or 0).

        The mask also defines the geometry of the 1D data structure it is paired to, for example how every pixel
        coordinate on the 1D line of data converts to physical units via the ``pixel_scales`` and ``origin``
        parameters and a sub-grid which is used for performing calculations via super-sampling.

        Parameters
        ----------
        mask: np.ndarray
            The ``ndarray`` of shape [total_pixels] containing the ``bool'``s representing the mask, where
            `False` signifies an entry is unmasked and used in calculations..
        pixel_scales: (float, float)
            The scaled units to pixel units conversion factor of each pixel.
        origin : (float, float)
            The x origin of the mask's coordinate system in scaled units.
        centre : (float, float)
            The x centre of the mask in scaled units provided it is a standard geometric shape (e.g. a circle).
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
            self.origin = (0.0,)

    def output_to_fits(self, file_path: str, overwrite: bool = False):
        """
        Write the 1D mask to a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the file that is output, including the file name and ``.fits`` extension.
        overwrite : bool
            If `True` and a file already exists with the input file_path the .fits file is overwritten. If `False`,
            an error is raised.

        Returns
        -------
        None

        Examples
        --------
        mask = Mask1D(mask=np.full(shape=(5,), fill_value=False))
        mask.output_to_fits(file_path='/path/to/file/filename.fits', overwrite=True)
        """
        array_util.numpy_array_1d_to_fits(
            array_2d=self.astype("float"), file_path=file_path, overwrite=overwrite
        )


class Mask1D(AbstractMask1d):
    @classmethod
    def manual(cls, mask, pixel_scales, sub_size=1, origin=(0.0,), invert=False):

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
    def unmasked(cls, shape_1d, pixel_scales, sub_size=1, origin=(0.0,), invert=False):
        """Setup a mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : float or (float, float)
            The scaled units to pixel units conversion factor of each pixel.
        """
        return cls.manual(
            mask=np.full(shape=shape_1d, fill_value=False),
            pixel_scales=pixel_scales,
            origin=origin,
            sub_size=sub_size,
            invert=invert,
        )

    @classmethod
    def from_fits(cls, file_path, pixel_scales, sub_size=1, hdu=0, origin=(0.0,)):
        """
        Loads the image from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales : float or (float, float)
            The scaled units to pixel units conversion factor of each pixel.
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
