import logging

import numpy as np
from typing import List, Tuple, Union

from autoarray import exc
from autoarray.mask import abstract_mask
from autoarray.structures.grids.one_d import grid_1d
from autoarray.structures.grids.one_d import grid_1d_util
from autoarray.structures.arrays.two_d import array_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractMask1d(abstract_mask.AbstractMask):
    def __new__(
        cls,
        mask: np.ndarray,
        pixel_scales: Tuple[float,],
        sub_size: int = 1,
        origin: Tuple[float,] = (0.0,),
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
        pixel_scales: (float,)
            The scaled units to pixel units conversion factor of each pixel.
        origin : (float,)
            The x origin of the mask's coordinate system in scaled units..
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

    @property
    def shape_native(self) -> Tuple[int]:
        return self.shape

    @property
    def sub_shape_native(self) -> Tuple[int]:
        return (self.shape[0] * self.sub_size,)

    @property
    def mask_sub_1(self) -> "Mask1D":
        """
        Returns the mask on the same scaled coordinate system but with a sub-grid of ``sub_size`` `.
        """
        return Mask1D(
            mask=self, sub_size=1, pixel_scales=self.pixel_scales, origin=self.origin
        )

    @property
    def unmasked_mask(self) -> "Mask1D":
        """
        The indicies of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge (e.g. next to at least one pixel with a `True` value but not central pixels like those within \
        an annulus mask).
        """
        return Mask1D.unmasked(
            shape_slim=self.shape_slim,
            sub_size=self.sub_size,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def unmasked_grid_sub_1(self) -> grid_1d.Grid1D:
        """ The scaled-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x \
        value y value in scaled units.
        """
        grid_slim = grid_1d_util.grid_1d_slim_via_mask_from(
            mask_1d=self, pixel_scales=self.pixel_scales, sub_size=1, origin=self.origin
        )

        return grid_1d.Grid1D(
            grid=grid_slim, mask=self.unmasked_mask.mask_sub_1, store_slim=True
        )

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
        array_2d_util.numpy_array_1d_to_fits(
            array_1d=self.astype("float"), file_path=file_path, overwrite=overwrite
        )


class Mask1D(AbstractMask1d):
    @classmethod
    def manual(
        cls,
        mask: Union[List, np.ndarray],
        pixel_scales: Union[float, Tuple[float]],
        sub_size: int = 1,
        origin: Tuple[float] = (0.0,),
        invert: bool = False,
    ) -> "Mask1D":

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
        cls,
        shape_slim,
        pixel_scales: Union[float, Tuple[float]],
        sub_size: int = 1,
        origin: Tuple[float] = (0.0,),
        invert: bool = False,
    ) -> "Mask1D":
        """
        Setup a 1D mask where all pixels are unmasked.

        Parameters
        ----------
        shape : (int, int)
            The (y,x) shape of the mask in units of pixels.
        pixel_scales : float or (float, float)
            The scaled units to pixel units conversion factor of each pixel.
        """
        return cls.manual(
            mask=np.full(shape=shape_slim, fill_value=False),
            pixel_scales=pixel_scales,
            origin=origin,
            sub_size=sub_size,
            invert=invert,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: str,
        pixel_scales: Union[float, Tuple[float]],
        sub_size: int = 1,
        hdu: int = 0,
        origin: Tuple[float] = (0.0,),
    ) -> "Mask1D":
        """
        Loads the 1D mask from a .fits file.

        Parameters
        ----------
        file_path : str
            The full path of the fits file.
        hdu : int
            The HDU number in the fits file containing the image image.
        pixel_scales : float or (float, float)
            The scaled units to pixel units conversion factor of each pixel.
        """

        return cls.manual(
            array_2d_util.numpy_array_1d_from_fits(file_path=file_path, hdu=hdu),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    def output_to_fits(self, file_path: str, overwrite: bool = False):

        array_2d_util.numpy_array_1d_to_fits(
            array_1d=self.astype("float"), file_path=file_path, overwrite=overwrite
        )

    @property
    def pixels_in_mask(self) -> int:
        return int(np.size(self) - np.sum(self))

    @property
    def is_all_false(self) -> bool:
        return self.pixels_in_mask == self.shape_slim[0]

    @property
    def shape_slim(self) -> Tuple[int]:
        return self.shape

    @property
    def shape_slim_scaled(self) -> Tuple[float]:
        return (float(self.pixel_scales[0] * self.shape_slim[0]),)

    @property
    def scaled_maxima(self) -> Tuple[float]:
        return (float(self.shape_slim_scaled[0] / 2.0 + self.origin[0]),)

    @property
    def scaled_minima(self) -> Tuple[float]:
        return (-float(self.shape_slim_scaled[0] / 2.0) + self.origin[0],)

    @property
    def extent(self):
        return np.array([self.scaled_minima[0], self.scaled_maxima[0]])
