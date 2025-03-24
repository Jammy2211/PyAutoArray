from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union

from autoarray.mask.abstract_mask import Mask

from autoarray.mask.derive.grid_1d import DeriveGrid1D
from autoarray.mask.derive.mask_1d import DeriveMask1D
from autoarray.geometry.geometry_1d import Geometry1D
from autoarray.structures.abstract_structure import Structure
from autoarray.structures.arrays import array_1d_util

from autoarray import exc
from autoarray import type as ty

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask1D(Mask):
    def __init__(
        self,
        mask: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float,] = (0.0,),
        invert: bool = False,
    ):
        """
        A 1D mask, representing 1D data on a uniform line of pixels with equal spacing.

        When applied to 1D data it extracts or masks the unmasked image pixels corresponding to mask entries that
        are `False` or 0).

        The mask also defines the geometry of the 1D data structure it is paired to, for example how every pixel
        coordinate on the 1D line of data converts to physical units via the `pixel_scales` and `origin`
        parameters and a grid which is used for performing calculations.

        Parameters
        ----------
        mask
            The ndarray of shape [total_pixels] containing the bool's representing the mask, where `False`
            signifies an entry is unmasked and used in calculations.
        pixel_scales
             The scaled units to pixel units conversion factor of each pixel.
        origin
            The x origin of the mask's coordinate system in scaled units.
        """

        if type(mask) is list:
            mask = np.asarray(mask).astype("bool")

        if invert:
            mask = np.invert(mask)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales,)

        if len(mask.shape) != 1:
            raise exc.MaskException("The input mask is not a one dimensional array")

        # noinspection PyArgumentList
        super().__init__(
            mask=mask,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj=obj)

        if isinstance(obj, Mask1D):
            pass
        else:
            self.origin = (0.0,)

    @property
    def native(self) -> Structure:
        raise NotImplemented()

    @property
    def geometry(self) -> Geometry1D:
        """
        Return the 1D geometry of the mask, representing its uniform rectangular grid of (x) coordinates defined by
        its ``shape_native``.
        """
        return Geometry1D(
            shape_native=self.shape_native,
            pixel_scales=self.pixel_scales,
            origin=self.origin,
        )

    @property
    def derive_mask(self) -> DeriveMask1D:
        return DeriveMask1D(mask=self)

    @property
    def derive_grid(self) -> DeriveGrid1D:
        return DeriveGrid1D(mask=self)

    @classmethod
    def all_false(
        cls,
        shape_slim,
        pixel_scales: ty.PixelScales,
        origin: Tuple[float] = (0.0,),
        invert: bool = False,
    ) -> "Mask1D":
        """
        Setup a 1D mask where all pixels are unmasked.

        Parameters
        ----------
        shape_slim
            The (y,x) shape of the mask in units of pixels.
        pixel_scales
            The scaled units to pixel units conversion factor of each pixel.
        """
        return cls(
            mask=np.full(shape=shape_slim, fill_value=False),
            pixel_scales=pixel_scales,
            origin=origin,
            invert=invert,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: Union[Path, str],
        pixel_scales: ty.PixelScales,
        hdu: int = 0,
        origin: Tuple[float] = (0.0,),
    ) -> "Mask1D":
        """
        Loads the 1D mask from a .fits file.

        Parameters
        ----------
        file_path
            The full path of the fits file.
        hdu
            The HDU number in the fits file containing the image image.
        pixel_scales
            The scaled units to pixel units conversion factor of each pixel.
        """

        return cls(
            array_1d_util.numpy_array_1d_via_fits_from(file_path=file_path, hdu=hdu),
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @property
    def shape_native(self) -> Tuple[int]:
        return self.shape

    @property
    def shape_slim(self) -> Tuple[int]:
        return self.shape

    @property
    def header_dict(self) -> Dict:
        """
        Returns the pixel scales of the mask as a header dictionary, which can be written to a .fits file.

        A 2D mask has different pixel scale variables for each dimension, the header therefore contain both pixel
        scales as separate y and x entries.

        Returns
        -------
        A dictionary containing the pixel scale of the mask, which can be output to a .fits file.
        """
        return {
            "PIXSCA": self.pixel_scales[0],
            "ORIGIN": self.origin[0],
        }
