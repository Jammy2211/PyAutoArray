from __future__ import annotations

from abc import ABC
import logging

from autoarray.numpy_wrapper import np, use_jax

if use_jax:
    import jax
from pathlib import Path
from typing import Dict, Union

from autoarray.abstract_ndarray import AbstractNDArray

from autoarray import exc
from autoarray import type as ty

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mask(AbstractNDArray, ABC):
    pixel_scales = None

    # noinspection PyUnusedLocal
    def __init__(
        self,
        mask: np.ndarray,
        origin: tuple,
        pixel_scales: ty.PixelScales,
        *args,
        **kwargs,
    ):
        """
        An abstract class for a mask that represents data structure that can be in 1D, 2D or other shapes.

        When applied to data it extracts or masks the unmasked image pixels corresponding to mask entries
        that are (`False` or 0).

        The mask also defines the geometry of the data structure it is paired with, for example how its pixels convert
        to physical units via the pixel_scales and origin parameters and a grid which is used for
        perform calculations via super-sampling.

        Parameters
        ----------
        mask
            The ndarray containing the bool's representing the mask, where `False` signifies an entry is
            unmasked and used in calculations.
        pixel_scales
            The scaled units to pixel units conversion factors of every pixel. If this is input as a float, it is
            converted to a (float, float) structure.
        origin
            The origin of the mask's coordinate system in scaled units.
        """

        # noinspection PyArgumentList
        mask = mask.astype("bool")
        super().__init__(mask)

        self.pixel_scales = pixel_scales
        self.origin = origin

    @property
    def mask(self):
        return self._array

    def __array_finalize__(self, obj):
        if isinstance(obj, Mask):
            self.pixel_scales = obj.pixel_scales
            self.origin = obj.origin
        else:
            self.pixel_scales = None

    @property
    def pixel_scale(self) -> float:
        """
        For a mask with dimensions two or above check that are pixel scales are the same, and if so return this
        single value as a float.
        """
        return self.pixel_scales[0]

    @property
    def pixel_scale_header(self) -> Dict:
        """
        Returns the pixel scale of the mask as a header dictionary, which can be written to a .fits file.

        If the array has different pixel scales in 2 dimensions, the header will contain both pixel scales as separate
        y and x entries.

        Returns
        -------
        A dictionary containing the pixel scale of the mask, which can be output to a .fits file.
        """
        try:
            return {"PIXSCALE": self.pixel_scale}
        except exc.MaskException:
            return {
                "PIXSCALEY": self.pixel_scales[0],
                "PIXSCALEX": self.pixel_scales[1],
            }

    @property
    def dimensions(self) -> int:
        return len(self.shape)

    def output_to_fits(self, file_path: Union[Path, str], overwrite: bool = False):
        """
        Overwrite with method to output the mask to a `.fits` file.
        """

    @property
    def pixels_in_mask(self) -> int:
        """
        The total number of unmasked pixels (values are `False`) in the mask.
        """
        return (np.size(self._array) - np.sum(self._array)).astype(int)

    @property
    def is_all_true(self) -> bool:
        """
        Returns `True` if all pixels in a mask are `True`, else returns `False`.
        """
        return self.pixels_in_mask == 0

    @property
    def is_all_false(self) -> bool:
        """
        Returns `False` if all pixels in a mask are `False`, else returns `True`.
        """
        return self.pixels_in_mask == np.size(self._array)

    @property
    def shape_slim(self) -> int:
        """
        The 1D shape of the mask, which is equivalent to the total number of unmasked pixels in the mask.
        """
        return self.pixels_in_mask
