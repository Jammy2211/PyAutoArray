from __future__ import annotations
import logging
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D

from autoconf import cached_property

from autoarray.mask import mask_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Indexes2D:
    def __init__(self, mask: Mask2D):

        self.mask = mask

    @property
    def native_for_slim(self) -> np.ndarray:
        """
        A 1D array of mappings between every unmasked pixel and its 2D pixel coordinates.
        """
        return mask_2d_util.native_index_for_slim_index_2d_from(
            mask_2d=self.mask, sub_size=1
        ).astype("int")

    @cached_property
    def sub_mask_index_for_sub_mask_1d_index(self) -> np.ndarray:
        """
        A 1D array of mappings between every unmasked sub pixel and its 2D sub-pixel coordinates.
        """
        return mask_2d_util.native_index_for_slim_index_2d_from(
            mask_2d=self.mask, sub_size=self.mask.sub_size
        ).astype("int")

    @cached_property
    def slim_index_for_sub_slim_index(self) -> np.ndarray:
        """
        The util between every sub-pixel and its host pixel.

        For example:

        sub_to_pixel[8] = 2 -  The ninth sub-pixel is within the 3rd pixel.
        sub_to_pixel[20] = 4 -  The twenty first sub-pixel is within the 5th pixel.
        """
        return mask_2d_util.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask_2d=self.mask, sub_size=self.mask.sub_size
        ).astype("int")

    @property
    def unmasked_slim(self) -> np.ndarray:
        """
        The 1D indexes of the mask's unmasked pixels (e.g. `value=False`).
        """
        return mask_2d_util.mask_slim_indexes_from(
            mask_2d=self.mask, return_masked_indexes=False
        ).astype("int")

    @property
    def masked_slim(self) -> np.ndarray:
        """
        The 1D indexes of the mask's unmasked pixels (e.g. `value=False`).
        """
        return mask_2d_util.mask_slim_indexes_from(
            mask_2d=self.mask, return_masked_indexes=True
        ).astype("int")

    @property
    def edge_slim(self) -> np.ndarray:
        """
        The indexes of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge
        (next to at least one pixel with a `True` value).
        """
        return mask_2d_util.edge_1d_indexes_from(mask_2d=self.mask).astype("int")

    @property
    def edge_native(self) -> np.ndarray:
        """
        The indexes of the mask's edge pixels, where an edge pixel is any unmasked pixel on its edge
        (next to at least one pixel with a `True` value).
        """
        return self.native_for_slim[self.edge_slim].astype("int")

    @property
    def border_slim(self) -> np.ndarray:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        return mask_2d_util.border_slim_indexes_from(mask_2d=self.mask).astype("int")

    @property
    def border_native(self) -> np.ndarray:
        """The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        return self.native_for_slim[self.border_slim].astype("int")

    @cached_property
    def sub_border_slim(self) -> np.ndarray:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        return mask_2d_util.sub_border_pixel_slim_indexes_from(
            mask_2d=self.mask, sub_size=self.mask.sub_size
        ).astype("int")
