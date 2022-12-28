from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask1D
    from autoarray.structures.grids.uniform_1d import Grid1D

from autoarray.structures.grids import grid_1d_util


logging.basicConfig()
logger = logging.getLogger(__name__)


class DerivedGrids1D:
    def __init__(self, mask: Mask1D):
        """
        Missing

        Parameters
        ----------
        mask
            The 2D mask from which indexes are computed.
        """
        self.mask = mask

    @property
    def unmasked_sub_1(self) -> Grid1D:
        """
        The scaled-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x
        value y value in scaled units.
        """
        from autoarray.structures.grids.uniform_1d import Grid1D

        grid_slim = grid_1d_util.grid_1d_slim_via_mask_from(
            mask_1d=self.mask,
            pixel_scales=self.mask.pixel_scales,
            sub_size=1,
            origin=self.mask.origin,
        )

        return Grid1D(
            grid=grid_slim,
            mask=self.mask.derived_masks.unmasked.derived_masks.sub_1,
        )
