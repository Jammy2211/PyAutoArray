from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask1D
    from autoarray.mask.mask_2d import Mask2D

logging.basicConfig()
logger = logging.getLogger(__name__)


class DerivedMasks1D:
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
    def mask_sub_1(self) -> Mask1D:
        """
        Returns the mask on the same scaled coordinate system but with a sub-grid of `sub_size`.
        """

        from autoarray.mask.mask_1d import Mask1D

        return Mask1D(
            mask=self.mask,
            sub_size=1,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def unmasked_mask(self) -> Mask1D:

        from autoarray.mask.mask_1d import Mask1D

        return Mask1D.unmasked(
            shape_slim=self.mask.shape_slim,
            sub_size=self.mask.sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def to_mask_2d(self) -> Mask2D:
        """
        Map the Mask1D to a Mask2D of shape [total_mask_1d_pixel, 1].

        The change in shape and dimensions of the mask is necessary for mapping results from 1D data structures to 2D.

        Returns
        -------
        mask_2d
            The 1D mask mapped to a 2D mask of shape [total_mask_1d_pixel, 1].
        """

        from autoarray.mask.mask_2d import Mask2D

        return Mask2D.manual(
            [self.mask],
            pixel_scales=(self.mask.pixel_scale, self.mask.pixel_scale),
            sub_size=self.mask.sub_size,
            origin=(0.0, 0.0),
        )
