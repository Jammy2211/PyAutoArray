from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D

from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.structures.grids import grid_2d_util


logging.basicConfig()
logger = logging.getLogger(__name__)


class DerivedGrids2D:
    
    def __init__(self, mask: Mask2D):
        """
        Missing

        Parameters
        ----------
        mask
            The 2D mask from which indexes are computed.
        """
        self.mask = mask
        
    @property
    def unmasked_grid_sub_1(self) -> Grid2D:
        """
        The scaled-grid of (y,x) coordinates of every pixel.

        This is defined from the top-left corner, such that the first pixel at location [0, 0] will have a negative x
        value y value in scaled units.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=self.mask.shape,
            pixel_scales=self.mask.pixel_scales,
            sub_size=1,
            origin=self.mask.origin,
        )

        return Grid2D(
            grid=grid_slim,
            mask=self.mask.derived_masks.unmasked.derived_masks.sub_1,
        )

    @property
    def masked_grid(self) -> Grid2D:

        from autoarray.structures.grids.uniform_2d import Grid2D

        sub_grid_1d = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=self.mask,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
            origin=self.mask.origin,
        )
        return Grid2D(
            grid=sub_grid_1d, mask=self.mask.derived_masks.edge.derived_masks.sub_1
        )

    @property
    def masked_grid_sub_1(self) -> Grid2D:

        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_slim = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=self.mask, pixel_scales=self.mask.pixel_scales, sub_size=1, origin=self.mask.origin
        )
        return Grid2D(grid=grid_slim, mask=self.mask.derived_masks.sub_1)

    @property
    def edge_grid_sub_1(self) -> Grid2D:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """

        from autoarray.structures.grids.uniform_2d import Grid2D

        edge_grid_1d = self.masked_grid_sub_1[self.mask.indexes.edge_slim]
        return Grid2D(
            grid=edge_grid_1d,
            mask=self.mask.derived_masks.edge.derived_masks.sub_1,
        )

    @property
    def border_grid_1d(self) -> Grid2D:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        return self.masked_grid[self.mask.indexes.sub_border_slim]

    @property
    def border_grid_sub_1(self) -> Grid2D:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        border_grid_1d = self.masked_grid_sub_1[self.mask.indexes.border_slim]
        return Grid2D(
            grid=border_grid_1d,
            mask=self.mask.derived_masks.border.derived_masks.sub_1,
        )