import numpy as np

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.structures.grids import grid_2d_util
from autoarray.mask.mask_2d import mask_2d_util

class AbstractOverSample:

    def oversampled_grid_2d_via_mask_from(self, mask : Mask2D, sub_size : int) -> Grid2D:

        over_sample_mask = mask_2d_util.oversample_mask_2d_from(
            mask=mask,
            sub_size=sub_size
        )

        pixel_scales = (mask.pixel_scales[0] / sub_size, mask.pixel_scales[1] / sub_size)

        mask = Mask2D(
            mask_2d=over_sample_mask,
            pixel_scales=pixel_scales,
            origin=mask.origin
        )

        sub_grid_1d = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=np.array(mask),
            pixel_scales=mask.pixel_scales,
            origin=mask.origin,
        )

        return Grid2D(values=sub_grid_1d, mask=mask, over_sample=self)