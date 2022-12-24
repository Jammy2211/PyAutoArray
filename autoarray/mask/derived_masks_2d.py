from __future__ import annotations
import logging
import copy
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray import type as ty
from autoarray.geometry.geometry_2d import Geometry2D
from autoarray.mask.indexes_2d import Indexes2D

from autoarray.structures.arrays import array_2d_util
from autoarray.geometry import geometry_util
from autoarray.structures.grids import grid_2d_util
from autoarray.mask import mask_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class DerivedMasks2D:
    def __init__(self, mask: Mask2D):
        """
        Computes the ``slim`` and ``native`` indexes of specific ``Mask2D`` quantities.

        A 2D mask has two data representations, ``slim`` and ``native``, which are described fully at ?.

        The ``Indexes2D`` class contains methods for computing 1D ``ndarrays`` of specific indexes
        of certain predefined quantities associated with the 2D mask.

        For example, the property ``native_for_slim`` returns an array of shape [total_unmasked_pixels*sub_size] that
        maps every unmasked sub-pixel to its corresponding native 2D pixel using its (y,x) pixel indexes.

        For example, for a sub-grid size of 2x2, if pixel [2,5] corresponds to the first pixel in the masked slim array:

        - The first sub-pixel in this pixel on the 1D array is sub_native_index_for_sub_slim_index_2d[4] = [2,5]
        - The second sub-pixel in this pixel on the 1D array is sub_native_index_for_sub_slim_index_2d[5] = [2,6]
        - The third sub-pixel in this pixel on the 1D array is sub_native_index_for_sub_slim_index_2d[5] = [3,5]

        Parameters
        ----------
        mask
            The 2D mask from which indexes are computed.
        """
        self.mask = mask

    @property
    def indexes(self) -> Indexes2D:
        return self.mask.indexes

    def rescaled_mask_from(self, rescale_factor) -> Mask2D:

        from autoarray.mask.mask_2d import Mask2D

        rescaled_mask = mask_2d_util.rescaled_mask_2d_from(
            mask_2d=self.mask, rescale_factor=rescale_factor
        )

        return Mask2D(
            mask=rescaled_mask,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
            origin=self.mask.origin,
        )

    @property
    def sub_mask(self) -> np.ndarray:

        sub_shape = (self.mask.shape[0] * self.mask.sub_size, self.mask.shape[1] * self.mask.sub_size)

        return mask_2d_util.mask_2d_via_shape_native_and_native_for_slim(
            shape_native=sub_shape,
            native_for_slim=self.indexes.sub_mask_native_for_sub_mask_slim,
        ).astype("bool")

    @property
    def unmasked_mask(self) -> Mask2D:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """

        from autoarray.mask.mask_2d import Mask2D

        return Mask2D.unmasked(
            shape_native=self.mask.shape_native,
            sub_size=self.mask.sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    def blurring_mask_from(self, kernel_shape_native) -> Mask2D:
        """
        Returns a blurring mask, which represents all masked pixels whose light will be blurred into unmasked
        pixels via PSF convolution (see grid.Grid2D.blurring_grid_from).

        Parameters
        ----------
        kernel_shape_native
           The shape of the psf which defines the blurring region (e.g. the shape of the PSF)
        """

        from autoarray.mask.mask_2d import Mask2D

        if kernel_shape_native[0] % 2 == 0 or kernel_shape_native[1] % 2 == 0:
            raise exc.MaskException("psf_size of exterior region must be odd")

        blurring_mask = mask_2d_util.blurring_mask_2d_from(
            mask_2d=self.mask, kernel_shape_native=kernel_shape_native
        )

        return Mask2D(
            mask=blurring_mask,
            sub_size=1,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def edge_mask(self) -> Mask2D:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """

        from autoarray.mask.mask_2d import Mask2D

        mask = np.full(fill_value=True, shape=self.mask.shape)
        mask[self.indexes.edge_native[:, 0], self.indexes.edge_native[:, 1]] = False
        return Mask2D(
            mask=mask,
            sub_size=self.mask.sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def edge_buffed_mask(self) -> Mask2D:

        from autoarray.mask.mask_2d import Mask2D

        edge_buffed_mask = mask_2d_util.buffed_mask_2d_from(mask_2d=self.mask).astype("bool")

        return Mask2D(
            mask=edge_buffed_mask,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
            origin=self.mask.origin,
        )

    @property
    def border_mask(self) -> Mask2D:
        """
        The indexes of the mask's border pixels, where a border pixel is any unmasked pixel on an
        exterior edge e.g. next to at least one pixel with a `True` value but not central pixels like those within
        an annulus mask.
        """

        from autoarray.mask.mask_2d import Mask2D

        mask = np.full(fill_value=True, shape=self.mask.shape)
        mask[self.indexes.border_native[:, 0], self.indexes.border_native[:, 1]] = False
        return Mask2D(
            mask=mask,
            sub_size=self.mask.sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )