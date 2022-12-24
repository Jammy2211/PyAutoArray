from __future__ import annotations
import logging
import copy
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.mask.indexes_2d import Indexes2D

from autoarray.structures.arrays import array_2d_util
from autoarray.mask import mask_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class DerivedMasks2D:
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
    def indexes(self) -> Indexes2D:
        return self.mask.indexes

    @property
    def sub_1(self) -> Mask2D:
        """
        Returns the mask on the same scaled coordinate system but with a sub-grid of `sub_size`.
        """

        from autoarray.mask.mask_2d import Mask2D

        return Mask2D(
            mask=self.mask,
            sub_size=1,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    def rescaled_from(self, rescale_factor) -> Mask2D:

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

    def resized_from(self, new_shape, pad_value: int = 0.0) -> Mask2D:
        """
        Resized the array to a new shape and at a new origin.

        Parameters
        ----------
        new_shape
            The new two-dimensional shape of the array.
        """

        from autoarray.mask.mask_2d import Mask2D

        mask = copy.deepcopy(self.mask)

        resized_mask = array_2d_util.resized_array_2d_from(
            array_2d=mask, resized_shape=new_shape, pad_value=pad_value
        ).astype("bool")

        return Mask2D(
            mask=resized_mask,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
            origin=self.mask.origin,
        )

    @property
    def sub(self) -> np.ndarray:

        sub_shape = (
            self.mask.shape[0] * self.mask.sub_size,
            self.mask.shape[1] * self.mask.sub_size,
        )

        return mask_2d_util.mask_2d_via_shape_native_and_native_for_slim(
            shape_native=sub_shape,
            native_for_slim=self.indexes.sub_mask_native_for_sub_mask_slim,
        ).astype("bool")

    @property
    def unmasked(self) -> Mask2D:
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

    def blurring_from(self, kernel_shape_native) -> Mask2D:
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
    def edge(self) -> Mask2D:
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
    def edge_buffed(self) -> Mask2D:

        from autoarray.mask.mask_2d import Mask2D

        edge_buffed_mask = mask_2d_util.buffed_mask_2d_from(mask_2d=self.mask).astype(
            "bool"
        )

        return Mask2D(
            mask=edge_buffed_mask,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
            origin=self.mask.origin,
        )

    @property
    def border(self) -> Mask2D:
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
