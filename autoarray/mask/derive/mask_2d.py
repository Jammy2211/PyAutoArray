from __future__ import annotations
import logging
import copy
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.mask.derive.indexes_2d import DeriveIndexes2D

from autoarray.structures.arrays import array_2d_util
from autoarray.mask import mask_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class DeriveMask2D:
    def __init__(self, mask: Mask2D):
        """
        Derives ``Mask2D`` objects from a ``Mask2D``.

        A ``Mask2D`` masks values which are associated with a a uniform rectangular grid of pixels, where entries
        which are ``False`` are unmasked and therefore used in subsequent calculations (for a full description see
        the :meth:`Mask2D class API documentation <autoarray.structures.arrays.uniform_2d.AbstractArray2D.__new__>`).

        From a ``Mask2D`` other ``Mask2D``s can be derived, which represent a subset of pixels with particular
        significance. For example:

        - An ``edge`` ``Mask2D``: has unmasked ``False`` values wherever the ``Mask2D`` that it is derived from
          neighbors at least one ``True`` value (i.e. they are on the edge of the original mask).

        - A ``recaled`` ``Mask``: The original mask rescaled to a larger or small 2D shape.

        Parameters
        ----------
        mask
            The ``Mask2D`` from which new ``Mask2D`` objects are derived.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D.manual(
                mask=[
                    [True,  True,  True,  True, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True,  True,  True,  True, True],
                ],
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.edge)

        """
        self.mask = mask

    @property
    def derive_indexes(self) -> DeriveIndexes2D:
        """
        Derives ``ndarrays``s of useful indexes from a ``Mask2D``.
        """
        return self.mask.derive_indexes

    @property
    def sub_1(self) -> Mask2D:
        """
        Returns the same ``Mask2D`` but with its geometry and (y,x) Cartersian coordinates reduced to a `sub_size=1`.
        """

        from autoarray.mask.mask_2d import Mask2D

        return Mask2D(
            mask=self.mask,
            sub_size=1,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    def rescaled_from(self, rescale_factor) -> Mask2D:
        """
        Returns the ``Mask2D`` rescaled to a bigger or small shape via input ``rescale_factor``.

        Parameters
        ----------
        rescale_factor
            The factor by which the ``Mask2D`` is rescaled (less than 1.0 produces a smaller mask, greater than 1.0
            produces a bigger mask).
        """
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
        Returns the ``Mask2D`` resized to a small or bigger ``ndarraay``, but with the same distribution of
         ``False`` and ``True`` entries.

        Resizing which increases the ``Mask2D`` shape pads it with values on its edge.

        Resizing which reduces the ``Mask2D`` shape removes entries on its edge.

        Parameters
        ----------
        new_shape
            The new two-dimensional shape of the resized ``Mask2D``.
        pad_value
            The value new values are padded using if the resized ``Mask2D`` is bigger.
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
            native_for_slim=self.derive_indexes.sub_mask_native_for_sub_mask_slim,
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
        mask[self.derive_indexes.edge_native[:, 0], self.derive_indexes.edge_native[:, 1]] = False
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
        mask[self.derive_indexes.border_native[:, 0], self.derive_indexes.border_native[:, 1]] = False
        return Mask2D(
            mask=mask,
            sub_size=self.mask.sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )
