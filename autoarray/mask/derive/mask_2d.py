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

        A ``Mask2D`` masks values which are associated with a uniform 2D rectangular grid of pixels, where unmasked
        entries (which are ``False``) are used in subsequent calculations and masked values (which are ``True``) are
        omitted (for a full description see
        the :meth:`Mask2D class API documentation <autoarray.mask.mask_2d.Mask2D.__new__>`).

        From a ``Mask2D`` other ``Mask2D``s can be derived, which represent a subset of pixels with significance.
        For example:

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

            mask_2d = aa.Mask2D(
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
        Derives ``ndarrays``s of useful indexes from a ``Mask2D``, which are often used in order to create the
        derived ``Mask2D`` objects.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
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

            print(derive_mask_2d.derive_indexes.edge)
        """
        return self.mask.derive_indexes

    @property
    def sub_1(self) -> Mask2D:
        """
        Returns the same ``Mask2D`` but with its geometry and (y,x) Cartesian coordinates reduced to a `sub_size=1`.

        For example, for the following ``Mask2D``:

        ::
           [[ True,  True],
            [False, False]]

        The ``sub_1``` mask (given via ``mask_2d.derive_mask.sub_1``) returns the same mask but its ``sub_size``
        parameter will be reduced to 1 if it was above 1 before.

        ::
           [[ True,  True],
            [False, False]]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                     [ True,  True],
                     [False, False]
                ],
                pixel_scales=1.0,
                sub_size=2,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            mask_sub_1 = derive_mask_2d.sub_1

            print(mask_sub_1)

            # The sub_size of the mask is 1.
            print(mask_sub_1.sub_size)
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

        For example, for a ``rescale_factor=2.0`` the following mask:

        ::
           [[ True,  True],
            [False, False]]

        Will double in size and become:

        ::
            [[True,   True,  True,  True],
             [True,   True,  True,  True],
             [False, False, False, False],
             [False, False, False, False]]

        Parameters
        ----------
        rescale_factor
            The factor by which the ``Mask2D`` is rescaled (less than 1.0 produces a smaller mask, greater than 1.0
            produces a bigger mask).

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                     [ True,  True],
                     [False, False]
                ],
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.rescaled_from(rescale_factor=2.0)
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

        For example, for a ``new_shape=(4,4)`` the following mask:

        ::
           [[ True,  True],
            [False, False]]

        Will be padded with zeros (``False`` values) and become:

        ::
          [[True,  True,  True, True]
           [True,  True,  True, True],
           [True, False, False, True],
           [True,  True,  True, True]]

        Parameters
        ----------
        new_shape
            The new two-dimensional shape of the resized ``Mask2D``.
        pad_value
            The value new values are padded using if the resized ``Mask2D`` is bigger.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                     [ True,  True],
                     [False, False]
                ],
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.resized_from(new_shape=(4,4))
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
        """
        Returns the sub-mask of the ``Mask2D``, which is the mask on the sub-grid which has ``False``  / ``True``
        entries where the original mask is ``False`` / ``True``.

        For example, for the following ``Mask2D``:

        ::
           [[ True,  True],
            [False, False]]

        The sub-mask (given via ``mask_2d.derive_mask.sub``) for a ``sub_size=2`` is:

        ::
            [[True,   True,  True,  True],
             [True,   True,  True,  True],
             [False, False, False, False],
             [False, False, False, False]]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                     [ True,  True],
                     [False, False]
                ],
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.sub)
        """
        sub_shape = (
            self.mask.shape[0] * self.mask.sub_size,
            self.mask.shape[1] * self.mask.sub_size,
        )

        return mask_2d_util.mask_2d_via_shape_native_and_native_for_slim(
            shape_native=sub_shape,
            native_for_slim=self.derive_indexes.sub_mask_native_for_sub_mask_slim,
        ).astype("bool")

    @property
    def all_false(self) -> Mask2D:
        """
        Returns a ``Mask2D`` which has the same
        geometry (``shape_native`` / ``sub_size`` / ``pixel_scales`` / ``origin``) as this ``Mask2D`` but all
        entries are unmasked (given by``False``).

        For example, for the following ``Mask2D``:

        ::
            [[True,  True],
            [False, False]]

        The unmasked mask (given via ``mask_2d.derive_mask.all_false``) is:

        ::
            [[False, False],
            [False, False]]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                     [ True,  True],
                     [False, False]
                ],
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.all_false)
        """

        from autoarray.mask.mask_2d import Mask2D

        return Mask2D.all_false(
            shape_native=self.mask.shape_native,
            sub_size=self.mask.sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    def blurring_from(self, kernel_shape_native: Tuple[int, int]) -> Mask2D:
        """
        Returns a blurring ``Mask2D``, representing all masked pixels (given by ``True``) whose values are blurred
        into unmasked pixels (given by ``False``) when a 2D convolution is performed.

        This mask is used by the ``Convolver2D`` object to ensure that 2D convolution can be performed on masked
        data structures without missing values.

        For example, for the following ``Mask2D``:

        ::

            [[True,  True,  True,  True, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True,  True,  True,  True, True]]

        The blurring-mask for `kernel_shape_native=(3,3)` (given
        via ``mask_2d.derive_mask.blurring_from(kernel_shape_native=(3,3)``) is:

        ::

            [[False, False, False, False, False],
             [False,  True,  True,  True, False],
             [False,  True,  True,  True, False],
             [False,  True,  True,  True, False],
             [False, False, False, False, False]]

        Parameters
        ----------
        kernel_shape_native
           The 2D shape of the 2D convolution ``Kernel2D`` which defines the blurring region.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True,  True, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True,  True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            blurring_mask = derive_mask_2d.blurring_from(kernel_shape_native=(3,3))

            print(blurring_mask)
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
        Returns an edge ``Mask2D``, representing all unmasked pixels (given by ``False``) which neighbor any masked
        value (give by ``True``) and therefore are on the edge of the 2D mask.

        For example, for the following ``Mask2D``:

        ::
            [[True,  True,  True,  True, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True,  True,  True,  True, True]]

        The edge-mask (given via ``mask_2d.derive_mask.edge``) is given by:

        ::
             [[True,  True,  True,  True, True],
              [True, False, False, False, True],
              [True, False,  True, False, True],
              [True, False, False, False, True],
              [True, True,   True, True, True]]

        The central pixel, which does not neighbor a ``True`` value in any one of the eight neighboring directions,
        is switched to masked.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True,  True, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True,  True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.edge)
        """

        from autoarray.mask.mask_2d import Mask2D

        mask = np.full(fill_value=True, shape=self.mask.shape)
        mask[
            self.derive_indexes.edge_native[:, 0], self.derive_indexes.edge_native[:, 1]
        ] = False
        return Mask2D(
            mask=mask,
            sub_size=self.mask.sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def edge_buffed(self) -> Mask2D:
        """
        Returns a buffed edge ``Mask2D``, representing all unmasked pixels (given by ``False``) which neighbor any
        masked value (give by ``True``) and therefore are on the edge of the 2D mask, but with a buffer of 1 pixel
        applied such that everu pixel 1 pixel further out are included.

        For example, for the following ``Mask2D``:

        ::
            [[True,  True,  True,  True, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True,  True,  True,  True, True]]

        The edge-mask (given via ``mask_2d.derive_mask.edge``) is given by:

        ::
             [[False, False, False,  False, False],
              [False, False, False,  False, False],
              [False, False,  True,  False, False],
              [False, False, False,  False, False],
              [False, False,  False, False, False]]

        The central pixel, which does not neighbor a ``True`` value in any one of the eight neighboring directions,
        is switched to masked.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True,  True, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True,  True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.edge_buffed)
        """
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
        Returns border ``Mask2D``, representing all unmasked pixels (given by ``False``) which neighbor any masked
        value (give by ``True``) and which are on the extreme exterior of the mask.

        This is therefore close related to the edge mask, but may not include central pixels unlike the edge mask.

        For example, for the following ``Mask2D``:

        ::
            [[True,  True,  True,  True,  True,  True,  True,  True, True],
             [True, False, False, False, False, False, False, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False,  True, False, False, False,  True, False, True],
             [True, False,  True, False,  True, False,  True, False, True],
             [True, False,  True, False, False, False,  True, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False, False, False, False, False, False, False, True],
             [True,  True,  True,  True,  True,  True,  True,  True, True]]

        The border-mask (given via ``mask_2d.derive_mask.edge``) is given by:

        ::
            [[True,  True,  True,  True,  True,  True,  True,  True, True],
             [True, False, False, False, False, False, False, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False, False, False, False, False, False, False, True],
             [True,  True,  True,  True,  True,  True,  True,  True, True]]

        The central ``False`` pixels would be included in an edge-mask, but are not included in the border mask
        because they are not on the extreme outer edge.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True,  True,  True,  True,  True,  True, True],
                      [True, False, False, False, False, False, False, False, True],
                      [True, False,  True,  True,  True,  True,  True, False, True],
                      [True, False,  True, False, False, False,  True, False, True],
                      [True, False,  True, False,  True, False,  True, False, True],
                      [True, False,  True, False, False, False,  True, False, True],
                      [True, False,  True,  True,  True,  True,  True, False, True],
                      [True, False, False, False, False, False, False, False, True],
                      [True,  True,  True,  True,  True,  True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.border)
        """

        from autoarray.mask.mask_2d import Mask2D

        mask = np.full(fill_value=True, shape=self.mask.shape)
        mask[
            self.derive_indexes.border_native[:, 0],
            self.derive_indexes.border_native[:, 1],
        ] = False
        return Mask2D(
            mask=mask,
            sub_size=self.mask.sub_size,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )
