from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask1D
    from autoarray.mask.mask_2d import Mask2D

logging.basicConfig()
logger = logging.getLogger(__name__)


class DeriveMask1D:
    def __init__(self, mask: Mask1D):
        """
        Derives ``Mask1D`` objects from a ``Mask1D``.

        A ``Mask1D`` masks values which are associated with a 1D uniform grid of pixels, where unmasked
        entries (which are ``False``) are used in subsequent calculations and masked values (which are ``True``) are
        omitted (for a full description see
        the :meth:`Mask1D class API documentation <autoarray.mask.mask_1d.Mask1D.__new__>`).

        From a ``Mask1D`` other ``Mask1D``s can be derived, which represent a subset of pixels with significance.
        For example:

        - An ``all_false`` ``Mask1D``: the same shape as the original ``Mask1D`` but has unmasked ``False`` values
          everywhere.

        Parameters
        ----------
        mask
            The ``Mask1D`` from which new ``Mask1D`` objects are derived.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_1d = aa.Mask1D.manual(
                mask=[True, False, False, False, True],
                pixel_scales=1.0,
            )

            derive_mask_1d = aa.DeriveMask1D(mask=mask_1d)

            print(derive_mask_1d.edge)
        """
        self.mask = mask

    @property
    def sub_1(self) -> Mask1D:
        """
        Returns the same ``Mask1D`` but with its geometry and (y,x) Cartesian coordinates reduced to a `sub_size=1`.

        For example, for the following ``Mask1D``:

        ::
           [True, False, False, False, True]

        The ``sub_1``` mask (given via ``mask_2d.derive_mask.sub_1``) returns the same mask but its ``sub_size``
        parameter will be reduced to 1 if it was above 1 before.

        ::
           [True, False, False, False, True]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_1d = aa.Mask1D.manual(
                mask[True, False, False, False, True],
                pixel_scales=1.0,
                sub_size=2,
            )

            derive_mask_1d = aa.DeriveMask1D(mask=mask_1d)

            mask_sub_1 = derive_mask_1d.sub_1

            print(mask_sub_1)

            # The sub_size of the mask is 1.
            print(mask_sub_1.sub_size)
        """

        from autoarray.mask.mask_1d import Mask1D

        return Mask1D(
            mask=self.mask,
            sub_size=1,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def all_false(self) -> Mask1D:
        """
        Returns a ``Mask1D`` which has the same
        geometry (``shape_native`` / ``sub_size`` / ``pixel_scales`` / ``origin``) as this ``Mask1D`` but all
        entries are unmasked (given by``False``).

        For example, for the following ``Mask1D``:

        ::
           [True, False, False, False, True]

        The unmasked mask (given via ``mask_1d.derive_mask.all_false``) is:

        ::
           [False, False, False, False, False]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_1d = aa.Mask1D.manual(
                mask[True, False, False, False, True],
                pixel_scales=1.0,
                sub_size=2,
            )

            derive_mask_1d = aa.DeriveMask1D(mask=mask_1d)

            print(derive_mask_1d.all_false)
        """
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
        Map the ``Mask1D`` to a Mask2D of shape [total_mask_1d_pixel, 1].

        The change in shape and dimensions of the mask is necessary for mapping results from 1D to 2D data
        structures (e.g. an ``Array1D`` to ``Array2D``).

        For example, for the following ``Mask1D``:

        ::
           [True, False, False, False, True]

        The corresponding ``Mask2D`` (given via ``mask_1d.derive_mask.to_mask_1d``) is:

        ::
           [[False, False, False, False, False]]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_1d = aa.Mask1D.manual(
                mask[True, False, False, False, True],
                pixel_scales=1.0,
                sub_size=2,
            )

            derive_mask_1d = aa.DeriveMask1D(mask=mask_1d)

            print(derive_mask_1d.to_mask_2d)
        """

        from autoarray.mask.mask_2d import Mask2D

        return Mask2D.manual(
            [self.mask],
            pixel_scales=(self.mask.pixel_scale, self.mask.pixel_scale),
            sub_size=self.mask.sub_size,
            origin=(0.0, 0.0),
        )
