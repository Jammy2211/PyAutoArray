from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask1D
    from autoarray.structures.grids.uniform_1d import Grid1D

from autoarray.structures.grids import grid_1d_util


logging.basicConfig()
logger = logging.getLogger(__name__)


class DeriveGrid1D:
    def __init__(self, mask: Mask1D):
        """
        Derives ``Grid1D`` objects from a ``Mask1D``.

        A ``Mask1D`` masks values which are associated with a 1D uniform grid of pixels, where unmasked
        entries (which are ``False``) are used in subsequent calculations and masked values (which are ``True``) are
        omitted (for a full description see
        the :meth:`Mask1D class API documentation <autoarray.mask.mask_1d.Mask1D.__new__>`).

        From a ``Mask1D``,``Grid1D``s can be derived, which represent the (y,x) Cartesian coordinates of a subset of
        pixels with significance.

        For example:

        - An ``all_false`` ``Grid1D``: the same shape as the original ``Mask1D`` but has unmasked ``False`` values
          everywhere.

        Parameters
        ----------
        mask
            The ``Mask2D`` from which new ``Grid2D`` objects are derived.

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

            derive_grid_2d = aa.DeriveGrid2D(mask=mask_2d)

            print(derive_grid_2d.border)
        """
        self.mask = mask

    @property
    def all_false_sub_1(self) -> Grid1D:
        """
        Returns a non-subgridded ``Grid1D`` which uses the ``Mask1D``
        geometry (``shape_native`` / ``sub_size`` / ``pixel_scales`` / ``origin``) and every pixel in the ``Mask2D``
        irrespective of whether pixels are masked or unmasked (given by ``True`` or``False``).

        For example, for the following ``Mask2D``:

        ::
            mask_2d = aa.Mask1D(
                mask=[False, False, True,  True]
                pixel_scales=1.0,
            )

        The ``all_false_sub_1`` ``Grid1D`` (given via ``mask_1d.derive_grid.all_false_sub_1``) is:

        ::
            [-2.0, -1.0, 1.0, 2.0]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_1d = aa.Mask2D(
                mask=[False, False, True,  True],
                pixel_scales=1.0,
                sub_size=2
            )

            derive_grid_1d = aa.DeriveGrid1D(mask=mask_1d)

            print(derive_grid_1d.all_false_sub_1)
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
            mask=self.mask.derive_mask.all_false.derive_mask.sub_1,
        )
