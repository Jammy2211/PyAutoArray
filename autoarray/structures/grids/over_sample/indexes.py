from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from autoconf import cached_property

from autoarray.numpy_wrapper import register_pytree_node_class

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D

from autoarray.mask import mask_2d_util


@register_pytree_node_class
class OverSampleIndexes:
    def __init__(self, mask: Mask2D, sub_size: int):
        self.mask = mask
        self.sub_size = sub_size

    def tree_flatten(self):
        return (self.mask,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(mask=children[0])

    @cached_property
    def sub_mask_native_for_sub_mask_slim(self) -> np.ndarray:
        """
        Derives a 1D ``ndarray`` which maps every subgridded 1D ``slim`` index of the ``Mask2D`` to its
        subgridded 2D ``native`` index.

        For example, for the following ``Mask2D`` for ``sub_size=1``:

        ::
            [[True,  True,  True, True]
             [True, False, False, True],
             [True, False,  True, True],
             [True,  True,  True, True]]

        This has three unmasked (``False`` values) which have the ``slim`` indexes:

        ::
            [0, 1, 2]

        The array ``sub_mask_native_for_sub_mask_slim`` is therefore:

        ::
            [[1,1], [1,2], [2,1]]

        For a ``Mask2D`` with ``sub_size=2`` each unmasked ``False`` entry is split into a sub-pixel of size 2x2 and
        there are therefore 12 ``slim`` indexes:

        ::
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        The array ``native_for_slim`` is therefore:

        ::
            [[2,2], [2,3], [2,4], [2,5], [3,2], [3,3], [3,4], [3,5], [4,2], [4,3], [5,2], [5,3]]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True, True]
                      [True, False, False, True],
                      [True, False,  True, True],
                      [True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.sub_mask_native_for_sub_mask_slim)
        """
        return mask_2d_util.native_index_for_slim_index_2d_from(
            mask_2d=self.mask.array, sub_size=self.sub_size
        ).astype("int")

    @cached_property
    def slim_for_sub_slim(self) -> np.ndarray:
        """
        Derives a 1D ``ndarray`` which maps every subgridded 1D ``slim`` index of the ``Mask2D`` to its
        non-subgridded 1D ``slim`` index.

        For example, for the following ``Mask2D`` for ``sub_size=1``:

        ::
            [[True,  True,  True, True]
             [True, False, False, True],
             [True, False,  True, True],
             [True,  True,  True, True]]

        This has three unmasked (``False`` values) which have the ``slim`` indexes:

        ::
            [0, 1, 2]

        The array ``slim_for_sub_slim`` is therefore:

        ::
            [0, 1, 2]

        For a ``Mask2D`` with ``sub_size=2`` each unmasked ``False`` entry is split into a sub-pixel of size 2x2.
        Therefore the array ``slim_for_sub_slim`` becomes:

        ::
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True, True]
                      [True, False, False, True],
                      [True, False,  True, True],
                      [True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.slim_for_sub_slim)
        """
        return mask_2d_util.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask_2d=np.array(self.mask), sub_size=self.sub_size
        ).astype("int")

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
            self.mask.shape[0] * self.sub_size,
            self.mask.shape[1] * self.sub_size,
        )

        return mask_2d_util.mask_2d_via_shape_native_and_native_for_slim(
            shape_native=sub_shape,
            native_for_slim=self.derive_indexes.sub_mask_native_for_sub_mask_slim,
        ).astype("bool")
