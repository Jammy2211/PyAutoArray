from __future__ import annotations
import logging
import numpy as np
from autoarray.numpy_wrapper import register_pytree_node_class
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D

from autoconf import cached_property

from autoarray.mask import mask_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


@register_pytree_node_class
class DeriveIndexes2D:
    def __init__(self, mask: Mask2D):
        """
        Derives 1D and 2D indexes of significance from a ``Mask2D``.

        A ``Mask2D`` masks values which are associated with a uniform 2D rectangular grid of pixels, where unmasked
        entries (which are ``False``) are used in subsequent calculations and masked values (which are ``True``) are
        omitted (for a full description see
        the :meth:`Mask2D class API documentation <autoarray.mask.mask_2d.Mask2D.__new__>`).

        The 2D mask has two data representations, ``slim`` and ``native``, which are described fully at ?.
        Derived indexes provides ``ndarrays`` which map between the two representations, for example:

        - ``native_for_slim``: returns an array of shape [total_unmasked_pixels*sub_size] that
        maps every unmasked sub-pixel to its corresponding native 2D pixel using its (y,x) pixel indexes.

        The ``DeriveIndexes2D`` class also contains methods for computing ``ndarrays`` with indexes
        of significant quantities of the ``Mask2D``, in both the ``slim`` and ``native`` data formats. For example:

        - The ``edge_slim`` and ``edge_native`` indexes: the locations of unmasked ``False`` values wherever
          the ``Mask2D`` that it is derived from neighbors at least one ``True`` value (i.e. they are on the edge of
          the original mask).

        Parameters
        ----------
        mask
            The 2D mask from which indexes are computed.

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

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.edge_native)
        """
        self.mask = mask

    def tree_flatten(self):
        return (self.mask,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(mask=children[0])

    @property
    def native_for_slim(self) -> np.ndarray:
        """
        Derives a 1D ``ndarray`` which maps every non-subgridded 1D ``slim`` index of the ``Mask2D`` to its
        non-subgridded 2D ``native`` index.

        For example, for the following ``Mask2D`` for ``sub_size=1``:

        ::
            [[True,  True,  True, True]
             [True, False, False, True],
             [True, False,  True, True],
             [True,  True,  True, True]]

        This has three unmasked (``False`` values) which have the ``slim`` indexes:

        ::
            [0, 1, 2]

        The array ``native_for_slim`` is therefore:

        ::
            [[1,1], [1,2], [2,1]]

        For a ``Mask2D`` with ``sub_size=2`` each unmasked ``False`` entry is split into a sub-pixel of size 2x2.
        However, this method ignores sub-gridding and therefore will still produce the same arrays above, as if
        ``sub_size=1``.

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

            print(derive_indexes_2d.native_for_slim)
        """
        return mask_2d_util.native_index_for_slim_index_2d_from(
            mask_2d=np.array(self.mask), sub_size=1
        ).astype("int")

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
            mask_2d=self.mask.array, sub_size=self.mask.sub_size
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
            mask_2d=np.array(self.mask), sub_size=self.mask.sub_size
        ).astype("int")

    @property
    def unmasked_slim(self) -> np.ndarray:
        """
        Derives a 1D ``ndarray`` comprising the 1D ``slim`` indexes of the ``masks``'s
        unmasked pixels (e.g. ``value=False``).

        For example, for the following ``Mask2D`` for ``sub_size=1``:

        ::
            [[True,  True,  True, True]
             [True, False, False, True],
             [True, False,  True, True],
             [True,  True,  True, True]]

        This has three unmasked (``False`` values) which have the ``slim`` indexes, there ``unmasked_slim`` is:

        ::
            [0, 1, 2]

        For a ``Mask2D`` with ``sub_size=2`` each unmasked ``False`` entry is split into a sub-pixel of size 2x2.
        Therefore the array ``unmasked_slim`` becomes:

        ::
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        The ``slim`` indexes are by definition the unmasked pixels, therefore this will always return an array of
        ascending integers with size the total number of unmasked pixels multiplied by the ``sub_size**2``.

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

            print(derive_indexes_2d.unmasked_slim)
        """
        return mask_2d_util.mask_slim_indexes_from(
            mask_2d=np.array(self.mask), return_masked_indexes=False
        ).astype("int")

    @property
    def masked_slim(self) -> np.ndarray:
        """
        Derives a 1D ``ndarray`` comprising the 1D ``slim`` indexes of the ``masks``'s
        masked pixels (e.g. ``value=True``).

        For example, for the following ``Mask2D`` for ``sub_size=1``:

        ::
            [[False,  True,  False, False]
             [False, False, True, False],
             [False, False,  True, False],
             [False,  False,  True, False]]

        This has four masked (``True`` values) which have the ``slim`` indexes, there ``masked_slim`` is:

        ::
            [0, 1, 2, 3]

        For a ``Mask2D`` with ``sub_size=2`` each masked ``True`` entry is split into a sub-pixel of size 2x2.
        Therefore the array ``masked_slim`` becomes:

        ::
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[False,  True, False, False]
                      [False, False,  True, False],
                      [False, False,  True, False],
                      [False, False,  True, False]]
                pixel_scales=1.0,
            )

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.masked_slim)
        """
        return mask_2d_util.mask_slim_indexes_from(
            mask_2d=np.array(self.mask), return_masked_indexes=True
        ).astype("int")

    @property
    def edge_slim(self) -> np.ndarray:
        """
        Returns the non-subgridded 1D ``slim`` indexes of edge pixels in the ``Mask2D``, representing all unmasked pixels (given
        by ``False``) which neighbor any masked value (give by ``True``) and therefore are on the edge of the 2D mask.

        For example, for the following ``Mask2D``:

        ::
            [[True,  True,  True,  True, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True,  True,  True,  True, True]]

        The `edge_slim` indexes (given via ``mask_2d.derive_indexes.edge_slim``) is given by:

        ::
             [0, 1, 2, 3, 5, 6, 7, 8]

        Note that index 4 is skipped, which corresponds to the ``False`` value in the centre of the mask, because it
        does not neighbor a ``True`` value in any one of the eight neighboring directions and is therefore not at
        an edge.

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

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.edge_slim)
        """
        return mask_2d_util.edge_1d_indexes_from(mask_2d=np.array(self.mask)).astype(
            "int"
        )

    @property
    def edge_native(self) -> np.ndarray:
        """
        Returns the non-subgridded 2D ``native`` indexes of edge pixels in the ``Mask2D``, representing all unmasked pixels (given
        by ``False``) which neighbor any masked value (give by ``True``) and therefore are on the edge of the 2D mask.

        For example, for the following ``Mask2D``:

        ::
            [[True,  True,  True,  True, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True,  True,  True,  True, True]]

        The `edge_native` indexes (given via ``mask_2d.derive_indexes.edge_native``) is given by:

        ::
             [[1,1], [1,2], [1,3], [2,1], [2,3], [3,1], [3,2], [3,3]]

        Note that index ``[2,2]`` is skipped, which corresponds to the ``False`` value in the centre of the mask,
        because it does not neighbor a ``True`` value in any one of the eight neighboring directions and is therefore
        not at an edge.

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

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.edge_native)
        """
        return self.native_for_slim[self.edge_slim].astype("int")

    @property
    def border_slim(self) -> np.ndarray:
        """
        Returns the non-subgridded 1D ``slim`` indexes of border pixels in the ``Mask2D``, representing all unmasked pixels (given
        by ``False``) which neighbor any masked value (give by ``True``) and which are on the extreme exterior of the
        mask.

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

        The `border_slim` indexes (given via ``mask_2d.derive_indexes.border_slim``) is given by:

        ::
             [0, 1, 2, 3, 5, 6, 7, 11, 12, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

        The interior 8 ``False`` values are omitted, because although they are edge pixels (neighbor a ``True``) they
        are not on the extreme exterior edge.

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

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.border_slim)
        """
        return mask_2d_util.border_slim_indexes_from(
            mask_2d=np.array(self.mask)
        ).astype("int")

    @property
    def border_native(self) -> np.ndarray:
        """
        Returns the non-subgridded 2D ``native`` indexes of border pixels in the ``Mask2D``, representing all unmasked pixels (given
        by ``False``) which neighbor any masked value (give by ``True``) and which are on the extreme exterior of the
        mask.

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

        The `border_native` indexes (given via ``mask_2d.derive_indexes.border_native``) is given by:

        ::
             [
              [1,1], [1,2], [1,3], [1,4], [1,5], [1,6], [1,7],
              [2,1], [2,7], [3,1], [3,7], [4,1], [4,7], [5,1], [5,7], [6,1], [6,7],
              [7,1], [7,2], 71,3], [7,4], [7,5], [7,6], [7,7]
             ]

        The interior 8 ``False`` values (e.g. ``native`` index ``[3,3]``) are omitted, because although they are edge
        pixels (neighbor a ``True``) they are not on the extreme exterior edge.

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

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.border_native)
        """
        return self.native_for_slim[self.border_slim].astype("int")

    @cached_property
    def sub_border_slim(self) -> np.ndarray:
        """
        Returns the subgridded 1D ``slim`` indexes of border pixels in the ``Mask2D``, representing all unmasked
        sub-pixels (given by ``False``) which neighbor any masked value (give by ``True``) and which are on the
        extreme exterior of the mask.

        The indexes are the sub-gridded extension of the ``border_slim`` which is illustrated above.

        This quantity is too complicated to write-out in a docstring, and it is recommended you print it in
        Python code to understand it if anything is unclear.

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
                sub_size=2,
            )

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.sub_border_slim)
        """
        return mask_2d_util.sub_border_pixel_slim_indexes_from(
            mask_2d=np.array(self.mask), sub_size=self.mask.sub_size
        ).astype("int")
