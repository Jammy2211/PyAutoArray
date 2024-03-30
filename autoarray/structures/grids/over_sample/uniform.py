import numpy as np
from typing import Tuple

from autoconf import cached_property

from autoarray.structures.grids.over_sample.abstract import AbstractOverSample

from autoarray.mask.mask_2d import mask_2d_util


class OverSampleUniform(AbstractOverSample):
    def __init__(self, sub_size: int = 1):
        """
        Over samples grid calculations using a uniform sub-grid that is the same size in every pixel.

        When a 2D grid of (y,x) coordinates is input into a function, the result is evaluated at every coordinate
        on the grid. When the grid is paired to a 2D image (e.g. an `Array2D`) the solution needs to approximate
        the 2D integral of that function in each pixel. Over sample objects define how this over-sampling is performed.

        This object inputs a uniform sub-grid, where every image-pixel is split into a uniform grid of sub-pixels. The
        function is evaluated at every sub-pixel, and the final value in each pixel is computed by summing the
        contribution from all sub-pixels. The sub-grid is the same size in every pixel.

        This is the simplest over-sampling method, but may not provide precise solutions for functions that vary
        significantly within a pixel. To achieve precision in these pixels a high `sub_size` is required, which can
        be computationally expensive as it is applied to every pixel.

        - ``native_for_slim``: returns an array of shape [total_unmasked_pixels*sub_size] that
        maps every unmasked sub-pixel to its corresponding native 2D pixel using its (y,x) pixel indexes.

        **Case 2 (sub-size>1, slim)**

        If the mask's `sub_size` is > 1, the grid is defined as a sub-grid where each entry corresponds to the (y,x)
        coordinates at the centre of each sub-pixel of an unmasked pixel. The Grid2D is therefore stored as an ndarray
        of shape [total_unmasked_coordinates*sub_size**2, 2]

        The sub-grid indexes are ordered such that pixels begin from the first (top-left) sub-pixel in the first
        unmasked pixel. Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel.
        Therefore, the sub-grid is an ndarray of shape [total_unmasked_coordinates*(sub_grid_shape)**2, 2].

        For example:

        - grid[9, 1] - using a 2x2 sub-grid, gives the 3rd unmasked pixel's 2nd sub-pixel x-coordinate.
        - grid[9, 1] - using a 3x3 sub-grid, gives the 2nd unmasked pixel's 1st sub-pixel x-coordinate.
        - grid[27, 0] - using a 3x3 sub-grid, gives the 4th unmasked pixel's 1st sub-pixel y-coordinate.

        Below is a visual illustration of a sub grid. Indexing of each sub-pixel goes from the top-left corner. In
        contrast to the grid above, our illustration below restricts the mask to just 2 pixels, to keep the
        illustration brief.

        .. code-block:: bash

             x x x x x x x x x x
             x x x x x x x x x x     This is an example mask.Mask2D, where:
             x x x x x x x x x x
             x x x x x x x x x x     x = `True` (Pixel is masked and excluded from lens)
             x x x x O O x x x x     O = `False` (Pixel is not masked and included in lens)
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        Our grid with a sub-size looks like it did before:

        .. code-block:: bash

            pixel_scales = 1.0"

            <--- -ve  x  +ve -->

             x x x x x x x x x x  ^
             x x x x x x x x x x  I
             x x x x x x x x x x  I                        y     x
             x x x x x x x x x x +ve  grid[0] = [0.5,  -1.5]
             x x x x 0 1 x x x x  y   grid[1] = [0.5,  -0.5]
             x x x x x x x x x x -ve
             x x x x x x x x x x  I
             x x x x x x x x x x  I
             x x x x x x x x x x \/
             x x x x x x x x x x

        However, if the sub-size is 2, we go to each unmasked pixel and allocate sub-pixel coordinates for it. For
        example, for pixel 0, if *sub_size=2*, we use a 2x2 sub-grid:

        .. code-block:: bash

            Pixel 0 - (2x2):
                                y      x
                   grid[0] = [0.66, -1.66]
            I0I1I  grid[1] = [0.66, -1.33]
            I2I3I  grid[2] = [0.33, -1.66]
                   grid[3] = [0.33, -1.33]

        If we used a sub_size of 3, for the pixel we we would create a 3x3 sub-grid:

        .. code-block:: bash

                                  y      x
                     grid[0] = [0.75, -0.75]
                     grid[1] = [0.75, -0.5]
                     grid[2] = [0.75, -0.25]
            I0I1I2I  grid[3] = [0.5,  -0.75]
            I3I4I5I  grid[4] = [0.5,  -0.5]
            I6I7I8I  grid[5] = [0.5,  -0.25]
                     grid[6] = [0.25, -0.75]
                     grid[7] = [0.25, -0.5]
                     grid[8] = [0.25, -0.25]

        **Case 4 (sub_size>1 native)**

        The properties of this grid can be derived by combining Case's 2 and 3 above, whereby the grid is stored as
        an ndarray of shape [total_y_coordinates*sub_size, total_x_coordinates*sub_size, 2].

        All sub-pixels in masked pixels have values (0.0, 0.0).

        Parameters
        ----------
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        """

        self.sub_size = sub_size

    @property
    def sub_shape_native_1d(self) -> Tuple[int]:
        return (self.shape[0] * self.sub_size,)

    @property
    def sub_shape_native_2d(self) -> Tuple[int, int]:
        return (self.shape[0] * self.sub_size, self.shape[1] * self.sub_size)

    @property
    def sub_shape_slim(self) -> int:
        """
        The 1D shape of the mask's sub-grid, which is equivalent to the total number of unmasked pixels in the mask.
        """
        return int(self.pixels_in_mask * self.sub_size**self.dimensions)

    @property
    def native_for_slim(self) -> np.ndarray:
        """
        Derives a 1D ``ndarray`` which maps every non-subgridded 1D ``slim`` index of the ``Mask2D`` to its
        non-subgridded 2D ``native`` index.

        For example, for the following ``Mask2D``:

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

        For `sub_size=2`` each unmasked ``False`` entry is split into a sub-pixel of size 2x2.
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
