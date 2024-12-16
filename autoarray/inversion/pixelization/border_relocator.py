from __future__ import annotations
import numpy as np
from typing import Union

from autoconf import cached_property

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.mask import mask_2d_util

from autoarray.operators.over_sampling import over_sample_util
from autoarray.structures.grids import grid_2d_util


def sub_slim_indexes_for_slim_index_via_mask_2d_from(
    mask_2d: np.ndarray, sub_size: Array2D
) -> [list]:
    """ "
    For pixels on a native 2D array of shape (total_y_pixels, total_x_pixels), compute a list of lists which, for every
    unmasked pixel giving its slim pixel indexes of its corresponding sub-pixels.

    For example, for a sub-grid size of 2, the following mappings from sub-pixels to 2D array pixels are:

    - sub_slim_indexes_for_slim_index[0] = [0, 1, 2, 3] -> The first pixel maps to the first 4 subpixels in 1D.
    - sub_slim_indexes_for_slim_index[1] = [4, 5, 6, 7] -> The seond pixel maps to the next 4 subpixels in 1D.

    Parameters
    ----------
    mask_2d
        The mask whose indexes are mapped.
    sub_size
        The size of the sub-grid in each mask pixel.


    Returns
    -------
    [list]
        The lists of the 1D sub-pixel indexes in every unmasked pixel in the mask.
    The term 'grid' is used because the grid is defined as the grid of coordinates on the centre of every
    pixel on the 2D array. Thus, this array maps sub-pixels on a sub-grid to pixels on a grid.

    Examples
    --------
    mask = ([[True, False, True]])
    sub_mask_1d_indexes_for_mask_1d_index = sub_mask_1d_indexes_for_mask_1d_index_from(mask=mask, sub_size=2)
    """

    total_pixels = mask_2d_util.total_pixels_2d_from(mask_2d=mask_2d)

    sub_slim_indexes_for_slim_index = [[] for _ in range(total_pixels)]

    slim_index_for_sub_slim_indexes = (
        over_sample_util.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask_2d=mask_2d, sub_size=np.array(sub_size)
        ).astype("int")
    )

    for sub_slim_index, slim_index in enumerate(slim_index_for_sub_slim_indexes):
        sub_slim_indexes_for_slim_index[slim_index].append(sub_slim_index)

    return sub_slim_indexes_for_slim_index


def sub_border_pixel_slim_indexes_from(
    mask_2d: np.ndarray, sub_size: Array2D
) -> np.ndarray:
    """
    Returns a slim array of shape [total_unmasked_border_pixels] listing all sub-borders pixel indexes in
    the mask.

    A borders pixel is a pixel which:

    1) is not fully surrounding by `False` mask values.
    2) Can reach the edge of the array without hitting a masked pixel in one of four directions (upwards, downwards,
       left, right).

    The borders pixels are thus pixels which are on the exterior edge of the mask. For example, the inner ring of
    edge pixels in an annular mask are edge pixels but not borders pixels.

    A sub-border pixel is, for a border-pixel, the pixel within that border pixel which is furthest from the origin
    of the mask.

    Parameters
    ----------
    mask_2d
        The mask for which the 1D border pixel indexes are calculated.
    sub_size
        The size of the sub-grid in each mask pixel.

    Returns
    -------
    np.ndarray
        The 1D indexes of all border sub-pixels on the mask.
    """

    border_pixels = mask_2d_util.border_slim_indexes_from(mask_2d=mask_2d)

    sub_border_pixels = np.zeros(shape=border_pixels.shape[0])

    sub_slim_indexes_for_slim_index = sub_slim_indexes_for_slim_index_via_mask_2d_from(
        mask_2d=mask_2d, sub_size=sub_size
    )

    sub_grid_2d_slim = over_sample_util.grid_2d_slim_over_sampled_via_mask_from(
        mask_2d=mask_2d,
        pixel_scales=(1.0, 1.0),
        sub_size=np.array(sub_size),
        origin=(0.0, 0.0),
    )
    mask_centre = grid_2d_util.grid_2d_centre_from(grid_2d_slim=sub_grid_2d_slim)

    for border_1d_index, border_pixel in enumerate(border_pixels):
        sub_border_pixels_of_border_pixel = sub_slim_indexes_for_slim_index[
            int(border_pixel)
        ]

        sub_border_pixels[
            border_1d_index
        ] = grid_2d_util.furthest_grid_2d_slim_index_from(
            grid_2d_slim=sub_grid_2d_slim,
            slim_indexes=sub_border_pixels_of_border_pixel,
            coordinate=mask_centre,
        )

    return sub_border_pixels


class BorderRelocator:
    def __init__(self, mask: Mask2D, sub_size: Union[int, Array2D]):
        self.mask = mask

        if isinstance(sub_size, int):
            sub_size = Array2D(
                values=np.full(fill_value=sub_size, shape=mask.shape_slim), mask=mask
            )

        self.sub_size = sub_size

    @cached_property
    def border_slim(self):
        """
        Returns the  1D ``slim`` indexes of border pixels in the ``Mask2D``, representing all unmasked
        sub-pixels (given by ``False``) which neighbor any masked value (give by ``True``) and which are on the
        extreme exterior of the mask.

        The indexes are the extended below to form the ``sub_border_slim`` which is illustrated above.

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
            )

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.border_slim)
        """
        return self.mask.derive_indexes.border_slim

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
            )

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.sub_border_slim)
        """
        return sub_border_pixel_slim_indexes_from(
            mask_2d=np.array(self.mask), sub_size=self.sub_size
        ).astype("int")

    @cached_property
    def border_grid(self) -> np.ndarray:
        """
        The (y,x) grid of all sub-pixels which are at the border of the mask.

        This is NOT all sub-pixels which are in mask pixels at the mask's border, but specifically the sub-pixels
        within these border pixels which are at the extreme edge of the border.
        """
        return self.mask.derive_grid.border

    @cached_property
    def sub_border_grid(self) -> np.ndarray:
        """
        The (y,x) grid of all sub-pixels which are at the border of the mask.

        This is NOT all sub-pixels which are in mask pixels at the mask's border, but specifically the sub-pixels
        within these border pixels which are at the extreme edge of the border.
        """
        sub_grid = over_sample_util.grid_2d_slim_over_sampled_via_mask_from(
            mask_2d=np.array(self.mask),
            pixel_scales=self.mask.pixel_scales,
            sub_size=np.array(self.sub_size),
            origin=self.mask.origin,
        )

        return sub_grid[self.sub_border_slim]

    def relocated_grid_from(self, grid: Grid2D) -> Grid2D:
        """
        Relocate the coordinates of a grid to the border of this grid if they are outside the border, where the
        border is defined as all pixels at the edge of the grid's mask (see *mask._border_1d_indexes*).

        This is performed as follows:

        1: Use the mean value of the grid's y and x coordinates to determine the origin of the grid.
        2: Compute the radial distance of every grid coordinate from the origin.
        3: For every coordinate, find its nearest pixel in the border.
        4: Determine if it is outside the border, by comparing its radial distance from the origin to its paired
        border pixel's radial distance.
        5: If its radial distance is larger, use the ratio of radial distances to move the coordinate to the
        border (if its inside the border, do nothing).

        The method can be used on uniform or irregular grids, however for irregular grids the border of the
        'image-plane' mask is used to define border pixels.

        Parameters
        ----------
        grid : Grid2D
            The grid (uniform or irregular) whose pixels are to be relocated to the border edge if outside it.
        """

        if len(self.sub_border_grid) == 0:
            return grid

        values = grid_2d_util.relocated_grid_via_jit_from(
            grid=np.array(grid),
            border_grid=np.array(grid[self.border_slim]),
        )

        over_sampled = grid_2d_util.relocated_grid_via_jit_from(
            grid=np.array(grid.over_sampled),
            border_grid=np.array(grid.over_sampled[self.sub_border_slim]),
        )

        return Grid2D(
            values=values,
            mask=grid.mask,
            over_sample_size=self.sub_size,
            over_sampled=over_sampled,
        )

    def relocated_mesh_grid_from(
        self, grid, mesh_grid: Grid2DIrregular
    ) -> Grid2DIrregular:
        """
        Relocate the coordinates of a pixelization grid to the border of this grid. See the
        method ``relocated_grid_from()`` for a full description of how this grid relocation works.

        Parameters
        ----------
        grid
            The pixelization grid whose pixels are relocated to the border edge if outside it.
        """

        if len(self.sub_border_grid) == 0:
            return mesh_grid

        return Grid2DIrregular(
            values=grid_2d_util.relocated_grid_via_jit_from(
                grid=np.array(mesh_grid),
                border_grid=np.array(grid[self.sub_border_slim]),
            ),
        )
