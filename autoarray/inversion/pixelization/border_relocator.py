from __future__ import annotations
import numpy as np

from autoconf import cached_property

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.mask import mask_2d_util
from autoarray.structures.grids import grid_2d_util


class BorderRelocator:
    def __init__(self, mask: Mask2D, sub_size: int):
        self.mask = mask
        self.sub_size = sub_size

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
        return mask_2d_util.sub_border_pixel_slim_indexes_from(
            mask_2d=np.array(self.mask), sub_size=self.sub_size
        ).astype("int")

    @property
    def sub_grid(self):
        return grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=np.array(self.mask),
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.sub_size,
            origin=self.mask.origin,
        )

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
        return self.sub_grid[self.sub_border_slim]

    def relocated_grid_from(self, grid: "Grid2D") -> "Grid2D":
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

        print(grid[2])

        print(
            Grid2D(
                values=grid_2d_util.relocated_grid_via_jit_from(
                    grid=np.array(grid),
                    border_grid=np.array(grid[self.sub_border_slim]),
                ),
                mask=grid.mask,
                over_sampling=grid.over_sampling,
            )[2]
        )

        return Grid2D(
            values=grid_2d_util.relocated_grid_via_jit_from(
                grid=np.array(grid),
                border_grid=np.array(grid[self.sub_border_slim]),
            ),
            mask=grid.mask,
            over_sampling=grid.over_sampling,
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
