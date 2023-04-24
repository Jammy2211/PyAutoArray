from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Union, Tuple

if TYPE_CHECKING:
    from autoarray.structures.arrays.uniform_1d import Array1D
    from autoarray.structures.grids.uniform_2d import Grid2D
    from autoarray.structures.grids.transformed_2d import Grid2DTransformed
    from autoarray.structures.grids.transformed_2d import Grid2DTransformedNumpy

from autoarray.structures.abstract_structure import Structure
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.mask.mask_1d import Mask1D

from autoarray.structures.grids import grid_1d_util
from autoarray.structures.grids import grid_2d_util
from autoarray.geometry import geometry_util
from autoarray import type as ty


class Grid1D(Structure):
    def __new__(
        cls,
        values: Union[np.ndarray, List],
        mask: Mask1D,
        store_native: bool = False,
        *args,
        **kwargs,
    ):
        """
        A grid of 1D (x) coordinates, which are paired to a uniform 1D mask of pixels and sub-pixels. Each entry
        on the grid corresponds to the (x) coordinates at the centre of a sub-pixel of an unmasked pixel.

        A `Grid1D` is ordered such that pixels begin from the left (e.g. index [0]) of the corresponding mask
        and go right. The positive x-axis is to the right.

        The grid can be stored in two formats:

        - slimmed: all masked entries are removed so the ndarray is shape [total_unmasked_coordinates*sub_size]
        - native: it retains the original shape of the grid so the ndarray is
          shape [total_y_coordinates*sub_size, total_x_coordinates*sub_size, 2].


        **Case 1 (sub-size=1, slim)**

        The Grid1D is an ndarray of shape [total_unmasked_coordinates].

        The first element of the ndarray corresponds to the pixel index. For example:

        - grid[3] = the 4th unmasked pixel's x-coordinate.
        - grid[6] = the 7th unmasked pixel's x-coordinate.

        Below is a visual illustration of a grid, where a total of 3 pixels are unmasked and are included in the grid.

        .. code-block:: bash

            <--- -ve  x  +ve -->

            x x x O o x O x x x

        This is an example mask.Mask1D, where:

        .. code-block:: bash

            x = `True` (Pixel is masked and excluded from the grid)
            O = `False` (Pixel is not masked and included in the grid)

        The mask pixel index's will come out like this (and the direction of scaled coordinates is highlighted
        around the mask.

        .. code-block:: bash

            pixel_scales = 1.0"

            <--- -ve  x  +ve -->

            x x x 0 1 x 2 x x x

            grid[0] = [-1.5]
            grid[1] = [-0.5]
            grid[2] = [1.5]


        **Case 2 (sub-size>1, slim)

        If the mask's `sub_size` is > 1, the grid is defined as a sub-grid where each entry corresponds to the (x)
        coordinates at the centre of each sub-pixel of an unmasked pixel. The Grid1D is therefore stored as an ndarray
        of shape [total_unmasked_coordinates*sub_size].

        The sub-grid indexes are ordered such that pixels begin from the first (leftmost) sub-pixel in the first
        unmasked pixel. Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel.
        Therefore, the sub-grid is an ndarray of shape [total_unmasked_coordinates*sub_grid_shape]. For example:

        - grid[5] - using `sub_size=2`, gives the 3rd unmasked pixel's 2nd sub-pixel x-coordinate.
        - grid[3] - using `sub_size=3`, gives the 2nd unmasked pixel's 1st sub-pixel x-coordinate.
        - grid[10] - using `sub_size=3`, gives the 4th unmasked pixel's 1st sub-pixel y-coordinate.

        Below is a visual illustration of a sub grid. Indexing of each sub-pixel goes from the top-left corner. In
        contrast to the grid above, our illustration below restricts the mask to just 2 pixels, to keep the
        illustration brief.

        .. code-block:: bash

            x x x O x O x x x

            This is an example mask.Mask1D, where:

            x = `True` (Pixel is masked and excluded from the grid)
            O = `False` (Pixel is not masked and included in the grid)

        Our grid with a sub-size=1 looks like it did before:

        .. code-block:: bash

            pixel_scales = 1.0"

            <--- -ve  x  +ve -->

             x x x 0 x 1 x x x

        However, if the sub-size is 2, we go to each unmasked pixel and allocate sub-pixel coordinates for it. For
        example, for pixel 0, if `sub_size=2`:

        .. code-block:: bash

            grid[0] = [-0.75]
            grid[1] = [-0.25]

        If we used a sub_size of 3, for the pixel we we would create a 3x3 sub-grid:

        .. code-block:: bash

            grid[0] = [-0.833]
            grid[1] = [-0.5]
            grid[2] = [-0.166]


        **Case 3 (sub_size=1 native)**

        The Grid2D has the same properties as Case 1, but is stored as an an ndarray of shape [total_x_coordinates].

        All masked entries on the grid has (y,x) values of (0.0, 0.0).

        For the following example mask:

        .. code-block:: bash

            x x x O O x O x x x

            - grid[0] = 0.0 (it is masked, thus zero)
            - grid[1] = 0.0 (it is masked, thus zero)
            - grid[2] = 0.0 (it is masked, thus zero)
            - grid[3] = -1.5
            - grid[4] = -0.5
            - grid[5] = 0.0 (it is masked, thus zero)
            - grid[6] = 0.5


        **Case 4 (sub_size>1 native)**

        The properties of this grid can be derived by combining Case's 2 and 3 above, whereby the grid is stored as
        an ndarray of shape [total_x_coordinates*sub_size,].

        All sub-pixels in masked pixels have value 0.0.


        **Grid1D Mapping**

        Every set of (x) coordinates in a pixel of the sub-grid maps to an unmasked pixel in the mask. For a uniform
        grid, every x coordinate directly corresponds to the location of its paired unmasked pixel.

        It is not a requirement that grid is uniform and that their coordinates align with the mask. The input grid
        could be an irregular set of x coordinates where the indexing signifies that the x coordinate
        *originates* or *is paired with* the mask's pixels but has had its value change by some aspect of the
        calculation.

        This is important for the child project *PyAutoLens*, where grids in the image-plane are ray-traced and
        deflected to perform lensing calculations. The grid indexing is used to map pixels between the image-plane and
        source-plane.

        Parameters
        ----------
        values
            The (y,x) coordinates of the grid.
        mask
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        """

        values = grid_1d_util.convert_grid_1d(
            grid_1d=values, mask_1d=mask, store_native=store_native
        )

        obj = values.view(cls)
        obj.mask = mask

        return obj

    @classmethod
    def no_mask(
        cls,
        values: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float] = (0.0,),
    ) -> "Grid1D":
        """
        Create a Grid1D (see *Grid1D.__new__*) by inputting the grid coordinates in 1D.

        Parameters
        ----------
        values
            The (y,x) coordinates of the grid input as an ndarray of shape [total_unmasked_pixells*(sub_size**2), 2]
            or a list of lists.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.

        Examples
        --------

        .. code-block:: python

            import autogrid as aa

            # Make Grid1D from input np.ndgrid.

            grid_1d = aa.Grid1D.no_mask(grid=np.grid([1.0, 2.0, 3.0, 4.0]), pixel_scales=1.0)

            # Make Grid2D from input list.

            grid_1d = aa.Grid1D.no_mask(grid=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

            # Print grid's slim (masked 1D data representation) and
            # native (masked 1D data representation)

            print(grid_1d.slim)
            print(grid_1d.native)
        """

        pixel_scales = geometry_util.convert_pixel_scales_1d(pixel_scales=pixel_scales)

        values = grid_2d_util.convert_grid(grid=values)

        mask = Mask1D.all_false(
            shape_slim=values.shape[0] // sub_size,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return Grid1D(values=values, mask=mask)

    @classmethod
    def from_mask(cls, mask: Mask1D) -> "Grid1D":
        """
        Create a Grid1D (see *Grid1D.__new__*) from a mask, where only unmasked pixels are included in the grid (if the
        grid is represented in its native 1D masked values are 0.0).

        The mask's pixel_scales, sub_size and origin properties are used to compute the grid (x) coordinates.

        Parameters
        ----------
        mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        """

        sub_grid_1d = grid_1d_util.grid_1d_slim_via_mask_from(
            mask_1d=mask,
            pixel_scales=mask.pixel_scales,
            sub_size=mask.sub_size,
            origin=mask.origin,
        )

        return Grid1D(values=sub_grid_1d, mask=mask)

    @classmethod
    def uniform(
        cls,
        shape_native: Tuple[int],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float] = (0.0, 0.0),
    ) -> "Grid1D":
        """
        Create a `Grid1D` (see `Grid`D.__new__`) as a uniform grid of (x) values given an input `shape_native` and
        `pixel_scales` of the grid.

        Parameters
        ----------
        shape_native
            The 1D shape of the uniform grid and the mask that it is paired with.
        pixel_scales
            The (x) scaled units to pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float,) tuple.
        sub_size
            The size (sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask and coordinate system.
        """
        pixel_scales = geometry_util.convert_pixel_scales_1d(pixel_scales=pixel_scales)

        grid_slim = grid_1d_util.grid_1d_slim_via_shape_slim_from(
            shape_slim=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return cls.no_mask(
            values=grid_slim,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def uniform_from_zero(
        cls, shape_native: Tuple[int], pixel_scales: ty.PixelScales, sub_size: int = 1
    ) -> "Grid1D":
        """
        Create a `Grid1D` (see `Grid`D.__new__`) as a uniform grid of (x) values given an input `shape_native` and
        `pixel_scales` of the grid, where the first (x) coordinate of the grid is 0.0 and all other values ascend
        positively.

        Parameters
        ----------
        shape_native
            The 1D shape of the uniform grid and the mask that it is paired with.
        pixel_scales
            The (x) scaled units to pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float,) tuple.
            it is converted to a (float,) tuple.
        sub_size
            The size (sub_size) of each unmasked pixels sub-grid.
        """
        pixel_scales = geometry_util.convert_pixel_scales_1d(pixel_scales=pixel_scales)

        grid_slim = grid_1d_util.grid_1d_slim_via_shape_slim_from(
            shape_slim=shape_native, pixel_scales=pixel_scales, sub_size=sub_size
        )

        grid_slim -= np.min(grid_slim)

        return cls.no_mask(
            values=grid_slim, pixel_scales=pixel_scales, sub_size=sub_size
        )

    @property
    def slim(self) -> "Grid1D":
        """
        Return a `Grid1D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels * sub_size, 2].

        If it is already stored in its `slim` representation  the `Grid1D` is returned as it is. If not, it is
        mapped from  `native` to `slim` and returned as a new `Grid1D`.
        """
        return Grid1D(values=self, mask=self.mask)

    @property
    def native(self) -> "Grid1D":
        """
        Return a `Grid1D` where the data is stored in its `native` representation, which is an ndarray of shape
        [sub_size*total_x_pixels, 2].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Grid1D`.
        """
        return Grid1D(values=self, mask=self.mask, store_native=True)

    @property
    def binned(self) -> "Grid1D":
        """
        Convenience method to access the binned-up grid in its 1D representation, which is a Grid2D stored as an
        ndarray of shape [total_unmasked_pixels, 2].

        The binning up process converts a grid from (y,x) values where each value is a coordinate on the sub-grid to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a grid with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.
        """

        grid_1d_slim = self.slim

        binned_grid_1d_slim = np.multiply(
            self.mask.sub_fraction,
            grid_1d_slim.reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        return Grid1D(values=binned_grid_1d_slim, mask=self.mask.derive_mask.sub_1)

    def grid_2d_radial_projected_from(self, angle: float = 0.0) -> Grid2DIrregular:
        """
        Project the 1D grid of (y,x) coordinates to an irregular 2d grid of (y,x) coordinates. The projection works
        as follows:

        1) Map the 1D (x) coordinates to 2D along the x-axis, such that the x value of every 2D coordinate is the
        corresponding (x) value in the 1D grid, and every y value is 0.0.

        2) Rotate this projected 2D grid clockwise by the input angle.

        Parameters
        ----------
        angle
            The angle with which the project 2D grid of coordinates is rotated clockwise.

        Returns
        -------
        Grid2DIrregular
            The projected and rotated 2D grid of (y,x) coordinates.
        """
        grid = np.zeros((self.sub_shape_slim, 2))

        grid[:, 1] = self.slim

        grid = geometry_util.transform_grid_2d_to_reference_frame(
            grid_2d=grid, centre=(0.0, 0.0), angle=angle
        )

        return Grid2DIrregular(values=grid)

    def structure_2d_from(
        self, result: np.ndarray
    ) -> Union[Array1D, Grid2D, Grid2DTransformed, Grid2DTransformedNumpy]:
        """
        Convert a result from an ndarray to an aa.Array2D or aa.Grid2D structure, where the conversion depends on
        type(result) as follows:

        .. code-block:: bash

            - 1D np.ndarray   -> aa.Array2D
            - 2D np.ndarray   -> aa.Grid2D

        This function is used by the grid_2d_to_structure decorator to convert the output result of a function
        to an autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        from autoarray.structures.arrays.uniform_1d import Array1D
        from autoarray.structures.grids.transformed_2d import Grid2DTransformed
        from autoarray.structures.grids.transformed_2d import Grid2DTransformedNumpy
        from autoarray.structures.grids.uniform_2d import Grid2D

        if len(result.shape) == 1:
            return Array1D(values=result, mask=self.mask)

        if isinstance(result, Grid2DTransformedNumpy):
            return Grid2DTransformed(values=result, mask=self.mask)
        return Grid2D(values=result, mask=self.mask.derive_mask.to_mask_2d)

    def structure_2d_list_from(
        self, result_list: List
    ) -> List[Union[Array1D, Grid2D, Grid2DTransformed, Grid2DTransformedNumpy]]:
        """
        Convert a result from a list of ndarrays to a list of aa.Array2D or aa.Grid2D structure, where the conversion
        depends on type(result) as follows:

        .. code-block:: bash

            - [1D np.ndarray] -> [aa.Array2D]
            - [2D np.ndarray] -> [aa.Grid2D]

        This function is used by the grid_like_list_to_structure-list decorator to convert the output result of a
        function to a list of autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result_list
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        return [self.structure_2d_from(result=result) for result in result_list]
