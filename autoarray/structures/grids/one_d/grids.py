from autoarray.structures.grids import abstract_grid


class Grid1D(abstract_grid.AbstractGrid1D):
    def __new__(cls, grid, mask, store_slim=True, *args, **kwargs):
        """
        A grid of 1D (x) coordinates, which are paired to a uniform 1D mask of pixels and sub-pixels. Each entry
        on the grid corresponds to the (x) coordinates at the centre of a sub-pixel of an unmasked pixel.

        A `Grid1D` is ordered such that pixels begin from the left (e.g. index [0]) of the corresponding mask
        and go right. The positive x-axis is to the right.

        The grid can be stored in two formats:

        - slimmed: all masked entries are removed so the ndarray is shape [total_unmasked_coordinates*sub_size]
        - native: it retains the original shape of the grid so the ndarray is
          shape [total_y_coordinates*sub_size, total_x_coordinates*sub_size, 2].

        Case 1: [sub-size=1, store_slim=True]:
        -----------------------------------------

        The Grid1D is an ndarray of shape [total_unmasked_coordinates].

        The first element of the ndarray corresponds to the pixel index. For example:

        - grid[3] = the 4th unmasked pixel's x-coordinate.
        - grid[6] = the 7th unmasked pixel's x-coordinate.

        Below is a visual illustration of a grid, where a total of 3 pixels are unmasked and are included in \
        the grid.

        x x x o o x o x x x

        This is an example mask.Mask1D, where:

        x = `True` (Pixel is masked and excluded from the grid)
        o = `False` (Pixel is not masked and included in the grid)

        The mask pixel index's will come out like this (and the direction of scaled coordinates is highlighted
        around the mask.

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                   x
         x x x 0 1 x 2 x x x

         grid[0] = [-1.5]
         grid[1] = [-0.5]
         grid[2] = [1.5]

        Case 2: [sub-size>1, store_slim=True]:
        ------------------

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

        x x x o x o x x x

        This is an example mask.Mask1D, where:

        x = `True` (Pixel is masked and excluded from the grid)
        o = `False` (Pixel is not masked and included in the grid)

        Our grid with a sub-size=1 looks like it did before:

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                   x
         x x x 0 x 1 x x x

        However, if the sub-size is 2, we go to each unmasked pixel and allocate sub-pixel coordinates for it. For
        example, for pixel 0, if `sub_size=2`:

        grid[0] = [-0.75]
        grid[1] = [-0.25]

        If we used a sub_size of 3, for the pixel we we would create a 3x3 sub-grid:

        grid[0] = [-0.833]
        grid[1] = [-0.5]
        grid[2] = [-0.166]

        Case 3: [sub_size=1 store_slim=False]
        --------------------------------------

        The Grid2D has the same properties as Case 1, but is stored as an an ndarray of shape
        [total_x_coordinates].

        All masked entries on the grid has (y,x) values of (0.0, 0.0).

        For the following example mask:

        x x x o o x o x x x

        - grid[0] = 0.0 (it is masked, thus zero)
        - grid[1] = 0.0 (it is masked, thus zero)
        - grid[2] = 0.0 (it is masked, thus zero)
        - grid[3] = -1.5
        - grid[4] = -0.5
        - grid[5] = 0.0 (it is masked, thus zero)
        - grid[6] = 0.5

        Case 4: [sub_size>1 store_slim=False]
        --------------------------------------

        The properties of this grid can be derived by combining Case's 2 and 3 above, whereby the grid is stored as
        an ndarray of shape [total_x_coordinates*sub_size,].

        All sub-pixels in masked pixels have value 0.0.

        Grid1D Mapping
        ------------

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
        grid : np.ndarray
            The (y,x) coordinates of the grid.
        mask : msk.Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_coordinates, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_coordinates, total_x_coordinates, 2].
        """

        obj = grid.view(cls)
        obj.mask = mask
        obj.store_slim = store_slim

        abstract_grid.check_grid_2d(grid_2d=obj)

        return obj

    @classmethod
    def manual_slim(
        cls,
        grid,
        shape_native,
        pixel_scales,
        sub_size=1,
        origin=(0.0, 0.0),
        store_slim=True,
    ):
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates in 1D, for example:

        grid=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_unmasked_pixells*(sub_size**2), 2]
            or a list of lists.
        shape_native : (float, float)
            The 2D shape of the mask the grid is paired with.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin : (float, float)
            The origin of the grid's mask.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_coordinates, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_coordinates, total_x_coordinates, 2].
        """

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = msk.Mask2D.unmasked(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        grid = abstract_grid.convert_manual_grid_2d_slim(
            grid_2d_slim=grid, mask_2d=mask, store_slim=store_slim
        )

        return Grid2D(grid=grid, mask=mask, store_slim=store_slim)
