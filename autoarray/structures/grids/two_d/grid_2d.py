import numpy as np
from sklearn.cluster import KMeans
from autoarray import exc
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray.structures.grids import abstract_grid
from autoarray.structures.grids.two_d import abstract_grid_2d
from autoarray.mask import mask_2d as msk, mask_2d_util
from autoarray.structures.grids.two_d import grid_2d_util, sparse_util
from autoarray.geometry import geometry_util


class Grid2D(abstract_grid_2d.AbstractGrid2D):
    def __new__(cls, grid, mask, store_slim=True, *args, **kwargs):
        """
        A grid of 2D (y,x) coordinates, which are paired to a uniform 2D mask of pixels and sub-pixels. Each entry
        on the grid corresponds to the (y,x) coordinates at the centre of a sub-pixel of an unmasked pixel.

        A `Grid2D` is ordered such that pixels begin from the top-row (e.g. index [0, 0]) of the corresponding mask
        and go right and down. The positive y-axis is upwards and positive x-axis to the right.

        The grid can be stored in two formats:

        - slimmed: all masked entries are removed so the ndarray is shape [total_unmasked_coordinates*sub_size**2, 2]
        - native: it retains the original shape of the grid so the ndarray is
          shape [total_y_coordinates*sub_size, total_x_coordinates*sub_size, 2].

        Case 1: [sub-size=1, store_slim = True]:
        -----------------------------------------

        The Grid2D is an ndarray of shape [total_unmasked_coordinates, 2], therefore when store_slim=True the shape of
        the grid is 2, not 1.

        The first element of the ndarray corresponds to the pixel index and second element the y or x coordinate value.
        For example:

        - grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Below is a visual illustration of a grid, where a total of 10 pixels are unmasked and are included in \
        the grid.

         x x x x x x x x x x
         x x x x x x x x x x     This is an example mask.Mask2D, where:
         x x x x x x x x x x
         x x x xIoIo x x x x     x = `True` (Pixel is masked and excluded from the grid)
         x x xIoIoIoIo x x x     o = `False` (Pixel is not masked and included in the grid)
         x x xIoIoIoIo x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x

        The mask pixel index's will come out like this (and the direction of scaled coordinates is highlighted
        around the mask.

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                        y      x
         x x x x x x x x x x  ^   grid[0] = [ 1.5, -0.5]
         x x x x x x x x x x  I   grid[1] = [ 1.5,  0.5]
         x x x x x x x x x x  I   grid[2] = [ 0.5, -1.5]
         x x x xI0I1 x x x x +ve  grid[3] = [ 0.5, -0.5]
         x x xI2I3I4I5 x x x  y   grid[4] = [ 0.5,  0.5]
         x x xI6I7I8I9 x x x -ve  grid[5] = [ 0.5,  1.5]
         x x x x x x x x x x  I   grid[6] = [-0.5, -1.5]
         x x x x x x x x x x  I   grid[7] = [-0.5, -0.5]
         x x x x x x x x x x \/   grid[8] = [-0.5,  0.5]
         x x x x x x x x x x      grid[9] = [-0.5,  1.5]

        Case 2: [sub-size>1, store_slim=True]:
        ------------------

        If the mask's `sub_size` is > 1, the grid is defined as a sub-grid where each entry corresponds to the (y,x)
        coordinates at the centre of each sub-pixel of an unmasked pixel. The Grid2D is therefore stored as an ndarray
        of shape [total_unmasked_coordinates*sub_size**2, 2]

        The sub-grid indexes are ordered such that pixels begin from the first (top-left) sub-pixel in the first
        unmasked pixel. Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel.
        Therefore, the sub-grid is an ndarray of shape [total_unmasked_coordinates*(sub_grid_shape)**2, 2]. For example:

        - grid[9, 1] - using a 2x2 sub-grid, gives the 3rd unmasked pixel's 2nd sub-pixel x-coordinate.
        - grid[9, 1] - using a 3x3 sub-grid, gives the 2nd unmasked pixel's 1st sub-pixel x-coordinate.
        - grid[27, 0] - using a 3x3 sub-grid, gives the 4th unmasked pixel's 1st sub-pixel y-coordinate.

        Below is a visual illustration of a sub grid. Indexing of each sub-pixel goes from the top-left corner. In
        contrast to the grid above, our illustration below restricts the mask to just 2 pixels, to keep the
        illustration brief.

         x x x x x x x x x x
         x x x x x x x x x x     This is an example mask.Mask2D, where:
         x x x x x x x x x x
         x x x x x x x x x x     x = `True` (Pixel is masked and excluded from lens)
         x x x xIoIo x x x x     o = `False` (Pixel is not masked and included in lens)
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x
         x x x x x x x x x x

        Our grid with a sub-size looks like it did before:

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->

         x x x x x x x x x x  ^
         x x x x x x x x x x  I
         x x x x x x x x x x  I                        y     x
         x x x x x x x x x x +ve  grid[0] = [0.5,  -1.5]
         x x x 0 1 x x x x x  y   grid[1] = [0.5,  -0.5]
         x x x x x x x x x x -ve
         x x x x x x x x x x  I
         x x x x x x x x x x  I
         x x x x x x x x x x \/
         x x x x x x x x x x

        However, if the sub-size is 2, we go to each unmasked pixel and allocate sub-pixel coordinates for it. For
        example, for pixel 0, if *sub_size=2*, we use a 2x2 sub-grid:

        Pixel 0 - (2x2):
                            y      x
               grid[0] = [0.66, -1.66]
        I0I1I  grid[1] = [0.66, -1.33]
        I2I3I  grid[2] = [0.33, -1.66]
               grid[3] = [0.33, -1.33]

        If we used a sub_size of 3, for the pixel we we would create a 3x3 sub-grid:

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

        Case 3: [sub_size=1, store_slim=False]
        --------------------------------------

        The Grid2D has the same properties as Case 1, but is stored as an an ndarray of shape
        [total_y_coordinates, total_x_coordinates, 2]. Therefore when store_slim=False the shape of the
        grid is 3, not 2.

        All masked entries on the grid has (y,x) values of (0.0, 0.0).

        For the following example mask:

         x x x x x x x x x xI
         x x x x x x x x x xI     This is an example mask.Mask2D, where:
         x x x x x x x x x xI
         x x x x o o x x x xI     x = `True` (Pixel is masked and excluded from the grid)
         x x x o o o o x x xI     o = `False` (Pixel is not masked and included in the grid)
         x x x o o o o x x xI
         x x x x x x x x x xI
         x x x x x x x x x xI
         x x x x x x x x x xI
         x x x x x x x x x xI

        - grid[0,0,0] = 0.0 (it is masked, thus zero)
        - grid[0,0,1] = 0.0 (it is masked, thus zero)
        - grid[3,3,0] = 0.0 (it is masked, thus zero)
        - grid[3,3,1] = 0.0 (it is masked, thus zero)
        - grid[3,4,0] = 1.5
        - grid[3,4,1] = -0.5

        Case 4: [sub_size>1 store_slim=False]
        --------------------------------------

        The properties of this grid can be derived by combining Case's 2 and 3 above, whereby the grid is stored as
        an ndarray of shape [total_y_coordinates*sub_size, total_x_coordinates*sub_size, 2].

        All sub-pixels in masked pixels have values (0.0, 0.0).

        Grid2D Mapping
        ------------

        Every set of (y,x) coordinates in a pixel of the sub-grid maps to an unmasked pixel in the mask. For a uniform
        grid, every (y,x) coordinate directly corresponds to the location of its paired unmasked pixel.

        It is not a requirement that grid is uniform and that their coordinates align with the mask. The input grid
        could be an irregular set of (y,x) coordinates where the indexing signifies that the (y,x) coordinate
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

        abstract_grid_2d.check_grid_2d(grid_2d=obj)

        return obj

    def _new_structure(self, grid, mask, store_slim):
        """Conveninence method for creating a new instance of the Grid2D class from this grid.

        This method is over-written by other grids (e.g. Grid2DIterate) such that the slim and native methods return
        instances of that Grid2D's type.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_sub_coordinates, 2] or list of lists.
        mask : msk.Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_coordinates, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_coordinates, total_x_coordinates, 2].
        """
        return Grid2D(grid=grid, mask=mask, store_slim=store_slim)

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
        """Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates in 1D, for example:

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

        grid = abstract_grid_2d.convert_manual_grid_2d_slim(
            grid_2d_slim=grid, mask_2d=mask, store_slim=store_slim
        )

        return Grid2D(grid=grid, mask=mask, store_slim=store_slim)

    @classmethod
    def manual_native(
        cls, grid, pixel_scales, sub_size=1, origin=(0.0, 0.0), store_slim=True
    ):
        """Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates in 2D, for example:

        grid=np.ndarray([[[1.0, 1.0], [2.0, 2.0]],
                         [[3.0, 3.0], [4.0, 4.0]]])

        grid=[[[1.0, 1.0], [2.0, 2.0]],
              [[3.0, 3.0], [4.0, 4.0]]]

        The 2D shape of the grid and its mask are determined from the input grid and the mask is setup as an
        unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape
            [total_y_coordinates*sub_size, total_x_pixel*sub_size, 2] or a list of lists.
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

        grid = abstract_grid.convert_grid(grid=grid)

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        shape = (int(grid.shape[0] / sub_size), int(grid.shape[1] / sub_size))

        mask = msk.Mask2D.unmasked(
            shape_native=shape,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        grid = abstract_grid_2d.convert_manual_grid_2d_native(
            grid_2d_native=grid, mask_2d=mask, store_slim=store_slim
        )

        return Grid2D(grid=grid, mask=mask, store_slim=store_slim)

    @classmethod
    def manual(
        cls,
        grid,
        pixel_scales,
        shape_native=None,
        sub_size=1,
        origin=(0.0, 0.0),
        store_slim=True,
    ):
        """Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates in 1D or 2D, automatically
        determining whether to use the 'manual_slim' or 'manual_native' methods.

        See the manual_slim and manual_native methods for examples.

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
        if len(grid.shape) == 2:
            return cls.manual_slim(
                grid=grid,
                shape_native=shape_native,
                pixel_scales=pixel_scales,
                sub_size=sub_size,
                origin=origin,
                store_slim=store_slim,
            )
        return cls.manual_native(
            grid=grid,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_slim=store_slim,
        )

    @classmethod
    def manual_mask(cls, grid, mask, store_slim=True):
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates in their native or slimmed format with
        their corresponding mask, automatically determining whether to use the 'manual_slim' or 'manual_native' methods.

        See the manual_slim and manual_native methods for examples.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_sub_coordinates, 2] or list of lists.
        mask : msk.Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_coordinates, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_coordinates, total_x_coordinates, 2].
        """

        grid = abstract_grid.convert_grid(grid=grid)
        abstract_grid_2d.check_grid_2d_and_mask_2d(grid_2d=grid, mask_2d=mask)

        grid = abstract_grid_2d.convert_manual_grid_2d(
            grid_2d=grid, mask_2d=mask, store_slim=store_slim
        )

        return Grid2D(grid=grid, mask=mask, store_slim=store_slim)

    @classmethod
    def manual_yx_1d(
        cls,
        y,
        x,
        shape_native,
        pixel_scales,
        sub_size=1,
        origin=(0.0, 0.0),
        store_slim=True,
    ):
        """Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates as 1D y and x values, for example:

        y = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = [1.0, 2.0, 3.0, 4.0]
        x = [1.0, 2.0, 3.0, 4.0]

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        y : np.ndarray or list
            The y coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        x : np.ndarray or list
            The x coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
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
        if type(y) is list:
            y = np.asarray(y)

        if type(x) is list:
            x = np.asarray(x)

        return cls.manual_slim(
            grid=np.stack((y, x), axis=-1),
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_slim=store_slim,
        )

    @classmethod
    def manual_yx_2d(
        cls, y, x, pixel_scales, sub_size=1, origin=(0.0, 0.0), store_slim=True
    ):
        """Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates as 2D y and x values, for example:

        y = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
        x = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        y = [[1.0, 2.0],
             [3.0, 4.0]]
        x = [[1.0, 2.0],
             [3.0, 4.0]]

        The 2D shape of the grid and its mask are determined from the input grid and the mask is setup as an
        unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        y : np.ndarray or list
            The y coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        x : np.ndarray or list
            The x coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
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
        if type(y) is list:
            y = np.asarray(y)

        if type(x) is list:
            x = np.asarray(x)

        return cls.manual_native(
            grid=np.stack((y, x), axis=-1),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_slim=store_slim,
        )

    @classmethod
    def uniform(
        cls, shape_native, pixel_scales, sub_size=1, origin=(0.0, 0.0), store_slim=True
    ):
        """Create a Grid2D (see *Grid2D.__new__*) as a uniform grid of (y,x) values given an input shape_native and pixel
        scale of the grid:

        Parameters
        ----------
        shape_native : (float, float)
            The 2D shape of the uniform grid and the mask that it is paired with.
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

        grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return cls.manual_slim(
            grid=grid_slim,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_slim=store_slim,
        )

    @classmethod
    def bounding_box(
        cls,
        bounding_box,
        shape_native,
        sub_size=1,
        buffer_around_corners=False,
        store_slim=True,
    ):
        """Create a Grid2D (see *Grid2D.__new__*) from an input bounding box with coordinates [y_min, y_max, x_min, x_max],
        where the shape_native is used to compute the (y,x) grid values within this bounding box.

        If buffer_around_corners=True, the grid's (y,x) values fully align with the input bounding box values. This
        means the mask's edge pixels extend beyond the bounding box by pixel_scale/2.0. If buffer_around_corners=False,
        the grid (y,x) coordinates are defined within the bounding box such that the mask's edge pixels align with
        the bouning box.

        Parameters
        ----------
        shape_native : (float, float)
            The 2D shape of the uniform grid and the mask that it is paired with.
        pixel_scales: (float, float) or float
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin : (float, float)
            The origin of the grid's mask.
        buffer_around_corners : bool
            Whether the grid is buffered such that the (y,x) values in the centre of its masks' edge pixels align
            with the input bounding box values.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_coordinates, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_coordinates, total_x_coordinates, 2].
        """
        y_min, y_max, x_min, x_max = bounding_box

        if not buffer_around_corners:

            pixel_scales = (
                (y_max - y_min) / (shape_native[0]),
                (x_max - x_min) / (shape_native[1]),
            )

        else:

            pixel_scales = (
                (y_max - y_min) / (shape_native[0] - 1),
                (x_max - x_min) / (shape_native[1] - 1),
            )
        origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)

        return cls.uniform(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_slim=store_slim,
        )

    @classmethod
    def from_mask(cls, mask, store_slim=True):
        """Create a Grid2D (see *Grid2D.__new__*) from a mask, where only unmasked pixels are included in the grid (if the
        grid is represented in 2D masked values are (0.0, 0.0)).

        The mask's pixel_scales, sub_size and origin properties are used to compute the grid (y,x) coordinates.

        Parameters
        ----------
        mask : Mask2D
            The mask whose masked pixels are used to setup the sub-pixel grid.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_coordinates, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_coordinates, total_x_coordinates, 2].
        """

        sub_grid_1d = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=mask,
            pixel_scales=mask.pixel_scales,
            sub_size=mask.sub_size,
            origin=mask.origin,
        )

        if store_slim:
            return Grid2D(grid=sub_grid_1d, mask=mask, store_slim=store_slim)

        sub_grid_2d = grid_2d_util.grid_2d_native_from(
            grid_2d_slim=sub_grid_1d, mask_2d=mask, sub_size=mask.sub_size
        )

        return Grid2D(grid=sub_grid_2d, mask=mask, store_slim=store_slim)

    @classmethod
    def from_fits(
        cls, file_path, pixel_scales, sub_size=1, origin=(0.0, 0.0), store_slim=True
    ):
        """Create a Grid2D (see *Grid2D.__new__*) from a mask, where only unmasked pixels are included in the grid (if the
        grid is represented in 2D masked values are (0.0, 0.0)).

        The mask's pixel_scales, sub_size and origin properties are used to compute the grid (y,x) coordinates.

        Parameters
        ----------
        mask : Mask2D
            The mask whose masked pixels are used to setup the sub-pixel grid.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_coordinates, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_coordinates, total_x_coordinates, 2].
        """

        sub_grid_2d = array_2d_util.numpy_array_2d_from_fits(file_path=file_path, hdu=0)

        return Grid2D.manual(
            grid=sub_grid_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_slim=store_slim,
        )

    @classmethod
    def blurring_grid_from_mask_and_kernel_shape(
        cls, mask, kernel_shape_native, store_slim=True
    ):
        """
        Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked (and
        therefore have their values set to (0.0, 0.0)), but are close enough to the unmasked pixels that their values
        will be convolved into the unmasked those pixels. This occurs in *PyAutoGalaxy* when computing images from
        light profile objects.

        The mask's pixel_scales, sub_size and origin properties are used to compute the blurring grid's (y,x)
        coordinates.

        For example, if our mask is as follows:

         x x x x x x x x x xI
         x x x x x x x x x xI     This is an imaging.Mask2D, where
         x x x x x x x x x xI
         x x x x x x x x x xI     x = `True` (Pixel is masked and excluded from lens)
         x x xIoIoIo x x x xI     o = `False` (Pixel is not masked and included in lens)
         x x xIoIoIo x x x xI
         x x xIoIoIo x x x xI
         x x x x x x x x x xI
         x x x x x x x x x xI
         x x x x x x x x x xI

        For a PSF of shape (3,3), the following blurring mask is computed (noting that only pixels that are direct
        neighbors of the unmasked pixels above will blur light into an unmasked pixel)

         x x x x x x x x xI     This is an example grid.Mask2D, where
         x x x x x x x x xI
         x xIoIoIoIoIo x xI     x = `True` (Pixel is masked and excluded from lens)
         x xIo x x xIo x xI     o = `False` (Pixel is not masked and included in lens)
         x xIo x x xIo x xI
         x xIo x x xIo x xI
         x xIoIoIoIoIo x xI
         x x x x x x x x xI
         x x x x x x x x xI

        Thus, the blurring grid coordinates and indexes will be as follows

        pixel_scales = 1.0"

        positive    negative
                                                            y     x                          y     x
         x x x  x  x  x  x  x xI  I   blurring_grid[0] = [2.0, -2.0]  blurring_grid[9] =  [-1.0, -2.0]
         x x x  x  x  x  x  x xI  I   blurring_grid[1] = [2.0, -1.0]  blurring_grid[10] = [-1.0,  2.0]
         x xI0 I1 I2 I3 I4  x xI pos  blurring_grid[2] = [2.0,  0.0]  blurring_grid[11] = [-2.0, -2.0]
         x xI5  x  x  x I6  x xI  y   blurring_grid[3] = [2.0,  1.0]  blurring_grid[12] = [-2.0, -1.0]
         x xI7  x  x  x I8  x xI  I   blurring_grid[4] = [2.0,  2.0]  blurring_grid[13] = [-2.0,  0.0]
         x xI9  x  x  x I10 x xI neg  blurring_grid[5] = [1.0, -2.0]  blurring_grid[14] = [-2.0,  1.0]
         x xI11I12I13I14I15 x xI  I   blurring_grid[6] = [1.0,  2.0]  blurring_grid[15] = [-2.0,  2.0]
         x x x  x  x  x  x  x xI  I   blurring_grid[7] = [0.0, -2.0]
         x x x  x  x  x  x  x xI  I   blurring_grid[8] = [0.0,  2.0]

        For a PSF of shape (5,5), the following blurring mask is computed (noting that pixels are 2 pixels from a
        direct unmasked pixels now blur light into an unmasked pixel)

         x x x x x x x x xI     This is an example grid.Mask2D, where
         xIoIoIoIoIoIoIo xI
         xIoIoIoIoIoIoIo xI     x = `True` (Pixel is masked and excluded from lens)
         xIoIo x x xIoIo xI     o = `False` (Pixel is not masked and included in lens)
         xIoIo x x xIoIo xI
         xIoIo x x xIoIo xI
         xIoIoIoIoIoIoIo xI
         xIoIoIoIoIoIoIo xI
         x x x x x x x x xI

        Parameters
        ----------
        mask : Mask2D
            The mask whose masked pixels are used to setup the blurring grid.
        kernel_shape_native : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        store_slim : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_coordinates, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_coordinates, total_x_coordinates, 2].
        """

        blurring_mask = mask.blurring_mask_from_kernel_shape(
            kernel_shape_native=kernel_shape_native
        )

        return cls.from_mask(mask=blurring_mask, store_slim=store_slim)

    def grid_from_deflection_grid(self, deflection_grid):
        """
        Returns a new Grid2D from this grid, where the (y,x) coordinates of this grid have a grid of (y,x) values,
             termed the deflection grid, subtracted from them to determine the new grid of (y,x) values.

            This is used by PyAutoLens to perform grid ray-tracing.

            Parameters
            ----------
            deflection_grid : np.ndarray
                The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return Grid2D(
            grid=self - deflection_grid, mask=self.mask, store_slim=self.store_slim
        )

    def blurring_grid_from_kernel_shape(self, kernel_shape_native):
        """
        Returns the blurring grid from a grid, via an input 2D kernel shape.

            For a full description of blurring grids, checkout *blurring_grid_from_mask_and_kernel_shape*.

            Parameters
            ----------
            kernel_shape_native : (float, float)
                The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """

        return Grid2D.blurring_grid_from_mask_and_kernel_shape(
            mask=self.mask,
            kernel_shape_native=kernel_shape_native,
            store_slim=self.store_slim,
        )

    def grid_with_coordinates_within_distance_removed(
        self, coordinates, distance
    ) -> "Grid2D":
        """Remove all coordinates from this Grid2D which are within a certain distance of an input list of coordinates.

        For example, if the grid has the coordinate (0.0, 0.0) and coordinates=[(0.0, 0.0)], distance=0.1 is input into
        this function, a new Grid2D will be created which removes the coordinate (0.0, 0.0).

        Parameters
        ----------
        coordinates : [(float, float)]
            The list of coordinates which are removed from the grid if they are within the distance threshold.
        distance : float
            The distance threshold that coordinates are removed if they are within that of the input coordinates.
        """

        if not isinstance(coordinates, list):
            coordinates = [coordinates]

        distance_mask = np.full(fill_value=False, shape=self.shape_native)

        for coordinate in coordinates:

            distances = self.distances_from_coordinate(coordinate=coordinate)

            distance_mask += distances.native < distance

        mask = msk.Mask2D.manual(
            mask=distance_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

        return Grid2D.from_mask(mask=mask, store_slim=self.store_slim)

    def structure_from_result(self, result: np.ndarray):
        """Convert a result from an ndarray to an aa.Array2D or aa.Grid2D structure, where the conversion depends on
        type(result) as follows:

        - 1D np.ndarray   -> aa.Array2D
        - 2D np.ndarray   -> aa.Grid2D

        This function is used by the grid_like_to_structure decorator to convert the output result of a function
        to an autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result : np.ndarray or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        if len(result.shape) == 1:
            return array_2d.Array2D(array=result, mask=self.mask, store_slim=True)
        else:
            if isinstance(result, Grid2DTransformedNumpy):
                return Grid2DTransformed(grid=result, mask=self.mask, store_slim=True)
            return Grid2D(grid=result, mask=self.mask, store_slim=True)

    def structure_list_from_result_list(self, result_list: list):
        """Convert a result from a list of ndarrays to a list of aa.Array2D or aa.Grid2D structure, where the conversion
        depends on type(result) as follows:

        - [1D np.ndarray] -> [aa.Array2D]
        - [2D np.ndarray] -> [aa.Grid2D]

        This function is used by the grid_like_list_to_structure-list decorator to convert the output result of a
        function to a list of autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result_list : np.ndarray or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        return [self.structure_from_result(result=result) for result in result_list]


class Grid2DSparse(abstract_grid_2d.AbstractGrid2D):
    def __new__(cls, grid, sparse_index_for_slim_index):
        """
        A sparse grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of a
        pixel on the sparse grid. To setup the sparse-grid, it is laid over a grid of unmasked pixels, such
        that all sparse-grid pixels which map inside of an unmasked grid pixel are included on the sparse grid.

        To setup this sparse grid, we thus have two sparse grid:

        - The unmasked sparse-grid, which corresponds to a uniform 2D array of pixels. The edges of this grid
          correspond to the 4 edges of the mask (e.g. the higher and lowest (y,x) scaled unmasked pixels) and the
          grid's shape is speciifed by the unmasked_sparse_grid_shape parameter.

        - The (masked) sparse-grid, which is all pixels on the unmasked sparse-grid above which fall within unmasked
          grid pixels. These are the pixels which are actually used for other modules in PyAutoArray.

        The origin of the unmasked sparse grid can be changed to allow off-center pairings with sparse-grid pixels,
        which is necessary when a mask has a centre offset from (0.0", 0.0"). However, the sparse grid itself
        retains an origin of (0.0", 0.0"), ensuring its scaled grid uses the same coordinate system as the
        other grid.

        The sparse grid is used to determine the pixel centers of an adaptive grid pixelization.

        Parameters
        ----------
        sparse_grid : np.ndarray or Grid2D
            The (y,x) grid of sparse coordinates.
        sparse_index_for_slim_index : np.ndarray
            An array whose indexes map pixels from a Grid2D's mask to the closest (y,x) coordinate on the sparse_grid.
        """

        obj = grid.view(cls)
        obj.sparse_index_for_slim_index = sparse_index_for_slim_index

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "sparse_index_for_slim_index"):
            self.sparse_index_for_slim_index = obj.sparse_index_for_slim_index

    @classmethod
    def from_grid_and_unmasked_2d_grid_shape(cls, grid, unmasked_sparse_shape):
        """Calculate a Grid2DSparse a Grid2D from the unmasked 2D shape of the sparse grid.

        This is performed by overlaying the 2D sparse grid (computed from the unmaksed sparse shape) over the edge
        values of the Grid2D.

        This function is used in the `Inversion` package to set up the VoronoiMagnification Pixelization.

        Parameters
        -----------
        grid : Grid2D
            The grid of (y,x) scaled coordinates at the centre of every image value (e.g. image-pixels).
        unmasked_sparse_shape : (int, int)
            The 2D shape of the sparse grid which is overlaid over the grid.
        """

        pixel_scales = grid.mask.pixel_scales

        pixel_scales = (
            (grid.shape_native_scaled[0] + pixel_scales[0])
            / (unmasked_sparse_shape[0]),
            (grid.shape_native_scaled[1] + pixel_scales[1])
            / (unmasked_sparse_shape[1]),
        )

        origin = grid.mask.mask_centre

        unmasked_sparse_grid_1d = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=unmasked_sparse_shape,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
        )

        unmasked_sparse_grid_pixel_centres = grid_2d_util.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=unmasked_sparse_grid_1d,
            shape_native=grid.mask.shape,
            pixel_scales=grid.mask.pixel_scales,
        ).astype(
            "int"
        )

        total_sparse_pixels = mask_2d_util.total_sparse_pixels_2d_from(
            mask_2d=grid.mask,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        sparse_for_unmasked_sparse = sparse_util.sparse_for_unmasked_sparse_from(
            mask=grid.mask,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=total_sparse_pixels,
        ).astype("int")

        unmasked_sparse_for_sparse = sparse_util.unmasked_sparse_for_sparse_from(
            total_sparse_pixels=total_sparse_pixels,
            mask=grid.mask,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        ).astype("int")

        regular_to_unmasked_sparse = grid_2d_util.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid,
            shape_native=unmasked_sparse_shape,
            pixel_scales=pixel_scales,
            origin=origin,
        ).astype("int")

        sparse_index_for_slim_index = sparse_util.sparse_slim_index_for_mask_slim_index_from(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            sparse_for_unmasked_sparse=sparse_for_unmasked_sparse,
        ).astype(
            "int"
        )

        sparse_grid = sparse_util.sparse_grid_via_unmasked_from(
            unmasked_sparse_grid=unmasked_sparse_grid_1d,
            unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
        )

        return Grid2DSparse(
            grid=sparse_grid, sparse_index_for_slim_index=sparse_index_for_slim_index
        )

    @classmethod
    def from_total_pixels_grid_and_weight_map(
        cls,
        total_pixels,
        grid,
        weight_map,
        n_iter=1,
        max_iter=5,
        seed=None,
        stochastic=False,
    ):
        """Calculate a Grid2DSparse from a Grid2D and weight map.

        This is performed by running a KMeans clustering algorithm on the weight map, such that Grid2DSparse (y,x)
        coordinates cluster around the weight map values with higher values.

        This function is used in the `Inversion` package to set up the VoronoiMagnification Pixelization.

        Parameters
        -----------
        total_pixels : int
            The total number of pixels in the Grid2DSparse and input into the KMeans algortihm.
        grid : Grid2D
            The grid of (y,x) coordinates corresponding to the weight map.
        weight_map : np.ndarray
            The 2D array of weight values that the KMeans clustering algorithm adapts to to determine the Grid2DSparse.
        n_iter : int
            The number of times the KMeans algorithm is repeated.
        max_iter : int
            The maximum number of iterations in one run of the KMeans algorithm.
        seed : int or None
            The random number seed, which can be used to reproduce Grid2DSparse's for the same inputs.
        stochastic : bool
            If True, the random number seed is randommly chosen every time the function is called, ensuring every
            pixel-grid is randomly determined and thus stochastic.
        """

        if stochastic:
            seed = np.random.randint(low=1, high=2 ** 31)

        if total_pixels > grid.shape[0]:
            raise exc.GridException

        kmeans = KMeans(
            n_clusters=total_pixels, random_state=seed, n_init=n_iter, max_iter=max_iter
        )

        try:
            kmeans = kmeans.fit(X=grid.slim_binned, sample_weight=weight_map)
        except ValueError or OverflowError:
            raise exc.InversionException()

        return Grid2DSparse(
            grid=kmeans.cluster_centers_,
            sparse_index_for_slim_index=kmeans.labels_.astype("int"),
        )

    @property
    def total_sparse_pixels(self):
        return len(self)


class Grid2DTransformed(Grid2D):

    pass


class Grid2DTransformedNumpy(np.ndarray):
    def __new__(cls, grid, *args, **kwargs):
        return grid.view(cls)
