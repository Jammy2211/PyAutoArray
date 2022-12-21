import numpy as np
from typing import List, Optional, Tuple, Union

from autoconf import conf
from autoconf import cached_property

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.abstract_structure import Structure
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.values import ValuesIrregular
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.sparse_2d import Grid2DSparse

from autoarray.structures.arrays import array_2d_util
from autoarray.structures.grids import grid_2d_util
from autoarray.geometry import geometry_util

from autoarray import type as ty


class Grid2D(Structure):
    def __new__(cls, grid: np.ndarray, mask: Mask2D, *args, **kwargs):
        """
        A grid of 2D (y,x) coordinates, which are paired to a uniform 2D mask of pixels and sub-pixels. Each entry
        on the grid corresponds to the (y,x) coordinates at the centre of a sub-pixel of an unmasked pixel.

        A `Grid2D` is ordered such that pixels begin from the top-row (e.g. index [0, 0]) of the corresponding mask
        and go right and down. The positive y-axis is upwards and positive x-axis to the right.

        The grid can be stored in two formats:

        - slimmed: all masked entries are removed so the ndarray is shape [total_unmasked_coordinates*sub_size**2, 2]
        - native: it retains the original shape of the grid so the ndarray is
          shape [total_y_coordinates*sub_size, total_x_coordinates*sub_size, 2].


        **Case 1 (sub-size=1, slim):

        The Grid2D is an ndarray of shape [total_unmasked_coordinates, 2], therefore when `slim` the shape of
        the grid is 2, not 1.

        The first element of the ndarray corresponds to the pixel index and second element the y or x coordinate value.
        For example:

        - grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Below is a visual illustration of a grid, where a total of 10 pixels are unmasked and are included in \
        the grid.

        .. code-block:: bash

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

        .. code-block:: bash

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


        **Case 2 (sub-size>1, slim):**

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

        .. code-block:: bash

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

        .. code-block:: bash

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


        **Case 3 (sub_size=1, native):**

        The Grid2D has the same properties as Case 1, but is stored as an an ndarray of shape
        [total_y_coordinates, total_x_coordinates, 2]. Therefore when `native` the shape of the
        grid is 3, not 2.

        All masked entries on the grid has (y,x) values of (0.0, 0.0).

        For the following example mask:

        .. code-block:: bash

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


        **Case 4 (sub_size>1 native):**

        The properties of this grid can be derived by combining Case's 2 and 3 above, whereby the grid is stored as
        an ndarray of shape [total_y_coordinates*sub_size, total_x_coordinates*sub_size, 2].

        All sub-pixels in masked pixels have values (0.0, 0.0).


        **Grid2D Mapping:**

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
        grid
            The (y,x) coordinates of the grid.
        mask :Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        """

        obj = grid.view(cls)
        obj.mask = mask

        grid_2d_util.check_grid_2d(grid_2d=obj)

        return obj

    @classmethod
    def manual_slim(
        cls,
        grid: Union[np.ndarray, List],
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates in 1D, for example:

        grid=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        grid or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_unmasked_pixells*(sub_size**2), 2]
            or a list of lists.
        shape_native
            The 2D shape of the mask the grid is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.
        """

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        mask = Mask2D.unmasked(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        grid = grid_2d_util.convert_grid_2d(grid_2d=grid, mask_2d=mask)

        return Grid2D(grid=grid, mask=mask)

    @classmethod
    def manual_native(
        cls,
        grid: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates in 2D, for example:

        grid=np.ndarray([[[1.0, 1.0], [2.0, 2.0]],
                         [[3.0, 3.0], [4.0, 4.0]]])

        grid=[[[1.0, 1.0], [2.0, 2.0]],
              [[3.0, 3.0], [4.0, 4.0]]]

        The 2D shape of the grid and its mask are determined from the input grid and the mask is setup as an
        unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        grid or list
            The (y,x) coordinates of the grid input as an ndarray of shape
            [total_y_coordinates*sub_size, total_x_pixel*sub_size, 2] or a list of lists.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.
        """

        grid = grid_2d_util.convert_grid(grid=grid)

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        shape = (int(grid.shape[0] / sub_size), int(grid.shape[1] / sub_size))

        mask = Mask2D.unmasked(
            shape_native=shape,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        grid = grid_2d_util.convert_grid_2d(grid_2d=grid, mask_2d=mask)

        return Grid2D(grid=grid, mask=mask)

    @classmethod
    def manual(
        cls,
        grid: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        shape_native: Tuple[int, int] = None,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates in 1D or 2D, automatically
        determining whether to use the 'manual_slim' or 'manual_native' methods.

        See the manual_slim and manual_native methods for examples.

        Parameters
        ----------
        grid or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_unmasked_pixells*(sub_size**2), 2]
            or a list of lists.
        shape_native
            The 2D shape of the mask the grid is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.
        """
        if len(grid.shape) == 2:
            return cls.manual_slim(
                grid=grid,
                shape_native=shape_native,
                pixel_scales=pixel_scales,
                sub_size=sub_size,
                origin=origin,
            )
        return cls.manual_native(
            grid=grid, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

    @classmethod
    def manual_mask(cls, grid: Union[np.ndarray, List], mask: Mask2D) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates in their native or slimmed format with
        their corresponding mask, automatically determining whether to use the 'manual_slim' or 'manual_native' methods.

        See the manual_slim and manual_native methods for examples.

        Parameters
        ----------
        grid or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_sub_coordinates, 2] or list of lists.
        mask :Mask2D
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        """

        grid = grid_2d_util.convert_grid(grid=grid)
        grid_2d_util.check_grid_2d_and_mask_2d(grid_2d=grid, mask_2d=mask)

        grid = grid_2d_util.convert_grid_2d(grid_2d=grid, mask_2d=mask)

        return Grid2D(grid=grid, mask=mask)

    @classmethod
    def manual_yx_1d(
        cls,
        y: Union[np.ndarray, List],
        x: np.ndarray,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates as 1D y and x values, for example:

        y = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = [1.0, 2.0, 3.0, 4.0]
        x = [1.0, 2.0, 3.0, 4.0]

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        y or list
            The y coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        x or list
            The x coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        shape_native
            The 2D shape of the mask the grid is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.
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
        )

    @classmethod
    def manual_yx_2d(
        cls,
        y: Union[np.ndarray, List],
        x: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the grid coordinates as 2D y and x values, for example:

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
        y or list
            The y coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        x or list
            The x coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.
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
        )

    @classmethod
    def manual_extent(
        cls,
        extent: Tuple[float, float, float, float],
        shape_native: Tuple[int, int],
        sub_size: int = 1,
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) by inputting the extent of the (y,x) grid coordinates as an input
        (x0, x1, y0, y1) tuple.

        The (y,x) `shape_native` in pixels is also input which determines the resolution of the `Grid2D`.

        (The **PyAutoArray** API typically uses a (y,x) notation, however extent variables begin with x currently.
        This will be updated in a future release):

        extent = (x0, x1, y0, y1) = (2.0, 4.0, -2.0, 6.0)
        shape_native = (y,x) = (10, 20)

        Parameters
        ----------
        extent
            The (x0, x1, y0, y1) extent of the grid in scaled coordinates over which the grid is created.
        shape_native
            The 2D shape of the grid that is created within this extent.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.
        """

        x0, x1, y0, y1 = extent

        ys = np.linspace(y1, y0, shape_native[0] * sub_size)
        xs = np.linspace(x0, x1, shape_native[1] * sub_size)

        xs_grid, ys_grid = np.meshgrid(xs, ys)

        xs_grid_1d = xs_grid.ravel()
        ys_grid_1d = ys_grid.ravel()

        grid_2d = np.vstack((ys_grid_1d, xs_grid_1d)).T

        grid_2d = grid_2d.reshape(
            (shape_native[0] * sub_size, shape_native[1] * sub_size, 2)
        )

        pixel_scales = (
            abs(grid_2d[0, 0, 0] - grid_2d[1, 0, 0]),
            abs(grid_2d[0, 0, 1] - grid_2d[0, 1, 1]),
        )

        return Grid2D.manual_native(grid=grid_2d, pixel_scales=pixel_scales)

    @classmethod
    def uniform(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "Grid2D":
        """
        Create a `Grid2D` (see *Grid2D.__new__*) as a uniform grid of (y,x) values given an input `shape_native` and
        `pixel_scales` of the grid:

        Parameters
        ----------
        shape_native
            The 2D shape of the uniform grid and the mask that it is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) tuple.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.
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
        )

    @classmethod
    def bounding_box(
        cls,
        bounding_box: np.ndarray,
        shape_native: Tuple[int, int],
        sub_size: int = 1,
        buffer_around_corners: bool = False,
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) from an input bounding box with coordinates [y_min, y_max, x_min, x_max],
        where the shape_native is used to compute the (y,x) grid values within this bounding box.

        If buffer_around_corners=True, the grid's (y,x) values fully align with the input bounding box values. This
        means the mask's edge pixels extend beyond the bounding box by pixel_scale/2.0. If buffer_around_corners=False,
        the grid (y,x) coordinates are defined within the bounding box such that the mask's edge pixels align with
        the bouning box.

        Parameters
        ----------
        shape_native
            The 2D shape of the uniform grid and the mask that it is paired with.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a ``float``,
            it is converted to a (float, float) structure.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin
            The origin of the grid's mask.
        buffer_around_corners
            Whether the grid is buffered such that the (y,x) values in the centre of its masks' edge pixels align
            with the input bounding box values.
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
        )

    @classmethod
    def from_mask(cls, mask: Mask2D) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) from a mask, where only unmasked pixels are included in the grid (if the
        grid is represented in its native 2D masked values are (0.0, 0.0)).

        The mask's pixel_scales, sub_size and origin properties are used to compute the grid (y,x) coordinates.

        Parameters
        ----------
        mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        """

        sub_grid_1d = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=mask,
            pixel_scales=mask.pixel_scales,
            sub_size=mask.sub_size,
            origin=mask.origin,
        )

        return Grid2D(grid=sub_grid_1d, mask=mask)

    @classmethod
    def from_fits(
        cls,
        file_path: str,
        pixel_scales: ty.PixelScales,
        sub_size: int = 1,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> "Grid2D":
        """
        Create a Grid2D (see *Grid2D.__new__*) from a mask, where only unmasked pixels are included in the grid (if the
        grid is represented in its native 2D masked values are (0.0, 0.0)).

        The mask's pixel_scales, sub_size and origin properties are used to compute the grid (y,x) coordinates.

        Parameters
        ----------
        mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        """

        sub_grid_2d = array_2d_util.numpy_array_2d_via_fits_from(
            file_path=file_path, hdu=0
        )

        return Grid2D.manual(
            grid=sub_grid_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

    @classmethod
    def blurring_grid_from(
        cls, mask: Mask2D, kernel_shape_native: Tuple[int, int]
    ) -> "Grid2D":
        """
        Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked (and
        therefore have their values set to (0.0, 0.0)), but are close enough to the unmasked pixels that their values
        will be convolved into the unmasked those pixels. This when computing images from
        light profile objects.

        The mask's pixel_scales, sub_size and origin properties are used to compute the blurring grid's (y,x)
        coordinates.

        For example, if our mask is as follows:

        .. code-block:: bash

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

        .. code-block:: bash

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

        .. code-block:: bash

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

        .. code-block:: bash

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
        mask
            The mask whose masked pixels are used to setup the blurring grid.
        kernel_shape_native
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """

        blurring_mask = mask.blurring_mask_from(kernel_shape_native=kernel_shape_native)

        return cls.from_mask(mask=blurring_mask)

    @property
    def slim(self) -> "Grid2D":
        """
        Return a `Grid2D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels * sub_size**2, 2].

        If it is already stored in its `slim` representation  it is returned as it is. If not, it is  mapped from
        `native` to `slim` and returned as a new `Grid2D`.
        """

        if len(self.shape) == 2:
            return self

        grid_2d_slim = grid_2d_util.grid_2d_slim_from(
            grid_2d_native=self, mask=self.mask, sub_size=self.mask.sub_size
        )

        return Grid2D(grid=grid_2d_slim, mask=self.mask)

    @property
    def native(self) -> "Grid2D":
        """
        Return a `Grid2D` where the data is stored in its `native` representation, which has shape
        [sub_size*total_y_pixels, sub_size*total_x_pixels, 2].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Grid2D`.

        This method is used in the child `Grid2D` classes to create their `native` properties.
        """

        if len(self.shape) != 2:
            return self

        grid_native = grid_2d_util.grid_2d_native_from(
            grid_2d_slim=self, mask_2d=self.mask, sub_size=self.mask.sub_size
        )

        return Grid2D(grid=grid_native, mask=self.mask)

    @property
    def binned(self) -> "Grid2D":
        """
        Return a `Grid2D` of the binned-up grid in its 1D representation, which is stored with
        shape [total_unmasked_pixels, 2].

        The binning up process converts a grid from (y,x) values where each value is a coordinate on the sub-grid to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a grid with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.
        """

        grid_2d_slim = self.slim

        grid_2d_slim_binned_y = np.multiply(
            self.mask.sub_fraction,
            grid_2d_slim[:, 0].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        grid_2d_slim_binned_x = np.multiply(
            self.mask.sub_fraction,
            grid_2d_slim[:, 1].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        grid_2d_binned = np.stack(
            (grid_2d_slim_binned_y, grid_2d_slim_binned_x), axis=-1
        )

        return Grid2D(grid=grid_2d_binned, mask=self.mask.mask_sub_1)

    @property
    def flipped(self) -> "Grid2D":
        """
        Return the grid as an ndarray of shape [total_unmasked_pixels, 2] with flipped values such that coordinates
        are given as (x,y) values.

        This is used to interface with Python libraries that require the grid in (x,y) format.
        """
        return np.fliplr(self)

    @property
    def in_radians(self) -> "Grid2D":
        """
        Return the grid as an ndarray where all (y,x) values are converted to Radians.

        This grid is used by the interferometer module.
        """
        return (self * np.pi) / 648000.0

    def grid_2d_via_deflection_grid_from(self, deflection_grid: "Grid2D") -> "Grid2D":
        """
        Returns a new Grid2D from this grid, where the (y,x) coordinates of this grid have a grid of (y,x) values,
        termed the deflection grid, subtracted from them to determine the new grid of (y,x) values.

        This is used by PyAutoLens to perform grid ray-tracing.

        Parameters
        ----------
        deflection_grid
            The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return Grid2D(grid=self - deflection_grid, mask=self.mask)

    def blurring_grid_via_kernel_shape_from(
        self, kernel_shape_native: Tuple[int, int]
    ) -> "Grid2D":
        """
        Returns the blurring grid from a grid, via an input 2D kernel shape.

            For a full description of blurring grids, checkout *blurring_grid_from*.

            Parameters
            ----------
            kernel_shape_native
                The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """

        return Grid2D.blurring_grid_from(
            mask=self.mask, kernel_shape_native=kernel_shape_native
        )

    def grid_with_coordinates_within_distance_removed_from(
        self, coordinates: Union[np.ndarray, List], distance: float
    ) -> "Grid2D":
        """Remove all coordinates from this Grid2D which are within a certain distance of an input list of coordinates.

        For example, if the grid has the coordinate (0.0, 0.0) and coordinates=[(0.0, 0.0)], distance=0.1 is input into
        this function, a new Grid2D will be created which removes the coordinate (0.0, 0.0).

        Parameters
        ----------
        coordinates : [(float, float)]
            The list of coordinates which are removed from the grid if they are within the distance threshold.
        distance
            The distance threshold that coordinates are removed if they are within that of the input coordinates.
        """

        if not isinstance(coordinates, list):
            coordinates = [coordinates]

        distance_mask = np.full(fill_value=False, shape=self.shape_native)

        for coordinate in coordinates:

            distances = self.distances_to_coordinate_from(coordinate=coordinate)

            distance_mask += distances.native < distance

        mask = Mask2D.manual(
            mask=distance_mask,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.origin,
        )

        return Grid2D.from_mask(mask=mask)

    def structure_2d_from(self, result: np.ndarray) -> Union[Array2D, "Grid2D"]:
        """
        Convert a result from an ndarray to an aa.Array2D or aa.Grid2D structure, where the conversion depends on
        type(result) as follows:

        - 1D np.ndarray   -> aa.Array2D
        - 2D np.ndarray   -> aa.Grid2D

        This function is used by the grid_2d_to_structure decorator to convert the output result of a function
        to an autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        from autoarray.structures.grids.transformed_2d import Grid2DTransformed
        from autoarray.structures.grids.transformed_2d import Grid2DTransformedNumpy

        if len(result.shape) == 1:
            return Array2D(array=result, mask=self.mask)
        else:
            if isinstance(result, Grid2DTransformedNumpy):
                return Grid2DTransformed(grid=result, mask=self.mask)
            return Grid2D(grid=result, mask=self.mask)

    def structure_2d_list_from(
        self, result_list: List
    ) -> List[Union[Array2D, "Grid2D"]]:
        """
        Convert a result from a list of ndarrays to a list of aa.Array2D or aa.Grid2D structure, where the conversion
        depends on type(result) as follows:

        - [1D np.ndarray] -> [aa.Array2D]
        - [2D np.ndarray] -> [aa.Grid2D]

        This function is used by the grid_like_list_to_structure-list decorator to convert the output result of a
        function to a list of autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result_list or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        return [self.structure_2d_from(result=result) for result in result_list]

    def values_from(self, array_slim: np.ndarray) -> ValuesIrregular:
        """
        Create a *ValuesIrregular* object from a 1D NumPy array of values of shape [total_coordinates]. The
        *ValuesIrregular* are structured following this `Grid2DIrregular` instance.
        """
        return ValuesIrregular(values=array_slim)

    def squared_distances_to_coordinate_from(
        self, coordinate: Tuple[float, float] = (0.0, 0.0)
    ) -> Array2D:
        """
        Returns the squared distance of every coordinate on the grid from an input coordinate.

        Parameters
        ----------
        coordinate
            The (y,x) coordinate from which the squared distance of every grid (y,x) coordinate is computed.
        """
        squared_distances = np.square(self[:, 0] - coordinate[0]) + np.square(
            self[:, 1] - coordinate[1]
        )
        return Array2D.manual_mask(array=squared_distances, mask=self.mask)

    def distances_to_coordinate_from(
        self, coordinate: Tuple[float, float] = (0.0, 0.0)
    ) -> Array2D:
        """
        Returns the distance of every coordinate on the grid from an input (y,x) coordinate.

        Parameters
        ----------
        coordinate
            The (y,x) coordinate from which the distance of every grid (y,x) coordinate is computed.
        """
        distances = np.sqrt(
            self.squared_distances_to_coordinate_from(coordinate=coordinate)
        )
        return Array2D.manual_mask(array=distances, mask=self.mask)

    def grid_2d_radial_projected_shape_slim_from(
        self, centre: Tuple[float, float] = (0.0, 0.0)
    ) -> int:
        """
        The function `grid_scaled_2d_slim_radial_projected_from()` determines a projected radial grid of points from a
        2D region of coordinates defined by an extent [xmin, xmax, ymin, ymax] and with a (y,x) centre.

        To do this, the function first performs these 3 steps:

        1) Given the region defined by the extent [xmin, xmax, ymin, ymax], the algorithm finds the longest 1D distance
           of the 4 paths from the (y,x) centre to the edge of the region (e.g. following the positive / negative y and
           x axes).

        2) Use the pixel-scale corresponding to the direction chosen (e.g. if the positive x-axis was the longest, the
           pixel_scale in the x dimension is used).

        3) Determine the number of pixels between the centre and the edge of the region using the longest path between
           the two chosen above.

        A schematic is shown below:

        .. code-block:: bash

            -------------------
            |                 |
            |<- - -  - ->x    | x = centre
            |                 | <-> = longest radial path from centre to extent edge
            |                 |
            -------------------

        Using the centre x above, this function finds the longest radial path to the edge of the extent window.

        This function returns the integer number of pixels given by this radial grid, which is then used to create
        the radial grid.

        Parameters
        ----------
        extent
            The extent of the grid the radii grid is computed using, with format [xmin, xmax, ymin, ymax]
        centre : (float, flloat)
            The (y,x) central coordinate which the radial grid is traced outwards from.
        pixel_scales
            The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
        sub_size
            The size of the sub-grid that each pixel of the 2D mask array is divided into.

        Returns
        -------
        int
            The 1D integer shape of a radial set of points sampling the longest distance from the centre to the edge of the
            extent in along the positive x-axis.
        """

        return grid_2d_util._radial_projected_shape_slim_from(
            extent=self.extent,
            centre=centre,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
        )

    def grid_2d_radial_projected_from(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        shape_slim: Optional[int] = 0,
    ) -> Grid2DIrregular:
        """
        Determine a projected radial grid of points from a 2D region of coordinates defined by an
        extent [xmin, xmax, ymin, ymax] and with a (y,x) centre.

        This functions operates as follows:

        1) Given the region defined by the extent [xmin, xmax, ymin, ymax], the algorithm finds the longest 1D distance
           of the 4 paths from the (y,x) centre to the edge of the region e.g. following the positive / negative y and
           x axes.

        2) Use the pixel-scale corresponding to the direction chosen e.g. if the positive x-axis was the longest, the
           pixel_scale in the x dimension is used.

        3) Determine the number of pixels between the centre and the edge of the region using the longest path between
           the two chosen above.

        4) Create a (y,x) grid of radial points where all points are at the centre's y value = 0.0 and the x values
           iterate from the centre in increasing steps of the pixel-scale.

        5) Rotate these radial coordinates by the input `angle` clockwise.

        A schematic is shown below:

        .. code-block:: bash

            -------------------
            |                 |
            |<- - -  - ->x    | x = centre
            |                 | <-> = longest radial path from centre to extent edge
            |                 |
            -------------------

        Parameters
        ----------
        centre
            The (y,x) central coordinate which the radial grid is traced outwards from.
        angle
            The angle with which the radial coordinates are rotated clockwise.

        Returns
        -------
        Grid2DIrregular
            A radial set of points sampling the longest distance from the centre to the edge of the extent in along the
            positive x-axis.
        """

        grid_radial_projected_2d = (
            grid_2d_util.grid_scaled_2d_slim_radial_projected_from(
                extent=self.extent,
                centre=centre,
                pixel_scales=self.mask.pixel_scales,
                sub_size=self.mask.sub_size,
                shape_slim=shape_slim,
            )
        )

        grid_radial_projected_2d = geometry_util.transform_grid_2d_to_reference_frame(
            grid_2d=grid_radial_projected_2d, centre=centre, angle=angle
        )

        grid_radial_projected_2d = geometry_util.transform_grid_2d_from_reference_frame(
            grid_2d=grid_radial_projected_2d, centre=centre, angle=0.0
        )

        if conf.instance["general"]["grid"]["remove_projected_centre"]:

            grid_radial_projected_2d = grid_radial_projected_2d[1:, :]

        return Grid2DIrregular(grid=grid_radial_projected_2d)

    @property
    def shape_native_scaled(self) -> Tuple[float, float]:
        """
        The (y,x) 2D shape of the grid in scaled units, computed from the minimum and maximum y and x values of the
        grid.
        """
        return (
            np.amax(self[:, 0]) - np.amin(self[:, 0]),
            np.amax(self[:, 1]) - np.amin(self[:, 1]),
        )

    @property
    def scaled_maxima(self) -> Tuple[float, float]:
        """
        The maximum values of the grid in scaled coordinates returned as a tuple (y_max, x_max).
        """
        return (
            self.mask.origin[0] + (self.mask.shape_native_scaled[0] / 2.0),
            self.mask.origin[1] + (self.mask.shape_native_scaled[1] / 2.0),
        )

    @property
    def scaled_minima(self) -> Tuple[float, float]:
        """
        The minium values of the grid in scaled coordinates returned as a tuple (y_min, x_min).
        """
        return (
            (self.mask.origin[0] - (self.mask.shape_native_scaled[0] / 2.0)),
            (self.mask.origin[1] - (self.mask.shape_native_scaled[1] / 2.0)),
        )

    @property
    def extent(self) -> np.ndarray:
        """
        The extent of the grid in scaled units returned as an ndarray of the form [x_min, x_max, y_min, y_max].

        This follows the format of the extent input parameter in the matplotlib method imshow (and other methods) and
        is used for visualization in the plot module.
        """
        return np.asarray(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )

    def extent_with_buffer_from(self, buffer: float = 1.0e-8) -> List[float]:
        """
        The extent of the grid in scaled units returned as a list [x_min, x_max, y_min, y_max], where all values are
        buffed such that their extent is further than the grid's extent..

        This follows the format of the extent input parameter in the matplotlib method imshow (and other methods) and
        is used for visualization in the plot module.
        """
        return [
            self.scaled_minima[1] - buffer,
            self.scaled_maxima[1] + buffer,
            self.scaled_minima[0] - buffer,
            self.scaled_maxima[0] + buffer,
        ]

    @cached_property
    def sub_border_grid(self) -> np.ndarray:
        """
        The (y,x) grid of all sub-pixels which are at the border of the mask.

        This is NOT all sub-pixels which are in mask pixels at the mask's border, but specifically the sub-pixels
        within these border pixels which are at the extreme edge of the border.
        """
        return self[self.mask.sub_border_flat_indexes]

    def padded_grid_from(self, kernel_shape_native: Tuple[int, int]) -> "Grid2D":
        """
        When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will
        be 'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the padded grid is used, which is 'buffed' such that it includes all pixels
        whose signal will be convolved into the unmasked pixels given the 2D kernel shape.

        Parameters
        ----------
        kernel_shape_native
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """

        shape = self.mask.shape

        padded_shape = (
            shape[0] + kernel_shape_native[0] - 1,
            shape[1] + kernel_shape_native[1] - 1,
        )

        padded_mask = Mask2D.unmasked(
            shape_native=padded_shape,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
        )

        return Grid2D.from_mask(mask=padded_mask)

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

        return Grid2D(
            grid=grid_2d_util.relocated_grid_via_jit_from(
                grid=grid, border_grid=self.sub_border_grid
            ),
            mask=grid.mask,
            sub_size=grid.mask.sub_size,
        )

    def relocated_mesh_grid_from(self, mesh_grid: Grid2DSparse) -> Grid2DSparse:
        """
        Relocate the coordinates of a pixelization grid to the border of this grid. See the method
        `relocated_grid_from()`for a full description of how this grid relocation works.

        This function operates the same as other grid relocation functions but instead returns the grid as a
        `Grid2DSparse` instance, which contains information pairing the grid to a pixelization.

        Parameters
        ----------
        grid
            The pixelization grid whose pixels are relocated to the border edge if outside it.
        """

        if len(self.sub_border_grid) == 0:
            return mesh_grid

        return Grid2DSparse(
            grid=grid_2d_util.relocated_grid_via_jit_from(
                grid=mesh_grid, border_grid=self.sub_border_grid
            ),
            sparse_index_for_slim_index=mesh_grid.sparse_index_for_slim_index,
        )
