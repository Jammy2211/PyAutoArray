import numpy as np
from sklearn.cluster import KMeans

import typing

import autoarray as aa

from autoconf import conf
from autoarray import decorator_util
from autoarray import exc
from autoarray.structures import abstract_structure, arrays, grids
from autoarray.mask import mask as msk
from autoarray.util import sparse_util, array_util, grid_util, mask_util


def convert_and_check_grid(grid, mask=None):

    if type(grid) is list:
        grid = np.asarray(grid)

    if grid.shape[-1] != 2:
        raise exc.GridException(
            "The final dimension of the input grid is not equal to 2 (e.g. the (y,x) coordinates)"
        )

    if 2 < len(grid.shape) > 3:
        raise exc.GridException("The dimensions of the input grid array is not 2 or 3")

    if mask is not None:

        if len(grid.shape) == 2:

            if grid.shape[0] != mask.sub_pixels_in_mask:
                raise exc.GridException(
                    "The input 1D grid does not have the same number of entries as sub-pixels in"
                    "the mask."
                )

        elif len(grid.shape) == 3:

            if (grid.shape[0], grid.shape[1]) != mask.sub_shape_2d:
                raise exc.GridException(
                    "The input grid is 2D but not the same dimensions as the sub-mask "
                    "(e.g. the mask 2D shape multipled by its sub size."
                )

    return grid


def convert_pixel_scales(pixel_scales):

    if type(pixel_scales) is float:
        pixel_scales = (pixel_scales, pixel_scales)

    return pixel_scales


class Grid(abstract_structure.AbstractStructure):
    def __new__(cls, grid, mask, store_in_1d=True, *args, **kwargs):
        """A grid of coordinates, which are paired to a uniform 2D mask of pixels and sub-pixels. Each entry
        on the grid corresponds to the (y,x) coordinates at the centre of a sub-pixel in an unmasked pixel.

        A *Grid* is ordered such that pixels begin from the top-row of the corresponding mask and go right and down.
        The positive y-axis is upwards and positive x-axis to the right.

        The grid can be stored in 1D or 2D, as detailed below.

        Case 1: [sub-size=1, store_in_1d = True]:
        -----------------------------------------

        The Grid is an ndarray of shape [total_unmasked_pixels, 2], therefore when store_in_1d=True the shape of the
        grid is 2, not 1.

        The first element of the ndarray corresponds to the pixel index and second element the y or x coordinate value.
        For example:

        - grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Below is a visual illustration of a grid, where a total of 10 pixels are unmasked and are included in \
        the grid.

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from the grid)
        |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in the grid)
        |x|x|x|o|o|o|o|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        The mask pixel index's will come out like this (and the direction of scaled coordinates is highlighted
        around the mask.

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                        y      x
        |x|x|x|x|x|x|x|x|x|x|  ^   grid[0] = [ 1.5, -0.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[1] = [ 1.5,  0.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[2] = [ 0.5, -1.5]
        |x|x|x|x|0|1|x|x|x|x| +ve  grid[3] = [ 0.5, -0.5]
        |x|x|x|2|3|4|5|x|x|x|  y   grid[4] = [ 0.5,  0.5]
        |x|x|x|6|7|8|9|x|x|x| -ve  grid[5] = [ 0.5,  1.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[6] = [-0.5, -1.5]
        |x|x|x|x|x|x|x|x|x|x|  |   grid[7] = [-0.5, -0.5]
        |x|x|x|x|x|x|x|x|x|x| \/   grid[8] = [-0.5,  0.5]
        |x|x|x|x|x|x|x|x|x|x|      grid[9] = [-0.5,  1.5]

        Case 2: [sub-size>1, store_in_1d=True]:
        ------------------

        If the masks's sub size is > 1, the grid is defined as a sub-grid where each entry corresponds to the (y,x)
        coordinates at the centre of each sub-pixel of an unmasked pixel.

        The sub-grid indexes are ordered such that pixels begin from the first (top-left) sub-pixel in the first
        unmasked pixel. Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel.
        Therefore, the sub-grid is an ndarray of shape [total_unmasked_pixels*(sub_grid_shape)**2, 2]. For example:

        - grid[9, 1] - using a 2x2 sub-grid, gives the 3rd unmasked pixel's 2nd sub-pixel x-coordinate.
        - grid[9, 1] - using a 3x3 sub-grid, gives the 2nd unmasked pixel's 1st sub-pixel x-coordinate.
        - grid[27, 0] - using a 3x3 sub-grid, gives the 4th unmasked pixel's 1st sub-pixel y-coordinate.

        Below is a visual illustration of a sub grid. Indexing of each sub-pixel goes from the top-left corner. In
        contrast to the grid above, our illustration below restricts the mask to just 2 pixels, to keep the
        illustration brief.

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|x|x|o|o|x|x|x|x|     o = False (Pixel is not masked and included in lens)
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        Our grid with a sub-size looks like it did before:

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->

        |x|x|x|x|x|x|x|x|x|x|  ^
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x|  |                        y     x
        |x|x|x|x|x|x|x|x|x|x| +ve  grid[0] = [0.5,  -1.5]
        |x|x|x|0|1|x|x|x|x|x|  y   grid[1] = [0.5,  -0.5]
        |x|x|x|x|x|x|x|x|x|x| -ve
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x|  |
        |x|x|x|x|x|x|x|x|x|x| \/
        |x|x|x|x|x|x|x|x|x|x|

        However, if the sub-size is 2, we go to each unmasked pixel and allocate sub-pixel coordinates for it. For
        example, for pixel 0, if *sub_size=2*, we use a 2x2 sub-grid:

        Pixel 0 - (2x2):
                            y      x
               grid[0] = [0.66, -1.66]
        |0|1|  grid[1] = [0.66, -1.33]
        |2|3|  grid[2] = [0.33, -1.66]
               grid[3] = [0.33, -1.33]

        If we used a sub_size of 3, for the pixel we we would create a 3x3 sub-grid:

                              y      x
                 grid[0] = [0.75, -0.75]
                 grid[1] = [0.75, -0.5]
                 grid[2] = [0.75, -0.25]
        |0|1|2|  grid[3] = [0.5,  -0.75]
        |3|4|5|  grid[4] = [0.5,  -0.5]
        |6|7|8|  grid[5] = [0.5,  -0.25]
                 grid[6] = [0.25, -0.75]
                 grid[7] = [0.25, -0.5]
                 grid[8] = [0.25, -0.25]

        Case 3: [sub_size=1 store_in_1d=False]
        --------------------------------------

        The Grid has the same properties as Case 1, but is stored as an an ndarray of shape
        [total_y_coordinates, total_x_coordinates, 2]. Therefore when store_in_1d=False the shape of the
        grid is 3, not 2.

        All masked entries on the grid has (y,x) values of (0.0, 0.0).

        For the following example mask:

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an example mask.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|o|o|x|x|x|x|     x = True (Pixel is masked and excluded from the grid)
        |x|x|x|o|o|o|o|x|x|x|     o = False (Pixel is not masked and included in the grid)
        |x|x|x|o|o|o|o|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        - grid[0,0,0] = 0.0 (it is masked, thus zero)
        - grid[0,0,1] = 0.0 (it is masked, thus zero)
        - grid[3,3,0] = 0.0 (it is masked, thus zero)
        - grid[3,3,1] = 0.0 (it is masked, thus zero)
        - grid[3,4,0] = 1.5
        - grid[3,4,1] = -0.5

        Case 4: [sub_size>1 store_in_1d=False]
        --------------------------------------

        The properties of this grid can be derived by combining Case's 2 and 3 above, whereby the grid is stored as
        an ndarray of shape [total_y_coordinates*sub_size, total_x_coordinates*sub_size, 2].

        All sub-pixels in masked pixels have values (0.0, 0.0).

        Grid Mapping
        ------------

        Every set of (y,x) coordinates in a pixel of the sub-grid maps to an unmasked pixel in the mask. For a uniform
        grid, every (y,x) coordinate directly corresponds to the location of its paired unmasked pixel.

        It is not a requirement that grid is uniform and that their coordinates align with the mask. The input grid
        could be an irregular set of (y,x) coordinates where the indexing signifies that the (y,x) coordinate
        *originates* or *is paired with* the mask's pixels but has had its value change by some aspect of the
        calculation.

        This is important for *PyAutoLens*, where grids in the image-plane are ray-traced and deflected to perform
        lensig calculations. The grid indexing is used to map pixels between the image-plane and source-plane.

        Parameters
        ----------
        grid : np.ndarray
            The (y,x) coordinates of the grid.
        mask : msk.Mask
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        obj = super(Grid, cls).__new__(
            cls=cls, structure=grid, mask=mask, store_in_1d=store_in_1d
        )
        return obj

    @classmethod
    def manual_1d(
        cls,
        grid,
        shape_2d,
        pixel_scales,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=True,
    ):
        """Create a Grid (see *Grid.__new__*) by inputting the grid coordinates in 1D, for example:

        grid=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked *Mask* of shape_2d.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_unmasked_pixells*(sub_size**2), 2]
            or a list of lists.
        shape_2d : (float, float)
            The 2D shape of the mask the grid is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin : (float, float)
            The origin of the grid's mask.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        grid = convert_and_check_grid(grid=grid)
        pixel_scales = convert_pixel_scales(pixel_scales=pixel_scales)

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        if store_in_1d:
            return Grid(grid=grid, mask=mask, store_in_1d=store_in_1d)

        grid_2d = grid_util.sub_grid_2d_from(
            sub_grid_1d=grid, mask=mask, sub_size=sub_size
        )

        return Grid(grid=grid_2d, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def manual_2d(
        cls, grid, pixel_scales, sub_size=1, origin=(0.0, 0.0), store_in_1d=True
    ):
        """Create a Grid (see *Grid.__new__*) by inputting the grid coordinates in 2D, for example:

        grid=np.ndarray([[[1.0, 1.0], [2.0, 2.0]],
                         [[3.0, 3.0], [4.0, 4.0]]])

        grid=[[[1.0, 1.0], [2.0, 2.0]],
              [[3.0, 3.0], [4.0, 4.0]]]

        The 2D shape of the grid and its mask are determined from the input grid and the mask is setup as an
        unmasked *Mask* of shape_2d.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape
            [total_y_pixels*sub_size, total_x_pixel*sub_size, 2] or a list of lists.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin : (float, float)
            The origin of the grid's mask.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        grid = convert_and_check_grid(grid=grid)
        pixel_scales = convert_pixel_scales(pixel_scales=pixel_scales)

        shape = (int(grid.shape[0] / sub_size), int(grid.shape[1] / sub_size))

        mask = msk.Mask.unmasked(
            shape_2d=shape, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

        if not store_in_1d:
            return Grid(grid=grid, mask=mask, store_in_1d=store_in_1d)

        grid_1d = grid_util.sub_grid_1d_from(
            sub_grid_2d=grid, mask=mask, sub_size=sub_size
        )

        return Grid(grid=grid_1d, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def manual_yx_1d(
        cls,
        y,
        x,
        shape_2d,
        pixel_scales,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=True,
    ):
        """Create a Grid (see *Grid.__new__*) by inputting the grid coordinates as 1D y and x values, for example:

        y = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = [1.0, 2.0, 3.0, 4.0]
        x = [1.0, 2.0, 3.0, 4.0]

        From 1D input the method cannot determine the 2D shape of the grid and its mask, thus the shape_2d must be
        input into this method. The mask is setup as a unmasked *Mask* of shape_2d.

        Parameters
        ----------
        y : np.ndarray or list
            The y coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        x : np.ndarray or list
            The x coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        shape_2d : (float, float)
            The 2D shape of the mask the grid is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin : (float, float)
            The origin of the grid's mask.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        if type(y) is list:
            y = np.asarray(y)

        if type(x) is list:
            x = np.asarray(x)

        return cls.manual_1d(
            grid=np.stack((y, x), axis=-1),
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def manual_yx_2d(
        cls, y, x, pixel_scales, sub_size=1, origin=(0.0, 0.0), store_in_1d=True
    ):
        """Create a Grid (see *Grid.__new__*) by inputting the grid coordinates as 2D y and x values, for example:

        y = np.array([[1.0, 2.0],
                     [3.0, 4.0]])
        x = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        y = [[1.0, 2.0],
             [3.0, 4.0]]
        x = [[1.0, 2.0],
             [3.0, 4.0]]

        The 2D shape of the grid and its mask are determined from the input grid and the mask is setup as an
        unmasked *Mask* of shape_2d.

        Parameters
        ----------
        y : np.ndarray or list
            The y coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        x : np.ndarray or list
            The x coordinates of the grid input as an ndarray of shape [total_coordinates] or list.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin : (float, float)
            The origin of the grid's mask.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        if type(y) is list:
            y = np.asarray(y)

        if type(x) is list:
            x = np.asarray(x)

        return cls.manual_2d(
            grid=np.stack((y, x), axis=-1),
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def uniform(
        cls, shape_2d, pixel_scales, sub_size=1, origin=(0.0, 0.0), store_in_1d=True
    ):
        """Create a Grid (see *Grid.__new__*) as a uniform grid of (y,x) values given an input shape_2d and pixel
        scale of the grid:

        Parameters
        ----------
        shape_2d : (float, float)
            The 2D shape of the uniform grid and the mask that it is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin : (float, float)
            The origin of the grid's mask.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        pixel_scales = convert_pixel_scales(pixel_scales=pixel_scales)

        grid_1d = grid_util.grid_1d_via_shape_2d_from(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        return cls.manual_1d(
            grid=grid_1d,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def bounding_box(
        cls,
        bounding_box,
        shape_2d,
        sub_size=1,
        buffer_around_corners=False,
        store_in_1d=True,
    ):
        """Create a Grid (see *Grid.__new__*) from an input bounding box with coordinates [y_min, y_max, x_min, x_max],
        where the shape_2d is used to compute the (y,x) grid values within this bounding box.

        If buffer_around_corners=True, the grid's (y,x) values fully align with the input bounding box values. This
        means the mask's edge pixels extend beyond the bounding box by pixel_scale/2.0. If buffer_around_corners=False,
        the grid (y,x) coordinates are defined within the bounding box such that the mask's edge pixels align with
        the bouning box.

        Parameters
        ----------
        shape_2d : (float, float)
            The 2D shape of the uniform grid and the mask that it is paired with.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        origin : (float, float)
            The origin of the grid's mask.
        buffer_around_corners : bool
            Whether the grid is buffered such that the (y,x) values in the centre of its masks' edge pixels align
            with the input bounding box values.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        y_min, y_max, x_min, x_max = bounding_box

        if not buffer_around_corners:

            pixel_scales = (
                (y_max - y_min) / (shape_2d[0]),
                (x_max - x_min) / (shape_2d[1]),
            )

        else:

            pixel_scales = (
                (y_max - y_min) / (shape_2d[0] - 1),
                (x_max - x_min) / (shape_2d[1] - 1),
            )
        origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)

        return cls.uniform(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def from_mask(cls, mask, store_in_1d=True):
        """Create a Grid (see *Grid.__new__*) from a mask, where only unmasked pixels are included in the grid (if the
        grid is represented in 2D masked values are (0.0, 0.0)).

        The mask's pixel_scales, sub_size and origin properties are used to compute the grid (y,x) coordinates.

        Parameters
        ----------
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """

        sub_grid_1d = grid_util.grid_1d_via_mask_from(
            mask=mask,
            pixel_scales=mask.pixel_scales,
            sub_size=mask.sub_size,
            origin=mask.origin,
        )

        if store_in_1d:
            return grids.Grid(grid=sub_grid_1d, mask=mask, store_in_1d=store_in_1d)

        sub_grid_2d = grid_util.sub_grid_2d_from(
            sub_grid_1d=sub_grid_1d, mask=mask, sub_size=mask.sub_size
        )

        return grids.Grid(grid=sub_grid_2d, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def blurring_grid_from_mask_and_kernel_shape(
        cls, mask, kernel_shape_2d, store_in_1d=True
    ):
        """Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked (and
        therefore have their values set to (0.0, 0.0)), but are close enough to the unmasked pixels that their values
        will be convolved into the unmasked those pixels. This occurs in *PyAutoGalaxy* when computing images from
        light profile objects.

        The mask's pixel_scales, sub_size and origin properties are used to compute the blurring grid's (y,x)
        coordinates.

        For example, if our mask is as follows:

        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     This is an imaging.Mask, where:
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|x|o|o|o|x|x|x|x|     o = False (Pixel is not masked and included in lens)
        |x|x|x|o|o|o|x|x|x|x|
        |x|x|x|o|o|o|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|x|

        For a PSF of shape (3,3), the following blurring mask is computed (noting that only pixels that are direct \
        neighbors of the unmasked pixels above will blur light into an unmasked pixel):

        |x|x|x|x|x|x|x|x|x|     This is an example grid.Mask, where:
        |x|x|x|x|x|x|x|x|x|
        |x|x|o|o|o|o|o|x|x|     x = True (Pixel is masked and excluded from lens)
        |x|x|o|x|x|x|o|x|x|     o = False (Pixel is not masked and included in lens)
        |x|x|o|x|x|x|o|x|x|
        |x|x|o|x|x|x|o|x|x|
        |x|x|o|o|o|o|o|x|x|
        |x|x|x|x|x|x|x|x|x|
        |x|x|x|x|x|x|x|x|x|

        Thus, the blurring grid coordinates and indexes will be as follows:

        pixel_scales = 1.0"

        <--- -ve  x  +ve -->
                                                            y     x
        |x|x|x |x |x |x |x |x|x|  |   blurring_grid[0] = [2.0, -2.0]  blurring_grid[9] =  [-1.0, -2.0]
        |x|x|x |x |x |x |x |x|x|  |   blurring_grid[1] = [2.0, -1.0]  blurring_grid[10] = [-1.0,  2.0]
        |x|x|0 |1 |2 |3 |4 |x|x| +ve  blurring_grid[2] = [2.0,  0.0]  blurring_grid[11] = [-2.0, -2.0]
        |x|x|5 |x |x |x |6 |x|x|  y   blurring_grid[3] = [2.0,  1.0]  blurring_grid[12] = [-2.0, -1.0]
        |x|x|7 |x |x |x |8 |x|x| -ve  blurring_grid[4] = [2.0,  2.0]  blurring_grid[13] = [-2.0,  0.0]
        |x|x|9 |x |x |x |10|x|x|  |   blurring_grid[5] = [1.0, -2.0]  blurring_grid[14] = [-2.0,  1.0]
        |x|x|11|12|13|14|15|x|x|  |   blurring_grid[6] = [1.0,  2.0]  blurring_grid[15] = [-2.0,  2.0]
        |x|x|x |x |x |x |x |x|x| \/   blurring_grid[7] = [0.0, -2.0]
        |x|x|x |x |x |x |x |x|x|      blurring_grid[8] = [0.0,  2.0]

        For a PSF of shape (5,5), the following blurring mask is computed (noting that pixels are 2 pixels from a
        direct unmasked pixels now blur light into an unmasked pixel):

        |x|x|x|x|x|x|x|x|x|     This is an example grid.Mask, where:
        |x|o|o|o|o|o|o|o|x|
        |x|o|o|o|o|o|o|o|x|     x = True (Pixel is masked and excluded from lens)
        |x|o|o|x|x|x|o|o|x|     o = False (Pixel is not masked and included in lens)
        |x|o|o|x|x|x|o|o|x|
        |x|o|o|x|x|x|o|o|x|
        |x|o|o|o|o|o|o|o|x|
        |x|o|o|o|o|o|o|o|x|
        |x|x|x|x|x|x|x|x|x|

        Parameters
        ----------
        mask : Mask
            The mask whose masked pixels are used to setup the blurring grid.
        kernel_shape_2d : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """

        blurring_mask = mask.regions.blurring_mask_from_kernel_shape(
            kernel_shape_2d=kernel_shape_2d
        )

        return cls.from_mask(mask=blurring_mask, store_in_1d=store_in_1d)

    def grid_from_deflection_grid(self, deflection_grid):
        """Compute a new Grid from this grid, where the (y,x) coordinates of this grid have a grid of (y,x) values,
         termed the deflection grid, subtracted from them to determine the new grid of (y,x) values.

        This is used by PyAutoLens to perform grid ray-tracing.

        Parameters
        ----------
        deflection_grid : ndarray
            The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return Grid(
            grid=self - deflection_grid, mask=self.mask, store_in_1d=self.store_in_1d
        )

    def blurring_grid_from_kernel_shape(self, kernel_shape_2d):
        """Compute the blurring grid from a grid, via an input 2D kernel shape.

        For a full description of blurring grids, checkout *blurring_grid_from_mask_and_kernel_shape*.

        Parameters
        ----------
        kernel_shape_2d : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """

        return Grid.blurring_grid_from_mask_and_kernel_shape(
            mask=self.mask,
            kernel_shape_2d=kernel_shape_2d,
            store_in_1d=self.store_in_1d,
        )

    def __array_finalize__(self, obj):

        super(Grid, self).__array_finalize__(obj)

        if hasattr(obj, "_sub_border_1d_indexes"):
            self._sub_border_1d_indexes = obj._sub_border_1d_indexes

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super().__setstate__(state[0:-1])

    def _new_grid(self, grid, mask, store_in_1d):
        """Conveninence method for creating a new instance of the Grid class from this grid.

        This method is over-written by other grids (e.g. GridIterate) such that the in_1d and in_2d methods return
        instances of that Grid's type.

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_sub_coordinates, 2] or list of lists.
        mask : msk.Mask
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
            """
        return Grid(grid=grid, mask=mask, store_in_1d=store_in_1d)

    @property
    def in_1d(self):
        """Convenience method to access the grid's 1D representation, which is a Grid stored as an ndarray of shape
        [total_unmasked_pixels*(sub_size**2), 2].

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D."""
        if self.store_in_1d:
            return self

        sub_grid_1d = grid_util.sub_grid_1d_from(
            sub_grid_2d=self, mask=self.mask, sub_size=self.mask.sub_size
        )

        return self._new_grid(grid=sub_grid_1d, mask=self.mask, store_in_1d=True)

    @property
    def in_2d(self):
        """Convenience method to access the grid's 2D representation, which is a Grid stored as an ndarray of shape
        [sub_size*total_y_pixels, sub_size*total_x_pixels, 2] where all masked values are given values (0.0, 0.0).

        If the grid is stored in 2D it is return as is. If it is stored in 1D, it must first be mapped from 1D to 2D."""

        if self.store_in_1d:
            sub_grid_2d = grid_util.sub_grid_2d_from(
                sub_grid_1d=self, mask=self.mask, sub_size=self.mask.sub_size
            )
            return self._new_grid(grid=sub_grid_2d, mask=self.mask, store_in_1d=False)

        return self

    @property
    def in_1d_binned(self):
        """Convenience method to access the binned-up grid in its 1D representation, which is a Grid stored as an
        ndarray of shape [total_unmasked_pixels, 2].

        The binning up process converts a grid from (y,x) values where each value is a coordinate on the sub-grid to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a grid with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D."""
        if not self.store_in_1d:

            sub_grid_1d = grid_util.sub_grid_1d_from(
                sub_grid_2d=self, mask=self.mask, sub_size=self.mask.sub_size
            )

        else:

            sub_grid_1d = self

        binned_grid_1d_y = np.multiply(
            self.mask.sub_fraction,
            sub_grid_1d[:, 0].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        binned_grid_1d_x = np.multiply(
            self.mask.sub_fraction,
            sub_grid_1d[:, 1].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        return self._new_grid(
            grid=np.stack((binned_grid_1d_y, binned_grid_1d_x), axis=-1),
            mask=self.mask.mask_sub_1,
            store_in_1d=True,
        )

    @property
    def in_2d_binned(self):
        """Convenience method to access the binned-up grid in its 2D representation, which is a Grid stored as an
        ndarray of shape [total_y_pixels, total_x_pixels, 2].

        The binning up process conerts a grid from (y,x) values where each value is a coordinate on the sub-grid to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a grid with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the grid is stored in 2D it is return as is. If it is stored in 1D, it must first be mapped from 1D to 2D."""
        if not self.store_in_1d:

            sub_grid_1d = grid_util.sub_grid_1d_from(
                sub_grid_2d=self, mask=self.mask, sub_size=self.mask.sub_size
            )

        else:

            sub_grid_1d = self

        binned_grid_1d_y = np.multiply(
            self.mask.sub_fraction,
            sub_grid_1d[:, 0].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        binned_grid_1d_x = np.multiply(
            self.mask.sub_fraction,
            sub_grid_1d[:, 1].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        binned_grid_1d = np.stack((binned_grid_1d_y, binned_grid_1d_x), axis=-1)

        binned_grid_2d = grid_util.sub_grid_2d_from(
            sub_grid_1d=binned_grid_1d, mask=self.mask, sub_size=1
        )

        return self._new_grid(
            grid=binned_grid_2d, mask=self.mask.mask_sub_1, store_in_1d=False
        )

    @property
    def in_1d_flipped(self) -> np.ndarray:
        """Return the grid as an ndarray of shape [total_unmasked_pixels, 2] with flipped values such that coordinates
        are given as (x,y) values.

        This is used to interface with Python libraries that require the grid in (x,y) format."""
        return np.fliplr(self)

    @property
    def in_2d_flipped(self):
        """Return the grid as an ndarray array of shape [total_x_pixels, total_y_pixels, 2[ with flipped values such
        that coordinates are given as (x,y) values.

        This is used to interface with Python libraries that require the grid in (x,y) format."""
        return np.stack((self.in_2d[:, :, 1], self.in_2d[:, :, 0]), axis=-1)

    @property
    @array_util.Memoizer()
    def in_radians(self):
        """Return the grid as an ndarray where all (y,x) values are converted to Radians.

        This grid is used by the interferometer module."""
        return (self * np.pi) / 648000.0

    def squared_distances_from_coordinate(
        self, coordinate=(0.0, 0.0)
    ) -> arrays.MaskedArray:
        """Compute the squared distance of every coordinate on the grid from an input coordinate.

        Parameters
        ----------
        coordinate : (float, float)
            The (y,x) coordinate from which the squared distance of every grid (y,x) coordinate is computed.
        """
        squared_distances = np.square(self[:, 0] - coordinate[0]) + np.square(
            self[:, 1] - coordinate[1]
        )
        return aa.MaskedArray(array=squared_distances, mask=self.mask)

    def distances_from_coordinate(self, coordinate=(0.0, 0.0)) -> arrays.MaskedArray:
        """Compute the distance of every coordinate on the grid from an input (y,x) coordinate.

        Parameters
        ----------
        coordinate : (float, float)
            The (y,x) coordinate from which the distance of every grid (y,x) coordinate is computed.
        """
        distances = np.sqrt(
            self.squared_distances_from_coordinate(coordinate=coordinate)
        )
        return aa.MaskedArray(array=distances, mask=self.mask)

    @property
    def shape_2d_scaled(self) -> (float, float):
        """The two dimensional shape of the grid in scaled units, computed by taking the minimum and maximum values of
        the grid."""
        return (
            np.amax(self[:, 0]) - np.amin(self[:, 0]),
            np.amax(self[:, 1]) - np.amin(self[:, 1]),
        )

    @property
    def scaled_maxima(self) -> (float, float):
        """The maximum values of the grid in scaled coordinates returned as a tuple (y_max, x_max)."""
        return (
            self.origin[0] + (self.shape_2d_scaled[0] / 2.0),
            self.origin[1] + (self.shape_2d_scaled[1] / 2.0),
        )

    @property
    def scaled_minima(self) -> (float, float):
        """The minium values of the grid in scaled coordinates returned as a tuple (y_min, x_min)."""
        return (
            (self.origin[0] - (self.shape_2d_scaled[0] / 2.0)),
            (self.origin[1] - (self.shape_2d_scaled[1] / 2.0)),
        )

    @property
    def extent(self) -> np.ndarray:
        """The extent of the grid in scaled units returned as an ndarray of the form [x_min, x_max, y_min, y_max].

        This follows the format of the extent input parameter in the matplotlib method imshow (and other methods) and
        is used for visualization in the plot module."""
        return np.asarray(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )

    def extent_with_buffer(self, buffer=1.0e-8) -> [float, float, float, float]:
        """The extent of the grid in scaled units returned as a list [x_min, x_max, y_min, y_max], where all values are
        buffed such that their extent is further than the grid's extent..

        This follows the format of the extent input parameter in the matplotlib method imshow (and other methods) and
        is used for visualization in the plot module."""
        return [
            self.scaled_minima[1] - buffer,
            self.scaled_maxima[1] + buffer,
            self.scaled_minima[0] - buffer,
            self.scaled_maxima[0] + buffer,
        ]

    @property
    def yticks(self) -> np.ndarray:
        """Returns the ytick labels of this grid, used for plotting the y-axis ticks when visualizing a grid"""
        return np.linspace(np.min(self[:, 0]), np.max(self[:, 0]), 4)

    @property
    def xticks(self) -> np.ndarray:
        """Returns the xtick labels of this grid, used for plotting the x-axis ticks when visualizing a grid"""
        return np.linspace(np.min(self[:, 1]), np.max(self[:, 1]), 4)

    @staticmethod
    @decorator_util.jit()
    def relocated_grid_from_grid_jit(grid, border_grid):
        """ Relocate the coordinates of a grid to its border if they are outside the border, where the border is
        defined as all pixels at the edge of the grid's mask (see *mask.regions._border_1d_indexes*).

        This is performed as follows:

        1) Use the mean value of the grid's y and x coordinates to determine the origin of the grid.
        2) Compute the radial distance of every grid coordinate from the origin.
        3) For every coordinate, find its nearest pixel in the border.
        4) Determine if it is outside the border, by comparing its radial distance from the origin to its paired \
           border pixel's radial distance.
        5) If its radial distance is larger, use the ratio of radial distances to move the coordinate to the border \
           (if its inside the border, do nothing).

        The method can be used on uniform or irregular grids, however for irregular grids the border of the
        'image-plane' mask is used to define border pixels.

        Parameters
        ----------
        grid : Grid
            The grid (uniform or irregular) whose pixels are to be relocated to the border edge if outside it.
        border_grid : Grid
            The grid of border (y,x) coordinates.
        """

        border_origin = np.zeros(2)
        border_origin[0] = np.mean(border_grid[:, 0])
        border_origin[1] = np.mean(border_grid[:, 1])
        border_grid_radii = np.sqrt(
            np.add(
                np.square(np.subtract(border_grid[:, 0], border_origin[0])),
                np.square(np.subtract(border_grid[:, 1], border_origin[1])),
            )
        )
        border_min_radii = np.min(border_grid_radii)

        grid_radii = np.sqrt(
            np.add(
                np.square(np.subtract(grid[:, 0], border_origin[0])),
                np.square(np.subtract(grid[:, 1], border_origin[1])),
            )
        )

        for pixel_index in range(grid.shape[0]):

            if grid_radii[pixel_index] > border_min_radii:

                closest_pixel_index = np.argmin(
                    np.square(grid[pixel_index, 0] - border_grid[:, 0])
                    + np.square(grid[pixel_index, 1] - border_grid[:, 1])
                )

                move_factor = (
                    border_grid_radii[closest_pixel_index] / grid_radii[pixel_index]
                )

                if move_factor < 1.0:
                    grid[pixel_index, :] = (
                        move_factor * (grid[pixel_index, :] - border_origin[:])
                        + border_origin[:]
                    )

        return grid

    def padded_grid_from_kernel_shape(self, kernel_shape_2d):
        """When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the padded grid is used, which is 'buffed' such that it includes all pixels
        whose signal will be convolved into the unmasked pixels given the 2D kernel shape.

        Parameters
        ----------
        kernel_shape_2d : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        shape = self.mask.shape

        padded_shape = (
            shape[0] + kernel_shape_2d[0] - 1,
            shape[1] + kernel_shape_2d[1] - 1,
        )

        padded_mask = msk.Mask.unmasked(
            shape_2d=padded_shape,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
        )

        return Grid.from_mask(mask=padded_mask)

    @property
    def sub_border_grid(self):
        """The (y,x) grid of all sub-pixels which are at the border of the mask.

        This is NOT all sub-pixels which are in mask pixels at the mask's border, but specifically the sub-pixels
        within these border pixels which are at the extreme edge of the border."""
        return self[self.regions._sub_border_1d_indexes]

    def relocated_grid_from_grid(self, grid):
        """ Relocate the coordinates of a grid to the border of this grid if they are outside the border, where the
        border is defined as all pixels at the edge of the grid's mask (see *mask.regions._border_1d_indexes*).

        This is performed as follows:

        1) Use the mean value of the grid's y and x coordinates to determine the origin of the grid.
        2) Compute the radial distance of every grid coordinate from the origin.
        3) For every coordinate, find its nearest pixel in the border.
        4) Determine if it is outside the border, by comparing its radial distance from the origin to its paired \
           border pixel's radial distance.
        5) If its radial distance is larger, use the ratio of radial distances to move the coordinate to the border \
           (if its inside the border, do nothing).

        The method can be used on uniform or irregular grids, however for irregular grids the border of the
        'image-plane' mask is used to define border pixels.

        Parameters
        ----------
        grid : Grid
            The grid (uniform or irregular) whose pixels are to be relocated to the border edge if outside it.
        """

        return Grid(
            grid=self.relocated_grid_from_grid_jit(
                grid=grid, border_grid=self.sub_border_grid
            ),
            mask=grid.mask,
            sub_size=grid.mask.sub_size,
        )

    def relocated_pixelization_grid_from_pixelization_grid(self, pixelization_grid):
        """ Relocate the coordinates of a pixelization grid to the border of this grid, see the method
        *relocated_grid_from_grid* for a full description of grid relocation.

        This function operates the same as other grid relocation functions by returns the grid as a
        *GridVoronoi* instance.

        Parameters
        ----------
        grid : Grid
            The grid (uniform or irregular) whose pixels are to be relocated to the border edge if outside it.
        """

        if isinstance(pixelization_grid, grids.GridVoronoi):

            return grids.GridVoronoi(
                grid=self.relocated_grid_from_grid_jit(
                    grid=pixelization_grid, border_grid=self.sub_border_grid
                ),
                nearest_pixelization_1d_index_for_mask_1d_index=pixelization_grid.nearest_pixelization_1d_index_for_mask_1d_index,
            )

        return pixelization_grid

    def output_to_fits(self, file_path, overwrite=False):
        """Output the grid to a .fits file.

        Parameters
        ----------
        file_path : str
            The path the file is output to, including the filename and the '.fits' extension,
            e.g. '/path/to/filename.fits'
        overwrite : bool
            If a file already exists at the path, if overwrite=True it is overwritten else an error is raised."""
        array_util.numpy_array_1d_to_fits(
            array_2d=self.in_2d, file_path=file_path, overwrite=overwrite
        )

    def structure_from_result(self, result: np.ndarray) -> typing.Union[arrays.Array]:
        """Convert a result from an ndarray to an aa.Array or aa.Grid structure, where the conversion depends on
        type(result) as follows:

        - 1D np.ndarray   -> aa.Array
        - 2D np.ndarray   -> aa.Grid

        This function is used by the grid_like_to_structure decorator to convert the output result of a function
        to an autoarray structure when a *Grid* instance is passed to the decorated function.

        Parameters
        ----------
        result : np.ndarray or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        if len(result.shape) == 1:
            return arrays.Array(array=result, mask=self.mask, store_in_1d=True)
        else:
            if isinstance(result, GridTransformedNumpy):
                return GridTransformed(grid=result, mask=self.mask, store_in_1d=True)
            return Grid(grid=result, mask=self.mask, store_in_1d=True)

    def structure_list_from_result_list(
        self, result_list: list
    ) -> typing.Union[arrays.Array, list]:
        """Convert a result from a list of ndarrays to a list of aa.Array or aa.Grid structure, where the conversion
        depends on type(result) as follows:

        - [1D np.ndarray] -> [aa.Array]
        - [2D np.ndarray] -> [aa.Grid]

        This function is used by the grid_like_list_to_structure-list decorator to convert the output result of a
        function to a list of autoarray structure when a *Grid* instance is passed to the decorated function.

        Parameters
        ----------
        result_list : np.ndarray or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        return [self.structure_from_result(result=result) for result in result_list]


class GridSparse:
    def __init__(self, sparse_grid, sparse_1d_index_for_mask_1d_index):
        """A sparse grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of a
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
        sparse_grid : ndarray or Grid
            The (y,x) grid of sparse coordinates.
        sparse_1d_index_for_mask_1d_index : ndarray
            An array whose indexes map pixels from a Grid's mask to the closest (y,x) coordinate on the sparse_grid.
        """
        self.sparse = sparse_grid
        self.sparse_1d_index_for_mask_1d_index = sparse_1d_index_for_mask_1d_index

    @classmethod
    def from_grid_and_unmasked_2d_grid_shape(cls, grid, unmasked_sparse_shape):
        """Calculate a GridSparse a Grid from the unmasked 2D shape of the sparse grid.

        This is performed by overlaying the 2D sparse grid (computed from the unmaksed sparse shape) over the edge
        values of the Grid.

        This function is used in the *operators.inversion* package to set up the VoronoiMagnification Pixelization.

        Parameters
        -----------
        grid : Grid
            The grid of (y,x) scaled coordinates at the centre of every image value (e.g. image-pixels).
        unmasked_sparse_shape : (int, int)
            The 2D shape of the sparse grid which is overlaid over the grid.
        """

        pixel_scales = grid.mask.pixel_scales

        pixel_scales = (
            (grid.shape_2d_scaled[0] + pixel_scales[0]) / (unmasked_sparse_shape[0]),
            (grid.shape_2d_scaled[1] + pixel_scales[1]) / (unmasked_sparse_shape[1]),
        )

        origin = grid.geometry.mask_centre

        unmasked_sparse_grid_1d = grid_util.grid_1d_via_shape_2d_from(
            shape_2d=unmasked_sparse_shape,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
        )

        unmasked_sparse_grid_pixel_centres = grid_util.grid_pixel_centres_1d_from(
            grid_scaled_1d=unmasked_sparse_grid_1d,
            shape_2d=grid.mask.shape,
            pixel_scales=grid.mask.pixel_scales,
        ).astype("int")

        total_sparse_pixels = mask_util.total_sparse_pixels_from(
            mask=grid.mask,
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

        regular_to_unmasked_sparse = grid_util.grid_pixel_indexes_1d_from(
            grid_scaled_1d=grid,
            shape_2d=unmasked_sparse_shape,
            pixel_scales=pixel_scales,
            origin=origin,
        ).astype("int")

        sparse_1d_index_for_mask_1d_index = sparse_util.sparse_1d_index_for_mask_1d_index_from(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            sparse_for_unmasked_sparse=sparse_for_unmasked_sparse,
        ).astype(
            "int"
        )

        sparse_grid = sparse_util.sparse_grid_via_unmasked_from(
            unmasked_sparse_grid=unmasked_sparse_grid_1d,
            unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
        )

        return GridSparse(
            sparse_grid=sparse_grid,
            sparse_1d_index_for_mask_1d_index=sparse_1d_index_for_mask_1d_index,
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
        """Calculate a GridSparse from a Grid and weight map.

        This is performed by running a KMeans clustering algorithm on the weight map, such that GridSparse (y,x)
        coordinates cluster around the weight map values with higher values.

        This function is used in the *operators.inversion* package to set up the VoronoiMagnification Pixelization.

        Parameters
        -----------
        total_pixels : int
            The total number of pixels in the GridSparse and input into the KMeans algortihm.
        grid : Grid
            The grid of (y,x) coordinates corresponding to the weight map.
        weight_map : ndarray
            The 2D array of weight values that the KMeans clustering algorithm adapts to to determine the GridSparse.
        n_iter : int
            The number of times the KMeans algorithm is repeated.
        max_iter : int
            The maximum number of iterations in one run of the KMeans algorithm.
        seed : int or None
            The random number seed, which can be used to reproduce GridSparse's for the same inputs.
        stochastic : bool
            If True, the random number seed is randommly chosen every time the function is called, ensuring every
            pixel-grid is randomly determined and thus stochastic.
        """

        if stochastic:
            seed = np.random.randint(low=1, high=2 ** 32)

        if total_pixels > grid.shape[0]:
            raise exc.GridException

        kmeans = KMeans(
            n_clusters=total_pixels, random_state=seed, n_init=n_iter, max_iter=max_iter
        )

        try:
            kmeans = kmeans.fit(X=grid.in_1d_binned, sample_weight=weight_map)
        except ValueError or OverflowError:
            raise exc.InversionException()

        return GridSparse(
            sparse_grid=kmeans.cluster_centers_,
            sparse_1d_index_for_mask_1d_index=kmeans.labels_.astype("int"),
        )

    @property
    def total_sparse_pixels(self):
        return len(self.sparse)


class GridTransformed(Grid):

    pass


class GridTransformedNumpy(np.ndarray):
    def __new__(cls, grid, *args, **kwargs):
        return grid.view(cls)


class MaskedGrid(Grid):
    @classmethod
    def manual_1d(cls, grid, mask, store_in_1d=True):
        """Create a Grid (see *Grid.__new__*) by inputting the grid coordinates in 1D with their paired mask, for 
        example:

        mask = Mask([[True, False, False, False])
        grid=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape [total_sub_coordinates, 2] or list of lists.
        mask : msk.Mask
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        grid = convert_and_check_grid(grid=grid, mask=mask)

        if store_in_1d:
            return Grid(grid=grid, mask=mask, store_in_1d=store_in_1d)

        sub_grid_2d = grid_util.sub_grid_2d_from(
            sub_grid_1d=grid, mask=mask, sub_size=mask.sub_size
        )

        return Grid(grid=sub_grid_2d, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def manual_2d(cls, grid, mask, store_in_1d=True):
        """Create a Grid (see *Grid.__new__*) by inputting the grid coordinates in 2D with their paired mask, for
        example:

        mask = Mask([[True, False, False, False])
        grid=np.array([[[1.0, 1.0], [2.0, 2.0]],
                       [[3.0, 3.0], [4.0, 4.0]]])
        grid=[[[1.0, 1.0], [2.0, 2.0]],
              [[3.0, 3.0], [4.0, 4.0]]]

        Mask values are removed, such that the grid in 1D will be of length 3, omitting the values [1.0, 1.0].

        Parameters
        ----------
        grid : np.ndarray or list
            The (y,x) coordinates of the grid input as an ndarray of shape
            [total_y_pixels*sub_size, total_x_pixels*sub_size, 2] or a list of lists.
        mask : msk.Mask
            The 2D mask associated with the grid, defining the pixels each grid coordinate is paired with and
            originates from.
        store_in_1d : bool
            If True, the grid is stored in 1D as an ndarray of shape [total_unmasked_pixels, 2]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels, 2].
        """
        grid = convert_and_check_grid(grid=grid, mask=mask)

        sub_grid_1d = grid_util.sub_grid_1d_from(
            sub_grid_2d=grid, mask=mask, sub_size=mask.sub_size
        )

        if store_in_1d:
            return Grid(grid=sub_grid_1d, mask=mask, store_in_1d=store_in_1d)

        sub_grid_2d = grid_util.sub_grid_2d_from(
            sub_grid_1d=sub_grid_1d, mask=mask, sub_size=mask.sub_size
        )

        return Grid(grid=sub_grid_2d, mask=mask, store_in_1d=store_in_1d)
