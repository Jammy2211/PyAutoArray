import numpy as np
from sklearn.cluster import KMeans

import typing

import autoarray as aa

from autoarray import decorator_util
from autoarray import exc
from autoarray.structures import abstract_structure, arrays, grids
from autoarray.mask import mask as msk
from autoarray.util import sparse_util, array_util, grid_util, mask_util


def convert_and_check_grid(grid):

    if type(grid) is list:
        grid = np.asarray(grid)

    if grid.shape[-1] != 2:
        raise exc.GridException(
            "The final dimension of the input grid is not equal to 2 (e.g. the (y,x) coordinates)"
        )

    if 2 < len(grid.shape) > 3:
        raise exc.GridException("The dimensions of the input grid array is not 2 or 3")

    return grid


def convert_pixel_scales(pixel_scales):

    if type(pixel_scales) is float:
        pixel_scales = (pixel_scales, pixel_scales)

    return pixel_scales


class Grid(abstract_structure.AbstractStructure):
    def __new__(cls, grid, mask, store_in_1d=True, binned=None, *args, **kwargs):
        """A grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of an \
        unmasked pixel. The positive y-axis is upwards and positive x-axis to the right.

        Sub-size = 1 case:
        ------------------

        A *Grid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_unmasked_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixel index and second element the y or x arc -econd coordinates. For example:

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

        Sub-size > 1 case:
        ------------------

        If the input masks's sub-size is greater than 1, the grid is defined as a sub-grid where each entry corresponds
        to the (y,x) coordinates at the centre of each sub-pixel of an unmasked pixel. The sub-grid indexes are ordered
        such that pixels begin from the first (top-left) sub-pixel in the first unmasked pixel. Indexes then go over
        the sub-pixels in each unmasked pixel, for every unmasked pixel. Therefore, the sub-grid is an ndarray of shape
        [total_unmasked_pixels*(sub_grid_shape)**2, 2]. For example:

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

        Our grid looks like it did before:

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

        However, we now go to each unmasked pixel and derive a sub-pixel grid for it. For example, for pixel 0,
        if *sub_size=2*, we use a 2x2 sub-grid:

        Pixel 0 - (2x2):
                                y      x
               grid[0] = [0.66, -1.66]
        |0|1|  grid[1] = [0.66, -1.33]
        |2|3|  grid[2] = [0.33, -1.66]
               grid[3] = [0.33, -1.33]

        Now, we'd normally sub-grid all pixels using the same *sub_size*, but for this illustration lets
        pretend we used a sub_size of 3x3 for pixel 1:

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

        Grid Mapping
        ------------

        Every set of (y,x) coordinates in a pixel of the sub-grid correspond to an unmasked pixel in its input 2D mask.
        In this case, the grid is uniform and every (y,x) coordinates of the grid directly corresponds to the
        location of its paired unmasked pixel.

        However, it is not a requirement that grid coordinates align with their corresponding unmasked pixels or that
        the grid is uniform. The input grid could be an irregular set of (y,x) coordinates where the indexing signifies
        that the (y,x) coordinate *originates* from that pixel in the mask, but has had its value change by some
        aspect of the calculation.

        Parameters
        ----------
        grid : np.ndarray
            The (y,x) coordinates of the grid in a NumPy array of shape [total_coordinates, 2].
        mask : msk.Mask
            The 2D mask associated with the grid, defining the image-plane pixels of each grid coordinate.
        """
        obj = super(Grid, cls).__new__(
            cls=cls, structure=grid, mask=mask, store_in_1d=store_in_1d
        )
        obj.interpolator = None
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
            sub_grid_1d=grid, mask_2d=mask, sub_size=sub_size
        )

        return Grid(grid=grid_2d, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def manual_2d(
        cls, grid, pixel_scales, sub_size=1, origin=(0.0, 0.0), store_in_1d=True
    ):

        grid = convert_and_check_grid(grid=grid)
        pixel_scales = convert_pixel_scales(pixel_scales=pixel_scales)

        shape = (int(grid.shape[0] / sub_size), int(grid.shape[1] / sub_size))

        mask = msk.Mask.unmasked(
            shape_2d=shape, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

        if not store_in_1d:
            return Grid(grid=grid, mask=mask, store_in_1d=store_in_1d)

        grid_1d = grid_util.sub_grid_1d_from(
            sub_grid_2d=grid, mask_2d=mask, sub_size=sub_size
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
        """Setup a sub-grid of the unmasked pixels, using a mask and a specified sub-grid size. The center of \
        every unmasked pixel's sub-pixels give the grid's (y,x) scaled coordinates.

        Parameters
        -----------
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        """

        sub_grid_1d = grid_util.grid_1d_via_mask_2d_from(
            mask_2d=mask,
            pixel_scales=mask.pixel_scales,
            sub_size=mask.sub_size,
            origin=mask.origin,
        )

        if store_in_1d:
            return grids.Grid(grid=sub_grid_1d, mask=mask, store_in_1d=store_in_1d)

        sub_grid_2d = grid_util.sub_grid_2d_from(
            sub_grid_1d=sub_grid_1d, mask_2d=mask, sub_size=mask.sub_size
        )

        return grids.Grid(grid=sub_grid_2d, mask=mask, store_in_1d=store_in_1d)

    @classmethod
    def blurring_grid_from_mask_and_kernel_shape(
        cls, mask, kernel_shape_2d, store_in_1d=True
    ):
        """Setup a blurring-grid from a mask, where a blurring grid consists of all pixels that are masked, but they \
        are close enough to the unmasked pixels that a fraction of their light will be blurred into those pixels \
        via PSF convolution. For example, if our mask is as follows:

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

        For a PSF of shape (5,5), the following blurring mask is computed (noting that pixels that are 2 pixels from an
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
        """

        blurring_mask = mask.regions.blurring_mask_from_kernel_shape(
            kernel_shape_2d=kernel_shape_2d
        )

        blurring_grid_1d = grid_util.grid_1d_via_mask_2d_from(
            mask_2d=blurring_mask,
            pixel_scales=blurring_mask.pixel_scales,
            sub_size=blurring_mask.sub_size,
            origin=blurring_mask.origin,
        )

        if store_in_1d:
            return blurring_mask.mapping.grid_stored_1d_from_grid_1d(
                grid_1d=blurring_grid_1d
            )
        else:
            return blurring_mask.mapping.grid_stored_2d_from_grid_1d(
                grid_1d=blurring_grid_1d
            )

    def __array_finalize__(self, obj):

        super(Grid, self).__array_finalize__(obj)

        if isinstance(obj, Grid):

            if hasattr(obj, "interpolator"):
                self.interpolator = obj.interpolator

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

    @property
    def in_1d(self):
        if self.store_in_1d:
            return self
        else:
            return self.mask.mapping.grid_stored_1d_from_sub_grid_2d(sub_grid_2d=self)

    @property
    def in_2d(self):
        if self.store_in_1d:
            return self.mask.mapping.grid_stored_2d_from_sub_grid_1d(sub_grid_1d=self)
        else:
            return self

    @property
    def in_1d_binned(self):
        if self.store_in_1d:
            return self.mask.mapping.grid_stored_1d_binned_from_sub_grid_1d(
                sub_grid_1d=self
            )
        else:
            sub_grid_1d = self.mask.mapping.grid_stored_1d_from_sub_grid_2d(
                sub_grid_2d=self
            )
            return self.mask.mapping.grid_stored_1d_binned_from_sub_grid_1d(
                sub_grid_1d=sub_grid_1d
            )

    @property
    def in_2d_binned(self):
        if self.store_in_1d:
            return self.mask.mapping.grid_stored_2d_binned_from_sub_grid_1d(
                sub_grid_1d=self
            )
        else:
            sub_grid_1d = self.mask.mapping.grid_stored_1d_from_sub_grid_2d(
                sub_grid_2d=self
            )
            return self.mask.mapping.grid_stored_2d_binned_from_sub_grid_1d(
                sub_grid_1d=sub_grid_1d
            )

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
            The (y,x) coordinate from which the squared distance of every grid (y,x) coordinate is computed.
            """
        distances = np.sqrt(
            self.squared_distances_from_coordinate(coordinate=coordinate)
        )
        return aa.MaskedArray(array=distances, mask=self.mask)

    def blurring_grid_from_kernel_shape(self, kernel_shape_2d):
        """From this grid, determine the blurring grid.

        The blurring grid gives the (y,x) coordinates of all pixels which are masked but whose light will be blurred
        into unmasked due to 2D convolution. These pixels are determined by this grid's mask and the 2D shape of
        the *Kernel*.

        Parameters
        ----------
        kernel_shape_2d : (int, int)
            The 2D shape of the Kernel used to determine which masked pixel's values will be blurred into the grid's
            unmasked pixels by 2D convolution.
        """

        blurring_mask = self.mask.regions.blurring_mask_from_kernel_shape(
            kernel_shape_2d=kernel_shape_2d
        )

        blurring_grid_1d = grid_util.grid_1d_via_mask_2d_from(
            mask_2d=blurring_mask,
            pixel_scales=blurring_mask.pixel_scales,
            sub_size=blurring_mask.sub_size,
            origin=blurring_mask.origin,
        )

        return blurring_mask.mapping.grid_stored_1d_from_grid_1d(
            grid_1d=blurring_grid_1d
        )

    def new_grid_with_interpolator(self, interpolation_pixel_scale):
        # noinspection PyAttributeOutsideInit
        # TODO: This function doesn't do what it says on the tin. The returned grid would be the same as the grid
        # TODO: on which the function was called but with a new interpolator set.
        self.interpolator = grids.GridInterpolate.from_mask_grid_and_interpolation_pixel_scales(
            mask=self.mask,
            grid=self[:, :],
            interpolation_pixel_scale=interpolation_pixel_scale,
        )
        return self

    @property
    def in_1d_flipped(self) -> np.ndarray:
        """Return the grid as a NumPy array of shape (total_pixels, 2) with flipped values such that coordinates are
        given as (x,y) values.

        This is used to interface with Python libraries that require the grid in (x,y) format."""
        return np.fliplr(self)

    @property
    def in_2d_flipped(self):
        """Return the grid as a NumPy array of shape (total_x_pixels, total_y_pixels, 2) with flipped values such that
        coordinates are given as (x,y) values.

        This is used to interface with Python libraries that require the grid in (x,y) format."""
        return np.stack((self.in_2d[:, :, 1], self.in_2d[:, :, 0]), axis=-1)

    @property
    @array_util.Memoizer()
    def in_radians(self):
        """Return the grid with a conversion to Radians.

        This grid is used by the interferometer module."""
        return (self * np.pi) / 648000.0

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
        """The extent of the grid in scaled units returned as a NumPy array [x_min, x_max, y_min, y_max].

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

        if self.interpolator is None and not isinstance(self, grids.GridIterator):
            return Grid.from_mask(mask=padded_mask)
        elif self.interpolator is None and isinstance(self, grids.GridIterator):
            return grids.GridIterator.from_mask(
                mask=padded_mask,
                fractional_accuracy=self.fractional_accuracy,
                sub_steps=self.sub_steps,
            )
        else:
            return Grid.from_mask(mask=padded_mask).new_grid_with_interpolator(
                interpolation_pixel_scale=self.interpolator.interpolation_pixel_scale
            )

    @property
    def sub_border_grid(self):
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

        This function operatess the same as other grid relocation functions by returns the grid as a
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

        array_util.numpy_array_1d_to_fits(
            array_2d=self.in_2d, file_path=file_path, overwrite=overwrite
        )

    def structure_from_result(self, result: np.ndarray) -> typing.Union[arrays.Array]:
        """Convert a result from a non autoarray structure to an aa.Array or aa.Grid structure, where the conversion
        depends on type(result) as follows:

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
            return self.mapping.array_stored_1d_from_sub_array_1d(sub_array_1d=result)
        else:
            return self.mapping.grid_stored_1d_from_sub_grid_1d(
                sub_grid_1d=result,
                is_transformed=isinstance(result, GridTransformedNumpy),
            )

    def structure_list_from_result_list(
        self, result_list: list
    ) -> typing.Union[arrays.Array, list]:
        """Convert a result from a list of non autoarray structures to an aa.Array or aa.Grid structure, where the
        conversion depends on type(result) as follows:

        - [1D np.ndarray] -> [aa.Array]
        - [2D np.ndarray] -> [aa.Grid]

        This function is used by the grid_like_list_to_structure-list decorator to convert the output result of a
        function to a list of autoarray structure when a *Grid* instance is passed to the decorated function.

        Parameters
        ----------
        result_list : np.ndarray or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        if len(result_list[0].shape) == 1:
            return [
                self.mapping.array_stored_1d_from_sub_array_1d(sub_array_1d=value)
                for value in result_list
            ]
        elif len(result_list[0].shape) == 2:
            return [
                self.mapping.grid_stored_1d_from_sub_grid_1d(sub_grid_1d=value)
                for value in result_list
            ]


class GridSparse:
    def __init__(self, sparse_grid, sparse_1d_index_for_mask_1d_index):
        """A sparse grid of coordinates, where each entry corresponds to the (y,x) coordinates at the centre of a \
        pixel on the sparse grid. To setup the sparse-grid, it is laid over a grid of unmasked pixels, such \
        that all sparse-grid pixels which map inside of an unmasked grid pixel are included on the sparse grid.

        To setup this sparse grid, we thus have two sparse grid:

        - The unmasked sparse-grid, which corresponds to a uniform 2D array of pixels. The edges of this grid \
          correspond to the 4 edges of the mask (e.g. the higher and lowest (y,x) scaled unmasked pixels) and the \
          grid's shape is speciifed by the unmasked_sparse_grid_shape parameter.

        - The (masked) sparse-grid, which is all pixels on the unmasked sparse-grid above which fall within unmasked \
          grid pixels. These are the pixels which are actually used for other modules in PyAutoArray.

        The origin of the unmasked sparse grid can be changed to allow off-center pairings with sparse-grid pixels, \
        which is necessary when a mask has a centre offset from (0.0", 0.0"). However, the sparse grid itself \
        retains an origin of (0.0", 0.0"), ensuring its scaled grid uses the same coordinate system as the \
        other grid.

        The sparse grid is used to determine the pixel centers of an adaptive grid pixelization.

        Parameters
        ----------
        unmasked_sparse_shape : (int, int)
            The shape of the unmasked sparse-grid whose centres form the sparse-grid.
        pixel_scales : (float, float)
            The pixel conversion scale of a pixel in the y and x directions.
        grid : Grid
            The grid used to determine which pixels are in the sparse grid.
        origin : (float, float)
            The centre of the unmasked sparse grid, which matches the centre of the mask.
        """
        self.sparse = sparse_grid
        self.sparse_1d_index_for_mask_1d_index = sparse_1d_index_for_mask_1d_index

    @classmethod
    def from_grid_and_unmasked_2d_grid_shape(cls, grid, unmasked_sparse_shape):
        """Calculate the image-plane pixelization from a grid of coordinates (and its mask).

        See *grid_stacks.GridSparse* for details on how this grid is calculated.

        Parameters
        -----------
        grid : grids.Grid
            The grid of (y,x) scaled coordinates at the centre of every image value (e.g. image-pixels).
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
            mask_2d=grid.mask,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        sparse_for_unmasked_sparse = sparse_util.sparse_for_unmasked_sparse_from(
            mask_2d=grid.mask,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=total_sparse_pixels,
        ).astype("int")

        unmasked_sparse_for_sparse = sparse_util.unmasked_sparse_for_sparse_from(
            total_sparse_pixels=total_sparse_pixels,
            mask_2d=grid.mask,
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
        cls, total_pixels, grid, weight_map, n_iter=1, max_iter=5, seed=None
    ):
        """Calculate the image-plane pixelization from a grid of coordinates (and its mask).

        See *grid_stacks.GridSparse* for details on how this grid is calculated.

        Parameters
        -----------
        grid : grids.Grid
            The grid of (y,x) scaled coordinates at the centre of every image value (e.g. image-pixels).
        """

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

        if type(grid) is list:
            grid = np.asarray(grid)

        if grid.shape[0] != mask.sub_pixels_in_mask:
            raise exc.GridException(
                "The input 1D grid does not have the same number of entries as sub-pixels in"
                "the mask."
            )

        if store_in_1d:
            return mask.mapping.grid_stored_1d_from_sub_grid_1d(sub_grid_1d=grid)
        return mask.mapping.grid_stored_2d_from_sub_grid_1d(sub_grid_1d=grid)

    @classmethod
    def manual_2d(cls, grid, mask, store_in_1d=True):

        if type(grid) is list:
            grid = np.asarray(grid)

        if (grid.shape[0], grid.shape[1]) != mask.sub_shape_2d:
            raise exc.GridException(
                "The input grid is 2D but not the same dimensions as the sub-mask "
                "(e.g. the mask 2D shape multipled by its sub size."
            )

        if store_in_1d:
            return mask.mapping.grid_stored_1d_from_sub_grid_2d(sub_grid_2d=grid)
        sub_grid_1d = mask.mapping.grid_stored_1d_from_sub_grid_2d(sub_grid_2d=grid)
        return mask.mapping.grid_stored_2d_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

    @classmethod
    def from_mask(cls, mask, store_in_1d=True):
        """Setup a sub-grid of the unmasked pixels, using a mask and a specified sub-grid size. The center of \
        every unmasked pixel's sub-pixels give the grid's (y,x) arc-second coordinates.

        Parameters
        -----------
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        """

        sub_grid_1d = grid_util.grid_1d_via_mask_2d_from(
            mask_2d=mask,
            pixel_scales=mask.pixel_scales,
            sub_size=mask.sub_size,
            origin=mask.origin,
        )

        if store_in_1d:
            return mask.mapping.grid_stored_1d_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)
        return mask.mapping.grid_stored_2d_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)
