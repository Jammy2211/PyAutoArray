import ast
import numpy as np
import scipy.spatial
import scipy.spatial.qhull as qhull
from functools import wraps
from sklearn.cluster import KMeans
import os

import typing

import autoarray as aa

from autoarray import decorator_util
from autoarray import exc
from autoarray.structures import abstract_structure, arrays
from autoarray.mask import mask as msk
from autoarray.util import (
    sparse_util,
    array_util,
    grid_util,
    mask_util,
    pixelization_util,
)


class AbstractGrid(abstract_structure.AbstractStructure):
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
        obj = super(AbstractGrid, cls).__new__(
            cls=cls, structure=grid, mask=mask, store_in_1d=store_in_1d
        )
        obj.interpolator = None
        obj.binned = None
        return obj

    def __array_finalize__(self, obj):

        super(AbstractGrid, self).__array_finalize__(obj)

        if isinstance(obj, Grid):

            if hasattr(obj, "interpolator"):
                self.interpolator = obj.interpolator

            if hasattr(obj, "binned"):
                self.binned = obj.binned

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

    def new_grid_with_binned_grid(self, binned_grid):
        # noinspection PyAttributeOutsideInit
        # TODO: This function doesn't do what it says on the tin. The returned grid would be the same as the grid
        # TODO: on which the function was called but with a new interpolator set.
        self.binned = binned_grid
        return self

    def new_grid_with_interpolator(self, pixel_scale_interpolation_grid):
        # noinspection PyAttributeOutsideInit
        # TODO: This function doesn't do what it says on the tin. The returned grid would be the same as the grid
        # TODO: on which the function was called but with a new interpolator set.
        self.interpolator = GridInterpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=self.mask,
            grid=self[:, :],
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
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

        padded_grid_1d = grid_util.grid_1d_via_mask_2d_from(
            mask_2d=padded_mask,
            pixel_scales=padded_mask.pixel_scales,
            sub_size=padded_mask.sub_size,
            origin=padded_mask.origin,
        )

        padded_sub_grid = padded_mask.mapping.grid_stored_1d_from_sub_grid_1d(
            sub_grid_1d=padded_grid_1d
        )

        if self.interpolator is None:
            return padded_sub_grid
        else:
            return padded_sub_grid.new_grid_with_interpolator(
                pixel_scale_interpolation_grid=self.interpolator.pixel_scale_interpolation_grid
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

        if isinstance(pixelization_grid, GridVoronoi):

            return GridVoronoi(
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


class Grid(AbstractGrid):
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

        if type(grid) is list:
            grid = np.asarray(grid)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        if grid.shape[-1] != 2:
            raise exc.GridException(
                "The final dimension of the input grid is not equal to 2 (e.g. the (y,x) coordinates)"
            )

        if 2 < len(grid.shape) > 3:
            raise exc.GridException(
                "The dimensions of the input grid array is not 2 or 3"
            )

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
        )

        if store_in_1d:
            return mask.mapping.grid_stored_1d_from_sub_grid_1d(sub_grid_1d=grid)
        else:
            return mask.mapping.grid_stored_2d_from_sub_grid_1d(sub_grid_1d=grid)

    @classmethod
    def manual_2d(
        cls, grid, pixel_scales, sub_size=1, origin=(0.0, 0.0), store_in_1d=True
    ):

        if type(grid) is list:
            grid = np.asarray(grid)

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

        if grid.shape[-1] != 2:
            raise exc.GridException(
                "The final dimension of the input grid is not equal to 2 (e.g. the (y,x) coordinates)"
            )

        if 2 < len(grid.shape) > 3:
            raise exc.GridException(
                "The dimensions of the input grid array is not 2 or 3"
            )

        shape = (int(grid.shape[0] / sub_size), int(grid.shape[1] / sub_size))

        mask = msk.Mask.unmasked(
            shape_2d=shape, pixel_scales=pixel_scales, sub_size=sub_size, origin=origin
        )

        if store_in_1d:
            return mask.mapping.grid_stored_1d_from_sub_grid_2d(sub_grid_2d=grid)
        else:
            sub_grid_1d = mask.mapping.grid_stored_1d_from_sub_grid_2d(sub_grid_2d=grid)
            return mask.mapping.grid_stored_2d_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

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

        if type(pixel_scales) is float:
            pixel_scales = (pixel_scales, pixel_scales)

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
            return mask.mapping.grid_stored_1d_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)
        else:
            return mask.mapping.grid_stored_2d_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

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


class GridIterator(Grid):
    def __new__(
        cls,
        grid,
        mask,
        fractional_accuracy=0.9999,
        sub_steps=[2, 4, 8, 16],
        store_in_1d=True,
        *args,
        **kwargs,
    ):
        """Represents a grid of coordinates as described for the *Grid* class, but using an iterative sub-grid that
        adapts its resolution when it is input into a function that performs an analytic calculation.

        A *Grid* represents (y,x) coordinates using a sub-grid, where functions are evaluated once at every coordinate
        on the sub-grid and averaged to give a more precise evaluation an analytic function. A *GridIterator* does not
        have a specified sub-grid size, but instead iteratively recomputes the analytic function at increasing sub-grid
        sizes until an input fractional accuracy is reached.

        Iteration is performed on a per (y,x) coordinate basis, such that the sub-grid size will adopt low values
        wherever doing so can meet the fractional accuracy and high values only where it is required to meet the
        fractional accuracy. For functions where a wide range of sub-grid sizes allow fractional accuracy to be met
        this ensures the function can be evaluated accurate in a computaionally efficient manner.

        This overcomes a limitation of the *Grid* class whereby if a small subset of pixels require high levels of
        sub-gridding to be evaluated accuracy, the entire grid would require this sub-grid size thus leading to
        unecessary expensive function evaluations.


        """
        obj = super().__new__(cls=cls, grid=grid, mask=mask, store_in_1d=store_in_1d)
        obj.grid = MaskedGrid.manual_1d(grid=grid, mask=mask)
        obj.fractional_accuracy = fractional_accuracy
        obj.sub_steps = sub_steps
        obj.binned = True
        return obj

    @classmethod
    def manual_1d(
        cls,
        grid,
        shape_2d,
        pixel_scales,
        origin=(0.0, 0.0),
        fractional_accuracy=0.9999,
        sub_steps=[2, 4, 8, 16],
        store_in_1d=True,
    ):

        grid = Grid.manual_1d(
            grid=grid,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
            store_in_1d=store_in_1d,
        )

        return GridIterator(
            grid=np.asarray(grid),
            mask=grid.mask,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def uniform(
        cls,
        shape_2d,
        pixel_scales,
        origin=(0.0, 0.0),
        fractional_accuracy=0.9999,
        sub_steps=[2, 4, 8, 16],
        store_in_1d=True,
    ):

        grid = Grid.uniform(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
            store_in_1d=store_in_1d,
        )

        return GridIterator(
            grid=np.asarray(grid),
            mask=grid.mask,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            store_in_1d=store_in_1d,
        )

    @classmethod
    def from_mask(
        cls, mask, fractional_accuracy=1e-4, sub_steps=[2, 4, 8, 16], store_in_1d=True
    ):
        """Setup a sub-grid of the unmasked pixels, using a mask and a specified sub-grid size. The center of \
        every unmasked pixel's sub-pixels give the grid's (y,x) scaled coordinates.

        Parameters
        -----------
        mask : Mask
            The mask whose masked pixels are used to setup the sub-pixel grid.
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        """

        grid = Grid.from_mask(mask=mask.mask_sub_1, store_in_1d=store_in_1d)

        return GridIterator(
            grid=np.asarray(grid),
            mask=grid.mask,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            store_in_1d=store_in_1d,
        )

    def fractional_mask_from_arrays(
        self, array_lower_sub_2d, array_higher_sub_2d
    ) -> msk.Mask:
        """ Compute a fractional mask from a result array, where the fractional mask describes whether the evaluated
        value in the result array is within the *GridIterator*'s specified fractional accuracy. The fractional mask thus
        determines whether a pixel on the grid needs to be reevaluated at a higher level of sub-gridding to meet the
        specified fractional accuracy. If it must be re-evaluated, the fractional masks's entry is *False*.

        The fractional mask is computed by comparing the results evaluated at one level of sub-gridding to another
        at a higher level of sub-griding. Thus, the sub-grid size in chosen on a per-pixel basis until the function
        is evaluated at the specified fractional accuracy.

        Parameters
        ----------
        result_array_lower_sub : arrays.Array
            The results computed by a function using a lower sub-grid size
        result_array_lower_sub : arrays.Array
            The results computed by a function using a lower sub-grid size.
        """

        fractional_mask = msk.Mask.unmasked(
            shape_2d=array_lower_sub_2d.shape_2d, invert=True
        )

        fractional_mask = self.fractional_mask_jit_from_array(
            fractional_accuracy_threshold=self.fractional_accuracy,
            fractional_mask=fractional_mask,
            array_higher_sub_2d=array_higher_sub_2d,
            array_lower_sub_2d=array_lower_sub_2d,
            array_higher_mask_2d=array_higher_sub_2d.mask,
        )

        return msk.Mask(
            mask_2d=fractional_mask,
            pixel_scales=array_higher_sub_2d.pixel_scales,
            origin=array_higher_sub_2d.origin,
        )

    @staticmethod
    @decorator_util.jit()
    def fractional_mask_jit_from_array(
        fractional_accuracy_threshold,
        fractional_mask,
        array_higher_sub_2d,
        array_lower_sub_2d,
        array_higher_mask_2d,
    ):
        """Jitted functioon to determine the fractional mask, which is a mask where:

        - *True* entries signify the function has been evaluated in that pixel to desired fractional accuracy and
           therefore does not need to be iteratively computed at higher levels of sub-gridding.

        - *False* entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
           therefore must be iterative computed at higher levels of sub-gridding to meet this accuracy."""

        for y in range(fractional_mask.shape[0]):
            for x in range(fractional_mask.shape[1]):
                if not array_higher_mask_2d[y, x]:

                    fractional_accuracy = (
                        array_lower_sub_2d[y, x] / array_higher_sub_2d[y, x]
                    )

                    if fractional_accuracy > 1.0:
                        fractional_accuracy = 1.0 / fractional_accuracy

                    if fractional_accuracy < fractional_accuracy_threshold:
                        fractional_mask[y, x] = False

        return fractional_mask

    def iterated_array_from_func(
        self, func, profile, array_lower_sub_2d
    ) -> arrays.Array:
        """Iterate over a function that returns an array of values until the it meets a specified fractional accuracy.
        The function returns a result on a pixel-grid where evaluating it on more points on a higher resolution
        sub-grid followed by binning lead to a more precise evaluation of the function.

        The function is first called for a sub-grid size of 1 and a higher resolution grid. The ratio of values give
        the fractional accuracy of each function evaluation. Pixels which do not meet the fractional accuracy are
        iteratively revaluated on higher resolution sub-grids. This is repeated until all pixels meet the fractional
        accuracy or the highest sub-size specified in the *sub_steps* attribute is computed.

        An example use case of this function is when a "profile_image_from_grid" methods in **PyAutoGalaxy**'s
        *LightProfile* module is comomputed, which by evaluating the function on a higher resolution sub-grids sample
        the analytic light profile at more points and thus more precisely.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation."""

        iterated_array = np.zeros(shape=self.shape_2d)

        fractional_mask_lower_sub = self.mask

        for sub_size in self.sub_steps:

            mask_higher_sub = fractional_mask_lower_sub.mapping.mask_new_sub_size_from_mask(
                mask=fractional_mask_lower_sub, sub_size=sub_size
            )

            grid_compute = Grid.from_mask(mask=mask_higher_sub)
            array_higher_sub = func(profile, grid_compute)
            array_higher_sub = grid_compute.structure_from_result(
                result=array_higher_sub
            ).in_2d_binned

            fractional_mask_higher_sub = self.fractional_mask_from_arrays(
                array_lower_sub_2d=array_lower_sub_2d,
                array_higher_sub_2d=array_higher_sub,
            )

            iterated_array = self.iterated_array_jit_from(
                iterated_array=iterated_array,
                fractional_mask_higher_sub=fractional_mask_higher_sub,
                fractional_mask_lower_sub=fractional_mask_lower_sub,
                array_higher_sub_2d=array_higher_sub,
            )

            if fractional_mask_higher_sub.is_all_true:
                return self.mask.mapping.array_stored_1d_from_array_2d(
                    array_2d=iterated_array
                )

            array_lower_sub_2d = array_higher_sub
            fractional_mask_lower_sub = fractional_mask_higher_sub

        return self.mask.mapping.array_stored_1d_from_array_2d(
            array_2d=iterated_array + array_higher_sub.in_2d_binned
        )

    @staticmethod
    @decorator_util.jit()
    def iterated_array_jit_from(
        iterated_array,
        fractional_mask_higher_sub,
        fractional_mask_lower_sub,
        array_higher_sub_2d,
    ):
        """Create the iterated array from a result array that is computed at a higher sub size leel than the previous grid.

        The iterated array is only updated for pixels where the fractional accuracy is met."""

        for y in range(iterated_array.shape[0]):
            for x in range(iterated_array.shape[1]):
                if (
                    fractional_mask_higher_sub[y, x]
                    and not fractional_mask_lower_sub[y, x]
                ):
                    iterated_array[y, x] = array_higher_sub_2d[y, x]

        return iterated_array

    def fractional_mask_from_grids(
        self, grid_lower_sub_2d, grid_higher_sub_2d
    ) -> msk.Mask:
        """ Compute a fractional mask from a result array, where the fractional mask describes whether the evaluated
        value in the result array is within the *GridIterator*'s specified fractional accuracy. The fractional mask thus
        determines whether a pixel on the grid needs to be reevaluated at a higher level of sub-gridding to meet the
        specified fractional accuracy. If it must be re-evaluated, the fractional masks's entry is *False*.

        The fractional mask is computed by comparing the results evaluated at one level of sub-gridding to another
        at a higher level of sub-griding. Thus, the sub-grid size in chosen on a per-pixel basis until the function
        is evaluated at the specified fractional accuracy.

        Parameters
        ----------
        result_array_lower_sub : arrays.Array
            The results computed by a function using a lower sub-grid size
        result_array_lower_sub : grids.Array
            The results computed by a function using a lower sub-grid size.
        """

        fractional_mask = msk.Mask.unmasked(
            shape_2d=grid_lower_sub_2d.shape_2d, invert=True
        )

        fractional_mask = self.fractional_mask_jit_from_grid(
            fractional_accuracy_threshold=self.fractional_accuracy,
            fractional_mask=fractional_mask,
            grid_higher_sub_2d=grid_higher_sub_2d,
            grid_lower_sub_2d=grid_lower_sub_2d,
            grid_higher_mask_2d=grid_higher_sub_2d.mask,
        )

        return msk.Mask(
            mask_2d=fractional_mask,
            pixel_scales=grid_higher_sub_2d.pixel_scales,
            origin=grid_higher_sub_2d.origin,
        )

    @staticmethod
    @decorator_util.jit()
    def fractional_mask_jit_from_grid(
        fractional_accuracy_threshold,
        fractional_mask,
        grid_higher_sub_2d,
        grid_lower_sub_2d,
        grid_higher_mask_2d,
    ):
        """Jitted functioon to determine the fractional mask, which is a mask where:

        - *True* entries signify the function has been evaluated in that pixel to desired fractional accuracy and
           therefore does not need to be iteratively computed at higher levels of sub-gridding.

        - *False* entries signify the function has not been evaluated in that pixel to desired fractional accuracy and
           therefore must be iterative computed at higher levels of sub-gridding to meet this accuracy."""

        for y in range(fractional_mask.shape[0]):
            for x in range(fractional_mask.shape[1]):
                if not grid_higher_mask_2d[y, x]:

                    fractional_accuracy_y = (
                        grid_lower_sub_2d[y, x, 0] / grid_higher_sub_2d[y, x, 0]
                    )

                    fractional_accuracy_x = (
                        grid_lower_sub_2d[y, x, 1] / grid_higher_sub_2d[y, x, 1]
                    )

                    if fractional_accuracy_y > 1.0:
                        fractional_accuracy_y = 1.0 / fractional_accuracy_y

                    if fractional_accuracy_x > 1.0:
                        fractional_accuracy_x = 1.0 / fractional_accuracy_x

                    fractional_accuracy = min(
                        fractional_accuracy_y, fractional_accuracy_x
                    )

                    if fractional_accuracy < fractional_accuracy_threshold:
                        fractional_mask[y, x] = False

        return fractional_mask

    def iterated_grid_from_func(self, func, profile, grid_lower_sub_2d):
        """Iterate over a function that returns a grid of values until the it meets a specified fractional accuracy.
        The function returns a result on a pixel-grid where evaluating it on more points on a higher resolution
        sub-grid followed by binning lead to a more precise evaluation of the function. For the fractional accuracy of
        the grid to be met, both the y and x values must meet it.

        The function is first called for a sub-grid size of 1 and a higher resolution grid. The ratio of values give
        the fractional accuracy of each function evaluation. Pixels which do not meet the fractional accuracy are
        iteratively revaulated on higher resolution sub-grids. This is repeated until all pixels meet the fractional
        accuracy or the highest sub-size specified in the *sub_steps* attribute is computed.

        An example use case of this function is when a "deflections_from_grid" methods in **PyAutoLens**'s *MassProfile*
        module is computed, which by evaluating the function on a higher resolution sub-grid samples the analytic
        mass profile at more points and thus more precisely.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation."""

        iterated_grid = np.zeros(shape=(self.shape_2d[0], self.shape_2d[1], 2))

        fractional_mask_lower_sub = self.mask

        for sub_size in self.sub_steps:

            mask_higher_sub = fractional_mask_lower_sub.mapping.mask_new_sub_size_from_mask(
                mask=fractional_mask_lower_sub, sub_size=sub_size
            )

            grid_compute = Grid.from_mask(mask=mask_higher_sub)
            grid_higher_sub = func(profile, grid_compute)
            grid_higher_sub = grid_compute.structure_from_result(
                result=grid_higher_sub
            ).in_2d_binned

            fractional_mask_higher_sub = self.fractional_mask_from_grids(
                grid_lower_sub_2d=grid_lower_sub_2d, grid_higher_sub_2d=grid_higher_sub
            )

            iterated_grid = self.iterated_grid_jit_from(
                iterated_grid=iterated_grid,
                fractional_mask_higher_sub=fractional_mask_higher_sub,
                fractional_mask_lower_sub=fractional_mask_lower_sub,
                grid_higher_sub_2d=grid_higher_sub,
            )

            if fractional_mask_higher_sub.is_all_true:
                return self.mask.mapping.grid_stored_1d_from_grid_2d(
                    grid_2d=iterated_grid
                )

            grid_lower_sub_2d = grid_higher_sub
            fractional_mask_lower_sub = fractional_mask_higher_sub

        return self.mask.mapping.grid_stored_1d_from_grid_2d(
            grid_2d=iterated_grid + grid_higher_sub.in_2d_binned
        )

    @staticmethod
    @decorator_util.jit()
    def iterated_grid_jit_from(
        iterated_grid,
        fractional_mask_higher_sub,
        fractional_mask_lower_sub,
        grid_higher_sub_2d,
    ):
        """Create the iterated grid from a result grid that is computed at a higher sub size level than the previous
        grid.

        The iterated grid is only updated for pixels where the fractional accuracy is met in both the (y,x) coodinates."""

        for y in range(iterated_grid.shape[0]):
            for x in range(iterated_grid.shape[1]):
                if (
                    fractional_mask_higher_sub[y, x]
                    and not fractional_mask_lower_sub[y, x]
                ):
                    iterated_grid[y, x, :] = grid_higher_sub_2d[y, x, :]

        return iterated_grid

    def iterated_result_from_func(self, func, profile):
        """Iterate over a function that returns an array or grid of values until the it meets a specified fractional
        accuracy. The function returns a result on a pixel-grid where evaluating it on more points on a higher
        resolution sub-grid followed by binning lead to a more precise evaluation of the function.

        A full description of the iteration method can be found in the functions *iterated_array_from_func* and
        *iterated_grid_from_func*. This function computes the result on a grid with a sub-size of 1, and uses its
        shape to call the correct function.

        Parameters
        ----------
        func : func
            The function which is iterated over to compute a more precise evaluation."""
        result_sub_1_1d = func(profile, self.grid)
        result_sub_1_2d = self.structure_from_result(
            result=result_sub_1_1d
        ).in_2d_binned

        if len(result_sub_1_2d.shape) == 2:
            return self.iterated_array_from_func(
                func=func, profile=profile, array_lower_sub_2d=result_sub_1_2d
            )
        elif len(result_sub_1_2d.shape) == 3:
            return self.iterated_grid_from_func(
                func=func, profile=profile, grid_lower_sub_2d=result_sub_1_2d
            )


class GridInterpolator:
    def __init__(self, grid, interp_grid, pixel_scale_interpolation_grid):
        self.grid = grid
        self.interp_grid = interp_grid
        self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid
        self.vtx, self.wts = self.interp_weights

    @property
    def interp_weights(self):
        tri = qhull.Delaunay(self.interp_grid)
        simplex = tri.find_simplex(self.grid)
        # noinspection PyUnresolvedReferences
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = self.grid - temp[:, 2]
        bary = np.einsum("njk,nk->nj", temp[:, :2, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    @classmethod
    def from_mask_grid_and_pixel_scale_interpolation_grids(
        cls, mask, grid, pixel_scale_interpolation_grid
    ):

        rescale_factor = mask.pixel_scale / pixel_scale_interpolation_grid

        mask = mask.mapping.mask_sub_1

        rescaled_mask = mask.mapping.rescaled_mask_from_rescale_factor(
            rescale_factor=rescale_factor
        )

        interp_mask = rescaled_mask.mapping.edge_buffed_mask

        interp_grid = grid_util.grid_1d_via_mask_2d_from(
            mask_2d=interp_mask,
            pixel_scales=(
                pixel_scale_interpolation_grid,
                pixel_scale_interpolation_grid,
            ),
            sub_size=1,
            origin=mask.origin,
        )

        return GridInterpolator(
            grid=grid,
            interp_grid=interp_mask.mapping.grid_stored_1d_from_grid_1d(
                grid_1d=interp_grid
            ),
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        )

    def interpolated_values_from_values(self, values) -> np.ndarray:
        """This function uses the precomputed vertexes and weights of a Delaunay gridding to interpolate a set of
        values computed on the interpolation grid to the GridInterpolator's full grid.

        This function is taken from the SciPy interpolation method griddata
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html). It is adapted here
        to reuse pre-computed interpolation vertexes and weights for efficiency. """
        return np.einsum("nj,nj->n", np.take(values, self.vtx), self.wts)


class GridCoordinates(np.ndarray):
    def __new__(cls, coordinates):
        """ A collection of (y,x) coordinates structured in a way defining groups of coordinates which share a common
        origin (for example coordinates may be grouped if they are from a specific region of a dataset).

        Grouping is structured as follows:

        [[x0, x1], [x0, x1, x2]]

        Here, we have two groups of coordinates, where each group is associated.

        The coordinate object does not store the coordinates as a list of list of tuples, but instead a 2D NumPy array
        of shape [total_coordinates, 2]. Index information is stored so that this array can be mapped to the list of
        list of tuple structure above. They are stored as a NumPy array so the coordinates can be used efficiently for
        calculations.

        The coordinates input to this function can have any of the following forms:

        [[(y0,x0), (y1,x1)], [(y0,x0)]]
        [[[y0,x0], [y1,x1]], [[y0,x0)]]
        [(y0,x0), (y1,x1)]
        [[y0,x0], [y1,x1]]

        In all cases, they will be converted to a list of list of tuples followed by a 2D NumPy array.

        Print methods are overidden so a user always "sees" the coordinates as the list structure.

        In contrast to a *Grid* structure, *GridCoordinates* do not lie on a uniform grid or correspond to values that
        originate from a uniform grid. Therefore, when handling irregular data-sets *GridCoordinates* should be used.

        Parameters
        ----------
        coordinates : [[tuple]] or equivalent
            A collection of (y,x) coordinates that are grouped if they correpsond to a shared origin.
        """

        if len(coordinates) == 0:
            return []

        if isinstance(coordinates[0], tuple):
            coordinates = [coordinates]
        elif isinstance(coordinates[0], np.ndarray):
            if len(coordinates[0].shape) == 1:
                coordinates = [coordinates]
        elif isinstance(coordinates[0], list) and isinstance(
            coordinates[0][0], (float)
        ):
            coordinates = [coordinates]

        upper_indexes = []

        a = 0

        for coords in coordinates:
            a += len(coords)
            upper_indexes.append(a)

        coordinates_arr = np.concatenate([np.array(i) for i in coordinates])

        obj = coordinates_arr.view(cls)
        obj.upper_indexes = upper_indexes
        obj.lower_indexes = [0] + upper_indexes[:-1]

        return obj

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

    def __array_finalize__(self, obj):

        if hasattr(obj, "lower_indexes"):
            self.lower_indexes = obj.lower_indexes

        if hasattr(obj, "upper_indexes"):
            self.upper_indexes = obj.upper_indexes

    @classmethod
    def from_yx_1d(cls, y, x):
        """Create *GridCoordinates* from a list of y and x values.

        This function omits coordinate grouping."""
        return GridCoordinates(coordinates=np.stack((y, x), axis=-1))

    @classmethod
    def from_pixels_and_mask(cls, pixels, mask):
        """Create *GridCoordinates* from a list of coordinates in pixel units and a mask which allows these coordinates to
        be converted to scaled units."""
        coordinates = []
        for coordinate_set in pixels:
            coordinates.append(
                [
                    mask.geometry.scaled_coordinates_from_pixel_coordinates(
                        pixel_coordinates=coordinates
                    )
                    for coordinates in coordinate_set
                ]
            )
        return cls(coordinates=coordinates)

    @property
    def in_1d(self):
        return self

    @property
    def in_list(self):
        """Return the coordinates on a structured list which groups coordinates with a common origin."""
        return [
            list(map(tuple, self[i:j, :]))
            for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]

    def values_from_arr_1d(self, arr_1d):
        """Create a *Values* object from a 1D NumPy array of values of shape [total_coordinates]. The
        *Values* are structured and grouped following this *Coordinate* instance."""
        values_1d = [
            list(arr_1d[i:j]) for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]
        return arrays.Values(values=values_1d)

    def coordinates_from_grid_1d(self, grid_1d):
        """Create a *GridCoordinates* object from a 2D NumPy array of values of shape [total_coordinates, 2]. The
        *GridCoordinates* are structured and grouped following this *Coordinate* instance."""
        coordinates_1d = [
            list(map(tuple, grid_1d[i:j, :]))
            for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]

        return GridCoordinates(coordinates=coordinates_1d)

    @classmethod
    def from_file(cls, file_path):
        """Create a *GridCoordinates* object from a file which stores the coordinates as a list of list of tuples.

        Parameters
        ----------
        file_path : str
            The path to the coordinates .dat file containing the coordinates (e.g. '/path/to/coordinates.dat')
        """
        with open(file_path) as f:
            coordinate_string = f.readlines()

        coordinates = []

        for line in coordinate_string:
            coordinate_list = ast.literal_eval(line)
            coordinates.append(coordinate_list)

        return GridCoordinates(coordinates=coordinates)

    def output_to_file(self, file_path, overwrite=False):
        """Output this instance of the *GridCoordinates* object to a list of list of tuples.

        Parameters
        ----------
        file_path : str
            The path to the coordinates .dat file containing the coordinates (e.g. '/path/to/coordinates.dat')
        overwrite : bool
            If there is as exsiting file it will be overwritten if this is *True*.
        """

        if overwrite and os.path.exists(file_path):
            os.remove(file_path)
        elif not overwrite and os.path.exists(file_path):
            raise FileExistsError(
                "The file ",
                file_path,
                " already exists. Set overwrite=True to overwrite this" "file",
            )

        with open(file_path, "w") as f:
            for coordinate in self.in_list:
                f.write(f"{coordinate}\n")

    def squared_distances_from_coordinate(self, coordinate=(0.0, 0.0)):
        """Compute the squared distance of every (y,x) coordinate in this *Coordinate* instance from an input
        coordinate.

        Parameters
        ----------
        coordinate : (float, float)
            The (y,x) coordinate from which the squared distance of every *Coordinate* is computed.
        """
        squared_distances = np.square(self[:, 0] - coordinate[0]) + np.square(
            self[:, 1] - coordinate[1]
        )
        return self.values_from_arr_1d(arr_1d=squared_distances)

    def distances_from_coordinate(self, coordinate=(0.0, 0.0)):
        """Compute the distance of every (y,x) coordinate in this *Coordinate* instance from an input coordinate.

        Parameters
        ----------
        coordinate : (float, float)
            The (y,x) coordinate from which the distance of every *Coordinate* is computed.
        """
        distances = np.sqrt(
            self.squared_distances_from_coordinate(coordinate=coordinate)
        )
        return self.values_from_arr_1d(arr_1d=distances)

    @property
    def shape_2d_scaled(self):
        """The two dimensional shape of the coordinates spain in scaled units, computed by taking the minimum and
        maximum values of the coordinates."""
        return (
            np.amax(self[:, 0]) - np.amin(self[:, 0]),
            np.amax(self[:, 1]) - np.amin(self[:, 1]),
        )

    @property
    def scaled_maxima(self):
        """The maximum values of the coordinates returned as a tuple (y_max, x_max)."""
        return (np.amax(self[:, 0]), np.amax(self[:, 1]))

    @property
    def scaled_minima(self):
        """The minimum values of the coordinates returned as a tuple (y_max, x_max)."""
        return (np.amin(self[:, 0]), np.amin(self[:, 1]))

    @property
    def extent(self):
        """The extent of the coordinates returned as a NumPy array [x_min, x_max, y_min, y_max].

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

    def structure_from_result(
        self, result: np.ndarray or list
    ) -> typing.Union[arrays.Values, list]:
        """Convert a result from a non autoarray structure to an aa.Values or aa.GridCoordinates structure, where
        the conversion depends on type(result) as follows:

        - 1D np.ndarray   -> aa.Values
        - 2D np.ndarray   -> aa.GridCoordinates
        - [1D np.ndarray] -> [aa.Values]
        - [2D np.ndarray] -> [aa.GridCoordinates]

        This function is used by the grid_like_to_structure decorator to convert the output result of a function
        to an autoarray structure when a *GridCoordinates* instance is passed to the decorated function.

        Parameters
        ----------
        result : np.ndarray or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """

        if isinstance(result, np.ndarray):
            if len(result.shape) == 1:
                return self.values_from_arr_1d(arr_1d=result)
            elif len(result.shape) == 2:
                return self.coordinates_from_grid_1d(grid_1d=result)
        elif isinstance(result, list):
            if len(result[0].shape) == 1:
                return [self.values_from_arr_1d(arr_1d=value) for value in result]
            elif len(result[0].shape) == 2:
                return [
                    self.coordinates_from_grid_1d(grid_1d=value) for value in result
                ]

    def structure_list_from_result_list(self, result_list: list) -> typing.Union[list]:
        """Convert a result from a list of non autoarray structures to a list of aa.Values or aa.GridCoordinates
        structures, where the conversion depends on type(result) as follows:

        - [1D np.ndarray] -> [aa.Values]
        - [2D np.ndarray] -> [aa.GridCoordinates]

        This function is used by the grid_like_list_to_structure_list decorator to convert the output result of a
        function to a list of autoarray structure when a *GridCoordinates* instance is passed to the decorated function.

        Parameters
        ----------
        result_list : np.ndarray or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        if len(result_list[0].shape) == 1:
            return [self.values_from_arr_1d(arr_1d=value) for value in result_list]
        elif len(result_list[0].shape) == 2:
            return [
                self.coordinates_from_grid_1d(grid_1d=value) for value in result_list
            ]


class GridRectangular(Grid):
    def __new__(cls, grid, shape_2d, pixel_scales, origin=(0.0, 0.0), *args, **kwargs):
        """A pixelization-grid of (y,x) coordinates which are used to form the pixel centres of adaptive pixelizations in the \
        *pixelizations* module.

        A *PixGrid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_pix_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixelization pixel index and second element the y or x arc -econd coordinates. For example:

        - pix_grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - pix_grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Parameters
        -----------
        pix_grid : ndarray
            The grid of (y,x) scaled coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        nearest_pixelization_1d_index_for_mask_1d_index : ndarray
            A 1D array that maps every grid pixel to its nearest pixelization-grid pixel.
        """

        mask = msk.Mask.unmasked(
            shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=1, origin=origin
        )

        obj = super().__new__(cls=cls, grid=grid, mask=mask)

        pixel_neighbors, pixel_neighbors_size = pixelization_util.rectangular_neighbors_from(
            shape_2d=shape_2d
        )
        obj.pixel_neighbors = pixel_neighbors.astype("int")
        obj.pixel_neighbors_size = pixel_neighbors_size.astype("int")
        return obj

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

    @classmethod
    def overlay_grid(cls, shape_2d, grid, buffer=1e-8):
        """The geometry of a rectangular grid.

        This is used to map grid of (y,x) scaled coordinates to the pixels on the rectangular grid.

        Parameters
        -----------
        shape_2d : (int, int)
            The dimensions of the rectangular grid of pixels (y_pixels, x_pixel)
        pixel_scales : (float, float)
            The pixel conversion scale of a pixel in the y and x directions.
        origin : (float, float)
            The scaled origin of the rectangular pixelization's coordinate system.
        pixel_neighbors : ndarray
            An array of length (y_pixels*x_pixels) which provides the index of all neighbors of every pixel in \
            the rectangular grid (entries of -1 correspond to no neighbor).
        pixel_neighbors_size : ndarrayy
            An array of length (y_pixels*x_pixels) which gives the number of neighbors of every pixel in the \
            rectangular grid.
        """

        y_min = np.min(grid[:, 0]) - buffer
        y_max = np.max(grid[:, 0]) + buffer
        x_min = np.min(grid[:, 1]) - buffer
        x_max = np.max(grid[:, 1]) + buffer

        pixel_scales = (
            float((y_max - y_min) / shape_2d[0]),
            float((x_max - x_min) / shape_2d[1]),
        )

        origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)

        grid_1d = grid_util.grid_1d_via_shape_2d_from(
            shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=1, origin=origin
        )

        return GridRectangular(
            grid=grid_1d, shape_2d=shape_2d, pixel_scales=pixel_scales, origin=origin
        )

    @property
    def pixels(self):
        return self.shape_2d[0] * self.shape_2d[1]

    @property
    def shape_2d_scaled(self):
        return (
            (self.shape_2d[0] * self.pixel_scales[0]),
            (self.shape_2d[1] * self.pixel_scales[1]),
        )


class GridVoronoi(np.ndarray):
    """Determine the geometry of the Voronoi pixelization, by alligning it with the outer-most coordinates on a \
    grid plus a small buffer.

    Parameters
    -----------
    grid : ndarray
        The (y,x) grid of coordinates which determine the Voronoi pixelization's
    pixelization_grid : ndarray
        The (y,x) centre of every Voronoi pixel in scaleds.
    origin : (float, float)
        The scaled origin of the Voronoi pixelization's coordinate system.
    pixel_neighbors : ndarray
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in \
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size : ndarrayy
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the \
        Voronoi grid.
    """

    def __new__(
        cls, grid, nearest_pixelization_1d_index_for_mask_1d_index=None, *args, **kwargs
    ):
        """A pixelization-grid of (y,x) coordinates which are used to form the pixel centres of adaptive pixelizations in the \
        *pixelizations* module.

        A *PixGrid* is ordered such pixels begin from the top-row of the mask and go rightwards and then \
        downwards. Therefore, it is a ndarray of shape [total_pix_pixels, 2]. The first element of the ndarray \
        thus corresponds to the pixelization pixel index and second element the y or x arc -econd coordinates. For example:

        - pix_grid[3,0] = the 4th unmasked pixel's y-coordinate.
        - pix_grid[6,1] = the 7th unmasked pixel's x-coordinate.

        Parameters
        -----------
        pix_grid : ndarray
            The grid of (y,x) scaled coordinates of every image-plane pixelization grid used for adaptive source \
            -plane pixelizations.
        nearest_pixelization_1d_index_for_mask_1d_index : ndarray
            A 1D array that maps every grid pixel to its nearest pixelization-grid pixel.
        """

        if type(grid) is list:
            grid = np.asarray(grid)

        obj = grid.view(cls)
        obj.nearest_pixelization_1d_index_for_mask_1d_index = (
            nearest_pixelization_1d_index_for_mask_1d_index
        )
        obj.interpolator = None

        obj.pixels = grid.shape[0]

        try:
            obj.voronoi = scipy.spatial.Voronoi(
                np.asarray([grid[:, 1], grid[:, 0]]).T, qhull_options="Qbb Qc Qx Qm"
            )
        except ValueError or OverflowError or scipy.spatial.qhull.QhullError:
            raise exc.PixelizationException()

        pixel_neighbors, pixel_neighbors_size = pixelization_util.voronoi_neighbors_from(
            pixels=obj.pixels, ridge_points=np.asarray(obj.voronoi.ridge_points)
        )

        obj.pixel_neighbors = pixel_neighbors.astype("int")
        obj.pixel_neighbors_size = pixel_neighbors_size.astype("int")
        obj.nearest_pixelization_1d_index_for_mask_1d_index = (
            nearest_pixelization_1d_index_for_mask_1d_index
        )

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "pixel_neighbors"):
            self.pixel_neighbors = obj.pixel_neighbors

        if hasattr(obj, "pixel_neighbors_size"):
            self.pixel_neighbors_size = obj.pixel_neighbors_size

        if hasattr(obj, "nearest_pixelization_1d_index_for_mask_1d_index"):
            self.nearest_pixelization_1d_index_for_mask_1d_index = (
                obj.nearest_pixelization_1d_index_for_mask_1d_index
            )

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

    @classmethod
    def manual_1d(cls, grid):
        return GridVoronoi(grid=grid)

    @classmethod
    def from_grid_and_unmasked_2d_grid_shape(cls, unmasked_sparse_shape, grid):

        sparse_grid = GridSparse.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=unmasked_sparse_shape, grid=grid
        )

        return GridVoronoi(
            grid=sparse_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_grid.sparse_1d_index_for_mask_1d_index,
        )

    @property
    def shape_2d_scaled(self):
        return (
            np.amax(self[:, 0]) - np.amin(self[:, 0]),
            np.amax(self[:, 1]) - np.amin(self[:, 1]),
        )

    @property
    def scaled_maxima(self):
        return (np.amax(self[:, 0]), np.amax(self[:, 1]))

    @property
    def scaled_minima(self):
        return (np.amin(self[:, 0]), np.amin(self[:, 1]))

    @property
    def extent(self):
        return np.asarray(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )


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


def grid_like_to_structure(func):
    """ Checks whether any coordinates in the grid are radially near (0.0, 0.0), which can lead to numerical faults in \
    the evaluation of a light or mass profiles. If any coordinates are radially within the the radial minimum \
    threshold, their (y,x) coordinates are shifted to that value to ensure they are evaluated correctly.

    By default this radial minimum is not used, and users should be certain they use a value that does not impact \
    results.

    Parameters
    ----------
    func : (profile, *args, **kwargs) -> Object
        A function that takes a grid of coordinates which may have a singularity as (0.0, 0.0)

    Returns
    -------
        A function that can except cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(profile, grid, *args, **kwargs):
        """ This decorator homogenizes the input of a "grid_like" structure (*Grid*, *GridIterator*, *GridInterpolator*
        or  *GridCoordinate*) into a function. It allows these classes to be interchangeably input into a function,
        such that the grid is used to evalaute the function as every (y,x) coordinates of the grid.

        The grid_like objects *Grid* and *GridCoordinates* are input into the function as a flattened 2D NumPy array
        of shape [total_coordinates, 2] where second dimension stores the (y,x) values. If a *GridIterator* is input,
        the function is evaluated using the appropriate iterated_*_from_func* function.

        The outputs of the function are converted from a 1D or 2D NumPy Array to an *Array*, *Grid*, *Values* or
        *GridCoordinate* objects, whichever is applicable as follows:

        - If the function returns (y,x) coordinates at every input point, the returned results are returned as a
         *Grid* or *GridCoordinates* structure - the same structure as the input.

        - If the function returns scalar values at every input point and a *Grid* is input, the returned results are
          an *Array* structure which uses the same dimensions and mask as the *Grid*.

        - If the function returns scalar values at every input point and *GridCoordinates* are input, the returned
          results are a *Values* object with structure resembling that of the *GridCoordinates*..

        If the input array is not a *Grid* structure (e.g. it is a 2D NumPy array) the output is a NumPy array.

        Parameters
        ----------
        profile : Profile
            A Profile object which uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid or GridCoordinates
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        if isinstance(grid, GridIterator):
            return grid.iterated_result_from_func(func=func, profile=profile)
        elif isinstance(grid, GridCoordinates):
            result = func(profile, grid, *args, **kwargs)
            return grid.structure_from_result(result=result)
        elif isinstance(grid, Grid):
            result = func(profile, grid, *args, **kwargs)
            return grid.structure_from_result(result=result)

        if not isinstance(grid, GridCoordinates) and not isinstance(grid, Grid):
            return func(profile, grid, *args, **kwargs)

    return wrapper


def grid_like_to_structure_list(func):
    """ Checks whether any coordinates in the grid are radially near (0.0, 0.0), which can lead to numerical faults in \
    the evaluation of a light or mass profiles. If any coordinates are radially within the the radial minimum \
    threshold, their (y,x) coordinates are shifted to that value to ensure they are evaluated correctly.

    By default this radial minimum is not used, and users should be certain they use a value that does not impact \
    results.

    Parameters
    ----------
    func : (profile, *args, **kwargs) -> Object
        A function that takes a grid of coordinates which may have a singularity as (0.0, 0.0)

    Returns
    -------
        A function that can except cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(profile, grid, *args, **kwargs):
        """ This decorator homogenizes the input of a "grid_like" structure (*Grid*, *GridIterator*, *GridInterpolator*
        or  *GridCoordinate*) into a function. It allows these classes to be interchangeably input into a function,
        such that the grid is used to evalaute the function as every (y,x) coordinates of the grid.

        The grid_like objects *Grid* and *GridCoordinates* are input into the function as a flattened 2D NumPy array
        of shape [total_coordinates, 2] where second dimension stores the (y,x) values. If a *GridIterator* is input,
        the function is evaluated using the appropriate iterated_*_from_func* function.

        If a *GridIterator* is not input the outputs of the function are converted from a list of 1D or 2D NumPy Arrays
        to a list of *Array*, *Grid*,  *Values* or  *GridCoordinate* objects, whichever is applicable as follows:

        - If the function returns (y,x) coordinates at every input point, the returned results are returned as a
         *Grid* or *GridCoordinates* structure - the same structure as the input.

        - If the function returns scalar values at every input point and a *Grid* is input, the returned results are
          an *Array* structure which uses the same dimensions and mask as the *Grid*.

        - If the function returns scalar values at every input point and *GridCoordinates* are input, the returned
          results are a *Values* object with structure resembling that of the *GridCoordinates*.

        if a *GridIterator* is input, the iterated grid calculation is not applicable. Thus, the highest resolution
        sub_size grid in the *GridIterator* is used instead.

        If the input array is not a *Grid* structure (e.g. it is a 2D NumPy array) the output is a NumPy array.

        Parameters
        ----------
        profile : Profile
            A Profile object which uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid or GridCoordinates
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        if isinstance(grid, GridIterator):
            mask = grid.mask.mapping.mask_new_sub_size_from_mask(
                mask=grid.mask, sub_size=max(grid.sub_steps)
            )
            grid = Grid.from_mask(mask=mask)
            result_list = func(profile, grid, *args, **kwargs)
            result_list = [
                grid.structure_from_result(result=result) for result in result_list
            ]
            result_list = [result.in_1d_binned for result in result_list]
            return grid.structure_list_from_result_list(result_list=result_list)
        elif isinstance(grid, GridCoordinates):
            result_list = func(profile, grid, *args, **kwargs)
            return grid.structure_list_from_result_list(result_list=result_list)
        elif isinstance(grid, Grid):
            result_list = func(profile, grid, *args, **kwargs)
            return grid.structure_list_from_result_list(result_list=result_list)

        if not isinstance(grid, GridCoordinates) and not isinstance(grid, Grid):
            return func(profile, grid, *args, **kwargs)

    return wrapper


def interpolate(func):
    """
    Decorate a profile method that accepts a coordinate grid and returns a data_type grid.

    If an interpolator attribute is associated with the input grid then that interpolator is used to down sample the
    coordinate grid prior to calling the function and up sample the result of the function.

    If no interpolator attribute is associated with the input grid then the function is called as hyper.

    Parameters
    ----------
    func
        Some method that accepts a grid

    Returns
    -------
    decorated_function
        The function with optional interpolation
    """

    @wraps(func)
    def wrapper(profile, grid, grid_radial_minimum=None, *args, **kwargs):
        if hasattr(grid, "interpolator"):
            interpolator = grid.interpolator
            if grid.interpolator is not None:
                values = func(
                    profile,
                    interpolator.interp_grid,
                    grid_radial_minimum,
                    *args,
                    **kwargs,
                )
                if values.ndim == 1:
                    return interpolator.interpolated_values_from_values(values=values)
                elif values.ndim == 2:
                    y_values = interpolator.interpolated_values_from_values(
                        values=values[:, 0]
                    )
                    x_values = interpolator.interpolated_values_from_values(
                        values=values[:, 1]
                    )
                    return np.asarray([y_values, x_values]).T
        return func(profile, grid, grid_radial_minimum, *args, **kwargs)

    return wrapper


def transform(func):
    """Wrap the function in a function that checks whether the coordinates have been transformed. If they have not \
    been transformed then they are transformed.

    Parameters
    ----------
    func : (profile, grid *args, **kwargs) -> Object
        A function where the input grid is the grid whose coordinates are transformed.

    Returns
    -------
        A function that can except cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(profile, grid, *args, **kwargs):
        """

        Parameters
        ----------
        profile : GeometryProfile
            The profiles that owns the function.
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
            A grid_like object whose coordinates may be transformed.
        """

        if not isinstance(grid, (GridTransformed, GridTransformedNumpy)):
            result = func(
                profile,
                profile.transform_grid_to_reference_frame(grid),
                *args,
                **kwargs,
            )

            return result

        else:
            return func(profile, grid, *args, **kwargs)

    return wrapper


def cache(func):
    """
    Caches results of a call to a grid function. If a grid that evaluates to the same byte value is passed into the same
    function of the same instance as previously then the cached result is returned.

    Parameters
    ----------
    func
        Some instance method that takes a grid as its argument

    Returns
    -------
    result
        Some result, either newly calculated or recovered from the cache
    """

    def wrapper(instance, grid: np.ndarray, *args, **kwargs):
        if not hasattr(instance, "cache"):
            instance.cache = {}
        key = (func.__name__, grid.tobytes())
        if key not in instance.cache:
            instance.cache[key] = func(instance, grid)
        return instance.cache[key]

    return wrapper


def relocate_to_radial_minimum(func):
    """ Checks whether any coordinates in the grid are radially near (0.0, 0.0), which can lead to numerical faults in \
    the evaluation of a light or mass profiles. If any coordinates are radially within the the radial minimum \
    threshold, their (y,x) coordinates are shifted to that value to ensure they are evaluated correctly.

    By default this radial minimum is not used, and users should be certain they use a value that does not impact \
    results.

    Parameters
    ----------
    func : (profile, *args, **kwargs) -> Object
        A function that takes a grid of coordinates which may have a singularity as (0.0, 0.0)

    Returns
    -------
        A function that can except cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(profile, grid, *args, **kwargs):
        """

        Parameters
        ----------
        profile : SphericalProfile
            The profiles that owns the function
        grid : grid_like
            The (y, x) coordinates which are to be radially moved from (0.0, 0.0).

        Returns
        -------
            The grid_like object whose coordinates are radially moved from (0.0, 0.0).
        """
        radial_minimum_config = aa.conf.NamedConfig(
            f"{aa.conf.instance.config_path}/radial_minimum.ini"
        )
        grid_radial_minimum = radial_minimum_config.get(
            "radial_minimum", profile.__class__.__name__, float
        )

        with np.errstate(all="ignore"):  # Division by zero fixed via isnan

            grid_radii = profile.grid_to_grid_radii(grid=grid)

            grid_radial_scale = np.where(
                grid_radii < grid_radial_minimum, grid_radial_minimum / grid_radii, 1.0
            )
            grid = np.multiply(grid, grid_radial_scale[:, None])
        grid[np.isnan(grid)] = grid_radial_minimum

        return func(profile, grid, *args, **kwargs)

    return wrapper


class MaskedGrid(AbstractGrid):
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
