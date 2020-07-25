import numpy as np

from autoarray import decorator_util
from autoarray import exc
from autoarray.structures import abstract_structure, arrays, grids
from autoarray.mask import mask as msk
from autoarray.util import array_util, grid_util


def check_grid(grid):

    if grid.shape[-1] != 2:
        raise exc.GridException(
            "The final dimension of the input grid is not equal to 2 (e.g. the (y,x) coordinates)"
        )

    if 2 < len(grid.shape) > 3:
        raise exc.GridException("The dimensions of the input grid array is not 2 or 3")

    if grid.store_in_1d and len(grid.shape) != 2:
        raise exc.GridException(
            "An grid input into the grids.Grid.__new__ method has store_in_1d = True but"
            "the input shape of the array is not 1."
        )


def check_grid_and_mask(grid, mask):

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


def convert_grid(grid):

    if type(grid) is list:
        grid = np.asarray(grid)

    return grid


def convert_manual_1d_grid(grid_1d, mask, store_in_1d):
    """
    Manual 1D Grid functions take as input a list or ndarray which is to be returned as an Grid. This function
    performs the following and checks and conversions on the input:

    1) If the input is a list, convert it to a 2D ndarray of shape [total_coordinates, 2].
    2) Check that the number of sub-pixels in the grid is identical to that of the mask.
    3) Return the grid in 1D if it is to be stored in 1D, else return it in 2D.

    For Grids, `1D' refers to a 2D NumPy array of shape [total_coordinates ,2] and '2D' a 3D NumPy array of shape
    [total_y_coordinates, total_x_coordinates, 2].

    Parameters
    ----------
    grid_1d : ndarray or list
        The input structure which is converted to a 2D ndarray if it is a list.
    mask : Mask
        The mask of the output Array.
    store_in_1d : bool
        Whether the memory-representation of the grid is in 1D or 2D.
    """

    grid_1d = convert_grid(grid=grid_1d)

    if store_in_1d:
        return grid_1d

    return grid_util.sub_grid_2d_from(
        sub_grid_1d=grid_1d, mask=mask, sub_size=mask.sub_size
    )


def convert_manual_2d_grid(grid_2d, mask, store_in_1d):
    """
    Manual 2D Grid functions take as input a list or ndarray which is to be returned as a Grid. This function
    performs the following and checks and conversions on the input:

    1) If the input is a list, convert it to a 3D ndarray of shape  [total_y_coordinates, total_x_coordinates, 2]
    2) Check that the number of sub-pixels in the array is identical to that of the mask.
    3) Return the array in 1D if it is to be stored in 1D, else return it in 2D.

    For Grids, `1D' refers to a 2D NumPy array of shape [total_coordinates ,2] and '2D' a 3D NumPy array of shape
    [total_y_coordinates, total_x_coordinates, 2]

    Parameters
    ----------
    grid_2d : ndarray or list
        The input structure which is converted to a 3D ndarray if it is a list.
    mask : Mask
        The mask of the output Grid.
    store_in_1d : bool
        Whether the memory-representation of the array is in 1D or 2D.
    """

    grid_1d = grid_util.sub_grid_1d_from(
        sub_grid_2d=grid_2d, mask=mask, sub_size=mask.sub_size
    )

    if store_in_1d:
        return grid_1d

    return grid_util.sub_grid_2d_from(
        sub_grid_1d=grid_1d, mask=mask, sub_size=mask.sub_size
    )


def convert_manual_grid(grid, mask, store_in_1d):
    """
    Manual Grid functions take as input a list or ndarray which is to be returned as an Grid. This function
    performs the following and checks and conversions on the input:

    1) If the input is a list, convert it to an ndarray.
    2) Check that the number of sub-pixels in the array is identical to that of the mask.
    3) Return the array in 1D if it is to be stored in 1D, else return it in 2D.

    Parameters
    ----------
    array : ndarray or list
        The input structure which is converted to an ndarray if it is a list.
    mask : Mask
        The mask of the output Array.
    store_in_1d : bool
        Whether the memory-representation of the array is in 1D or 2D.
    """

    grid = convert_grid(grid=grid)

    if len(grid.shape) == 2:
        return convert_manual_1d_grid(grid_1d=grid, mask=mask, store_in_1d=store_in_1d)
    return convert_manual_2d_grid(grid_2d=grid, mask=mask, store_in_1d=store_in_1d)


class AbstractGrid(abstract_structure.AbstractStructure):
    def __array_finalize__(self, obj):

        super(AbstractGrid, self).__array_finalize__(obj)

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
        raise NotImplementedError

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

    def squared_distances_from_coordinate(self, coordinate=(0.0, 0.0)) -> arrays.Array:
        """Compute the squared distance of every coordinate on the grid from an input coordinate.

        Parameters
        ----------
        coordinate : (float, float)
            The (y,x) coordinate from which the squared distance of every grid (y,x) coordinate is computed.
        """
        squared_distances = np.square(self[:, 0] - coordinate[0]) + np.square(
            self[:, 1] - coordinate[1]
        )
        return arrays.Array.manual_mask(array=squared_distances, mask=self.mask)

    def distances_from_coordinate(self, coordinate=(0.0, 0.0)) -> arrays.Array:
        """Compute the distance of every coordinate on the grid from an input (y,x) coordinate.

        Parameters
        ----------
        coordinate : (float, float)
            The (y,x) coordinate from which the distance of every grid (y,x) coordinate is computed.
        """
        distances = np.sqrt(
            self.squared_distances_from_coordinate(coordinate=coordinate)
        )
        return arrays.Array.manual_mask(array=distances, mask=self.mask)

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

        return grids.Grid.from_mask(mask=padded_mask)

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

        return grids.Grid(
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
