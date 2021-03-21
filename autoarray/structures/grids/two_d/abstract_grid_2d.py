import numpy as np

from autoarray import decorator_util
from autoarray import exc
from autoarray.geometry import geometry_util
from autoarray.structures import abstract_structure
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.grids import abstract_grid
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures.grids.two_d import grid_2d_irregular
from autoarray.mask import mask_2d as msk
from autoarray.structures.grids.two_d import grid_2d_util
from autoarray.structures.arrays.two_d import array_2d_util


def check_grid_2d(grid_2d):

    if grid_2d.shape[-1] != 2:
        raise exc.GridException(
            "The final dimension of the input grid is not equal to 2 (e.g. the (y,x) coordinates)"
        )

    if 2 < len(grid_2d.shape) > 3:
        raise exc.GridException("The dimensions of the input grid array is not 2 or 3")

    if grid_2d.store_slim and len(grid_2d.shape) != 2:
        raise exc.GridException(
            "An grid input into the grid_2d.Grid2D.__new__ method has store_slim = `True` but"
            "the input shape of the array is not 1."
        )


def check_grid_2d_and_mask_2d(grid_2d, mask_2d):

    if len(grid_2d.shape) == 2:

        if grid_2d.shape[0] != mask_2d.sub_pixels_in_mask:
            raise exc.GridException(
                "The input 1D grid does not have the same number of entries as sub-pixels in"
                "the mask."
            )

    elif len(grid_2d.shape) == 3:

        if (grid_2d.shape[0], grid_2d.shape[1]) != mask_2d.sub_shape_native:
            raise exc.GridException(
                "The input grid is 2D but not the same dimensions as the sub-mask "
                "(e.g. the mask 2D shape multipled by its sub size."
            )


def convert_manual_grid_2d_slim(grid_2d_slim, mask_2d, store_slim):
    """
    Manual 1D Grid2D functions take as input a list or ndarray which is to be returned as an Grid2D. This function
    performs the following and checks and conversions on the input:

    1) If the input is a list, convert it to a 2D ndarray of shape [total_coordinates, 2].
    2) Check that the number of sub-pixels in the grid is identical to that of the mask.
    3) Return the grid in 1D if it is to be stored in 1D, else return it in 2D.

    For Grids, `1D' refers to a 2D NumPy array of shape [total_coordinates ,2] and '2D' a 3D NumPy array of shape
    [total_y_coordinates, total_x_coordinates, 2].

    Parameters
    ----------
    grid_2d_slim : np.ndarray or list
        The input structure which is converted to a 2D ndarray if it is a list.
    mask_2d : Mask2D
        The mask of the output Array2D.
    store_slim : bool
        Whether the memory-representation of the grid is in 1D or 2D.
    """

    grid_2d_slim = abstract_grid.convert_grid(grid=grid_2d_slim)

    if store_slim:
        return grid_2d_slim

    return grid_2d_util.grid_2d_native_from(
        grid_2d_slim=grid_2d_slim, mask_2d=mask_2d, sub_size=mask_2d.sub_size
    )


def convert_manual_grid_2d_native(grid_2d_native, mask_2d, store_slim):
    """
    Manual 2D Grid2D functions take as input a list or ndarray which is to be returned as a Grid2D. This function
    performs the following and checks and conversions on the input:

    1) If the input is a list, convert it to a 3D ndarray of shape  [total_y_coordinates, total_x_coordinates, 2]
    2) Check that the number of sub-pixels in the array is identical to that of the mask.
    3) Return the array in 1D if it is to be stored in 1D, else return it in 2D.

    For Grids, `1D' refers to a 2D NumPy array of shape [total_coordinates ,2] and '2D' a 3D NumPy array of shape
    [total_y_coordinates, total_x_coordinates, 2]

    Parameters
    ----------
    grid_2d_native : np.ndarray or list
        The input structure which is converted to a 3D ndarray if it is a list.
    mask_2d : Mask2D
        The mask of the output Grid2D.
    store_slim : bool
        Whether the memory-representation of the array is in 1D or 2D.
    """

    grid_slim = grid_2d_util.grid_2d_slim_from(
        grid_2d_native=grid_2d_native, mask=mask_2d, sub_size=mask_2d.sub_size
    )

    if store_slim:
        return grid_slim

    return grid_2d_util.grid_2d_native_from(
        grid_2d_slim=grid_slim, mask_2d=mask_2d, sub_size=mask_2d.sub_size
    )


def convert_manual_grid_2d(grid_2d, mask_2d, store_slim):
    """
    Manual Grid2D functions take as input a list or ndarray which is to be returned as an Grid2D. This function
    performs the following and checks and conversions on the input:

    1) If the input is a list, convert it to an ndarray.
    2) Check that the number of sub-pixels in the array is identical to that of the mask.
    3) Return the array in 1D if it is to be stored in 1D, else return it in 2D.

    Parameters
    ----------
    array : np.ndarray or list
        The input structure which is converted to an ndarray if it is a list.
    mask_2d : Mask2D
        The mask of the output Array2D.
    store_slim : bool
        Whether the memory-representation of the array is in 1D or 2D.
    """

    grid_2d = abstract_grid.convert_grid(grid=grid_2d)

    if len(grid_2d.shape) == 2:
        return convert_manual_grid_2d_slim(
            grid_2d_slim=grid_2d, mask_2d=mask_2d, store_slim=store_slim
        )
    return convert_manual_grid_2d_native(
        grid_2d_native=grid_2d, mask_2d=mask_2d, store_slim=store_slim
    )


class AbstractGrid2D(abstract_structure.AbstractStructure2D):
    def __array_finalize__(self, obj):

        super().__array_finalize__(obj)

        if hasattr(obj, "_sub_border_flat_indexes"):
            self.mask._sub_border_flat_indexes = obj._sub_border_flat_indexes

    @property
    def slim(self):
        """
        Convenience method to access the grid's 1D representation, which is a Grid2D stored as an ndarray of shape
        [total_unmasked_pixels*(sub_size**2), 2].

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.
        """
        if self.store_slim:
            return self

        sub_grid_2d_slim = grid_2d_util.grid_2d_slim_from(
            grid_2d_native=self, mask=self.mask, sub_size=self.mask.sub_size
        )

        return self._new_structure(
            grid=sub_grid_2d_slim, mask=self.mask, store_slim=True
        )

    @property
    def native(self):
        """
        Convenience method to access the grid's 2D representation, which is a Grid2D stored as an ndarray of shape
        [sub_size*total_y_pixels, sub_size*total_x_pixels, 2] where all masked values are given values (0.0, 0.0).

        If the grid is stored in 2D it is return as is. If it is stored in 1D, it must first be mapped from 1D to 2D.
        """

        if self.store_slim:

            grid_2d = grid_2d_util.grid_2d_native_from(
                grid_2d_slim=self, mask_2d=self.mask, sub_size=self.mask.sub_size
            )

            return self._new_structure(grid=grid_2d, mask=self.mask, store_slim=False)

        return self

    @property
    def slim_binned(self):
        """
        Convenience method to access the binned-up grid in its 1D representation, which is a Grid2D stored as an
        ndarray of shape [total_unmasked_pixels, 2].

        The binning up process converts a grid from (y,x) values where each value is a coordinate on the sub-grid to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a grid with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.
        """
        if not self.store_slim:

            grid_2d_slim = grid_2d_util.grid_2d_slim_from(
                grid_2d_native=self, mask=self.mask, sub_size=self.mask.sub_size
            )

        else:

            grid_2d_slim = self

        grid_2d_slim_binned_y = np.multiply(
            self.mask.sub_fraction,
            grid_2d_slim[:, 0].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        grid_2d_slim_binned_x = np.multiply(
            self.mask.sub_fraction,
            grid_2d_slim[:, 1].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        return self._new_structure(
            grid=np.stack((grid_2d_slim_binned_y, grid_2d_slim_binned_x), axis=-1),
            mask=self.mask.mask_sub_1,
            store_slim=True,
        )

    @property
    def native_binned(self):
        """
        Convenience method to access the binned-up grid in its 2D representation, which is a Grid2D stored as an
        ndarray of shape [total_y_pixels, total_x_pixels, 2].

        The binning up process conerts a grid from (y,x) values where each value is a coordinate on the sub-grid to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a grid with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the grid is stored in 2D it is return as is. If it is stored in 1D, it must first be mapped from 1D to 2D.
        """
        if not self.store_slim:

            grid_2d_slim = grid_2d_util.grid_2d_slim_from(
                grid_2d_native=self, mask=self.mask, sub_size=self.mask.sub_size
            )

        else:

            grid_2d_slim = self

        grid_2d_slim_binned_y = np.multiply(
            self.mask.sub_fraction,
            grid_2d_slim[:, 0].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        grid_2d_slim_binned_x = np.multiply(
            self.mask.sub_fraction,
            grid_2d_slim[:, 1].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        grid_2d_slim_binned = np.stack(
            (grid_2d_slim_binned_y, grid_2d_slim_binned_x), axis=-1
        )

        grid_2d_native_binned = grid_2d_util.grid_2d_native_from(
            grid_2d_slim=grid_2d_slim_binned, mask_2d=self.mask, sub_size=1
        )

        return self._new_structure(
            grid=grid_2d_native_binned, mask=self.mask.mask_sub_1, store_slim=False
        )

    @property
    def slim_flipped(self) -> np.ndarray:
        """Return the grid as an ndarray of shape [total_unmasked_pixels, 2] with flipped values such that coordinates
        are given as (x,y) values.

        This is used to interface with Python libraries that require the grid in (x,y) format."""
        return np.fliplr(self)

    @property
    def native_flipped(self):
        """Return the grid as an ndarray array of shape [total_x_pixels, total_y_pixels, 2[ with flipped values such
        that coordinates are given as (x,y) values.

        This is used to interface with Python libraries that require the grid in (x,y) format."""
        return np.stack((self.native[:, :, 1], self.native[:, :, 0]), axis=-1)

    @property
    @array_2d_util.Memoizer()
    def in_radians(self):
        """Return the grid as an ndarray where all (y,x) values are converted to Radians.

        This grid is used by the interferometer module."""
        return (self * np.pi) / 648000.0

    def squared_distances_from_coordinate(self, coordinate=(0.0, 0.0)):
        """
        Returns the squared distance of every coordinate on the grid from an input coordinate.

        Parameters
        ----------
        coordinate : (float, float)
            The (y,x) coordinate from which the squared distance of every grid (y,x) coordinate is computed.
        """
        squared_distances = np.square(self[:, 0] - coordinate[0]) + np.square(
            self[:, 1] - coordinate[1]
        )
        return array_2d.Array2D.manual_mask(array=squared_distances, mask=self.mask)

    def distances_from_coordinate(self, coordinate=(0.0, 0.0)):
        """
        Returns the distance of every coordinate on the grid from an input (y,x) coordinate.

        Parameters
        ----------
        coordinate : (float, float)
            The (y,x) coordinate from which the distance of every grid (y,x) coordinate is computed.
        """
        distances = np.sqrt(
            self.squared_distances_from_coordinate(coordinate=coordinate)
        )
        return array_2d.Array2D.manual_mask(array=distances, mask=self.mask)

    def grid_2d_radial_projected_from(self, centre=(0.0, 0.0), angle: float = 0.0) -> grid_2d_irregular.Grid2DIrregular:
        """
        Determine a projected radial grid of points from a 2D region of coordinates defined by an
        extent [xmin, xmax, ymin, ymax] and with a (y,x) centre. This functions operates as follows:

        1) Given the region defined by the extent [xmin, xmax, ymin, ymax], the algorithm finds the longest 1D distance
        of the 4 paths from the (y,x) centre to the edge of the region (e.g. following the positive / negative y and
        x axes).

        2) Use the pixel-scale corresponding to the direction chosen (e.g. if the positive x-axis was the longest, the
        pixel_scale in the x dimension is used).

        3) Determine the number of pixels between the centre and the edge of the region using the longest path between
        the two chosen above.

        4) Create a (y,x) grid of radial points where all points are at the centre's y value = 0.0 and the x values
        iterate from the centre in increasing steps of the pixel-scale.

        5) Rotate these radial coordinates by the input `angle` clockwise.

        A schematic is shown below:

        -------------------
        |                 |
        |<- - -  - ->x    | x = centre
        |                 | <-> = longest radial path from centre to extent edge
        |                 |
        -------------------

        Using the centre x above, this function finds the longest radial path to the edge of the extent window.

        The returned `grid_radii` represents a radial set of points that in 1D sample the 2D grid outwards from its
        centre. This grid stores the radial coordinates as (y,x) values (where all y values are the same) as opposed to
        a 1D data structure so that it can be used in functions which require that a 2D grid structure is input.

        Parameters
        ----------
        extent : np.ndarray
            The extent of the grid the radii grid is computed using, with format [xmin, xmax, ymin, ymax]
        centre : (float, flloat)
            The (y,x) central coordinate which the radial grid is traced outwards from.
        pixel_scales : (float, float)
            The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
        sub_size : int
            The size of the sub-grid that each pixel of the 2D mask array is divided into.
        angle : float
            The angle with which the radial coordinates are rotated clockwise.

        Returns
        -------
        grid_2d_irregular.Grid2DIrregular
            A radial set of points sampling the longest distance from the centre to the edge of the extent in along the
            positive x-axis.
        """
        grid_radial_projected_2d = grid_2d_util.grid_scaled_2d_slim_radial_projected_from(
            extent=self.extent,
            centre=centre,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
        )

        grid_radial_projected_2d = geometry_util.transform_grid_2d_to_reference_frame(
            grid_2d=grid_radial_projected_2d, centre=(0.0, 0.0), angle=angle
        )

        return grid_2d_irregular.Grid2DIrregular(grid=grid_radial_projected_2d)

    @property
    def shape_native_scaled(self) -> (float, float):
        """
        The two dimensional shape of the grid in scaled units, computed by taking the minimum and maximum values of
        the grid.
        """
        return (
            np.amax(self[:, 0]) - np.amin(self[:, 0]),
            np.amax(self[:, 1]) - np.amin(self[:, 1]),
        )

    @property
    def scaled_maxima(self) -> (float, float):
        """
        The maximum values of the grid in scaled coordinates returned as a tuple (y_max, x_max).
        """
        return (
            self.origin[0] + (self.shape_native_scaled[0] / 2.0),
            self.origin[1] + (self.shape_native_scaled[1] / 2.0),
        )

    @property
    def scaled_minima(self) -> (float, float):
        """
        The minium values of the grid in scaled coordinates returned as a tuple (y_min, x_min).
        """
        return (
            (self.origin[0] - (self.shape_native_scaled[0] / 2.0)),
            (self.origin[1] - (self.shape_native_scaled[1] / 2.0)),
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

    def extent_with_buffer(self, buffer=1.0e-8) -> [float, float, float, float]:
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

    def padded_grid_from_kernel_shape(self, kernel_shape_native):
        """When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the padded grid is used, which is 'buffed' such that it includes all pixels
        whose signal will be convolved into the unmasked pixels given the 2D kernel shape.

        Parameters
        ----------
        kernel_shape_native : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        shape = self.mask.shape

        padded_shape = (
            shape[0] + kernel_shape_native[0] - 1,
            shape[1] + kernel_shape_native[1] - 1,
        )

        padded_mask = msk.Mask2D.unmasked(
            shape_native=padded_shape,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
        )

        return grid_2d.Grid2D.from_mask(mask=padded_mask)

    @property
    def sub_border_grid(self):
        """
        The (y,x) grid of all sub-pixels which are at the border of the mask.

        This is NOT all sub-pixels which are in mask pixels at the mask's border, but specifically the sub-pixels
        within these border pixels which are at the extreme edge of the border.
        """
        return self[self.mask._sub_border_flat_indexes]

    def relocated_grid_from_grid(self, grid):
        """
        Relocate the coordinates of a grid to the border of this grid if they are outside the border, where the
        border is defined as all pixels at the edge of the grid's mask (see *mask._border_1d_indexes*).

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
        grid : Grid2D
            The grid (uniform or irregular) whose pixels are to be relocated to the border edge if outside it.
        """

        if len(self.sub_border_grid) == 0:
            return grid

        return grid_2d.Grid2D(
            grid=self.relocated_grid_from_grid_jit(
                grid=grid, border_grid=self.sub_border_grid
            ),
            mask=grid.mask,
            sub_size=grid.mask.sub_size,
        )

    def relocated_pixelization_grid_from_pixelization_grid(self, pixelization_grid):
        """
        Relocate the coordinates of a pixelization grid to the border of this grid, see the method
        *relocated_grid_from_grid* for a full description of grid relocation.

        This function operates the same as other grid relocation functions by returns the grid as a
        `Grid2DVoronoi` instance.

        Parameters
        ----------
        grid : Grid2D
            The grid (uniform or irregular) whose pixels are to be relocated to the border edge if outside it.
        """

        if len(self.sub_border_grid) == 0:
            return pixelization_grid

        return grid_2d.Grid2DSparse(
            grid=self.relocated_grid_from_grid_jit(
                grid=pixelization_grid, border_grid=self.sub_border_grid
            ),
            sparse_index_for_slim_index=pixelization_grid.sparse_index_for_slim_index,
        )

    @staticmethod
    @decorator_util.jit()
    def relocated_grid_from_grid_jit(grid, border_grid):
        """
        Relocate the coordinates of a grid to its border if they are outside the border, where the border is
        defined as all pixels at the edge of the grid's mask (see *mask._border_1d_indexes*).

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
        grid : Grid2D
            The grid (uniform or irregular) whose pixels are to be relocated to the border edge if outside it.
        border_grid : Grid2D
            The grid of border (y,x) coordinates.
        """

        grid_relocated = np.zeros(grid.shape)
        grid_relocated[:, :] = grid[:, :]

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

                    grid_relocated[pixel_index, :] = (
                        move_factor * (grid[pixel_index, :] - border_origin[:])
                        + border_origin[:]
                    )

        return grid_relocated

    def output_to_fits(self, file_path, overwrite=False):
        """
        Output the grid to a .fits file.

        Parameters
        ----------
        file_path : str
            The path the file is output to, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        overwrite : bool
            If a file already exists at the path, if overwrite=True it is overwritten else an error is raised."""
        array_2d_util.numpy_array_2d_to_fits(
            array_2d=self.native, file_path=file_path, overwrite=overwrite
        )
