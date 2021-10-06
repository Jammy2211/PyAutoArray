import numpy as np
from typing import List, Tuple, Union

from autoconf import conf
from autoconf import cached_property

from autoarray.structures.abstract_structure import AbstractStructure2D
from autoarray.structures.arrays.two_d import array_2d as a2d
from autoarray.structures.arrays.values import ValuesIrregular
from autoarray.structures.grids.two_d import grid_2d as g2d
from autoarray.structures.grids.two_d import grid_2d_irregular as g2d_irr
from autoarray.mask import mask_2d as m2d

from autoarray import exc
from autoarray.structures.grids import abstract_grid
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray.structures.grids.two_d import grid_2d_util
from autoarray.geometry import geometry_util


def check_grid_2d(grid_2d):

    if grid_2d.shape[-1] != 2:
        raise exc.GridException(
            "The final dimension of the input grid is not equal to 2 (e.g. the (y,x) coordinates)"
        )

    if 2 < len(grid_2d.shape) > 3:
        raise exc.GridException("The dimensions of the input grid array is not 2 or 3")


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
                "(e.g. the mask 2D shape multipled by its sub size.)"
            )


def convert_grid_2d(grid_2d: Union[np.ndarray, List], mask_2d) -> np.ndarray:
    """
    he `manual` classmethods in the Grid2D object take as input a list or ndarray which is returned as a Grid2D. 
    
    This function performs the following and checks and conversions on the input:

    1: If the input is a list, convert it to an ndarray.
    2: Check that the number of sub-pixels in the array is identical to that of the mask.
    3) Map the input ndarray to its `slim` or `native` representation, depending on the `general.ini` config file
    entry `store_slim`.

    For a Grid2D, `slim` refers to a 2D NumPy array of shape [total_coordinates, 2] and `native` a 3D NumPy array of
    shape [total_y_coordinates, total_x_coordinates, 2]

    Parameters
    ----------
    grid_2d
        The input (y,x) grid of coordinates which is converted to an ndarray if it is a list.
    mask_2d : Mask2D
        The mask of the output Array2D.
    """

    grid_2d = abstract_grid.convert_grid(grid=grid_2d)

    if conf.instance["general"]["structures"]["store_slim"]:
        return convert_grid_2d_to_slim(grid_2d=grid_2d, mask_2d=mask_2d)
    return convert_grid_2d_to_native(grid_2d=grid_2d, mask_2d=mask_2d)


def convert_grid_2d_to_slim(grid_2d: Union[np.ndarray, List], mask_2d) -> np.ndarray:
    """
    he `manual` classmethods in the Grid2D object take as input a list or ndarray which is returned as a Grid2D. 

    This function checks the dimensions of the input `grid_2d` and maps it to its `slim` representation.

    For a Grid2D, `slim` refers to a 2D NumPy array of shape [total_coordinates, 2].

    Parameters
    ----------
    grid_2d
        The input (y,x) grid of coordinates which is converted to its silm representation.
    mask_2d : Mask2D
        The mask of the output Array2D.
    """
    if len(grid_2d.shape) == 2:
        return grid_2d
    return grid_2d_util.grid_2d_slim_from(
        grid_2d_native=grid_2d, mask=mask_2d, sub_size=mask_2d.sub_size
    )


def convert_grid_2d_to_native(grid_2d: Union[np.ndarray, List], mask_2d) -> np.ndarray:
    """
    he `manual` classmethods in the Grid2D object take as input a list or ndarray which is returned as a Grid2D. 

    This function checks the dimensions of the input `grid_2d` and maps it to its `native` representation.

    For a Grid2D, `native` refers to a 2D NumPy array of shape [total_y_coordinates, total_x_coordinates, 2].

    Parameters
    ----------
    grid_2d
        The input (y,x) grid of coordinates which is converted to its native representation.
    mask_2d : Mask2D
        The mask of the output Array2D.
    """
    if len(grid_2d.shape) == 3:
        return grid_2d
    return grid_2d_util.grid_2d_native_from(
        grid_2d_slim=grid_2d, mask_2d=mask_2d, sub_size=mask_2d.sub_size
    )


class AbstractGrid2D(AbstractStructure2D):
    @property
    def slim(self):
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

        return self._new_structure(grid=grid_2d_slim, mask=self.mask)

    @property
    def native(self):
        """
        Return a `Grid2D` where the data is stored in its `native` representation, which is an ndarray of shape
        [sub_size*total_y_pixels, sub_size*total_x_pixels, 2].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Grid2D`.
        """

        if len(self.shape) != 2:
            return self

        grid_2d_native = grid_2d_util.grid_2d_native_from(
            grid_2d_slim=self, mask_2d=self.mask, sub_size=self.mask.sub_size
        )

        return self._new_structure(grid=grid_2d_native, mask=self.mask)

    @property
    def binned(self) -> "AbstractGrid2D":
        """
        Convenience method to access the binned-up grid in its 1D representation, which is a Grid2D stored as an
        ndarray of shape [total_unmasked_pixels, 2].

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

        return self._new_structure(
            grid=np.stack((grid_2d_slim_binned_y, grid_2d_slim_binned_x), axis=-1),
            mask=self.mask.mask_sub_1,
        )

    @property
    def flipped(self) -> np.ndarray:
        """
        Return the grid as an ndarray of shape [total_unmasked_pixels, 2] with flipped values such that coordinates
        are given as (x,y) values.

        This is used to interface with Python libraries that require the grid in (x,y) format.
        """
        return np.fliplr(self)

    @property
    def in_radians(self):
        """
        Return the grid as an ndarray where all (y,x) values are converted to Radians.

        This grid is used by the interferometer module.
        """
        return (self * np.pi) / 648000.0

    def values_from(self, array_slim):
        """
        Create a *ValuesIrregular* object from a 1D NumPy array of values of shape [total_coordinates]. The
        *ValuesIrregular* are structured following this `Grid2DIrregular` instance.
        """
        return ValuesIrregular(values=array_slim)

    def squared_distances_to_coordinate(
        self, coordinate: Tuple[float, float] = (0.0, 0.0)
    ):
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
        return a2d.Array2D.manual_mask(array=squared_distances, mask=self.mask)

    def distances_to_coordinate(self, coordinate: Tuple[float, float] = (0.0, 0.0)):
        """
        Returns the distance of every coordinate on the grid from an input (y,x) coordinate.

        Parameters
        ----------
        coordinate
            The (y,x) coordinate from which the distance of every grid (y,x) coordinate is computed.
        """
        distances = np.sqrt(self.squared_distances_to_coordinate(coordinate=coordinate))
        return a2d.Array2D.manual_mask(array=distances, mask=self.mask)

    def grid_2d_radial_projected_from(
        self, centre: Tuple[float, float] = (0.0, 0.0), angle: float = 0.0
    ) -> "g2d_irr.Grid2DIrregular":
        """
        Determine a projected radial grid of points from a 2D region of coordinates defined by an
        extent [xmin, xmax, ymin, ymax] and with a (y,x) centre. This functions operates as follows:

        1 Given the region defined by the extent [xmin, xmax, ymin, ymax], the algorithm finds the longest 1D distance
        of the 4 paths from the (y,x) centre to the edge of the region e.g. following the positive / negative y and
        x axes.

        2: Use the pixel-scale corresponding to the direction chosen e.g. if the positive x-axis was the longest, the
        pixel_scale in the x dimension is used.

        3: Determine the number of pixels between the centre and the edge of the region using the longest path between
        the two chosen above.

        4: Create a (y,x) grid of radial points where all points are at the centre's y value = 0.0 and the x values
        iterate from the centre in increasing steps of the pixel-scale.

        5: Rotate these radial coordinates by the input `angle` clockwise.

        A schematic is shown below:

        -------------------
        |                 |
        |<- - -  - ->x    | x = centre
        |                 | <-> = longest radial path from centre to extent edge
        |                 |
        -------------------

        4: Create a (y,x) grid of radial points where all points are at the centre's y value = 0.0 and the x values
        iterate from the centre in increasing steps of the pixel-scale.

        5: Rotate these radial coordinates by the input `angle` clockwise.

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

        if hasattr(self, "radial_projected_shape_slim"):
            shape_slim = self.radial_projected_shape_slim
        else:
            shape_slim = 0

        grid_radial_projected_2d = grid_2d_util.grid_scaled_2d_slim_radial_projected_from(
            extent=self.extent,
            centre=centre,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            shape_slim=shape_slim,
        )

        grid_radial_projected_2d = geometry_util.transform_grid_2d_to_reference_frame(
            grid_2d=grid_radial_projected_2d, centre=centre, angle=angle
        )

        grid_radial_projected_2d = geometry_util.transform_grid_2d_from_reference_frame(
            grid_2d=grid_radial_projected_2d, centre=centre, angle=0.0
        )

        if conf.instance["general"]["grid"]["remove_projected_centre"]:

            grid_radial_projected_2d = grid_radial_projected_2d[1:, :]

        return g2d_irr.Grid2DIrregular(grid=grid_radial_projected_2d)

    @property
    def shape_native_scaled(self) -> Tuple[float, float]:
        """
        The two dimensional shape of the grid in scaled units, computed by taking the minimum and maximum values of
        the grid.
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
            self.origin[0] + (self.shape_native_scaled[0] / 2.0),
            self.origin[1] + (self.shape_native_scaled[1] / 2.0),
        )

    @property
    def scaled_minima(self) -> Tuple[float, float]:
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

    def extent_with_buffer(self, buffer=1.0e-8) -> List[float]:
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

    def padded_grid_from(self, kernel_shape_native: Tuple[float, float]):
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

        padded_mask = m2d.Mask2D.unmasked(
            shape_native=padded_shape,
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.mask.sub_size,
        )

        return g2d.Grid2D.from_mask(mask=padded_mask)

    @cached_property
    def sub_border_grid(self):
        """
        The (y,x) grid of all sub-pixels which are at the border of the mask.

        This is NOT all sub-pixels which are in mask pixels at the mask's border, but specifically the sub-pixels
        within these border pixels which are at the extreme edge of the border.
        """
        return self[self.mask.sub_border_flat_indexes]

    def relocated_grid_from(self, grid):
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

        return g2d.Grid2D(
            grid=grid_2d_util.relocated_grid_via_jit_from(
                grid=grid, border_grid=self.sub_border_grid
            ),
            mask=grid.mask,
            sub_size=grid.mask.sub_size,
        )

    def relocated_pixelization_grid_from(self, pixelization_grid):
        """
        Relocate the coordinates of a pixelization grid to the border of this grid, see the method
        *relocated_grid_from* for a full description of grid relocation.

        This function operates the same as other grid relocation functions by returns the grid as a
        `Grid2DVoronoi` instance.

        Parameters
        ----------
        grid : Grid2D
            The grid (uniform or irregular) whose pixels are to be relocated to the border edge if outside it.
        """

        if len(self.sub_border_grid) == 0:
            return pixelization_grid

        return g2d.Grid2DSparse(
            grid=grid_2d_util.relocated_grid_via_jit_from(
                grid=pixelization_grid, border_grid=self.sub_border_grid
            ),
            sparse_index_for_slim_index=pixelization_grid.sparse_index_for_slim_index,
        )

    def output_to_fits(self, file_path: str, overwrite: bool = False):
        """
        Output the grid to a .fits file.

        Parameters
        ----------
        file_path
            The path the file is output to, including the filename and the .fits extension, e.g. '/path/to/filename.fits'
        overwrite
            If a file already exists at the path, if overwrite=True it is overwritten else an error is raised.
        """
        array_2d_util.numpy_array_2d_to_fits(
            array_2d=self.native, file_path=file_path, overwrite=overwrite
        )
