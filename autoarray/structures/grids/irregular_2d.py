import numpy as np
from typing import List, Optional, Tuple, Union

from autoarray.abstract_ndarray import AbstractNDArray
from autoarray.geometry.geometry_2d_irregular import Geometry2DIrregular
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.irregular import ArrayIrregular

from autoarray import exc
from autoarray.structures.grids import grid_2d_util
from autoarray.geometry import geometry_util


class Grid2DIrregular(AbstractNDArray):
    def __init__(self, values: Union[np.ndarray, List]):
        """
        An irregular grid of (y,x) coordinates.

        The `Grid2DIrregular` stores the (y,x) irregular grid of coordinates as 2D NumPy array of shape
        [total_coordinates, 2].

        Calculations should use the NumPy array structure wherever possible for efficient calculations.

        The coordinates input to this function can have any of the following forms (they will be converted to the
        1D NumPy array structure and can be converted back using the object's properties):

        ::

            [(y0,x0), (y1,x1)]
            [[y0,x0], [y1,x1]]

        If your grid lies on a 2D uniform grid of data the `Grid2D` data structure should be used.

        Parameters
        ----------
        values
            The irregular grid of (y,x) coordinates.
        """

        if len(values) == 0:
            super().__init__(values)
            return

        if type(values) is list:
            if isinstance(values[0], Grid2DIrregular):
                values = values
            else:
                values = np.asarray(values)

        super().__init__(values)

    @classmethod
    def from_yx_1d(cls, y: np.ndarray, x: np.ndarray) -> "Grid2DIrregular":
        """
        Create `Grid2DIrregular` from a list of y and x values.
        """
        return Grid2DIrregular(values=np.stack((y, x), axis=-1))

    @classmethod
    def from_pixels_and_mask(
        cls, pixels: Union[np.ndarray, List], mask: Mask2D
    ) -> "Grid2DIrregular":
        """
        Create `Grid2DIrregular` from a list of coordinates in pixel units and a mask which allows these
        coordinates to be converted to scaled units.
        """

        coorindates = [
            mask.geometry.scaled_coordinates_2d_from(
                pixel_coordinates_2d=pixel_coordinates_2d
            )
            for pixel_coordinates_2d in pixels
        ]

        return Grid2DIrregular(values=coorindates)

    @property
    def values(self):
        return self._array

    @property
    def geometry(self):
        """
        The (y,x) 2D shape of the irregular grid in scaled units, computed by taking the minimum and
        maximum values of (y,x) coordinates on the grid.
        """
        shape_native_scaled = (
            np.amax(self[:, 0]).astype("float") - np.amin(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float") - np.amin(self[:, 1]).astype("float"),
        )

        scaled_maxima = (
            np.amax(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float"),
        )

        scaled_minima = (
            np.amin(self[:, 0]).astype("float"),
            np.amin(self[:, 1]).astype("float"),
        )

        return Geometry2DIrregular(
            shape_native_scaled=shape_native_scaled,
            scaled_maxima=scaled_maxima,
            scaled_minima=scaled_minima,
        )

    @property
    def slim(self) -> "Grid2DIrregular":
        return self

    @property
    def native(self) -> "Grid2DIrregular":
        return self

    @property
    def in_list(self) -> List:
        """
        Return the coordinates in a list.
        """
        return [tuple(value) for value in self]

    @property
    def scaled_minima(self) -> Tuple:
        """
        The (y,x) minimum values of the grid in scaled units, buffed such that their extent is further than the grid's
        extent.
        """
        return (
            np.amin(self[:, 0]).astype("float"),
            np.amin(self[:, 1]).astype("float"),
        )

    @property
    def scaled_maxima(self) -> Tuple:
        """
        The (y,x) maximum values of the grid in scaled units, buffed such that their extent is further than the grid's
        extent.
        """
        return (
            np.amax(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float"),
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

    def grid_2d_via_deflection_grid_from(
        self, deflection_grid: np.ndarray
    ) -> "Grid2DIrregular":
        """
        Returns a new Grid2DIrregular from this grid coordinates, where the (y,x) coordinates of this grid have a
        grid of (y,x) values, termed the deflection grid, subtracted from them to determine the new grid of (y,x)
        values.

        This is to perform grid ray-tracing.

        Parameters
        ----------
        deflection_grid
            The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return Grid2DIrregular(values=self - deflection_grid)

    def squared_distances_to_coordinate_from(
        self, coordinate: Tuple[float, float] = (0.0, 0.0)
    ) -> ArrayIrregular:
        """
        Returns the squared distance of every (y,x) coordinate in this *Coordinate* instance from an input
        coordinate.

        Parameters
        ----------
        coordinate
            The (y,x) coordinate from which the squared distance of every *Coordinate* is computed.
        """
        squared_distances = np.square(self[:, 0] - coordinate[0]) + np.square(
            self[:, 1] - coordinate[1]
        )
        return ArrayIrregular(values=squared_distances)

    def distances_to_coordinate_from(
        self, coordinate: Tuple[float, float] = (0.0, 0.0)
    ) -> ArrayIrregular:
        """
        Returns the distance of every (y,x) coordinate in this *Coordinate* instance from an input coordinate.

        Parameters
        ----------
        coordinate
            The (y,x) coordinate from which the distance of every coordinate is computed.
        """
        distances = np.sqrt(
            self.squared_distances_to_coordinate_from(coordinate=coordinate)
        )
        return ArrayIrregular(values=distances)

    @property
    def furthest_distances_to_other_coordinates(self) -> ArrayIrregular:
        """
        For every (y,x) coordinate on the `Grid2DIrregular` returns the furthest radial distance of each coordinate
        to any other coordinate on the grid.

        For example, for the following grid:

        ::

            values=[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)]

        The returned further distances are:

        ::

            [3.0, 2.0, 3.0]

        Returns
        -------
        ArrayIrregular
            The further distances of every coordinate to every other coordinate on the irregular grid.
        """

        radial_distances_max = np.zeros((self.shape[0]))

        for i in range(self.shape[0]):
            x_distances = np.square(np.subtract(self[i, 0], self[:, 0]))
            y_distances = np.square(np.subtract(self[i, 1], self[:, 1]))

            radial_distances_max[i] = np.sqrt(np.max(np.add(x_distances, y_distances)))

        return ArrayIrregular(values=radial_distances_max)

    def grid_of_closest_from(self, grid_pair: "Grid2DIrregular") -> "Grid2DIrregular":
        """
        From an input grid, find the closest coordinates of this instance of the `Grid2DIrregular` to each coordinate
        on the input grid and return each closest coordinate as a new `Grid2DIrregular`.

        Parameters
        ----------
        grid_pair
            The grid of coordinates the closest coordinate of each (y,x) location is paired with.

        Returns
        -------
        The grid of coordinates corresponding to the closest coordinate of each coordinate of this instance of
        the `Grid2DIrregular` to the input grid.
        """

        grid_of_closest = np.zeros((grid_pair.shape[0], 2))

        for i in range(grid_pair.shape[0]):
            x_distances = np.square(np.subtract(grid_pair[i, 0], self[:, 0]))
            y_distances = np.square(np.subtract(grid_pair[i, 1], self[:, 1]))

            radial_distances = np.add(x_distances, y_distances)

            grid_of_closest[i, :] = self[np.argmin(radial_distances), :]

        return Grid2DIrregular(values=grid_of_closest)


class Grid2DIrregularTransformed(Grid2DIrregular):
    pass


class Grid2DIrregularUniform(Grid2DIrregular):
    def __init__(
        self,
        values: np.ndarray,
        shape_native: Optional[Tuple[float, float]] = None,
        pixel_scales: Optional[Tuple[float, float]] = None,
    ):
        """
        A collection of (y,x) coordinates which is structured as follows:

        ::

            [[x0, x1], [x0, x1]]

        The grid object does not store the coordinates as a list of tuples, but instead a 2D NumPy array of
        shape [total_coordinates, 2]. They are stored as a NumPy array so the coordinates can be used efficiently for
        calculations.

        The coordinates input to this function can have any of the following forms:

        ::

            [(y0,x0), (y1,x1)]

        In all cases, they will be converted to a list of tuples followed by a 2D NumPy array.

        Print methods are overidden so a user always "sees" the coordinates as the list structure.

        Like the `Grid2D` structure, `Grid2DIrregularUniform` lie on a uniform grid corresponding to values that
        originate from a uniform grid. This contrasts the `Grid2DIrregular` class above. However, although this class
        stores the pixel-scale and 2D shape of this grid, it does not store the mask that a `Grid2D` does that enables
        the coordinates to be mapped from 1D to 2D. This is for calculations that utilize the 2d information of the
        grid but do not want the memory overheads associated with the 2D mask.

        Parameters
        ----------
        values
            A collection of (y,x) coordinates that.
        """

        if len(values) == 0:
            super().__init__(values=values)
            return

        if isinstance(values[0], float):
            values = [values]

        if isinstance(values[0], tuple):
            values = [values]
        elif isinstance(values[0], (np.ndarray, AbstractNDArray)):
            if len(values[0].shape) == 1:
                values = [values]
        elif isinstance(values[0], list) and isinstance(values[0][0], (float)):
            values = [values]

        coordinates_arr = np.concatenate([np.array(i) for i in values])

        self._internal_list = values

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        self.shape_native = shape_native
        self.pixel_scales = pixel_scales

        super().__init__(coordinates_arr)

    def __array_finalize__(self, obj):
        if hasattr(obj, "_internal_list"):
            self._internal_list = obj._internal_list

        if hasattr(obj, "shape_native"):
            self.shape_native = obj.shape_native

        if hasattr(obj, "pixel_scales"):
            self.pixel_scales = obj.pixel_scales

    @property
    def pixel_scale(self) -> float:
        if self.pixel_scales[0] == self.pixel_scales[1]:
            return self.pixel_scales[0]
        else:
            raise exc.GridException(
                "Cannot return a pixel_scale for a grid where each dimension has a "
                "different pixel scale (e.g. pixel_scales[0] != pixel_scales[1])"
            )

    @classmethod
    def from_grid_sparse_uniform_upscale(
        cls,
        grid_sparse_uniform: np.ndarray,
        upscale_factor: int,
        pixel_scales,
        shape_native=None,
    ) -> "Grid2DIrregularUniform":
        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        grid_upscaled_1d = grid_2d_util.grid_2d_slim_upscaled_from(
            grid_slim=np.array(grid_sparse_uniform),
            upscale_factor=upscale_factor,
            pixel_scales=pixel_scales,
        )

        pixel_scales = (
            pixel_scales[0] / upscale_factor,
            pixel_scales[1] / upscale_factor,
        )

        return Grid2DIrregularUniform(
            values=grid_upscaled_1d,
            pixel_scales=pixel_scales,
            shape_native=shape_native,
        )

    def grid_from(self, grid_slim: np.ndarray) -> "Grid2DIrregularUniform":
        """
        Create a `Grid2DIrregularUniform` object from a 2D NumPy array of values of shape [total_coordinates, 2]. The
        `Grid2DIrregularUniform` are structured following this *GridIrregular2D* instance.
        """
        return Grid2DIrregularUniform(
            values=grid_slim,
            pixel_scales=self.pixel_scales,
            shape_native=self.shape_native,
        )

    def grid_2d_via_deflection_grid_from(
        self, deflection_grid: np.ndarray
    ) -> "Grid2DIrregularUniform":
        """
        Returns a new Grid2DIrregular from this grid coordinates, where the (y,x) coordinates of this grid have a
        grid of (y,x) values, termed the deflection grid, subtracted from them to determine the new grid of (y,x)
        values.

        This is used by PyAutoLens to perform grid ray-tracing.

        Parameters
        ----------
        deflection_grid
            The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return Grid2DIrregularUniform(
            values=self - deflection_grid,
            pixel_scales=self.pixel_scales,
            shape_native=self.shape_native,
        )
