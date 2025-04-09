import logging
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Union

from autoarray.abstract_ndarray import AbstractNDArray
from autoarray.geometry.geometry_2d_irregular import Geometry2DIrregular
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.irregular import ArrayIrregular

logger = logging.getLogger(__name__)


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
        squared_distances = jnp.square(self.array[:, 0] - coordinate[0]) + jnp.square(
            self.array[:, 1] - coordinate[1]
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
        distances = jnp.sqrt(
            self.squared_distances_to_coordinate_from(coordinate=coordinate).array
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

        def max_radial_distance(point):
            x_distances = jnp.square(point[0] - self.array[:, 0])
            y_distances = jnp.square(point[1] - self.array[:, 1])
            return jnp.sqrt(jnp.nanmax(x_distances + y_distances))

        return ArrayIrregular(values=jax.vmap(max_radial_distance)(self.array))

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

        jax_array = jnp.asarray(self.array)

        def closest_point(point):
            x_distances = jnp.square(point[0] - jax_array[:, 0])
            y_distances = jnp.square(point[1] - jax_array[:, 1])
            radial_distances = x_distances + y_distances
            return jax_array[jnp.argmin(radial_distances)]

        return jax.vmap(closest_point)(grid_pair.array)
