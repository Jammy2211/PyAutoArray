import numpy as np
import os
from os import path
import pickle
from typing import List, Optional, Tuple, Union
import json

from autoarray.abstract_ndarray import AbstractNDArray
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.values import ValuesIrregular

from autoarray import exc
from autoarray.structures.grids import grid_2d_util
from autoarray.geometry import geometry_util


class Grid2DIrregular(AbstractNDArray):
    def __new__(cls, grid: Union[np.ndarray, List]):
        """
        An irregular grid of (y,x) coordinates.

        The `Grid2DIrregular` stores the (y,x) irregular grid of coordinates as 2D NumPy array of shape
        [total_coordinates, 2].

        Calculations should use the NumPy array structure wherever possible for efficient calculations.

        The coordinates input to this function can have any of the following forms (they will be converted to the
        1D NumPy array structure and can be converted back using the object's properties):

        [(y0,x0), (y1,x1)]
        [[y0,x0], [y1,x1]]

        If your grid lies on a 2D uniform grid of data the `Grid2D` data structure should be used.

        Parameters
        ----------
        grid : Grid2DIrregular
            The irregular grid of (y,x) coordinates.
        """

        if len(grid) == 0:
            return []

        if type(grid) is list:

            if isinstance(grid[0], Grid2DIrregular):
                return grid

            grid = np.asarray(grid)

        obj = grid.view(cls)

        return obj

    @property
    def shape_native_scaled(self) -> Tuple[float, float]:
        """
        The (y,x) 2D shape of the irregular grid in scaled units, computed by taking the minimum and
        maximum values of (y,x) coordinates on the grid.
        """
        return (
            np.amax(self[:, 0]).astype("float") - np.amin(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float") - np.amin(self[:, 1]).astype("float"),
        )

    @property
    def scaled_maxima(self) -> Tuple[int, int]:
        """The maximum values of the coordinates returned as a tuple (y_max, x_max)."""
        return (
            np.amax(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float"),
        )

    @property
    def scaled_minima(self) -> Tuple[int, int]:
        """The minimum values of the coordinates returned as a tuple (y_max, x_max)."""
        return (
            np.amin(self[:, 0]).astype("float"),
            np.amin(self[:, 1]).astype("float"),
        )

    @property
    def extent(self) -> np.ndarray:
        """The extent of the coordinates returned as a NumPy array [x_min, x_max, y_min, y_max].

        This follows the format of the extent input parameter in the matplotlib method imshow (and other methods) and
        is used for visualization in the plot module."""
        return np.array(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )

    @property
    def slim(self) -> "Grid2DIrregular":
        return self

    @property
    def native(self) -> "Grid2DIrregular":
        return self

    @classmethod
    def from_yx_1d(cls, y: np.ndarray, x: np.ndarray) -> "Grid2DIrregular":
        """
        Create `Grid2DIrregular` from a list of y and x values.
        """
        return Grid2DIrregular(grid=np.stack((y, x), axis=-1))

    @classmethod
    def from_pixels_and_mask(
        cls, pixels: Union[np.ndarray, List], mask: Mask2D
    ) -> "Grid2DIrregular":
        """
        Create `Grid2DIrregular` from a list of coordinates in pixel units and a mask which allows these coordinates to
        be converted to scaled units.
        """

        coorindates = [
            mask.scaled_coordinates_2d_from(pixel_coordinates_2d=pixel_coordinates_2d)
            for pixel_coordinates_2d in pixels
        ]

        return Grid2DIrregular(grid=coorindates)

    @property
    def in_list(self) -> List:
        """
        Return the coordinates in a list.
        """
        return [tuple(value) for value in self]

    def values_from(self, array_slim: np.ndarray) -> ValuesIrregular:
        """
        Create a *ValuesIrregular* object from a 1D NumPy array of values of shape [total_coordinates]. The
        *ValuesIrregular* are structured following this `Grid2DIrregular` instance.
        """
        return ValuesIrregular(values=array_slim)

    def values_via_value_from(self, value: float) -> ValuesIrregular:
        return self.values_from(
            array_slim=np.full(fill_value=value, shape=self.shape[0])
        )

    def grid_from(
        self, grid_slim: np.ndarray
    ) -> Union["Grid2DIrregular", "Grid2DIrregularTransformed"]:
        """
        Create a `Grid2DIrregular` object from a 2D NumPy array of values of shape [total_coordinates, 2]. The
        `Grid2DIrregular` are structured following this *Grid2DIrregular* instance."""

        from autoarray.structures.grids.transformed_2d import Grid2DTransformedNumpy

        if isinstance(grid_slim, Grid2DTransformedNumpy):
            return Grid2DIrregularTransformed(grid=grid_slim)
        return Grid2DIrregular(grid=grid_slim)

    def grid_via_deflection_grid_from(
        self, deflection_grid: np.ndarray
    ) -> "Grid2DIrregular":
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
        return Grid2DIrregular(grid=self - deflection_grid)

    def structure_2d_from(
        self, result: Union[np.ndarray, List]
    ) -> Union[ValuesIrregular, "Grid2DIrregular", "Grid2DIrregularTransformed", List]:
        """
        Convert a result from a non autoarray structure to an aa.ValuesIrregular or aa.Grid2DIrregular structure, where
        the conversion depends on type(result) as follows:

        - 1D np.ndarray   -> aa.ValuesIrregular
        - 2D np.ndarray   -> aa.Grid2DIrregular
        - [1D np.ndarray] -> [aa.ValuesIrregular]
        - [2D np.ndarray] -> [aa.Grid2DIrregular]

        This function is used by the grid_2d_to_structure decorator to convert the output result of a function
        to an autoarray structure when a `Grid2DIrregular` instance is passed to the decorated function.

        Parameters
        ----------
        result or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """

        if isinstance(result, np.ndarray):
            if len(result.shape) == 1:
                return self.values_from(array_slim=result)
            elif len(result.shape) == 2:
                return self.grid_from(grid_slim=result)
        elif isinstance(result, list):
            if len(result[0].shape) == 1:
                return [self.values_from(array_slim=value) for value in result]
            elif len(result[0].shape) == 2:
                return [self.grid_from(grid_slim=value) for value in result]

    def structure_2d_list_from(
        self, result_list: List
    ) -> List[Union[ValuesIrregular, "Grid2DIrregular", "Grid2DIrregularTransformed"]]:
        """
        Convert a result from a list of non autoarray structures to a list of aa.ValuesIrregular or aa.Grid2DIrregular
        structures, where the conversion depends on type(result) as follows:

        - [1D np.ndarray] -> [aa.ValuesIrregular]
        - [2D np.ndarray] -> [aa.Grid2DIrregular]

        This function is used by the grid_like_list_to_structure_list decorator to convert the output result of a
        function to a list of autoarray structure when a `Grid2DIrregular` instance is passed to the decorated function.

        Parameters
        ----------
        result_list or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        if len(result_list[0].shape) == 1:
            return [self.values_from(array_slim=value) for value in result_list]
        elif len(result_list[0].shape) == 2:
            return [self.grid_from(grid_slim=value) for value in result_list]

    def squared_distances_to_coordinate_from_from(
        self, coordinate: Tuple[float, float] = (0.0, 0.0)
    ) -> ValuesIrregular:
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
        return self.values_from(array_slim=squared_distances)

    def distances_to_coordinate_from(
        self, coordinate: Tuple[float, float] = (0.0, 0.0)
    ) -> ValuesIrregular:
        """
        Returns the distance of every (y,x) coordinate in this *Coordinate* instance from an input coordinate.

            Parameters
            ----------
            coordinate
                The (y,x) coordinate from which the distance of every *Coordinate* is computed.
        """
        distances = np.sqrt(
            self.squared_distances_to_coordinate_from_from(coordinate=coordinate)
        )
        return self.values_from(array_slim=distances)

    @property
    def furthest_distances_to_other_coordinates(self) -> ValuesIrregular:
        """
        For every (y,x) coordinate on the `Grid2DIrregular` returns the furthest radial distance of each coordinate
        to any other coordinate on the grid.

        For example, for the following grid:

        grid=[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)]

        The returned further distances are:

        [3.0, 2.0, 3.0]

        Returns
        -------
        ValuesIrregular
            The further distances of every coordinate to every other coordinate on the irregular grid.
        """

        radial_distances_max = np.zeros((self.shape[0]))

        for i in range(self.shape[0]):

            x_distances = np.square(np.subtract(self[i, 0], self[:, 0]))
            y_distances = np.square(np.subtract(self[i, 1], self[:, 1]))

            radial_distances_max[i] = np.sqrt(np.max(np.add(x_distances, y_distances)))

        return self.values_from(array_slim=radial_distances_max)

    def grid_of_closest_from(
        self, grid_pair: Union["Grid2DIrregular"]
    ) -> "Grid2DIrregular":
        """
        From an input grid, find the closest coordinates of this instance of the `Grid2DIrregular` to each coordinate
        on the input grid and return each closest coordinate as a new `Grid2DIrregular`.

        Parameters
        ----------
        grid_pair
            The grid of coordinates the closest coordinate of each (y,x) location is paired with.

        Returns
        -------
        Grid2DIrregular
            The grid of coordinates corresponding to the closest coordinate of each coordinate of this instance of
            the `Grid2DIrregular` to the input grid.

        """

        grid_of_closest = np.zeros((grid_pair.shape[0], 2))

        for i in range(grid_pair.shape[0]):

            x_distances = np.square(np.subtract(grid_pair[i, 0], self[:, 0]))
            y_distances = np.square(np.subtract(grid_pair[i, 1], self[:, 1]))

            radial_distances = np.add(x_distances, y_distances)

            grid_of_closest[i, :] = self[np.argmin(radial_distances), :]

        return Grid2DIrregular(grid=grid_of_closest)

    @classmethod
    def from_json(cls, file_path: str) -> "Grid2DIrregular":
        """
        Create a `Grid2DIrregular` object from a file which stores the coordinates as a list of list of tuples.

        Parameters
        ----------
        file_path
            The path to the coordinates .dat file containing the coordinates (e.g. '/path/to/coordinates.dat')
        """

        with open(file_path) as infile:
            grid = json.load(infile)

        return Grid2DIrregular(grid=grid)

    def output_to_json(self, file_path: str, overwrite: bool = False):
        """
        Output this instance of the `Grid2DIrregular` object to a list of list of tuples.

        Parameters
        ----------
        file_path
            The path to the coordinates .dat file containing the coordinates (e.g. '/path/to/coordinates.dat')
        overwrite
            If there is as exsiting file it will be overwritten if this is `True`.
        """

        file_dir = os.path.split(file_path)[0]

        if not path.exists(file_dir):
            os.makedirs(file_dir)

        if overwrite and path.exists(file_path):
            os.remove(file_path)
        elif not overwrite and path.exists(file_path):
            raise FileExistsError(
                "The file ",
                file_path,
                " already exists. Set overwrite=True to overwrite this" "file",
            )

        with open(file_path, "w+") as f:
            json.dump(self.in_list, f)


class Grid2DIrregularTransformed(Grid2DIrregular):

    pass


class Grid2DIrregularUniform(Grid2DIrregular):
    def __new__(
        cls,
        grid: np.ndarray,
        shape_native: Optional[Tuple[float, float]] = None,
        pixel_scales: Optional[Tuple[float, float]] = None,
    ):
        """
        A collection of (y,x) coordinates which is structured as follows:

        [[x0, x1], [x0, x1]]

        The grid object does not store the coordinates as a list of tuples, but instead a 2D NumPy array of
        shape [total_coordinates, 2]. They are stored as a NumPy array so the coordinates can be used efficiently for
        calculations.

        The coordinates input to this function can have any of the following forms:

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
        grid : [tuple] or equivalent
            A collection of (y,x) coordinates that.
        """

        #    obj = super(Grid2DIrregularUniform, cls).__new__(cls=cls, coordinates=coordinates)

        if len(grid) == 0:
            return []

        if isinstance(grid[0], float):
            grid = [grid]

        if isinstance(grid[0], tuple):
            grid = [grid]
        elif isinstance(grid[0], np.ndarray):
            if len(grid[0].shape) == 1:
                grid = [grid]
        elif isinstance(grid[0], list) and isinstance(grid[0][0], (float)):
            grid = [grid]

        coordinates_arr = np.concatenate([np.array(i) for i in grid])

        obj = coordinates_arr.view(cls)
        obj._internal_list = grid

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        obj.shape_native = shape_native
        obj.pixel_scales = pixel_scales

        return obj

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
            grid_slim=grid_sparse_uniform,
            upscale_factor=upscale_factor,
            pixel_scales=pixel_scales,
        )

        pixel_scales = (
            pixel_scales[0] / upscale_factor,
            pixel_scales[1] / upscale_factor,
        )

        return Grid2DIrregularUniform(
            grid=grid_upscaled_1d, pixel_scales=pixel_scales, shape_native=shape_native
        )

    def grid_from(self, grid_slim: np.ndarray) -> "Grid2DIrregularUniform":
        """
        Create a `Grid2DIrregularUniform` object from a 2D NumPy array of values of shape [total_coordinates, 2]. The
        `Grid2DIrregularUniform` are structured following this *GridIrregular2D* instance.
        """
        return Grid2DIrregularUniform(
            grid=grid_slim,
            pixel_scales=self.pixel_scales,
            shape_native=self.shape_native,
        )

    def grid_via_deflection_grid_from(
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
            grid=self - deflection_grid,
            pixel_scales=self.pixel_scales,
            shape_native=self.shape_native,
        )
