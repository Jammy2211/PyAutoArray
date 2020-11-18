import ast
import numpy as np
import os
from os import path
import pickle
import typing

from autoarray.structures import arrays
from autoarray.util import geometry_util, grid_util
from autoarray import exc


class AbstractGridIrregular(np.ndarray):
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
        return self

    @property
    def in_1d_list(self):
        """Return the coordinates on a structured list which groups coordinates with a common origin."""
        return [tuple(coordinate) for coordinate in self.in_1d]

    def values_from_arr_1d(self, arr_1d):
        raise NotImplementedError()

    def grid_from_grid_1d(self, grid_1d):
        raise NotImplementedError()

    def structure_from_result(
        self, result: np.ndarray or list
    ) -> typing.Union[arrays.Values, list]:
        """Convert a result from a non autoarray structure to an aa.Values or aa.GridIrregularGrouped structure, where
        the conversion depends on type(result) as follows:

        - 1D np.ndarray   -> aa.Values
        - 2D np.ndarray   -> aa.GridIrregularGrouped
        - [1D np.ndarray] -> [aa.Values]
        - [2D np.ndarray] -> [aa.GridIrregularGrouped]

        This function is used by the grid_like_to_structure decorator to convert the output result of a function
        to an autoarray structure when a `GridIrregularGrouped` instance is passed to the decorated function.

        Parameters
        ----------
        result : np.ndarray or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """

        if isinstance(result, np.ndarray):
            if len(result.shape) == 1:
                return self.values_from_arr_1d(arr_1d=result)
            elif len(result.shape) == 2:
                return self.grid_from_grid_1d(grid_1d=result)
        elif isinstance(result, list):
            if len(result[0].shape) == 1:
                return [self.values_from_arr_1d(arr_1d=value) for value in result]
            elif len(result[0].shape) == 2:
                return [self.grid_from_grid_1d(grid_1d=value) for value in result]

    def structure_list_from_result_list(self, result_list: list) -> typing.Union[list]:
        """Convert a result from a list of non autoarray structures to a list of aa.Values or aa.GridIrregularGrouped
        structures, where the conversion depends on type(result) as follows:

        - [1D np.ndarray] -> [aa.Values]
        - [2D np.ndarray] -> [aa.GridIrregularGrouped]

        This function is used by the grid_like_list_to_structure_list decorator to convert the output result of a
        function to a list of autoarray structure when a `GridIrregularGrouped` instance is passed to the decorated function.

        Parameters
        ----------
        result_list : np.ndarray or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        if len(result_list[0].shape) == 1:
            return [self.values_from_arr_1d(arr_1d=value) for value in result_list]
        elif len(result_list[0].shape) == 2:
            return [self.grid_from_grid_1d(grid_1d=value) for value in result_list]

    def squared_distances_from_coordinate(self, coordinate=(0.0, 0.0)):
        """
        Returns the squared distance of every (y,x) coordinate in this *Coordinate* instance from an input
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
        """
        Returns the distance of every (y,x) coordinate in this *Coordinate* instance from an input coordinate.

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

    @classmethod
    def load(cls, file_path, filename):
        with open(path.join(file_path, f"{filename}.pickle"), "rb") as f:
            return pickle.load(f)

    def save(self, file_path, filename):
        """
        Save the tracer by serializing it with pickle.
        """
        with open(path.join(file_path, f"{filename}.pickle"), "wb") as f:
            pickle.dump(self, f)


class GridIrregular(AbstractGridIrregular):
    def __new__(cls, grid):
        """
        An irregular grid of (y,x) coordinates.

        The `GridIrregular` stores the (y,x) irregular grid of coordinates as 2D NumPy array of shape
        [total_coordinates, 2].

        Calculations should use the NumPy array structure wherever possible for efficient calculations.

        The coordinates input to this function can have any of the following forms (they will be converted to the
        1D NumPy array structure and can be converted back using the object's properties):

        [(y0,x0), (y1,x1)]
        [[y0,x0], [y1,x1]]

        If your grid lies on a 2D uniform grid of data the `Grid` data structure should be used.

        Parameters
        ----------
        grid : grids.GridIrregular
            The irregular grid of (y,x) coordinates.
        """

        if len(grid) == 0:
            return []

        if isinstance(grid, dict):
            coordinates_dict = grid
            grid = [value for value in grid.values()]
        else:
            coordinates_dict = None

        if type(grid) is list:
            grid = np.asarray(grid)

        obj = grid.view(cls)

        if coordinates_dict is not None:
            obj.as_dict = coordinates_dict

        return obj

    def values_from_arr_1d(self, arr_1d):
        """Create a *Values* object from a 1D NumPy array of values of shape [total_coordinates]. The
        *Values* are structured and grouped following this *Coordinate* instance."""
        return arrays.Values(values=arr_1d)

    def grid_from_grid_1d(self, grid_1d):
        """Create a `GridIrregularGrouped` object from a 2D NumPy array of values of shape [total_coordinates, 2]. The
        `GridIrregularGrouped` are structured and grouped following this *Coordinate* instance."""
        return GridIrregular(grid=grid_1d)

    def grid_from_deflection_grid(self, deflection_grid):
        """
        Returns a new GridIrregularGrouped from this grid coordinates, where the (y,x) coordinates of this grid have a
            grid of (y,x) values, termed the deflection grid, subtracted from them to determine the new grid of (y,x)
            values.

            This is used by PyAutoLens to perform grid ray-tracing.

            Parameters
            ----------
            deflection_grid : np.ndarray
                The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return GridIrregular(grid=self - deflection_grid)


class GridIrregularGrouped(AbstractGridIrregular):
    def __new__(cls, grid):
        """
        An irregular grid of (y,x) coordinates structured in a way defining groups of coordinates which share a common
        origin (for example coordinates may be grouped if they are from a specific region of a dataset).

        Grouping is structured as follows:

        [[(y0,x0), (y1,x1)], [(y2,x2), (y3,x3), (y3,x3)]]

        Here, we have two groups of coordinates, where each group is associated.

        The `GridIrregularGrouped` stores the (y,x) irregular grid of coordinates as 2D NumPy array of shape
        [total_coordinates, 2]. Index information is stored so that this array can be mapped to the list of list of
        tuples structure above.

        Calculations should use the NumPy array structure wherever possible for efficient calculations.

        The coordinates input to this function can have any of the following forms (they will be converted to the
        1D NumPy array structure and can be converted back using the object's properties):

        [[(y0,x0), (y1,x1)], [(y2,x2)]]
        [[[y0,x0], [y1,x1]], [[y2,x2)]]
        [(y0,x0), (y1,x1)]
        [[y0,x0], [y1,x1]]

        In all cases, they will be converted to a list of list of tuples followed by a 2D NumPy array.

        Parameters
        ----------
        grid : [[tuple]] or equivalent
            An irregular grid of (y,x) coordinates that are grouped if they share a common property.
        """

        if isinstance(grid, dict):
            coordinates_dict = grid
            grid = [value for value in grid.values()]
        else:
            coordinates_dict = None

        if len(grid) == 0:
            return []

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

        if coordinates_dict is not None:
            obj.as_dict = coordinates_dict

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "_internal_list"):
            self._internal_list = obj._internal_list

    @property
    def upper_indexes(self):
        upper_indexes = []

        a = 0

        for coords in self._internal_list:
            a += len(coords)
            upper_indexes.append(a)

        return upper_indexes

    @property
    def lower_indexes(self):
        return [0] + self.upper_indexes[:-1]

    @property
    def in_grouped_list(self):
        """Return the coordinates on a structured list which groups coordinates with a common origin."""
        return [
            list(map(tuple, self[i:j, :]))
            for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]

    @classmethod
    def from_yx_1d(cls, y, x):
        """Create `GridIrregularGrouped` from a list of y and x values.

        This function omits coordinate grouping."""
        return GridIrregularGrouped(grid=np.stack((y, x), axis=-1))

    @classmethod
    def from_pixels_and_mask(cls, pixels, mask):
        """Create `GridIrregularGrouped` from a list of coordinates in pixel units and a mask which allows these coordinates to
        be converted to scaled units."""
        grouped_grids = []
        for grouped_grid in pixels:
            grouped_grids.append(
                [
                    mask.geometry.scaled_coordinates_2d_from(
                        pixel_coordinates_2d=coordinates
                    )
                    for coordinates in grouped_grid
                ]
            )
        return cls(grid=grouped_grids)

    @classmethod
    def from_file(cls, file_path):
        """Create a `GridIrregularGrouped` object from a file which stores the coordinates as a list of list of tuples.

        Parameters
        ----------
        file_path : str
            The path to the coordinates .dat file containing the coordinates (e.g. '/path/to/coordinates.dat')
        """

        with open(file_path) as f:
            grid_string = f.readlines()

        grouped_grids = []

        for line in grid_string:
            coordinate_list = ast.literal_eval(line)
            grouped_grids.append(coordinate_list)

        return cls(grid=grouped_grids)

    def output_to_file(self, file_path, overwrite=False):
        """Output this instance of the `GridIrregularGrouped` object to a list of list of tuples.

        Parameters
        ----------
        file_path : str
            The path to the coordinates .dat file containing the coordinates (e.g. '/path/to/coordinates.dat')
        overwrite : bool
            If there is as exsiting file it will be overwritten if this is `True`.
        """

        if overwrite and path.exists(file_path):
            os.remove(file_path)
        elif not overwrite and path.exists(file_path):
            raise FileExistsError(
                "The file ",
                file_path,
                " already exists. Set overwrite=True to overwrite this" "file",
            )

        with open(file_path, "w") as f:
            for coordinate in self.in_grouped_list:
                f.write(f"{coordinate}\n")

    def values_from_arr_1d(self, arr_1d):
        """Create a *Values* object from a 1D NumPy array of values of shape [total_coordinates]. The
        *Values* are structured and grouped following this *Coordinate* instance."""
        values_1d = [
            list(arr_1d[i:j]) for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]
        return arrays.Values(values=values_1d)

    def grid_from_grid_1d(self, grid_1d):
        """Create a `GridIrregularGrouped` object from a 2D NumPy array of values of shape [total_coordinates, 2]. The
        `GridIrregularGrouped` are structured and grouped following this *Coordinate* instance."""
        grid_1d = [
            list(map(tuple, grid_1d[i:j, :]))
            for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]

        return GridIrregularGrouped(grid=grid_1d)

    def grid_from_deflection_grid(self, deflection_grid):
        """
        Returns a new GridIrregularGrouped from this grid coordinates, where the (y,x) coordinates of this grid have a
            grid of (y,x) values, termed the deflection grid, subtracted from them to determine the new grid of (y,x)
            values.

            This is used by PyAutoLens to perform grid ray-tracing.

            Parameters
            ----------
            deflection_grid : np.ndarray
                The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return GridIrregularGrouped(grid=self - deflection_grid)


class GridIrregularGroupedUniform(GridIrregularGrouped):
    def __new__(cls, grid, shape_2d=None, pixel_scales=None):
        """A collection of (y,x) coordinates structured in a way defining groups of coordinates which share a common
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

        Like the `Grid` structure, `GridIrregularGroupedUniform` lie on a uniform grid corresponding to values that
        originate from a uniform grid. This contrasts the `GridIrregularGrouped` class above. However, although this class
        stores the pixel-scale and 2D shape of this grid, it does not store the mask that a `Grid` does that enables
        the coordinates to be mapped from 1D to 2D. This is for calculations that utilize the 2d information of the
        grid but do not want the memory overheads associated with the 2D mask.

        Parameters
        ----------
        grid : [[tuple]] or equivalent
            A collection of (y,x) coordinates that are grouped if they correpsond to a shared origin.
        """

        #    obj = super(GridIrregularGroupedUniform, cls).__new__(cls=cls, coordinates=coordinates)

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

        obj.shape_2d = shape_2d
        obj.pixel_scales = pixel_scales

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "_internal_list"):
            self._internal_list = obj._internal_list

        if hasattr(obj, "shape_2d"):
            self.shape_2d = obj.shape_2d

        if hasattr(obj, "pixel_scales"):
            self.pixel_scales = obj.pixel_scales

    @property
    def pixel_scale(self):
        if self.pixel_scales[0] == self.pixel_scales[1]:
            return self.pixel_scales[0]
        else:
            raise exc.GridException(
                "Cannot return a pixel_scale for a grid where each dimension has a "
                "different pixel scale (e.g. pixel_scales[0] != pixel_scales[1]"
            )

    @classmethod
    def from_grid_sparse_uniform_upscale(
        cls, grid_sparse_uniform, upscale_factor, pixel_scales, shape_2d=None
    ):

        pixel_scales = geometry_util.convert_pixel_scales_2d(pixel_scales=pixel_scales)

        grid_upscaled_1d = grid_util.grid_upscaled_1d_from(
            grid_1d=grid_sparse_uniform,
            upscale_factor=upscale_factor,
            pixel_scales=pixel_scales,
        )

        pixel_scales = (
            pixel_scales[0] / upscale_factor,
            pixel_scales[1] / upscale_factor,
        )

        return cls(grid=grid_upscaled_1d, pixel_scales=pixel_scales, shape_2d=shape_2d)

    def grid_from_grid_1d(self, grid_1d):
        """Create a `GridIrregularGroupedUniform` object from a 2D NumPy array of values of shape [total_coordinates, 2]. The
        `GridIrregularGroupedUniform` are structured and grouped following this *Coordinate* instance."""
        grouped_grid_1d = [
            list(map(tuple, grid_1d[i:j, :]))
            for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]

        return GridIrregularGroupedUniform(
            grid=grouped_grid_1d, pixel_scales=self.pixel_scales, shape_2d=self.shape_2d
        )

    def grid_from_deflection_grid(self, deflection_grid):
        """
        Returns a new GridIrregularGrouped from this grid coordinates, where the (y,x) coordinates of this grid have a
            grid of (y,x) values, termed the deflection grid, subtracted from them to determine the new grid of (y,x)
            values.

            This is used by PyAutoLens to perform grid ray-tracing.

            Parameters
            ----------
            deflection_grid : np.ndarray
                The grid of (y,x) coordinates which is subtracted from this grid.
        """
        return GridIrregularGroupedUniform(
            grid=self - deflection_grid,
            pixel_scales=self.pixel_scales,
            shape_2d=self.shape_2d,
        )
