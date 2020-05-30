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
