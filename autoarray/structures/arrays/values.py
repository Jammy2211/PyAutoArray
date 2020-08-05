import ast
import logging

import numpy as np
import os

from autoarray.structures import grids

logging.basicConfig()
logger = logging.getLogger(__name__)


class Values(np.ndarray):
    def __new__(cls, values):
        """ A collection of values structured in a way defining groups of values which share a common origin (for
        example values may be grouped if they are from a specific region of a dataset).

        Grouping is structured as follows:

        [[value0, value1], [value0, value1, value2]]

        Here, we have two groups of values, where each group is associated with the other values.

        The values object does not store the values as a list of list of floats, but instead a 1D NumPy array
        of shape [total_values]. Index information is stored so that this array can be mapped to the list of
        list of float structure above. They are stored as a NumPy array so the values can be used efficiently for
        calculations.

        The values input to this function can have any of the following forms:

        [[value0, value1], [value0]]
        [[value0, value1]]

        In all cases, they will be converted to a list of list of floats followed by a 1D NumPy array.

        Print methods are overridden so a user always "sees" the values as the list structure.

        In contrast to a *Array* structure, *Values* do not lie on a uniform grid or correspond to values that
        originate from a uniform grid. Therefore, when handling irregular data-sets *Values* should be used.

        Parameters
        ----------
        values : [[float]] or equivalent
            A collection of values that are grouped according to whether they share a common origin.
        """

        if len(values) == 0:
            return []

        if isinstance(values, dict):
            values_dict = values
            values = [value for value in values.values()]
        else:
            values_dict = None

        if isinstance(values[0], float):
            values = [values]

        upper_indexes = []

        a = 0

        for value in values:
            a = a + len(value)
            upper_indexes.append(a)

        values_arr = np.concatenate([np.array(i) for i in values])

        obj = values_arr.view(cls)
        obj.upper_indexes = upper_indexes
        obj.lower_indexes = [0] + upper_indexes[:-1]

        if values_dict is not None:
            obj.as_dict = values_dict

        return obj

    def __array_finalize__(self, obj):

        if hasattr(obj, "lower_indexes"):
            self.lower_indexes = obj.lower_indexes

        if hasattr(obj, "upper_indexes"):
            self.upper_indexes = obj.upper_indexes

    @property
    def in_1d(self):
        """Convenience method to access the Values in their 1D representation, which is an ndarray of shape
        [total_values]."""
        return self

    @property
    def in_list(self):
        """Convenience method to access the Values in their list representation, whcih is a list of lists of floatss."""
        return [list(self[i:j]) for i, j in zip(self.lower_indexes, self.upper_indexes)]

    @property
    def in_1d_list(self):
        """Return the coordinates on a structured list which groups coordinates with a common origin."""
        return [value for value in self.in_1d]

    def values_from_arr_1d(self, arr_1d):
        """Create a *Values* object from a 1D ndarray of values of shape [total_values].

        The *Values* are structured and grouped following this *Values* instance.

        Parameters
        ----------
        arr_1d : ndarray
            The 1D array (shape [total_values]) of values that are mapped to a *Values* object."""
        values_1d = [
            list(arr_1d[i:j]) for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]
        return Values(values=values_1d)

    def coordinates_from_grid_1d(self, grid_1d):
        """Create a *GridCoordinates* object from a 2D ndarray array of values of shape [total_values, 2].

        The *GridCoordinates* are structured and grouped following this *Coordinate* instance.

        Parameters
        ----------
        grid_1d : ndarray
            The 2d array (shape [total_coordinates, 2]) of (y,x) coordinates that are mapped to a *GridCoordinates*
            object."""
        coordinates_1d = [
            list(map(tuple, grid_1d[i:j, :]))
            for i, j in zip(self.lower_indexes, self.upper_indexes)
        ]

        return grids.GridCoordinates(coordinates=coordinates_1d)

    @classmethod
    def from_file(cls, file_path):
        """Create a *Values* object from a file which stores the values as a list of list of floats.

        Parameters
        ----------
        file_path : str
            The path to the values .dat file containing the values (e.g. '/path/to/values.dat')
        """
        with open(file_path) as f:
            values_lines = f.readlines()

        values = []

        for line in values_lines:
            values_list = ast.literal_eval(line)
            values.append(values_list)

        return Values(values=values)

    def output_to_file(self, file_path, overwrite=False):
        """Output this instance of the *Values* object to a list of list of floats.

        Parameters
        ----------
        file_path : str
            The path to the values .dat file containing the values (e.g. '/path/to/values.dat')
        overwrite : bool
            If there is as exsiting file it will be overwritten if this is *True*.
        """

        if os.path.exists(file_path):
            if overwrite:
                os.remove(file_path)
            else:
                raise FileExistsError(
                    f"The file {file_path} already exists. Set overwrite=True to overwrite this"
                    "file"
                )

        with open(file_path, "w+") as f:
            for value in self.in_list:
                f.write("%s\n" % value)
