import logging

import numpy as np
import os
import json
from os import path

from autoarray.structures import grids

logging.basicConfig()
logger = logging.getLogger(__name__)


class ValuesIrregular(np.ndarray):
    def __new__(cls, values):
        """
        A collection of values structured in a way defining groups of values which share a common origin (for
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

        In contrast to a *Array2D* structure, *ValuesIrregular* do not lie on a uniform grid or correspond to values that
        originate from a uniform grid. Therefore, when handling irregular data-sets *ValuesIrregular* should be used.

        Parameters
        ----------
        values : [[float]] or equivalent
            A collection of values that are grouped according to whether they share a common origin.
        """

        if len(values) == 0:
            return []

        if type(values) is list:
            values = np.asarray(values)

        obj = values.view(cls)

        return obj

    @property
    def slim(self):
        """The ValuesIrregular in their 1D representation, an ndarray of shape [total_values]."""
        return self

    @property
    def in_list(self):
        """Return the values on a structured list which groups values with a common origin."""
        return [value for value in self]

    @property
    def in_grouped_list(self):
        """Return the values on a structured list which groups values with a common origin."""
        return self.in_list

    def values_from_array_slim(self, array_slim):
        """Create a *ValuesIrregular* object from a 1D ndarray of values of shape [total_values].

        The *ValuesIrregular* are structured and grouped following this *ValuesIrregular* instance.

        Parameters
        ----------
        array_slim : np.ndarray
            The 1D array (shape [total_values]) of values that are mapped to a *ValuesIrregular* object."""
        return ValuesIrregular(values=array_slim)

    def grid_from_grid_slim(self, grid_slim):
        """Create a `Grid2DIrregular` object from a 2D ndarray array of values of shape [total_values, 2].

        The `Grid2DIrregular` are structured and grouped following this *Coordinate* instance.

        Parameters
        ----------
        grid_slim : np.ndarray
            The 2d array (shape [total_coordinates, 2]) of (y,x) coordinates that are mapped to a `Grid2DIrregular`
            object."""
        return grids.Grid2DIrregular(grid=grid_slim)

    @classmethod
    def from_file(cls, file_path):
        """Create a `Grid2DIrregular` object from a file which stores the coordinates as a list of list of tuples.

        Parameters
        ----------
        file_path : str
            The path to the coordinates .dat file containing the coordinates (e.g. '/path/to/coordinates.dat')
        """

        with open(file_path) as infile:
            values = json.load(infile)

        return ValuesIrregular(values=values)

    def output_to_json(self, file_path, overwrite=False):
        """Output this instance of the `Grid2DIrregular` object to a list of list of tuples.

        Parameters
        ----------
        file_path : str
            The path to the coordinates .dat file containing the coordinates (e.g. '/path/to/coordinates.dat')
        overwrite : bool
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
