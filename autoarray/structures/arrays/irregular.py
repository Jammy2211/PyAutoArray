from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import os
from os import path
from typing import List, Union

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.structures.abstract_structure import Structure

logging.basicConfig()
logger = logging.getLogger(__name__)


class ArrayIrregular(Structure):
    def structure_2d_list_from(self, result_list: list) -> List["Structure"]:
        raise NotImplementedError()

    def structure_2d_from(self, result: np.ndarray) -> "Structure":
        raise NotImplementedError()

    def trimmed_after_convolution_from(self, kernel_shape) -> "Structure":
        raise NotImplementedError()

    @property
    def native(self) -> Structure:
        raise NotImplementedError()

    def __init__(self, values: Union[List, np.ndarray]):
        """
        A collection of values which are structured as follows:

        [value0, value1, value3]

        The values object does not store the values as a list of floats, but instead a 1D NumPy array
        of shape [total_values]. This array can be mapped to the list of floats structure above. They are stored
        as a NumPy array so the values can be used efficiently for calculations.

        The values input to this function can have any of the following forms:

        [value0, value1]

        In all cases, they will be converted to a list of floats followed by a 1D NumPy array.

        Print methods are overridden so a user always "sees" the values as the list structure.

        In contrast to a `Array2D` structure, `ArrayIrregular` do not lie on a uniform grid or correspond to values
        that originate from a uniform grid. Therefore, when handling irregular data-sets `ArrayIrregular` should be
        used.

        Parameters
        ----------
        values : [float] or equivalent
            A collection of values.
        """

        # if len(values) == 0:
        #     return []

        # if isinstance(values, ArrayIrregular):
        #     return values

        if type(values) is list:
            values = np.asarray(values)

        super().__init__(values)

    @property
    def slim(self) -> "ArrayIrregular":
        """
        The ArrayIrregular in their `slim` representation, a 1D ndarray of shape [total_values].
        """
        return self

    @property
    def in_list(self) -> List:
        """
        Return the values in a list.
        """
        return [value for value in self]

    def values_from(self, array_slim: np.ndarray) -> "ArrayIrregular":
        """
        Create a `ArrayIrregular` object from a 1D ndarray of values of shape [total_values].

        The returned values have an identical structure to this `ArrayIrregular` instance.

        Parameters
        ----------
        array_slim
            The 1D ndarray with (hape [total_values] whose values are mapped to a `ArrayIrregular` object.
        """
        return ArrayIrregular(values=array_slim)

    def grid_from(self, grid_slim: np.ndarray) -> Grid2DIrregular:
        """
        Create a `Grid2DIrregular` object from a 2D ndarray array of values of shape [total_values, 2].

        The returned grid are structured following this `ArrayIrregular` instance.

        Parameters
        ----------
        grid_slim
            The 2d array (shape [total_coordinates, 2]) of (y,x) coordinates that are mapped to a `Grid2DIrregular`
            object.
        """
        from autoarray.structures.grids.irregular_2d import Grid2DIrregular

        return Grid2DIrregular(values=grid_slim)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> "ArrayIrregular":
        """
        Create a `ArrayIrregular` object from a  `.json` file which stores the coordinates as a list of list of tuples.

        Parameters
        ----------
        file_path
            The path to the coordinates .dat file containing the coordinates (e.g. '/path/to/coordinates.dat')
        """

        with open(file_path) as infile:
            values = json.load(infile)

        return ArrayIrregular(values=values)

    def output_to_json(self, file_path: Union[Path, str], overwrite: bool = False):
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
