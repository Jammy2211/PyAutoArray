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
    def native(self) -> Structure:
        return self

    @property
    def in_list(self) -> List:
        """
        Return the values in a list.
        """
        return [value for value in self]
