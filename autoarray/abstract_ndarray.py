from __future__ import annotations

from copy import copy

from abc import ABC
from abc import abstractmethod
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.structures.abstract_structure import Structure

from autoarray.structures.arrays import array_2d_util


def to_new_array(func):
    def wrapper(self, *args, **kwargs):
        new_array = copy(self)
        new_array._array = func(self, *args, **kwargs)
        return new_array

    return wrapper


class AbstractNDArray(ABC):
    def __init__(self, array):
        self._array = array

    def __reduce__(self):

        pickled_state = super().__reduce__()

        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)

        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        self._array.__setstate__(state[0:-1])

    @property
    @abstractmethod
    def native(self) -> Structure:
        """
        Returns the data structure in its `native` format which contains all unmaksed values to the native dimensions.
        """

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

    @property
    def shape(self):
        return self._array.shape

    @to_new_array
    def reshape(self, *args, **kwargs):
        return self._array.reshape(*args, **kwargs)

    def __getitem__(self, item):
        return self._array[item]
