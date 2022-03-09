from abc import ABC
from abc import abstractmethod
import numpy as np
from typing import Tuple, Union

from autoarray.abstract_ndarray import AbstractNDArray


class Structure(AbstractNDArray, ABC):
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

        super().__setstate__(state[0:-1])

    def __array_finalize__(self, obj):

        if hasattr(obj, "mask"):
            self.mask = obj.mask

        if hasattr(obj, "header"):
            self.header = obj.header

    def _new_structure(self, structure: "Structure", mask):
        """
        Conveninence method for creating a new instance of the Grid2D class from this grid.

        This method is over-written by other grids (e.g. Grid2DIterate) such that the slim and native methods return
        instances of that Grid2D's type.

        Parameters
        ----------
        data_structure : Structure
            The structure which is to be turned into a new structure.
        mask :Mask2D
            The mask associated with this structure.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def slim(self) -> "Structure":
        """
        Returns the data structure in its `slim` format which flattens all unmasked values to a 1D array.
        """

    @property
    @abstractmethod
    def native(self) -> "Structure":
        """
        Returns the data structure in its `native` format which contains all unmaksed values to the native dimensions.
        """

    @property
    def shape_slim(self) -> int:
        return self.mask.shape_slim

    @property
    def sub_shape_slim(self) -> int:
        return self.mask.sub_shape_slim

    @property
    def shape_native(self) -> Tuple[int, ...]:
        return self.mask.shape

    @property
    def sub_shape_native(self) -> Tuple[int, ...]:
        return self.mask.sub_shape_native

    @property
    def pixel_scales(self) -> Tuple[int, ...]:
        return self.mask.pixel_scales

    @property
    def pixel_scale(self) -> float:
        return self.mask.pixel_scale

    @property
    def origin(self) -> Tuple[int, ...]:
        return self.mask.origin

    @property
    def sub_size(self) -> int:
        return self.mask.sub_size

    @property
    def unmasked_grid(self) -> Union["Grid1D", "Grid2D"]:
        return self.mask.unmasked_grid_sub_1

    @property
    def total_pixels(self) -> int:
        return self.shape[0]

    def output_to_fits(self, file_path, overwrite):
        raise NotImplementedError


class Structure1D(Structure):

    pass


class Structure2D(Structure):
    def structure_2d_list_from(self, result_list: list):
        raise NotImplementedError

    def structure_2d_from(self, result: np.ndarray):
        raise NotImplementedError
