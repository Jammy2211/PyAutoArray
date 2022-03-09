from abc import ABC
from abc import abstractmethod
import numpy as np
from typing import Tuple, Union

from autoarray.abstract_ndarray import AbstractNDArray


class Structure(AbstractNDArray, ABC):
    def __array_finalize__(self, obj):

        if hasattr(obj, "mask"):
            self.mask = obj.mask

        if hasattr(obj, "header"):
            self.header = obj.header

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
