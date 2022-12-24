from __future__ import annotations
from abc import ABC
from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from autoarray.structures.grids.uniform_1d import Grid1D
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.abstract_ndarray import AbstractNDArray

from autoarray import exc


class Structure(AbstractNDArray, ABC):
    def __array_finalize__(self, obj):

        if hasattr(obj, "mask"):
            self.mask = obj.mask

    @property
    @abstractmethod
    def slim(self) -> "Structure":
        """
        Returns the data structure in its `slim` format which flattens all unmasked values to a 1D array.
        """

    @property
    def geometry(self):
        return self.mask.geometry

    @property
    def indexes(self):
        return self.mask.indexes

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
    def pixel_scales(self) -> Tuple[float, ...]:
        return self.mask.pixel_scales

    @property
    def pixel_scale(self) -> float:
        return self.mask.pixel_scale

    @property
    def pixel_area(self):

        if len(self.pixel_scales) != 2:

            raise exc.GridException("Cannot compute area of structure which is not 2D.")

        return self.pixel_scales[0] * self.pixel_scales[1]

    @property
    def total_area(self):
        return self.total_pixels * self.pixel_area

    @property
    def origin(self) -> Tuple[int, ...]:
        return self.mask.origin

    @property
    def sub_size(self) -> int:
        return self.mask.sub_size

    @property
    def unmasked_grid(self) -> Union[Grid1D, Grid2D]:
        return self.mask.unmasked_grid_sub_1

    @property
    def total_pixels(self) -> int:
        return self.shape[0]

    def structure_2d_list_from(self, result_list: list) -> List["Structure"]:
        raise NotImplementedError

    def structure_2d_from(self, result: np.ndarray) -> "Structure":
        raise NotImplementedError

    def trimmed_after_convolution_from(self, kernel_shape) -> "Structure":
        raise NotImplementedError
