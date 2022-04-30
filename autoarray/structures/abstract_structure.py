from abc import ABC
from abc import abstractmethod
import numpy as np
from typing import List, Tuple, Union

from autoarray.abstract_ndarray import AbstractNDArray

from autoarray.structures.arrays import array_2d_util

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
    def unmasked_grid(self) -> Union["Grid1D", "Grid2D"]:
        return self.mask.unmasked_grid_sub_1

    @property
    def total_pixels(self) -> int:
        return self.shape[0]

    def structure_2d_list_from(self, result_list: list) -> List["Structure"]:
        raise NotImplementedError

    def structure_2d_from(self, result: np.ndarray) -> "Structure":
        raise NotImplementedError

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
