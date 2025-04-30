from autoarray.abstract_ndarray import AbstractNDArray
from autoarray.structures.arrays.uniform_2d import Array2D

class Array2DRGB(Array2D):

    def __init__(self, values, mask):

        array = values

        while isinstance(array, AbstractNDArray):
            array = array.array

        self._array = array
        self.mask = mask

    @property
    def native(self) -> "Array2D":
        """
        Return a `Array2D` where the data is stored in its `native` representation, which is an ``ndarray`` of shape
        [total_y_pixels, total_x_pixels].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Array2D`.
        """
        return self