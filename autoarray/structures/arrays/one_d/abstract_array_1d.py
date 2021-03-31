from autoarray.structures import abstract_structure
from autoarray.structures.arrays.one_d import array_1d
from autoarray.structures.arrays.one_d import array_1d_util


class AbstractArray1D(abstract_structure.AbstractStructure1D):
    @property
    def slim(self):
        """
        Return an `Array1D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels * sub_size].

        If it is already stored in its `slim` representation  it is returned as it is. If not, it is  mapped from
        `native` to `slim` and returned as a new `Array1D`.
        """

        if self.shape[0] != self.mask.sub_shape_native[0]:
            return self

        array = array_1d_util.array_1d_slim_from(
            array_1d_native=self, mask_1d=self.mask, sub_size=self.mask.sub_size
        )

        return array_1d.Array1D(array=array, mask=self.mask)

    @property
    def native(self):
        """
        Return an `Array1D` where the data is stored in its `native` representation, which is an ndarray of shape
        [total_pixels * sub_size].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Array1D`.
        """

        if self.shape[0] == self.mask.sub_shape_native[0]:
            return self

        array = array_1d_util.array_1d_native_from(
            array_1d_slim=self, mask_1d=self.mask, sub_size=self.sub_size
        )

        return array_1d.Array1D(array=array, mask=self.mask)
