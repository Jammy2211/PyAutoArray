from autoarray.structures import abstract_structure
from autoarray.structures.arrays.one_d import array_1d_util


class AbstractArray1D(abstract_structure.AbstractStructure1D):
    @property
    def native(self):

        return array_1d_util.array_1d_native_from(
            array_1d_slim=self, mask_1d=self.mask, sub_size=self.sub_size
        )

    @property
    def slim(self):
        return self
