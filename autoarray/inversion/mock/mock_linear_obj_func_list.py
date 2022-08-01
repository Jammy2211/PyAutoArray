import numpy as np

from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList


class MockLinearObjFuncList(AbstractLinearObjFuncList):
    def __init__(
        self, pixels=None, grid=None, mapping_matrix=None, regularization=None
    ):

        super().__init__(grid=grid, regularization=regularization)

        self._pixels = pixels
        self._mapping_matrix = mapping_matrix

    @property
    def pixels(self) -> int:
        return self._pixels

    @property
    def mapping_matrix(self) -> np.ndarray:
        return self._mapping_matrix
