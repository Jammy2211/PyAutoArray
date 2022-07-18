import numpy as np
from typing import List, Union

from autoarray.inversion.linear_obj.func_list import LinearObjFuncList
from autoarray.inversion.linear_eqn.imaging.abstract import AbstractLEqImaging
from autoarray.inversion.linear_eqn.abstract import AbstractLEq

from autoarray.inversion.mock.mock_mapper import MockMapper


class MockLinearObjFunc(LinearObjFuncList):
    def __init__(
        self, grid=None, mapping_matrix=None, operated_mapping_matrix_override=None
    ):

        super().__init__(grid=grid)

        self._mapping_matrix = mapping_matrix
        self._operated_mapping_matrix_override = operated_mapping_matrix_override

    @property
    def mapping_matrix(self) -> np.ndarray:
        return self._mapping_matrix

    @property
    def operated_mapping_matrix_override(self) -> np.ndarray:
        return self._operated_mapping_matrix_override
