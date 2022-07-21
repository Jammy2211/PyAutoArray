import numpy as np

from autoarray.inversion.linear_obj.func_list import LinearObjFuncList


class MockLinearObjFuncList(LinearObjFuncList):
    def __init__(self, pixels=None, grid=None):

        super().__init__(grid=grid)

        self._pixels = pixels

    @property
    def pixels(self) -> int:
        return self._pixels
