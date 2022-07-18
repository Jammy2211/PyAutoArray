from autoarray.inversion.regularization.to_reg_matrix import LinearObjToRegMatrix


class MockLinearObjToRegMatrix(LinearObjToRegMatrix):
    def __init__(
        self,
        regularization=None,
        linear_obj=None,
        pixels=None,
        regularization_matrix=None,
    ):

        super().__init__(regularization=regularization, linear_obj=linear_obj)

        self._pixels = pixels
        self._regularization_matrix = regularization_matrix

    @property
    def regularization_matrix(self):
        if self._regularization_matrix is None:
            return super().regularization_matrix

        return self._regularization_matrix

    @property
    def pixels(self):
        if self._pixels is None:
            return super().pixels

        return self._pixels
