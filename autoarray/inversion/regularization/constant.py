import numpy as np

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util


class Constant(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0):
        """
        A constant regularization scheme (regularization is described in the `Regularization` class above) which
        uses a single value to apply smoothing on the solution of an `Inversion`.

        For this regularization scheme, there is only 1 regularization coefficient that is applied to
        all neighboring pixels. This means that the matrix B only needs to regularize pixels in one direction
        (e.g. pixel 0 regularizes pixel 1, but NOT visa versa). For example:

        B = [-1, 1]  [0->1]
            [0, -1]  1 does not regularization with 0

        A small numerical value of 1.0e-8 is added to all elements in constant regularization matrix, to ensure that
        it is positive definite.

        Parameters
        -----------
        coefficient
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        """

        self.coefficient = coefficient

        super().__init__()

    def regularization_weights_from(self, linear_obj) -> np.ndarray:
        return self.coefficient * np.ones(linear_obj.pixels)

    def regularization_matrix_from(self, linear_obj):

        return regularization_util.constant_regularization_matrix_from(
            coefficient=self.coefficient,
            neighbors=linear_obj.neighbors,
            neighbors_sizes=linear_obj.neighbors.sizes,
        )
