import numpy as np

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util


class Zeroth(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0):
        """
        A zeroth order regularization scheme (regularization is described in the `Regularization` class above) which
        uses a single value to apply smoothing on the solution of an `Inversion`.

        Zeroth order regularization assumes a prior on the solution that its values should be closer to zero,
        penalizing solutions where they deviate further from zero. This is typically applied to prevent solutions
        from going to large positive and negative values that alternate.

        For this regularization scheme, there is only 1 regularization coefficient that is applied to
        all pixels by themselves (e.g. no neighboring scheme is used) For example:

        B = [1, 0]  0 -> 0
            [0, 1]  1 -> 1

        A small numerical value of 1.0e-8 is added to all elements in constant regularization matrix, to ensure that
        it is positive definite.

        Parameters
        -----------
        coefficient
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        """

        self.coefficient = coefficient

        super().__init__()

    def regularization_weights_from(self, linear_obj: "LinearObj") -> np.ndarray:
        return self.coefficient * np.ones(linear_obj.pixels)

    def regularization_matrix_from(self, linear_obj: "LinearObj") -> np.ndarray:

        return regularization_util.zeroth_regularization_matrix_from(
            coefficient=self.coefficient, pixels=linear_obj.pixels
        )
