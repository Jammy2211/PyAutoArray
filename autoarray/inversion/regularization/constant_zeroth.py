import numpy as np

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util


class ConstantZeroth(AbstractRegularization):
    def __init__(self, coefficient_neighbour=1.0, coefficient_zeroth=1.0):

        super().__init__()

        self.coefficient_neighbour = coefficient_neighbour
        self.coefficient_zeroth = coefficient_zeroth

    def regularization_weights_from(self, linear_obj) -> np.ndarray:
        return self.coefficient_neighbour * np.ones(linear_obj.pixels)

    def regularization_matrix_from(self, linear_obj):
        return regularization_util.constant_zeroth_regularization_matrix_from(
            coefficient=self.coefficient_neighbour,
            coefficient_zeroth=self.coefficient_zeroth,
            neighbors=linear_obj.neighbors,
            neighbors_sizes=linear_obj.neighbors.sizes,
        )
