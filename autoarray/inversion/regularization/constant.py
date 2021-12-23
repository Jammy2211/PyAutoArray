import numpy as np

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util


class Constant(AbstractRegularization):
    def __init__(self, coefficient=1.0):
        """A instance-regularization scheme (regularization is described in the `Regularization` class above).

        For the instance regularization_matrix scheme, there is only 1 regularization coefficient that is applied to \
        all neighboring pixels. This means that we when write B, we only need to regularize pixels in one direction \
        (e.g. pixel 0 regularizes pixel 1, but NOT visa versa). For example:

        B = [-1, 1]  [0->1]
            [0, -1]  1 does not regularization with 0

        A small numerical value of 1.0e-8 is added to all elements in a instance regularization matrix, to ensure that \
        it is positive definite.

        Parameters
        -----------
        coefficient : (float,)
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        """
        self.coefficient = coefficient
        super(Constant, self).__init__()

    def regularization_weights_from(self, mapper) -> np.ndarray:
        return self.coefficient * np.ones(mapper.pixels)

    def regularization_matrix_from(self, mapper):
        return regularization_util.constant_regularization_matrix_from(
            coefficient=self.coefficient,
            pixel_neighbors=mapper.source_pixelization_grid.pixel_neighbors,
            pixel_neighbors_sizes=mapper.source_pixelization_grid.pixel_neighbors.sizes,
        )
