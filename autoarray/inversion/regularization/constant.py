import numpy as np
from typing import Tuple

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util

from autofit import exc


class Constant(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0, if_interpolated: bool = False):
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
        coefficient : (float,)
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        """
        self.coefficient = coefficient
        self.if_interpolated = if_interpolated

        super(Constant, self).__init__()

    def regularization_weights_from(self, mapper) -> np.ndarray:
        return self.coefficient * np.ones(mapper.pixels)

    def regularization_matrix_from(self, mapper, if_splitted=False):

        if self.if_interpolated is True:

            (
                splitted_mappings,
                splitted_sizes,
                splitted_weights,
            ) = mapper.splitted_pixelization_mappings_sizes_and_weights

            max_j = np.shape(splitted_weights)[1] - 1
            # The maximum neighbor index for a grid, if the maximum number of neighbors is 100, then it should be 99.

            # Usually, there should not be such a case where a grid can have over 100 (sometimes even 200) neighbours,
            # but when running the codes, this indeed happens.
            # From my experience, it only happens in the initializing phase (generating initializing points), when very weird lens mass
            # distribution can be considered. So, when coming across such extreme cases, we throw out a "exc.FitException" error to do a resample.
            # We should keep an eye on this.

            splitted_weights *= -1.0

            for i in range(len(splitted_mappings)):
                pixel_index = i // 4
                flag = 0
                for j in range(splitted_sizes[i]):
                    if splitted_mappings[i][j] == pixel_index:
                        splitted_weights[i][j] += 1.0
                        flag = 1

                if j >= max_j:
                    raise exc.FitException("neighbours exceeds!")

                if flag == 0:
                    splitted_mappings[i][j + 1] = pixel_index
                    splitted_sizes[i] += 1
                    splitted_weights[i][j + 1] = 1.0

            return (
                regularization_util.constant_pixel_splitted_regularization_matrix_from(
                    coefficient=self.coefficient,
                    splitted_mappings=splitted_mappings,
                    splitted_sizes=splitted_sizes,
                    splitted_weights=splitted_weights,
                )
            )

        else:
            return regularization_util.constant_regularization_matrix_from(
                coefficient=self.coefficient,
                pixel_neighbors=mapper.source_pixelization_grid.pixel_neighbors,
                pixel_neighbors_sizes=mapper.source_pixelization_grid.pixel_neighbors.sizes,
            )

