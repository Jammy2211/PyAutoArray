import numpy as np
from typing import Tuple

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util

from autofit import exc


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

    def regularization_weights_from(self, mapper) -> np.ndarray:
        return self.coefficient * np.ones(mapper.pixels)

    def regularization_matrix_from(self, mapper):

        return regularization_util.constant_regularization_matrix_from(
            coefficient=self.coefficient,
            pixel_neighbors=mapper.source_pixelization_grid.pixel_neighbors,
            pixel_neighbors_sizes=mapper.source_pixelization_grid.pixel_neighbors.sizes,
        )


class ConstantSplit(Constant):
    def __init__(self, coefficient: float = 1.0):
        """
        A constant regularization scheme which splits every source pixel into a cross of four regularization points
        (regularization is described in the `Regularization` class above) and interpolates to these points in order
        to apply smoothing on the solution of an `Inversion`.

        The size of this cross is determined via the size of the source-pixel, for example if the source pixel is a
        Voronoi pixel the area of the pixel is computed and the distance of each point of the cross is given by
        the area times 0.5.

        For this regularization scheme, there is only 1 regularization coefficient that is applied to
        all neighboring pixels. This means that the matrix B only needs to regularize pixels in one direction
        (e.g. pixel 0 regularizes pixel 1, but NOT visa versa). For example:

        B = [-1, 1]  [0->1]
            [0, -1]  1 does not regularization with 0

        Note that for this scheme the indexes of entries in the regularization matrix are not the source pixel indexes
        but the indexes of each source pixel index cross.

        A small numerical value of 1.0e-8 is added to all elements in constant regularization matrix, to ensure that
        it is positive definite.

        Parameters
        -----------
        coefficient
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        """
        self.coefficient = coefficient

        super().__init__()

    def regularization_matrix_from(self, mapper):

        (
            splitted_mappings,
            splitted_sizes,
            splitted_weights,
        ) = mapper.splitted_pixelization_mappings_sizes_and_weights

        (
            splitted_mappings,
            splitted_sizes,
            splitted_weights,
        ) = regularization_util.reg_split_from(
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
        )

        return regularization_util.constant_pixel_splitted_regularization_matrix_from(
            coefficient=self.coefficient,
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
        )
