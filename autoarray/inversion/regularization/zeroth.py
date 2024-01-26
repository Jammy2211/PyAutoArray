from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util


class Zeroth(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0):
        """
        A zeroth order regularization scheme which zeroth order regularization to pixels with low expected
        signal values.

        Zeroth order regularization assumes a prior on the solution that its values should be closer to zero,
        penalizing solutions where they deviate further from zero. This is typically applied to prevent solutions
        from going to large positive and negative values that alternate.

        For this regularization scheme, there is only 1 regularization coefficient that is applied to
        all pixels by themselves (e.g. no neighboring scheme is used) For example:

        B = [1, 0]  0 -> 0
            [0, 1]  1 -> 1

        A small numerical value of 1.0e-8 is added to all elements in constant regularization matrix, to ensure that
        it is positive definite.

        A full description of regularization and this matrix can be found in the parent `AbstractRegularization` class.

        Parameters
        ----------
        coefficient
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        """

        self.coefficient = coefficient

        super().__init__()

    def regularization_weights_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.

        The regularization weights define the level of regularization applied to each parameter in the linear object
        (e.g. the ``pixels`` in a ``Mapper``).

        For standard regularization (e.g. ``Constant``) are weights are equal, however for adaptive schemes
        (e.g. ``AdaptiveBrightness``) they vary to adapt to the data being reconstructed.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses these weights when performing regularization.

        Returns
        -------
        The regularization weights.
        """
        return self.coefficient * np.ones(linear_obj.params)

    def regularization_matrix_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        Returns the regularization matrix with shape [pixels, pixels].

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses this matrix to perform regularization.

        Returns
        -------
        The regularization matrix.
        """
        return regularization_util.zeroth_regularization_matrix_from(
            coefficient=self.coefficient, pixels=linear_obj.params
        )
