from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util


class ConstantZeroth(AbstractRegularization):
    def __init__(self, coefficient_neighbor=1.0, coefficient_zeroth=1.0):
        super().__init__()

        self.coefficient_neighbor = coefficient_neighbor
        self.coefficient_zeroth = coefficient_zeroth

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
        return self.coefficient_neighbor * np.ones(linear_obj.params)

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
        return regularization_util.constant_zeroth_regularization_matrix_from(
            coefficient=self.coefficient_neighbor,
            coefficient_zeroth=self.coefficient_zeroth,
            neighbors=linear_obj.neighbors,
            neighbors_sizes=linear_obj.neighbors.sizes,
        )
