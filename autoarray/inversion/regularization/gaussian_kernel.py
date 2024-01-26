from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util

class GaussianKernel(AbstractRegularization):

    def __init__(self, coefficient: float = 1.0, scale: float = 1.0):

        self.coefficient = coefficient
        self.scale = scale
        super().__init__()

    def regularization_matrix_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        points: the position of mesh that is regularized
        """
        return regularization_util.regularization_matrix_gp_from(
            coefficient=self.coefficient,
            scale=self.scale,
            nu=None,
            points=linear_obj.source_plane_mesh_grid,
            reg_type="gauss",
        )