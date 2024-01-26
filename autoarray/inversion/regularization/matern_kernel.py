import numpy as np

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util


class MaternKernel(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0, scale: float = 1.0, nu: float = 0.5):
        self.coefficient = coefficient
        self.scale = float(scale)
        self.nu = float(nu)
        super().__init__()

    def regularization_matrix_from(self, points) -> np.ndarray:
        """
        points: the position of mesh that is regularized
        """
        return regularization_util.regularization_matrix_gp_from(
            coefficient=self.coefficient,
            scale=self.scale,
            nu=self.nu,
            points=points,
            reg_type="matern",
        )
