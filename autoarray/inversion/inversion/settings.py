from autoconf import conf
import logging
from typing import Optional

logging.basicConfig()
logger = logging.getLogger(__name__)


class SettingsInversion:
    def __init__(
        self,
        use_w_tilde: bool = True,
        use_positive_only_solver: bool = False,
        no_regularization_add_to_curvature_diag: bool = True,
        check_solution: Optional[bool] = None,
        use_w_tilde_numpy: bool = False,
        use_source_loop: bool = False,
        use_linear_operators: bool = False,
        tolerance: float = 1e-8,
        maxiter: int = 250,
    ):
        """
        The settings of an Inversion, customizing how a linear set of equations are solved for.

        An Inversion is used to reconstruct a dataset, for example the luminous emission of a galaxy.

        Parameters
        ----------
        use_w_tilde
            Whether to use the w-tilde formalism to perform the inversion, which speeds up the construction of the
            simultaneous linear equations (by bypassing the construction of a `mapping_matrix`) for many dataset
            use cases.
        use_positive_only_solver
            Whether to use a positive-only linear system solver, which requires that every reconstucted value is
            positive but is computationally much slower than the default solver (which allows for positive and
            negative values).
        no_regularization_add_to_curvature_diag
            When True, if a linear object in the inversion has no regularization, values of 1.0e-8 are added to the
            diagonal of its `curvature_matrix` to stablelize the linear algebra solver.
        check_solution
            If True, the `reconstruction` of the inversion is checked to ensure that no two source pixels have
            numerically identical values, which indicates a spurious solution where the linear algebra solver
            reconstructs every value as an identical value (for reasons that are currently not understood).
        use_w_tilde_numpy
            If True, the curvature_matrix is computed via numpy matrix multiplication (as opposed to numba functions
            which exploit sparsity to do the calculation normally in a more efficient way).
        use_source_loop
            Shhhh its a secret.
        use_linear_operators
            For an interferometer inversion, whether to use the linear operator solution to solve the linear system
            or not (this input does nothing for imaging data).
        tolerance
            For an interferometer inversion using the linear operators method, sets the tolerance of the solver
            (this input does nothing for imaging data and other interferometer methods).
        maxiter
            For an interferometer inversion using the linear operators method, sets the maximum number of iterations
            of the solver (this input does nothing for imaging data and other interferometer methods).
        """

        self.use_w_tilde = use_w_tilde
        self.use_positive_only_solver = use_positive_only_solver
        self.use_linear_operators = use_linear_operators
        self.no_regularization_add_to_curvature_diag = (
            no_regularization_add_to_curvature_diag
        )
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.use_w_tilde_numpy = use_w_tilde_numpy
        self.use_source_loop = use_source_loop

        self._check_solution = check_solution

    @property
    def check_solution(self):

        if self._check_solution is None:

            return conf.instance["general"]["inversion"]["check_solution_default"]

        return self._check_solution
