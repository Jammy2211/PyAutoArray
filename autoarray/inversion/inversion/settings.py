import logging

logging.basicConfig()
logger = logging.getLogger(__name__)


class SettingsInversion:
    def __init__(
        self,
        use_w_tilde: bool = True,
        use_linear_operators: bool = False,
        linear_func_only_add_to_curvature_diag: bool = True,
        tolerance: float = 1e-8,
        maxiter: int = 250,
        check_solution: bool = True,
        use_curvature_matrix_preload: bool = True,
        use_w_tilde_numpy: bool = False,
    ):

        self.use_w_tilde = use_w_tilde
        self.use_linear_operators = use_linear_operators
        self.linear_func_only_add_to_curvature_diag = (
            linear_func_only_add_to_curvature_diag
        )
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.check_solution = check_solution
        self.use_curvature_matrix_preload = use_curvature_matrix_preload
        self.use_w_tilde_numpy = use_w_tilde_numpy
