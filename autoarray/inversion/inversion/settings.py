import logging
from typing import Optional

from autoconf import conf

logging.basicConfig()
logger = logging.getLogger(__name__)


class SettingsInversion:
    def __init__(
        self,
        use_w_tilde: bool = True,
        use_positive_only_solver: Optional[bool] = None,
        positive_only_uses_p_initial: Optional[bool] = None,
        relocate_pix_border : bool = True,
        force_edge_pixels_to_zeros: bool = True,
        force_edge_image_pixels_to_zeros: bool = False,
        image_pixels_source_zero=None,
        no_regularization_add_to_curvature_diag_value: float = None,
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
            Whether to use a positive-only linear system solver, which requires that every reconstructed value is
            positive but is computationally much slower than the default solver (which allows for positive and
            negative values).
        relocate_pix_border
            If `True`, all coordinates of all pixelization source mesh grids have pixels outside their border
            relocated to their edge.
        no_regularization_add_to_curvature_diag_value
            If a linear func object does not have a corresponding regularization, this value is added to its
            diagonal entries of the curvature regularization matrix to ensure the matrix is positive-definite.
        use_w_tilde_numpy
            If True, the curvature_matrix is computed via numpy matrix multiplication (as opposed to numba functions
            which exploit sparsity to do the calculation normally in a more efficient way).
        use_source_loop
            Shhhh its a secret.
        use_linear_operators
            For an interferometer inversion, whether to use the linear operator solution to solve the linear system
            or not (this input does nothing for dataset data).
        tolerance
            For an interferometer inversion using the linear operators method, sets the tolerance of the solver
            (this input does nothing for dataset data and other interferometer methods).
        maxiter
            For an interferometer inversion using the linear operators method, sets the maximum number of iterations
            of the solver (this input does nothing for dataset data and other interferometer methods).
        """

        self.use_w_tilde = use_w_tilde
        self._use_positive_only_solver = use_positive_only_solver
        self._positive_only_uses_p_initial = positive_only_uses_p_initial
        self.relocate_pix_border = relocate_pix_border
        self.use_linear_operators = use_linear_operators
        self.force_edge_pixels_to_zeros = force_edge_pixels_to_zeros
        self.force_edge_image_pixels_to_zeros = force_edge_image_pixels_to_zeros
        self.image_pixels_source_zero = image_pixels_source_zero
        self._no_regularization_add_to_curvature_diag_value = (
            no_regularization_add_to_curvature_diag_value
        )
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.use_w_tilde_numpy = use_w_tilde_numpy
        self.use_source_loop = use_source_loop

    @property
    def use_positive_only_solver(self):
        if self._use_positive_only_solver is None:
            return conf.instance["general"]["inversion"]["use_positive_only_solver"]

        return self._use_positive_only_solver

    @property
    def positive_only_uses_p_initial(self):
        if self._positive_only_uses_p_initial is None:
            return conf.instance["general"]["inversion"]["positive_only_uses_p_initial"]

        return self._positive_only_uses_p_initial

    @property
    def no_regularization_add_to_curvature_diag_value(self):
        if self._no_regularization_add_to_curvature_diag_value is None:
            return conf.instance["general"]["inversion"][
                "no_regularization_add_to_curvature_diag_value"
            ]

        return self._no_regularization_add_to_curvature_diag_value
