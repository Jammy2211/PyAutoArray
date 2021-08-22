import logging

from autoconf import conf

logging.basicConfig()
logger = logging.getLogger(__name__)


class SettingsInversion:
    def __init__(
        self,
        use_w_tilde: bool = True,
        use_linear_operators: bool = False,
        tolerance: float = 1e-8,
        maxiter: int = 250,
        check_solution: bool = True,
    ):

        self.use_w_tilde = use_w_tilde
        self.use_linear_operators = use_linear_operators
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.check_solution = check_solution

        self._use_sksparse = None


    @property
    def use_sksparse(self):

        if self._use_sksparse is None:

            if not conf.instance["general"]["inversion"]["use_sksparse_if_installed"]:
                self._use_sksparse = False

                return self._use_sksparse

            try:

                from sksparse.cholmod import cholesky
                self._use_sksparse = True

            except ImportError:

                logger.info(
                    "Scikit-Sparse is not installed, therefore NumPy will be used for linear algebra calculation."
                    "Installing the optional requirement scikit-sparse via the command 'pip install scikit-sparse==0.4.'"
                    "will give a x3+ speed up."
                )

                self._use_sksparse = False

        return self._use_sksparse