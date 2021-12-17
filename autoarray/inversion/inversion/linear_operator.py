import numpy as np
import pylops
from scipy import sparse

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.inversion.inversion.abstract import AbstractInversion


class InversionLinearOperator(AbstractInversion):
    @cached_property
    @profile_func
    def preconditioner_matrix(self):

        curvature_matrix_approx = np.multiply(
            np.sum(self.leq.noise_map.weight_list_ordered_1d),
            self.linear_obj_list[0].mapping_matrix.T
            @ self.linear_obj_list[0].mapping_matrix,
        )

        return np.add(curvature_matrix_approx, self.regularization_matrix)

    @cached_property
    @profile_func
    def preconditioner_matrix_inverse(self):
        return np.linalg.inv(self.preconditioner_matrix)

    @cached_property
    @profile_func
    def reconstruction(self):
        """
        Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
        of https://arxiv.org/pdf/astro-ph/0302587.pdf

        S is the vector of reconstructed inversion values.
        """

        Aop = pylops.MatrixMult(
            sparse.bsr_matrix(self.linear_obj_list[0].mapping_matrix)
        )

        Fop = self.leq.transformer

        Op = Fop * Aop

        MOp = pylops.MatrixMult(sparse.bsr_matrix(self.preconditioner_matrix_inverse))

        return pylops.NormalEquationsInversion(
            Op=Op,
            Regs=None,
            epsNRs=[1.0],
            data=self.data.ordered_1d,
            Weight=pylops.Diagonal(diag=self.leq.noise_map.weight_list_ordered_1d),
            NRegs=[pylops.MatrixMult(sparse.bsr_matrix(self.regularization_matrix))],
            M=MOp,
            tol=self.settings.tolerance,
            atol=self.settings.tolerance,
            **dict(maxiter=self.settings.maxiter),
        )

    @cached_property
    @profile_func
    def log_det_curvature_reg_matrix_term(self):
        return 2.0 * np.sum(
            np.log(np.diag(np.linalg.cholesky(self.preconditioner_matrix)))
        )

    @property
    def errors(self):
        return None
