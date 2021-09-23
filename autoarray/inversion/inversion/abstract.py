import numpy as np
import pylops
from scipy.sparse import csc_matrix
from scipy import sparse
from scipy.sparse.linalg import splu
from typing import Dict, Optional, Union

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.structures.visibilities import Visibilities
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular
from autoarray.inversion.linear_eqn.imaging import AbstractLinearEqnImaging
from autoarray.inversion.linear_eqn.interferometer import (
    AbstractLinearEqnInterferometer,
)
from autoarray.inversion.inversion.settings import SettingsInversion

from autoarray import exc
from autoarray.inversion.inversion import inversion_util


class AbstractInversion:
    def __init__(
        self,
        data: Union[Visibilities, Array2D],
        linear_eqn: Union[AbstractLinearEqnImaging, AbstractLinearEqnInterferometer],
        settings: SettingsInversion = SettingsInversion(),
        profiling_dict: Optional[Dict] = None,
    ):

        self.data = data

        self.linear_eqn = linear_eqn

        self.settings = settings

        self.profiling_dict = profiling_dict

    @property
    def noise_map(self):
        return self.linear_eqn.noise_map

    @property
    def mapper(self):
        return self.mapper_list[0]

    @property
    def mapper_list(self):
        return [eqn.mapper for eqn in [self.linear_eqn]]

    @property
    def regularization_list(self):
        return [eqn.regularization for eqn in [self.linear_eqn]]

    @cached_property
    @profile_func
    def data_vector(self) -> np.ndarray:
        """
        To solve for the source pixel fluxes we now pose the problem as a linear inversion which we use the NumPy
        linear  algebra libraries to solve. The linear algebra is based on
        the paper https://arxiv.org/pdf/astro-ph/0302587.pdf .

        This requires us to convert `w_tilde_data` into a data vector matrices of dimensions [image_pixels].

        The `data_vector` D is the first such matrix, which is given by equation (4)
        in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        The calculation is performed by the method `w_tilde_data_imaging_from`.
        """
        return self.linear_eqn.data_vector_from(data=self.data)

    @property
    @profile_func
    def curvature_matrix(self) -> np.ndarray:
        """
        The `curvature_matrix` F is the second matrix, given by equation (4)
        in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        This function computes F using the w_tilde formalism, which is faster as it precomputes the PSF convolution
        of different noise-map pixels (see `curvature_matrix_via_w_tilde_curvature_preload_imaging_from`).

        The `curvature_matrix` computed here is overwritten in memory when the regularization matrix is added to it,
        because for large matrices this avoids overhead. For this reason, `curvature_matrix` is not a cached property
        to ensure if we access it after computing the `curvature_reg_matrix` it is correctly recalculated in a new
        array of memory.
        """
        return self.linear_eqn.curvature_matrix

    @cached_property
    @profile_func
    def regularization_matrix(self) -> np.ndarray:
        """
        The regularization matrix H is used to impose smoothness on our inversion's reconstruction. This enters the
        linear algebra system we solve for using D and F above and is given by
        equation (12) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        A complete description of regularization is given in the `regularization.py` and `regularization_util.py`
        modules.
        """
        return self.linear_eqn.regularization_matrix

    @cached_property
    @profile_func
    def curvature_reg_matrix(self):
        """
        The linear system of equations solves for F + regularization_coefficient*H, which is computed below.

        This function overwrites the `curvature_matrix`, because for large matrices this avoids overhead. The
        `curvature_matrix` is not a cached property as a result, to ensure if we access it after computing the
        `curvature_reg_matrix` it is correctly recalculated in a new array of memory.
        """
        return self.linear_eqn.curvature_reg_matrix

    @cached_property
    @profile_func
    def curvature_reg_matrix_cholesky(self):
        """
        Performs a Cholesky decomposition of the `curvature_reg_matrix`, the result of which is used to solve the
        linear system of equations of the `LinearEqn`.

        The method `np.linalg.solve` is faster to do this, but the Cholesky decomposition is used later in the code
        to speed up the calculation of `log_det_curvature_reg_matrix_term`.
        """
        try:
            return np.linalg.cholesky(self.curvature_reg_matrix)
        except np.linalg.LinAlgError:
            raise exc.InversionException()

    @cached_property
    @profile_func
    def reconstruction(self):
        """
        Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
        of https://arxiv.org/pdf/astro-ph/0302587.pdf

        S is the vector of reconstructed inversion values.
        """

        try:
            return inversion_util.reconstruction_from(
                data_vector=self.data_vector,
                curvature_reg_matrix_cholesky=self.linear_eqn.curvature_reg_matrix_cholesky,
                settings=self.settings,
            )
        except NotImplementedError:
            Aop = pylops.MatrixMult(
                sparse.bsr_matrix(self.mapper_list[0].mapping_matrix)
            )

            Fop = self.linear_eqn.transformer

            Op = Fop * Aop

            MOp = pylops.MatrixMult(
                sparse.bsr_matrix(self.linear_eqn.preconditioner_matrix_inverse)
            )

            return pylops.NormalEquationsInversion(
                Op=Op,
                Regs=None,
                epsNRs=[1.0],
                data=self.data.ordered_1d,
                Weight=pylops.Diagonal(diag=self.noise_map.weight_list_ordered_1d),
                NRegs=[
                    pylops.MatrixMult(sparse.bsr_matrix(self.regularization_matrix))
                ],
                M=MOp,
                tol=self.settings.tolerance,
                atol=self.settings.tolerance,
                **dict(maxiter=self.settings.maxiter),
            )

    @cached_property
    @profile_func
    def mapped_reconstructed_image(self) -> Array2D:
        """
        Using the reconstructed source pixel fluxes we map each source pixel flux back to the image plane and
        reconstruct the image data.

        This uses the unique mappings of every source pixel to image pixels, which is a quantity that is already
        computed when using the w-tilde formalism.

        Returns
        -------
        Array2D
            The reconstructed image data which the inversion fits.
        """
        return self.linear_eqn.mapped_reconstructed_image_from(
            reconstruction=self.reconstruction
        )

    @cached_property
    @profile_func
    def regularization_term(self):
        """
        Returns the regularization term of an inversion. This term represents the sum of the difference in flux
        between every pair of neighboring pixels.

        This is computed as:

        s_T * H * s = solution_vector.T * regularization_matrix * solution_vector

        The term is referred to as *G_l* in Warren & Dye 2003, Nightingale & Dye 2015.

        The above works include the regularization_matrix coefficient (lambda) in this calculation. In PyAutoLens,
        this is already in the regularization matrix and thus implicitly included in the matrix multiplication.
        """
        return np.matmul(
            self.reconstruction.T,
            np.matmul(self.linear_eqn.regularization_matrix, self.reconstruction),
        )

    @cached_property
    @profile_func
    def log_det_curvature_reg_matrix_term(self):
        """
        The log determinant of [F + reg_coeff*H] is used to determine the Bayesian evidence of the solution.

        This uses the Cholesky decomposition which is already computed before solving the reconstruction.
        """
        return 2.0 * np.sum(
            np.log(np.diag(self.linear_eqn.curvature_reg_matrix_cholesky))
        )

    @cached_property
    @profile_func
    def log_det_regularization_matrix_term(self) -> float:
        """
        The Bayesian evidence of an inversion which quantifies its overall goodness-of-fit uses the log determinant
        of regularization matrix, Log[Det[Lambda*H]].

        Unlike the determinant of the curvature reg matrix, which uses an existing preloading Cholesky decomposition
        used for the source reconstruction, this uses scipy sparse linear algebra to solve the determinant efficiently.

        Returns
        -------
        float
            The log determinant of the regularization matrix.
        """
        if self.linear_eqn.preloads.log_det_regularization_matrix_term is not None:
            return self.linear_eqn.preloads.log_det_regularization_matrix_term

        try:

            lu = splu(csc_matrix(self.linear_eqn.regularization_matrix))
            diagL = lu.L.diagonal()
            diagU = lu.U.diagonal()
            diagL = diagL.astype(np.complex128)
            diagU = diagU.astype(np.complex128)

            return np.real(np.log(diagL).sum() + np.log(diagU).sum())

        except RuntimeError:

            try:
                return 2.0 * np.sum(
                    np.log(
                        np.diag(
                            np.linalg.cholesky(self.linear_eqn.regularization_matrix)
                        )
                    )
                )
            except np.linalg.LinAlgError:
                raise exc.InversionException()

    @property
    def brightest_reconstruction_pixel(self):
        return np.argmax(self.reconstruction)

    @property
    def brightest_reconstruction_pixel_centre(self):
        return Grid2DIrregular(
            grid=[
                self.linear_eqn.mapper.source_pixelization_grid[
                    self.brightest_reconstruction_pixel
                ]
            ]
        )

    @property
    def residual_map(self):
        raise NotImplementedError()

    @property
    def normalized_residual_map(self):
        raise NotImplementedError()

    @property
    def chi_squared_map(self):
        raise NotImplementedError()

    @property
    def errors(self):
        return self.linear_eqn.errors

    @property
    def regularization_weight_list(self):
        return self.linear_eqn.regularization_weight_list
