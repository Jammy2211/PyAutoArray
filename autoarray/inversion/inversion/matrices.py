import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.inversion.linear_eqn.mapper.imaging import AbstractLinearEqnImaging
from autoarray.inversion.inversion.abstract import AbstractInversion

from autoarray import exc
from autoarray.inversion.linear_eqn import linear_eqn_util
from autoarray.inversion.inversion import inversion_util


class InversionMatrices(AbstractInversion):
    @cached_property
    def mapping_matrix(self) -> np.ndarray:
        """
        For a given pixelization pixel on the mapping matrix, we can use it to map it to a set of image-pixels in the
        image  plane. This therefore creates a 'image' of the source pixel (which corresponds to a set of values that
        mostly zeros, but with 1's where mappings occur).

        Before reconstructing the source, we blur every one of these source pixel images with the Point Spread Function
        of our  dataset via 2D convolution. This uses the methods
        in `Convolver.__init__` and `Convolver.convolve_mapping_matrix`:
        """
        return self.linear_eqn.mapping_matrix

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        """
        For a given pixelization pixel on the mapping matrix, we can use it to map it to a set of image-pixels in the
        image  plane. This therefore creates a 'image' of the source pixel (which corresponds to a set of values that
        mostly zeros, but with 1's where mappings occur).

        Before reconstructing the source, we blur every one of these source pixel images with the Point Spread Function
        of our  dataset via 2D convolution. This uses the methods
        in `Convolver.__init__` and `Convolver.convolve_mapping_matrix`:
        """

        if self.preloads.operated_mapping_matrix is not None:
            return self.preloads.operated_mapping_matrix

        return self.linear_eqn.operated_mapping_matrix

    @cached_property
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
        return self.linear_eqn.data_vector_from(data=self.data, preloads=self.preloads)

    @cached_property
    @profile_func
    def curvature_matrix(self) -> np.ndarray:

        if (
            self.preloads.curvature_matrix_preload is None
            or not self.settings.use_curvature_matrix_preload
        ):
            return self.linear_eqn.curvature_matrix

        return linear_eqn_util.curvature_matrix_via_sparse_preload_from(
            mapping_matrix=self.operated_mapping_matrix,
            noise_map=self.noise_map,
            curvature_matrix_preload=self.preloads.curvature_matrix_preload,
            curvature_matrix_counts=self.preloads.curvature_matrix_counts,
        )

    @cached_property
    @profile_func
    def curvature_reg_matrix(self):
        """
        The linear system of equations solves for F + regularization_coefficient*H, which is computed below.

        For a single mapper, this function overwrites the cached `curvature_matrix`, because for large matrices this
        avoids overheads in memory allocation. The `curvature_matrix` is removed as a cached property as a result,
        to ensure if we access it after computing the `curvature_reg_matrix` it is correctly recalculated in a new
        array of memory.
        """
        if self.has_one_mapper:

            curvature_reg_matrix = inversion_util.curvature_reg_matrix_from(
                curvature_matrix=self.curvature_matrix,
                regularization_matrix=self.regularization_matrix,
                pixel_neighbors=self.mapper_list[
                    0
                ].source_pixelization_grid.pixel_neighbors,
                pixel_neighbors_sizes=self.mapper_list[
                    0
                ].source_pixelization_grid.pixel_neighbors.sizes,
            )

            del self.__dict__["curvature_matrix"]

            return curvature_reg_matrix

        return np.add(self.curvature_matrix, self.regularization_matrix)

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

        return inversion_util.reconstruction_from(
            data_vector=self.data_vector,
            curvature_reg_matrix_cholesky=self.curvature_reg_matrix_cholesky,
            settings=self.settings,
        )

    @cached_property
    @profile_func
    def log_det_curvature_reg_matrix_term(self):
        """
        The log determinant of [F + reg_coeff*H] is used to determine the Bayesian evidence of the solution.

        This uses the Cholesky decomposition which is already computed before solving the reconstruction.
        """
        return 2.0 * np.sum(np.log(np.diag(self.curvature_reg_matrix_cholesky)))

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
        if self.preloads.log_det_regularization_matrix_term is not None:
            return self.preloads.log_det_regularization_matrix_term

        try:

            lu = splu(csc_matrix(self.regularization_matrix))
            diagL = lu.L.diagonal()
            diagU = lu.U.diagonal()
            diagL = diagL.astype(np.complex128)
            diagU = diagU.astype(np.complex128)

            return np.real(np.log(diagL).sum() + np.log(diagU).sum())

        except RuntimeError:

            try:
                return 2.0 * np.sum(
                    np.log(np.diag(np.linalg.cholesky(self.regularization_matrix)))
                )
            except np.linalg.LinAlgError:
                raise exc.InversionException()

    @property
    def errors_with_covariance(self):
        return np.linalg.inv(self.curvature_reg_matrix)

    @property
    def errors(self):
        return np.diagonal(self.errors_with_covariance)

    @property
    def curvature_matrix_preload(self) -> np.ndarray:
        (
            curvature_matrix_preload,
            curvature_matrix_counts,
        ) = linear_eqn_util.curvature_matrix_preload_from(
            mapping_matrix=self.operated_mapping_matrix
        )

        return curvature_matrix_preload

    @property
    def curvature_matrix_counts(self) -> np.ndarray:
        (
            curvature_matrix_preload,
            curvature_matrix_counts,
        ) = linear_eqn_util.curvature_matrix_preload_from(
            mapping_matrix=self.operated_mapping_matrix
        )

        return curvature_matrix_counts
