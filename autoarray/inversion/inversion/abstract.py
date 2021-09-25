import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from typing import Dict, List, Optional, Union

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.visibilities import Visibilities
from autoarray.inversion.linear_eqn.imaging import AbstractLinearEqnImaging
from autoarray.inversion.linear_eqn.interferometer import (
    AbstractLinearEqnInterferometer,
)
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.inversion.settings import SettingsInversion

from autoarray import exc


class AbstractInversion:
    def __init__(
        self,
        data: Union[Visibilities, Array2D],
        linear_eqn_list: List[Union[AbstractLinearEqnImaging, AbstractLinearEqnInterferometer]],
        regularization: AbstractRegularization,
        settings: SettingsInversion = SettingsInversion(),
        profiling_dict: Optional[Dict] = None,
    ):

        self.data = data

        self.linear_eqn_list = linear_eqn_list
        self.regularization = regularization

        self.settings = settings

        self.profiling_dict = profiling_dict

    @property
    def preloads(self):
        return self.linear_eqn_list[0].preloads

    @property
    def noise_map(self):
        return self.linear_eqn_list[0].noise_map

    @property
    def mapper_list(self):
        return [eqn.mapper for eqn in self.linear_eqn_list]

    @property
    def regularization_list(self):
        return [self.regularization]

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
        if self.preloads.regularization_matrix is not None:
            return self.preloads.regularization_matrix
        return self.regularization.regularization_matrix_from_mapper(mapper=self.mapper_list[0])

    @cached_property
    @profile_func
    def reconstruction(self):
        raise NotImplementedError

    @cached_property
    @profile_func
    def reconstructions_of_mappers_list(self):
        return [self.reconstruction]

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
        return self.linear_eqn_list[0].mapped_reconstructed_image_from(
            reconstruction=self.reconstruction
        )

    @cached_property
    @profile_func
    def mapped_reconstructed_visibilities(self) -> Array2D:
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
        return self.linear_eqn_list[0].mapped_reconstructed_visibilities_from(
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
            np.matmul(self.regularization_matrix, self.reconstruction),
        )

    @cached_property
    @profile_func
    def log_det_curvature_reg_matrix_term(self):
        """
        The log determinant of [F + reg_coeff*H] is used to determine the Bayesian evidence of the solution.

        This uses the Cholesky decomposition which is already computed before solving the reconstruction.
        """
        raise NotImplementedError

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
        if self.linear_eqn_list[0].preloads.log_det_regularization_matrix_term is not None:
            return self.linear_eqn_list[0].preloads.log_det_regularization_matrix_term

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
        raise NotImplementedError

    @property
    def errors(self):
        raise NotImplementedError

    @property
    def brightest_reconstruction_pixel_list(self):

        brightest_reconstruction_pixel_list = []

        for eqn, reconstruction in zip(
            self.linear_eqn_list, self.reconstructions_of_mappers_list
        ):

            brightest_reconstruction_pixel_list.append(
                eqn.brightest_reconstruction_pixel_from(reconstruction=reconstruction)
            )

        return brightest_reconstruction_pixel_list

    @property
    def brightest_reconstruction_pixel_centre_list(self):

        brightest_reconstruction_pixel_centre_list = []

        for eqn, reconstruction in zip(
            self.linear_eqn_list, self.reconstructions_of_mappers_list
        ):
            brightest_reconstruction_pixel_centre_list.append(
                eqn.brightest_reconstruction_pixel_centre_from(
                    reconstruction=reconstruction
                )
            )

        return brightest_reconstruction_pixel_centre_list

    @property
    def regularization_weight_list(self):
        return self.regularization.regularization_weights_from_mapper(
            mapper=self.mapper_list[0]
        )
