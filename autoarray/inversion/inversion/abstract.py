import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from typing import Dict, List, Optional, Union

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular
from autoarray.structures.visibilities import Visibilities
from autoarray.inversion.linear_eqn.imaging import AbstractLinearEqnImaging
from autoarray.inversion.linear_eqn.interferometer import (
    AbstractLinearEqnInterferometer,
)
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads

from autoarray import exc
from autoarray.inversion.inversion import inversion_util


class AbstractInversion:
    def __init__(
        self,
        data: Union[Visibilities, Array2D],
        linear_eqn: Union[AbstractLinearEqnImaging, AbstractLinearEqnInterferometer],
        regularization_list: [AbstractRegularization],
        settings: SettingsInversion = SettingsInversion(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):

        self.data = data

        self.linear_eqn = linear_eqn
        self.regularization_list = regularization_list

        self.settings = settings
        self.preloads = preloads
        self.profiling_dict = profiling_dict

    @property
    def has_one_mapper(self):
        if len(self.mapper_list) == 1:
            return True
        return False

    @property
    def noise_map(self):
        return self.linear_eqn.noise_map

    @property
    def mapper_list(self):
        return self.linear_eqn.mapper_list

    @cached_property
    @profile_func
    def regularization_matrix(self) -> np.ndarray:
        """
        The regularization matrix H is used to impose smoothness on our inversion's reconstruction. This enters the
        linear algebra system we solve for using D and F above and is given by
        equation (12) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        A complete description of regularization is given in the `regularization.py` and `regularization_util.py`
        modules.

        For multiple mappers, the regularization matrix is computed as the block diagonal of each individual mapper.
        The scipy function `block_diag` has an overhead associated with it and if there is only one mapper and
        regularization it is bypassed.
        """
        if self.preloads.regularization_matrix is not None:
            return self.preloads.regularization_matrix

        if self.has_one_mapper:
            return self.regularization_list[0].regularization_matrix_from(
                mapper=self.mapper_list[0]
            )

        return block_diag(
            *[
                reg.regularization_matrix_from(mapper=mapper)
                for (reg, mapper) in zip(self.regularization_list, self.mapper_list)
            ]
        )

    @cached_property
    @profile_func
    def reconstruction(self):
        raise NotImplementedError

    @property
    def reconstruction_of_mappers(self):
        return self.linear_eqn.source_quantity_of_mappers_from(
            source_quantity=self.reconstruction
        )

    @property
    def mapped_reconstructed_data_of_mappers(
        self
    ) -> List[Union[Array2D, Visibilities]]:
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
        return self.linear_eqn.mapped_reconstructed_data_of_mappers_from(
            reconstruction=self.reconstruction
        )

    @property
    def mapped_reconstructed_image_of_mappers(self) -> List[Array2D]:
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
        return self.linear_eqn.mapped_reconstructed_image_of_mappers_from(
            reconstruction=self.reconstruction
        )

    @cached_property
    @profile_func
    def mapped_reconstructed_data(self) -> Union[Array2D, Visibilities]:
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
        return sum(self.mapped_reconstructed_data_of_mappers)

    @cached_property
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
        return sum(self.mapped_reconstructed_image_of_mappers)

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
        raise NotImplementedError

    @property
    def errors(self):
        raise NotImplementedError

    @property
    def brightest_reconstruction_pixel_list(self):

        brightest_reconstruction_pixel_list = []

        for reconstruction in self.reconstruction_of_mappers:

            brightest_reconstruction_pixel_list.append(np.argmax(reconstruction))

        return brightest_reconstruction_pixel_list

    @property
    def brightest_reconstruction_pixel_centre_list(self):

        brightest_reconstruction_pixel_centre_list = []

        for mapper, reconstruction in zip(
            self.mapper_list, self.reconstruction_of_mappers
        ):

            brightest_reconstruction_pixel = np.argmax(reconstruction)

            centre = Grid2DIrregular(
                grid=[mapper.source_pixelization_grid[brightest_reconstruction_pixel]]
            )

            brightest_reconstruction_pixel_centre_list.append(centre)

        return brightest_reconstruction_pixel_centre_list

    @property
    def errors_of_mappers(self):
        return self.linear_eqn.source_quantity_of_mappers_from(
            source_quantity=self.errors
        )

    @property
    def regularization_weights_of_mappers(self):
        return [
            regularization.regularization_weights_from(mapper=self.mapper_list[0])
            for regularization in self.regularization_list
        ]

    @property
    def residual_map_of_mappers(self,) -> List[np.ndarray]:

        residual_map_of_mappers = []

        for mapper_index in range(self.linear_eqn.total_mappers):

            mapper = self.mapper_list[mapper_index]
            reconstruction_of_mapper = self.reconstruction_of_mappers[mapper_index]

            residual_map = inversion_util.inversion_residual_map_from(
                reconstruction=reconstruction_of_mapper,
                data=self.data,
                slim_index_for_sub_slim_index=mapper.source_grid_slim.mask.slim_index_for_sub_slim_index,
                all_sub_slim_indexes_for_pixelization_index=mapper.all_sub_slim_indexes_for_pixelization_index,
            )

            residual_map_of_mappers.append(residual_map)

        return residual_map_of_mappers

    @property
    def normalized_residual_map_of_mappers(self) -> List[np.ndarray]:

        normalized_map_of_mappers = []

        for mapper_index in range(self.linear_eqn.total_mappers):

            mapper = self.mapper_list[mapper_index]
            reconstruction_of_mapper = self.reconstruction_of_mappers[mapper_index]

            normalized_map = inversion_util.inversion_normalized_residual_map_from(
                reconstruction=reconstruction_of_mapper,
                data=self.data,
                noise_map_1d=self.noise_map.slim,
                slim_index_for_sub_slim_index=mapper.source_grid_slim.mask.slim_index_for_sub_slim_index,
                all_sub_slim_indexes_for_pixelization_index=mapper.all_sub_slim_indexes_for_pixelization_index,
            )

            normalized_map_of_mappers.append(normalized_map)

        return normalized_map_of_mappers

    @property
    def chi_squared_map_of_mappers(self) -> List[np.ndarray]:

        chi_squared_map_of_mappers = []

        for mapper_index in range(self.linear_eqn.total_mappers):

            mapper = self.mapper_list[mapper_index]
            reconstruction_of_mapper = self.reconstruction_of_mappers[mapper_index]

            chi_squared_map = inversion_util.inversion_chi_squared_map_from(
                reconstruction=reconstruction_of_mapper,
                data=self.data,
                noise_map_1d=self.noise_map.slim,
                slim_index_for_sub_slim_index=mapper.source_grid_slim.mask.slim_index_for_sub_slim_index,
                all_sub_slim_indexes_for_pixelization_index=mapper.all_sub_slim_indexes_for_pixelization_index,
            )

            chi_squared_map_of_mappers.append(chi_squared_map)

        return chi_squared_map_of_mappers

    @property
    def total_mappers(self):
        return len(self.mapper_list)
