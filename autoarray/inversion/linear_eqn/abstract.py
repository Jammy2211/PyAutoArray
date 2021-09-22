import numpy as np
from typing import Dict, Optional, Union

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.preloads import Preloads
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.inversion.mappers.rectangular import MapperRectangular
from autoarray.inversion.mappers.voronoi import MapperVoronoi
from autoarray.inversion.regularizations.abstract import AbstractRegularization

from autoarray import exc
from autoarray.inversion.inversion import inversion_util


class AbstractLinearEqn:
    def __init__(
        self,
        noise_map: Union[Array2D, VisibilitiesNoiseMap],
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: AbstractRegularization,
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):

        self.noise_map = noise_map
        self.mapper = mapper
        self.regularization = regularization

        self.preloads = preloads

        self.profiling_dict = profiling_dict

    @profile_func
    def data_vector_from(self, data):
        raise NotImplementedError

    @property
    def curvature_matrix(self) -> np.ndarray:
        raise NotImplementedError

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
        return self.regularization.regularization_matrix_from_mapper(mapper=self.mapper)

    @cached_property
    @profile_func
    def curvature_reg_matrix(self):
        """
        The linear system of equations solves for F + regularization_coefficient*H, which is computed below.

        This function overwrites the `curvature_matrix`, because for large matrices this avoids overhead. The
        `curvature_matrix` is not a cached property as a result, to ensure if we access it after computing the
        `curvature_reg_matrix` it is correctly recalculated in a new array of memory.
        """

        return inversion_util.curvature_reg_matrix_from(
            curvature_matrix=self.curvature_matrix,
            regularization_matrix=self.regularization_matrix,
            pixel_neighbors=self.mapper.source_pixelization_grid.pixel_neighbors,
            pixel_neighbors_sizes=self.mapper.source_pixelization_grid.pixel_neighbors.sizes,
        )

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

    @profile_func
    def mapped_reconstructed_image_from(self, reconstruction) -> Array2D:
        raise NotImplementedError

    @property
    def errors_with_covariance(self):
        return np.linalg.inv(self.curvature_reg_matrix)

    @property
    def errors(self):
        return np.diagonal(self.errors_with_covariance)

    @cached_property
    def preconditioner_matrix(self):
        raise NotImplementedError

    @cached_property
    def preconditioner_matrix_inverse(self):
        return np.linalg.inv(self.preconditioner_matrix)

    @property
    def regularization_weight_list(self):
        return self.regularization.regularization_weight_list_from_mapper(
            mapper=self.mapper
        )
