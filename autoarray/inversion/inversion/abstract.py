import numpy as np
from scipy.interpolate import griddata
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from typing import Dict, Optional, Union

from autoconf import cached_property
from autoconf import conf
from autoarray.numba_util import profile_func

from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.preloads import Preloads
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular
from autoarray.inversion.mappers import MapperRectangular
from autoarray.inversion.mappers import MapperVoronoi
from autoarray.inversion.regularization import Regularization
from autoarray.inversion.inversion.settings import SettingsInversion

from autoarray import exc
from autoarray.inversion.inversion import inversion_util


class AbstractInversion:
    def __init__(
        self,
        noise_map: Union[Array2D, VisibilitiesNoiseMap],
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: Regularization,
        settings: SettingsInversion = SettingsInversion(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):

        self.noise_map = noise_map
        self.mapper = mapper
        self.regularization = regularization

        self.settings = settings
        self.preloads = preloads

        self.profiling_dict = profiling_dict

    def interpolated_reconstructed_data_from_shape_native(self, shape_native=None):
        return self.interpolated_values_from_shape_native(
            values=self.reconstruction, shape_native=shape_native
        )

    def interpolated_errors_from_shape_native(self, shape_native=None):
        return self.interpolated_values_from_shape_native(
            values=self.errors, shape_native=shape_native
        )

    def interpolated_values_from_shape_native(self, values, shape_native=None):

        if shape_native is not None:

            grid = Grid2D.bounding_box(
                bounding_box=self.mapper.source_pixelization_grid.extent,
                shape_native=shape_native,
                buffer_around_corners=False,
            )

        elif (
            conf.instance["general"]["inversion"]["interpolated_grid_shape"]
            in "image_grid"
        ):

            grid = self.mapper.source_grid_slim

        elif (
            conf.instance["general"]["inversion"]["interpolated_grid_shape"]
            in "source_grid"
        ):

            dimension = int(np.sqrt(self.mapper.pixels))
            shape_native = (dimension, dimension)

            grid = Grid2D.bounding_box(
                bounding_box=self.mapper.source_pixelization_grid.extent,
                shape_native=shape_native,
                buffer_around_corners=False,
            )

        else:

            raise exc.InversionException(
                "In the genenal.ini config file a valid option was not found for the"
                "interpolated_grid_shape. Must be {image_grid, source_grid}"
            )

        interpolated_reconstruction = griddata(
            points=self.mapper.source_pixelization_grid,
            values=values,
            xi=grid.binned.native,
            method="linear",
        )

        interpolated_reconstruction[np.isnan(interpolated_reconstruction)] = 0.0

        return Array2D.manual(
            array=interpolated_reconstruction, pixel_scales=grid.pixel_scales
        )

    @cached_property
    def data_vector(self) -> np.ndarray:
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
        linear system of equations of the `Inversion`.

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

        lu = splu(csc_matrix(self.regularization_matrix))
        diagL = lu.L.diagonal()
        diagU = lu.U.diagonal()
        diagL = diagL.astype(np.complex128)
        diagU = diagU.astype(np.complex128)

        return np.real(np.log(diagL).sum() + np.log(diagU).sum())

    @property
    def brightest_reconstruction_pixel(self):
        return np.argmax(self.reconstruction)

    @property
    def errors_with_covariance(self):
        return np.linalg.inv(self.curvature_reg_matrix)

    @property
    def errors(self):
        return np.diagonal(self.errors_with_covariance)

    @property
    def brightest_reconstruction_pixel_centre(self):
        return Grid2DIrregular(
            grid=[
                self.mapper.source_pixelization_grid[
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
    def regularization_weight_list(self):
        return self.regularization.regularization_weight_list_from_mapper(
            mapper=self.mapper
        )
