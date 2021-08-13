import numpy as np
from scipy.interpolate import griddata
from typing import Union

from autoconf import conf

from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular
from autoarray.inversion.mappers import MapperRectangular
from autoarray.inversion.mappers import MapperVoronoi
from autoarray.inversion.regularization import Regularization
from autoarray.inversion.inversion.settings import SettingsInversion

from autoarray import exc


def log_determinant_of_matrix_cholesky(matrix):
    """There are two terms in the inversion's Bayesian log likelihood function which require the log determinant of \
    a matrix. These are (Nightingale & Dye 2015, Nightingale, Dye and Massey 2018):

    ln[det(F + H)] = ln[det(curvature_reg_matrix)]
    ln[det(H)]     = ln[det(regularization_matrix)]

    The curvature_reg_matrix is positive-definite, which means the above log determinants can be computed \
    efficiently (compared to using np.det) by using a Cholesky decomposition first and summing the log of each \
    diagonal term.

    Parameters
    -----------
    matrix
        The positive-definite matrix the log determinant is computed for.
    """
    try:
        return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(matrix))))
    except np.linalg.LinAlgError:
        raise exc.InversionException()


class AbstractInversion:
    def __init__(
        self,
        noise_map: np.ndarray,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: Regularization,
        regularization_matrix: np.ndarray,
        reconstruction: np.ndarray,
        settings: SettingsInversion,
    ):

        self.noise_map = noise_map
        self.mapper = mapper
        self.regularization = regularization
        self.regularization_matrix = regularization_matrix
        self.reconstruction = reconstruction
        self.settings = settings

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

    @property
    def regularization_term(self):
        """
        Returns the regularization term of an inversion. This term represents the sum of the difference in flux \
        between every pair of neighboring pixels. This is computed as:

        s_T * H * s = solution_vector.T * regularization_matrix * solution_vector

        The term is referred to as *G_l* in Warren & Dye 2003, Nightingale & Dye 2015.

        The above works include the regularization_matrix coefficient (lambda) in this calculation. In PyAutoLens, \
        this is already in the regularization matrix and thus implicitly included in the matrix multiplication.
        """
        return np.matmul(
            self.reconstruction.T,
            np.matmul(self.regularization_matrix, self.reconstruction),
        )

    @property
    def log_det_regularization_matrix_term(self):
        return log_determinant_of_matrix_cholesky(self.regularization_matrix)

    @property
    def brightest_reconstruction_pixel(self):
        return np.argmax(self.reconstruction)

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


class AbstractInversionMatrix:
    def __init__(
        self,
        curvature_reg_matrix: np.ndarray,
        curvature_matrix: np.ndarray,
        regularization_matrix: np.ndarray,
    ):

        self.curvature_matrix = curvature_matrix
        self.curvature_reg_matrix = curvature_reg_matrix
        self.regularization_matrix = regularization_matrix

    @property
    def log_det_curvature_reg_matrix_term(self):
        return log_determinant_of_matrix_cholesky(self.curvature_reg_matrix)

    @property
    def errors_with_covariance(self):
        return np.linalg.inv(self.curvature_reg_matrix)

    @property
    def errors(self):
        return np.diagonal(self.errors_with_covariance)
