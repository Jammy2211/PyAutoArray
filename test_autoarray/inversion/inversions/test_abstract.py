from autoconf import conf
import autoarray as aa
from autoarray.inversion.inversion import abstract
from autoarray.inversion.inversion.imaging import InversionImagingMatrix
from autoarray.mock import mock
from autoarray.inversion import mappers
from autoarray.inversion import regularization as reg
from autoarray import exc

import numpy as np
from os import path
import pytest
from typing import Optional, Union

directory = path.dirname(path.realpath(__file__))


class MockInversionImagingMatrix(InversionImagingMatrix):
    def __init__(
        self,
        image: Optional[aa.Array2D] = None,
        noise_map: Optional[aa.Array2D] = None,
        convolver: Optional[aa.Convolver] = None,
        mapper: Optional[
            Union[mappers.MapperRectangular, mappers.MapperVoronoi, mock.MockMapper]
        ] = None,
        regularization: Optional[reg.Regularization] = None,
        curvature_matrix: Optional[np.ndarray] = None,
        regularization_matrix: Optional[np.ndarray] = None,
        curvature_reg_matrix: Optional[np.ndarray] = None,
        reconstruction: Optional[np.ndarray] = None,
        mapped_reconstructed_image: Optional[np.ndarray] = None,
        settings: Optional[aa.SettingsInversion] = None,
        preload_log_det_regularization_matrix_term=None,
    ):

        super().__init__(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            mapper=mapper,
            regularization=regularization,
            curvature_matrix=curvature_matrix,
            regularization_matrix=regularization_matrix,
            curvature_reg_matrix=curvature_reg_matrix,
            reconstruction=reconstruction,
            mapped_reconstructed_image=mapped_reconstructed_image,
            settings=settings,
            preload_log_det_regularization_matrix_term=preload_log_det_regularization_matrix_term,
        )


class TestAbstractInversion:
    def test__regularization_term(self):

        reconstruction = np.array([1.0, 1.0, 1.0])

        regularization_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        inversion = MockInversionImagingMatrix(
            reconstruction=reconstruction, regularization_matrix=regularization_matrix
        )

        # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

        # G_l = s_T * H * s

        # Matrix multiplication:

        # s_T * H = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [(1.0*1.0) + (1.0*0.0) + (1.0*0.0)] = [1.0, 1.0, 1.0]
        #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*1.0) + (1.0*0.0)]
        #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*0.0) + (1.0*1.0)]

        # (s_T * H) * s = [1.0, 1.0, 1.0] * [1.0] = 3.0
        #                                   [1.0]
        #                                   [1.0]

        assert inversion.regularization_term == 3.0

        reconstruction = np.array([2.0, 3.0, 5.0])

        regularization_matrix = np.array(
            [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
        )

        inversion = MockInversionImagingMatrix(
            reconstruction=reconstruction, regularization_matrix=regularization_matrix
        )

        # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

        # G_l = s_T * H * s

        # Matrix multiplication:

        # s_T * H = [2.0, 3.0, 5.0] * [2.0,  -1.0,  0.0] = [(2.0* 2.0) + (3.0*-1.0) + (5.0 *0.0)] = [1.0, -1.0, 7.0]
        #                             [-1.0,  2.0, -1.0]   [(2.0*-1.0) + (3.0* 2.0) + (5.0*-1.0)]
        #                             [ 0.0, -1.0,  2.0]   [(2.0* 0.0) + (3.0*-1.0) + (5.0 *2.0)]

        # (s_T * H) * s = [1.0, -1.0, 7.0] * [2.0] = 34.0
        #                                    [3.0]
        #                                    [5.0]

        assert inversion.regularization_term == 34.0

    def test__brightest_reconstruction_pixel_and_centre(self):

        matrix_shape = (9, 3)

        mapper = mock.MockMapper(
            matrix_shape,
            source_pixelization_grid=aa.Grid2DVoronoi.manual_slim(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 0.0]]
            ),
        )

        reconstruction = np.array([2.0, 3.0, 5.0, 0.0])

        inversion = MockInversionImagingMatrix(
            mapper=mapper, reconstruction=reconstruction
        )

        assert inversion.brightest_reconstruction_pixel == 2
        assert inversion.brightest_reconstruction_pixel_centre.in_list == [(5.0, 6.0)]

    def test__errors_and_errors_with_covariance(self,):

        curvature_reg_matrix = np.array(
            [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]]
        )

        inversion = MockInversionImagingMatrix(
            curvature_reg_matrix=curvature_reg_matrix
        )

        assert inversion.errors_with_covariance == pytest.approx(
            np.array([[2.5, -1.0, -0.5], [-1.0, 1.0, 0.0], [-0.5, 0.0, 0.5]]), 1.0e-2
        )
        assert inversion.errors == pytest.approx(np.array([2.5, 1.0, 0.5]), 1.0e-3)

    def test__interpolated_reconstruction_and_errors__config_is_image_grid(self,):

        conf.instance = conf.Config(
            path.join(directory, path.join("files", "inversion_image_grid")),
            path.join(directory, "output"),
        )

        matrix_shape = (25, 3)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, True, True],
                    [True, True, True, True, True],
                ]
            ),
            pixel_scales=1.0,
            sub_size=2,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        pixelization_grid = aa.Grid2D.uniform(
            shape_native=(3, 3), pixel_scales=1.0, sub_size=1
        )

        reconstruction = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        mapper = mock.MockMapper(
            matrix_shape=matrix_shape,
            source_grid_slim=grid,
            source_pixelization_grid=pixelization_grid,
        )

        curvature_reg_matrix = np.eye(N=9)

        inversion = MockInversionImagingMatrix(
            mapper=mapper,
            reconstruction=reconstruction,
            curvature_reg_matrix=curvature_reg_matrix,
        )

        interpolated_reconstruction = (
            inversion.interpolated_reconstructed_data_from_shape_native()
        )

        assert interpolated_reconstruction.shape_native == (5, 5)

        assert interpolated_reconstruction.slim == pytest.approx(
            np.ones(shape=(25,)), 1.0e-4
        )
        assert interpolated_reconstruction.native == pytest.approx(
            np.ones(shape=(5, 5)), 1.0e-4
        )
        assert interpolated_reconstruction.pixel_scales == pytest.approx(
            (1.0, 1.0), 1.0e-4
        )

        interpolated_errors = inversion.interpolated_errors_from_shape_native()

        assert interpolated_errors.shape_native == (5, 5)

        assert interpolated_errors.slim == pytest.approx(np.ones(shape=(25,)), 1.0e-4)
        assert interpolated_errors.native == pytest.approx(
            np.ones(shape=(5, 5)), 1.0e-4
        )
        assert interpolated_errors.pixel_scales == pytest.approx((1.0, 1.0), 1.0e-4)

    def test__interpolated_reconstruction__config_is_source_grid(self,):
        conf.instance = conf.Config(
            path.join(directory, path.join("files", "inversion_source_grid")),
            path.join(directory, "output"),
        )

        matrix_shape = (25, 3)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, True, True],
                    [True, True, True, True, True],
                ]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        pixelization_grid = aa.Grid2D.uniform(
            shape_native=(3, 3), pixel_scales=1.0, sub_size=1
        )

        mapper = mock.MockMapper(
            matrix_shape=matrix_shape,
            source_grid_slim=grid,
            source_pixelization_grid=pixelization_grid,
        )

        reconstruction = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        inversion = MockInversionImagingMatrix(
            mapper=mapper, reconstruction=reconstruction
        )

        interpolated_reconstruction = (
            inversion.interpolated_reconstructed_data_from_shape_native()
        )

        assert (
            interpolated_reconstruction.slim
            == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ).all()
        assert (
            interpolated_reconstruction.native
            == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()
        assert interpolated_reconstruction.pixel_scales == pytest.approx(
            (0.66666, 0.66666), 1.0e-4
        )

        interpolated_reconstruction = inversion.interpolated_reconstructed_data_from_shape_native(
            shape_native=(2, 2)
        )

        assert (
            interpolated_reconstruction.slim == np.array([1.0, 1.0, 1.0, 1.0])
        ).all()
        assert (
            interpolated_reconstruction.native == np.array([[1.0, 1.0], [1.0, 1.0]])
        ).all()
        assert interpolated_reconstruction.pixel_scales == pytest.approx(
            (1.0, 1.0), 1.0e-4
        )

        inversion.reconstruction = np.array(
            [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0]
        )

        interpolated_reconstruction = inversion.interpolated_reconstructed_data_from_shape_native(
            shape_native=(2, 2)
        )

        assert (
            interpolated_reconstruction.slim == np.array([3.0, 3.0, 3.0, 3.0])
        ).all()
        assert (
            interpolated_reconstruction.native == np.array([[3.0, 3.0], [3.0, 3.0]])
        ).all()
        assert interpolated_reconstruction.pixel_scales == (1.0, 1.0)

    def test__interp__manual_shape_native__uses_input_shape_native(self):

        matrix_shape = (25, 3)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, True, True],
                    [True, True, True, True, True],
                ]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        pixelization_grid = aa.Grid2D.uniform(
            shape_native=(3, 3), pixel_scales=1.0, sub_size=1
        )

        mapper = mock.MockMapper(
            matrix_shape=matrix_shape,
            source_grid_slim=grid,
            source_pixelization_grid=pixelization_grid,
        )

        reconstruction = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        inversion = MockInversionImagingMatrix(
            mapper=mapper, reconstruction=reconstruction
        )

        interpolated_reconstruction = inversion.interpolated_reconstructed_data_from_shape_native(
            shape_native=(2, 2)
        )

        assert (
            interpolated_reconstruction.slim == np.array([1.0, 1.0, 1.0, 1.0])
        ).all()
        assert (
            interpolated_reconstruction.native == np.array([[1.0, 1.0], [1.0, 1.0]])
        ).all()
        assert interpolated_reconstruction.pixel_scales == pytest.approx(
            (1.0, 1.0), 1.0e-4
        )

        inversion.reconstruction = np.array(
            [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0]
        )

        interpolated_reconstruction = inversion.interpolated_reconstructed_data_from_shape_native(
            shape_native=(2, 2)
        )

        assert (
            interpolated_reconstruction.slim == np.array([3.0, 3.0, 3.0, 3.0])
        ).all()
        assert (
            interpolated_reconstruction.native == np.array([[3.0, 3.0], [3.0, 3.0]])
        ).all()
        assert interpolated_reconstruction.pixel_scales == (1.0, 1.0)


class TestLogDetMatrixCholesky:
    def test__determinant_of_positive_definite_matrix_via_cholesky(self):

        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(
            abstract.log_determinant_of_matrix_cholesky(matrix), 1e-4
        )

    def test__determinant_of_positive_definite_matrix_2_via_cholesky(self):

        matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(
            abstract.log_determinant_of_matrix_cholesky(matrix), 1e-4
        )

    def test__matrix_not_positive_definite__raises_reconstruction_exception(self):

        matrix = np.array([[2.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 0.0]])

        with pytest.raises(exc.InversionException):
            assert pytest.approx(
                abstract.log_determinant_of_matrix_cholesky(matrix), 1e-4
            )

    def test__preload_of_log_det_regularizaation_term_overwrites_calculation(self):

        inversion = MockInversionImagingMatrix(
            preload_log_det_regularization_matrix_term=1.0
        )

        assert inversion.preload_log_det_regularization_matrix_term == 1.0
