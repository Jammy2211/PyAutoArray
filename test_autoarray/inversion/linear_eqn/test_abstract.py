import numpy as np
from os import path
import pytest


import autoarray as aa
from autoarray.mock.mock import MockMapper, MockLinearEqn, MockInversion

directory = path.dirname(path.realpath(__file__))


class TestAbstractLinearEqn:
    def test__regularization_term(self):

        reconstruction = np.array([1.0, 1.0, 1.0])

        regularization_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        linear_eqn = MockLinearEqn(regularization_matrix=regularization_matrix)

        linear_eqn = MockInversion(linear_eqn=linear_eqn, reconstruction=reconstruction)

        # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

        # G_l = s_T * H * s

        # Matrix multiplication:

        # s_T * H = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [(1.0*1.0) + (1.0*0.0) + (1.0*0.0)] = [1.0, 1.0, 1.0]
        #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*1.0) + (1.0*0.0)]
        #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*0.0) + (1.0*1.0)]

        # (s_T * H) * s = [1.0, 1.0, 1.0] * [1.0] = 3.0
        #                                   [1.0]
        #                                   [1.0]

        assert linear_eqn.regularization_term == 3.0

        reconstruction = np.array([2.0, 3.0, 5.0])

        regularization_matrix = np.array(
            [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
        )

        linear_eqn = MockLinearEqn(regularization_matrix=regularization_matrix)

        linear_eqn = MockInversion(linear_eqn=linear_eqn, reconstruction=reconstruction)

        # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

        # G_l = s_T * H * s

        # Matrix multiplication:

        # s_T * H = [2.0, 3.0, 5.0] * [2.0,  -1.0,  0.0] = [(2.0* 2.0) + (3.0*-1.0) + (5.0 *0.0)] = [1.0, -1.0, 7.0]
        #                             [-1.0,  2.0, -1.0]   [(2.0*-1.0) + (3.0* 2.0) + (5.0*-1.0)]
        #                             [ 0.0, -1.0,  2.0]   [(2.0* 0.0) + (3.0*-1.0) + (5.0 *2.0)]

        # (s_T * H) * s = [1.0, -1.0, 7.0] * [2.0] = 34.0
        #                                    [3.0]
        #                                    [5.0]

        assert linear_eqn.regularization_term == 34.0

    def test__brightest_reconstruction_pixel_and_centre(self):

        matrix_shape = (9, 3)

        mapper = MockMapper(
            matrix_shape,
            source_pixelization_grid=aa.Grid2DVoronoi.manual_slim(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 0.0]]
            ),
        )

        inversion = MockLinearEqn(mapper=mapper)

        reconstruction = np.array([2.0, 3.0, 5.0, 0.0])

        linear_eqn = MockInversion(linear_eqn=inversion, reconstruction=reconstruction)

        assert linear_eqn.brightest_reconstruction_pixel == 2
        assert linear_eqn.brightest_reconstruction_pixel_centre.in_list == [(5.0, 6.0)]


class TestLogDetMatrixCholesky:
    def test__determinant_of_positive_definite_matrix_via_cholesky(self):

        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        linear_eqn = MockInversion(
            linear_eqn=MockLinearEqn(curvature_reg_matrix=matrix)
        )

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(
            linear_eqn.log_det_curvature_reg_matrix_term, 1e-4
        )

        matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

        linear_eqn = MockInversion(
            linear_eqn=MockLinearEqn(curvature_reg_matrix=matrix)
        )

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(
            linear_eqn.log_det_curvature_reg_matrix_term, 1e-4
        )

    def test__preload_of_log_det_regularizaation_term_overwrites_calculation(self):

        inversion = MockLinearEqn(
            preloads=aa.Preloads(log_det_regularization_matrix_term=1.0)
        )

        linear_eqn = MockInversion(linear_eqn=inversion)

        assert linear_eqn.log_det_regularization_matrix_term == 1.0
