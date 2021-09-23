import numpy as np
from os import path
import pytest

import autoarray as aa

from autoarray.mock.mock import MockLinearEqn, MockInversion

from autoarray import exc


directory = path.dirname(path.realpath(__file__))


def test__regularization_term():

    reconstruction = np.array([1.0, 1.0, 1.0])

    regularization_matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    inversion = MockInversion(
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

    inversion = MockInversion(
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


def test__preload_of_regularization_matrix__overwrites_calculation():

    linear_eqn = MockLinearEqn(
        preloads=aa.Preloads(regularization_matrix=np.ones((2, 2)))
    )

    inversion = MockInversion(linear_eqn=linear_eqn)

    assert (inversion.regularization_matrix == np.ones((2, 2))).all()


def test__reconstruction_raises_exception_for_linalg_error():

    with pytest.raises(exc.InversionException):

        # noinspection PyTypeChecker
        inversion = MockInversion(
            data_vector=np.ones(3), curvature_reg_matrix=np.ones((3, 3))
        )

        # noinspection PyStatementEffect
        inversion.reconstruction


def test__determinant_of_positive_definite_matrix_via_cholesky():

    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    linear_eqn = MockInversion(curvature_reg_matrix=matrix)

    log_determinant = np.log(np.linalg.det(matrix))

    assert log_determinant == pytest.approx(
        linear_eqn.log_det_curvature_reg_matrix_term, 1e-4
    )

    matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

    linear_eqn = MockInversion(curvature_reg_matrix=matrix)

    log_determinant = np.log(np.linalg.det(matrix))

    assert log_determinant == pytest.approx(
        linear_eqn.log_det_curvature_reg_matrix_term, 1e-4
    )


def test__preload_of_log_det_regularization_term_overwrites_calculation():

    linear_eqn = MockLinearEqn(
        preloads=aa.Preloads(log_det_regularization_matrix_term=1.0)
    )

    inversion = MockInversion(linear_eqn=linear_eqn)

    assert inversion.log_det_regularization_matrix_term == 1.0
