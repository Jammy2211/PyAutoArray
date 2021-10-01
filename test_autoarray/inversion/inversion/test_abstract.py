import numpy as np
from os import path
import pytest

import autoarray as aa

from autoarray.mock.mock import (
    MockMapper,
    MockRegularization,
    MockLinearEqn,
    MockInversion,
)

from autoarray import exc


directory = path.dirname(path.realpath(__file__))


def test__regularization_matrix():

    linear_eqn_0 = MockLinearEqn(operated_mapping_matrix=np.ones((2, 2)))
    linear_eqn_1 = MockLinearEqn(operated_mapping_matrix=2.0 * np.ones((3, 3)))

    reg_0 = MockRegularization(regularization_matrix=np.ones((2, 2)))
    reg_1 = MockRegularization(regularization_matrix=2.0 * np.ones((3, 3)))

    inversion = MockInversion(
        regularization_list=[reg_0, reg_1], linear_eqn_list=[linear_eqn_0, linear_eqn_1]
    )

    regularization_matrix = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0, 2.0],
        ]
    )

    assert inversion.regularization_matrix == pytest.approx(regularization_matrix)


def test__preloads__operated_mapping_matrix_and_curvature_matrix_preload():

    operated_mapping_matrix = 2.0 * np.ones((9, 3))

    curvature_matrix_preload, curvature_matrix_counts = aa.util.linear_eqn.curvature_matrix_preload_from(
        mapping_matrix=operated_mapping_matrix
    )

    preloads = aa.Preloads(
        operated_mapping_matrix=operated_mapping_matrix,
        curvature_matrix_preload=curvature_matrix_preload.astype("int"),
        curvature_matrix_counts=curvature_matrix_counts.astype("int"),
    )

    # noinspection PyTypeChecker
    linear_eqn = MockLinearEqn(
        noise_map=np.ones(9),
        mapper=MockMapper(),
    )

    inversion = MockInversion(linear_eqn_list=[linear_eqn], preloads=preloads)

    assert inversion.operated_mapping_matrix[0, 0] == 2.0
    assert inversion._curvature_matrix_via_mapping[0, 0] == 36.0


def test__preload_of_regularization_matrix__overwrites_calculation():

    inversion = MockInversion(preloads=aa.Preloads(regularization_matrix=np.ones((2, 2))))

    assert (inversion.regularization_matrix == np.ones((2, 2))).all()


def test__reconstruction_of_mappers():

    reconstruction = np.ones(3)

    inversion = MockInversion(
        linear_eqn_list=[MockLinearEqn(mapper=MockMapper(pixels=3))],
        reconstruction=reconstruction,
    )

    assert (inversion.reconstruction_of_mappers[0] == np.ones(3)).all()

    reconstruction = np.array([1.0, 1.0, 2.0, 2.0, 2.0])

    inversion = MockInversion(
        linear_eqn_list=[
            MockLinearEqn(mapper=MockMapper(pixels=2)),
            MockLinearEqn(mapper=MockMapper(pixels=3)),
        ],
        reconstruction=reconstruction,
    )

    assert (inversion.reconstruction_of_mappers[0] == np.ones(2)).all()
    assert (inversion.reconstruction_of_mappers[1] == 2.0 * np.ones(3)).all()


def test__mapped_reconstructed_data():

    # noinspection PyTypeChecker
    inversion = MockInversion(
        linear_eqn_list=[MockLinearEqn(mapped_reconstructed_data=np.ones(3))],
        reconstruction_of_mappers=[None],
    )

    assert (inversion.mapped_reconstructed_data_of_mappers[0] == np.ones(3)).all()
    assert (inversion.mapped_reconstructed_data == np.ones(3)).all()

    # noinspection PyTypeChecker
    inversion = MockInversion(
        linear_eqn_list=[
            MockLinearEqn(mapped_reconstructed_data=np.ones(2)),
            MockLinearEqn(mapped_reconstructed_data=2.0 * np.ones(2)),
        ],
        reconstruction_of_mappers=[None, None],
    )

    assert (inversion.mapped_reconstructed_data_of_mappers[0] == np.ones(2)).all()
    assert (inversion.mapped_reconstructed_data_of_mappers[1] == 2.0 * np.ones(2)).all()
    assert (inversion.mapped_reconstructed_data == 3.0 * np.ones(2)).all()


def test__mapped_reconstructed_image():

    # noinspection PyTypeChecker
    inversion = MockInversion(
        linear_eqn_list=[MockLinearEqn(mapped_reconstructed_image=np.ones(3))],
        reconstruction_of_mappers=[None],
    )

    assert (inversion.mapped_reconstructed_image_of_mappers[0] == np.ones(3)).all()
    assert (inversion.mapped_reconstructed_image == np.ones(3)).all()

    # noinspection PyTypeChecker
    inversion = MockInversion(
        linear_eqn_list=[
            MockLinearEqn(mapped_reconstructed_image=np.ones(2)),
            MockLinearEqn(mapped_reconstructed_image=2.0 * np.ones(2)),
        ],
        reconstruction_of_mappers=[None, None],
    )

    assert (inversion.mapped_reconstructed_image_of_mappers[0] == np.ones(2)).all()
    assert (
        inversion.mapped_reconstructed_image_of_mappers[1] == 2.0 * np.ones(2)
    ).all()
    assert (inversion.mapped_reconstructed_image == 3.0 * np.ones(2)).all()


def test__reconstruction_raises_exception_for_linalg_error():

    with pytest.raises(exc.InversionException):

        # noinspection PyTypeChecker
        inversion = MockInversion(
            data_vector=np.ones(3), curvature_reg_matrix=np.ones((3, 3))
        )

        # noinspection PyStatementEffect
        inversion.reconstruction


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


def test__preload_of_log_det_regularization_term_overwrites_calculation():

    inversion = MockInversion( preloads=aa.Preloads(log_det_regularization_matrix_term=1.0))

    assert inversion.log_det_regularization_matrix_term == 1.0


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
