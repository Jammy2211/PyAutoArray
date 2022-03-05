import numpy as np
from os import path
import pytest

import autoarray as aa

from autoarray.inversion.mappers.mock.mock_mapper import MockMapper
from autoarray.inversion.regularization.mock.mock_regularization import (
    MockRegularization,
)
from autoarray.inversion.linear_eqn.mock.mock_leq import MockLinearObjFunc
from autoarray.inversion.linear_eqn.mock.mock_leq import MockLEq
from autoarray.inversion.inversion.mock.mock_inversion import MockInversion

from autoarray import exc


directory = path.dirname(path.realpath(__file__))


def test__mapper_list__filters_other_objects():

    inversion = MockInversion(
        leq=MockLEq(
            linear_obj_list=[
                MockMapper(pixels=1),
                MockLinearObjFunc(),
                MockMapper(pixels=2),
            ]
        )
    )

    assert inversion.mapper_list[0].pixels == 1
    assert inversion.mapper_list[1].pixels == 2


def test__regularization_matrix():

    leq = MockLEq(linear_obj_list=[MockMapper(), MockMapper()])

    reg_0 = MockRegularization(regularization_matrix=np.ones((2, 2)))
    reg_1 = MockRegularization(regularization_matrix=2.0 * np.ones((3, 3)))

    inversion = MockInversion(regularization_list=[reg_0, reg_1], leq=leq)

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

    curvature_matrix_preload, curvature_matrix_counts = aa.util.leq.curvature_matrix_preload_from(
        mapping_matrix=operated_mapping_matrix
    )

    preloads = aa.Preloads(
        operated_mapping_matrix=operated_mapping_matrix,
        curvature_matrix_preload=curvature_matrix_preload.astype("int"),
        curvature_matrix_counts=curvature_matrix_counts.astype("int"),
    )

    # noinspection PyTypeChecker
    leq = MockLEq(noise_map=np.ones(9), linear_obj_list=MockMapper())

    inversion = MockInversion(leq=leq, preloads=preloads)

    assert inversion.operated_mapping_matrix[0, 0] == 2.0
    assert inversion.curvature_matrix[0, 0] == 36.0


def test__preload_of_regularization_matrix__overwrites_calculation():

    inversion = MockInversion(
        preloads=aa.Preloads(regularization_matrix=np.ones((2, 2)))
    )

    assert (inversion.regularization_matrix == np.ones((2, 2))).all()


def test__reconstruction_dict():

    reconstruction = np.array([0.0, 1.0, 1.0, 1.0])

    linear_obj = MockLinearObjFunc()
    mapper = MockMapper(pixels=3)

    inversion = MockInversion(
        leq=MockLEq(linear_obj_list=[linear_obj, mapper]), reconstruction=reconstruction
    )

    assert (inversion.reconstruction_dict[mapper] == np.ones(3)).all()

    reconstruction = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    linear_obj = MockLinearObjFunc()
    mapper_0 = MockMapper(pixels=2)
    mapper_1 = MockMapper(pixels=3)

    inversion = MockInversion(
        leq=MockLEq(linear_obj_list=[linear_obj, mapper_0, mapper_1]),
        reconstruction=reconstruction,
    )

    assert (inversion.reconstruction_dict[mapper_0] == np.ones(2)).all()
    assert (inversion.reconstruction_dict[mapper_1] == 2.0 * np.ones(3)).all()


def test__mapped_reconstructed_data():

    linear_obj_0 = MockLinearObjFunc()

    mapped_reconstructed_data_dict = {linear_obj_0: np.ones(3)}

    # noinspection PyTypeChecker
    inversion = MockInversion(
        leq=MockLEq(mapped_reconstructed_data_dict=mapped_reconstructed_data_dict),
        reconstruction=np.ones(3),
        reconstruction_dict=[None],
    )

    assert (inversion.mapped_reconstructed_data_dict[linear_obj_0] == np.ones(3)).all()
    assert (inversion.mapped_reconstructed_data == np.ones(3)).all()

    linear_obj_1 = MockLinearObjFunc()

    mapped_reconstructed_data_dict = {
        linear_obj_0: np.ones(2),
        linear_obj_1: 2.0 * np.ones(2),
    }

    # noinspection PyTypeChecker
    inversion = MockInversion(
        leq=MockLEq(mapped_reconstructed_data_dict=mapped_reconstructed_data_dict),
        reconstruction=np.array([1.0, 1.0, 2.0, 2.0]),
        reconstruction_dict=[None, None],
    )

    assert (inversion.mapped_reconstructed_data_dict[linear_obj_0] == np.ones(2)).all()
    assert (
        inversion.mapped_reconstructed_data_dict[linear_obj_1] == 2.0 * np.ones(2)
    ).all()
    assert (inversion.mapped_reconstructed_data == 3.0 * np.ones(2)).all()


def test__mapped_reconstructed_image():

    linear_obj_0 = MockLinearObjFunc()

    mapped_reconstructed_image_dict = {linear_obj_0: np.ones(3)}

    # noinspection PyTypeChecker
    inversion = MockInversion(
        leq=MockLEq(mapped_reconstructed_image_dict=mapped_reconstructed_image_dict),
        reconstruction=np.ones(3),
        reconstruction_dict=[None],
    )

    assert (inversion.mapped_reconstructed_image_dict[linear_obj_0] == np.ones(3)).all()
    assert (inversion.mapped_reconstructed_image == np.ones(3)).all()

    linear_obj_1 = MockLinearObjFunc()

    mapped_reconstructed_image_dict = {
        linear_obj_0: np.ones(2),
        linear_obj_1: 2.0 * np.ones(2),
    }

    # noinspection PyTypeChecker
    inversion = MockInversion(
        leq=MockLEq(mapped_reconstructed_image_dict=mapped_reconstructed_image_dict),
        reconstruction=np.array([1.0, 1.0, 2.0, 2.0]),
        reconstruction_dict=[None, None],
    )

    assert (inversion.mapped_reconstructed_image_dict[linear_obj_0] == np.ones(2)).all()
    assert (
        inversion.mapped_reconstructed_image_dict[linear_obj_1] == 2.0 * np.ones(2)
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

    inversion = MockInversion(
        preloads=aa.Preloads(log_det_regularization_matrix_term=1.0)
    )

    assert inversion.log_det_regularization_matrix_term == 1.0


def test__determinant_of_positive_definite_matrix_via_cholesky():

    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    leq = MockInversion(curvature_reg_matrix=matrix)

    log_determinant = np.log(np.linalg.det(matrix))

    assert log_determinant == pytest.approx(leq.log_det_curvature_reg_matrix_term, 1e-4)

    matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

    leq = MockInversion(curvature_reg_matrix=matrix)

    log_determinant = np.log(np.linalg.det(matrix))

    assert log_determinant == pytest.approx(leq.log_det_curvature_reg_matrix_term, 1e-4)


def test__brightest_reconstruction_pixel_and_centre():

    matrix_shape = (9, 3)

    mapper = MockMapper(
        matrix_shape,
        source_pixelization_grid=aa.Grid2DVoronoi.manual_slim(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 0.0]]
        ),
    )

    inversion = MockInversion(
        leq=MockLEq(linear_obj_list=[mapper]),
        reconstruction=np.array([2.0, 3.0, 5.0, 0.0]),
    )

    assert inversion.brightest_reconstruction_pixel_list[0] == 2

    assert inversion.brightest_reconstruction_pixel_centre_list[0].in_list == [
        (5.0, 6.0)
    ]


def test__interpolated_reconstruction_list_from():

    interpolated_array = np.array([0.0, 1.0, 1.0, 1.0])

    mapper = MockMapper(pixels=3, interpolated_array=interpolated_array)

    inversion = MockInversion(
        leq=MockLEq(linear_obj_list=[mapper]), reconstruction=interpolated_array
    )

    interpolated_reconstruction_list = inversion.interpolated_reconstruction_list_from(
        shape_native=(3, 3)
    )

    assert (interpolated_reconstruction_list[0] == np.array([0.0, 1.0, 1.0, 1.0])).all()
