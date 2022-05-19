import numpy as np
from os import path
import pytest

import autoarray as aa

from autoarray import exc


directory = path.dirname(path.realpath(__file__))


def test__linear_obj_func_list__filters_other_objects():

    inversion = aa.m.MockInversion(
        leq=aa.m.MockLEq(
            linear_obj_list=[
                aa.m.MockMapper(pixels=1),
                aa.m.MockLinearObjFunc(grid=3),
                aa.m.MockMapper(pixels=2),
                aa.m.MockLinearObjFunc(grid=4),
            ]
        )
    )

    assert inversion.linear_obj_func_list[0].grid == 3
    assert inversion.linear_obj_func_list[1].grid == 4
    assert inversion.has_linear_obj_func == True


def test__mapper_list__filters_other_objects():

    inversion = aa.m.MockInversion(
        leq=aa.m.MockLEq(
            linear_obj_list=[
                aa.m.MockMapper(pixels=1),
                aa.m.MockLinearObjFunc(),
                aa.m.MockMapper(pixels=2),
            ]
        )
    )

    assert inversion.mapper_list[0].pixels == 1
    assert inversion.mapper_list[1].pixels == 2
    assert inversion.has_mapper == True


def test__regularization_matrix():

    leq = aa.m.MockLEq(linear_obj_list=[aa.m.MockMapper(), aa.m.MockMapper()])

    reg_0 = aa.m.MockRegularization(regularization_matrix=np.ones((2, 2)))
    reg_1 = aa.m.MockRegularization(regularization_matrix=2.0 * np.ones((3, 3)))

    inversion = aa.m.MockInversion(regularization_list=[reg_0, reg_1], leq=leq)

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
    leq = aa.m.MockLEq(noise_map=np.ones(9), linear_obj_list=aa.m.MockMapper())

    inversion = aa.m.MockInversion(leq=leq, preloads=preloads)

    assert inversion.operated_mapping_matrix[0, 0] == 2.0
    assert inversion.curvature_matrix[0, 0] == 36.0


def test__preload_of_regularization_matrix__overwrites_calculation():

    inversion = aa.m.MockInversion(
        preloads=aa.Preloads(regularization_matrix=np.ones((2, 2)))
    )

    assert (inversion.regularization_matrix == np.ones((2, 2))).all()


def test__reconstruction_dict():

    reconstruction = np.array([0.0, 1.0, 1.0, 1.0])

    linear_obj = aa.m.MockLinearObjFunc()
    mapper = aa.m.MockMapper(pixels=3)

    inversion = aa.m.MockInversion(
        leq=aa.m.MockLEq(linear_obj_list=[linear_obj, mapper]),
        reconstruction=reconstruction,
    )

    assert (inversion.reconstruction_dict[mapper] == np.ones(3)).all()

    reconstruction = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    linear_obj = aa.m.MockLinearObjFunc()
    mapper_0 = aa.m.MockMapper(pixels=2)
    mapper_1 = aa.m.MockMapper(pixels=3)

    inversion = aa.m.MockInversion(
        leq=aa.m.MockLEq(linear_obj_list=[linear_obj, mapper_0, mapper_1]),
        reconstruction=reconstruction,
    )

    assert (inversion.reconstruction_dict[mapper_0] == np.ones(2)).all()
    assert (inversion.reconstruction_dict[mapper_1] == 2.0 * np.ones(3)).all()


def test__mapped_reconstructed_data():

    linear_obj_0 = aa.m.MockLinearObjFunc()

    mapped_reconstructed_data_dict = {linear_obj_0: np.ones(3)}

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        leq=aa.m.MockLEq(mapped_reconstructed_data_dict=mapped_reconstructed_data_dict),
        reconstruction=np.ones(3),
        reconstruction_dict=[None],
    )

    assert (inversion.mapped_reconstructed_data_dict[linear_obj_0] == np.ones(3)).all()
    assert (inversion.mapped_reconstructed_data == np.ones(3)).all()

    linear_obj_1 = aa.m.MockLinearObjFunc()

    mapped_reconstructed_data_dict = {
        linear_obj_0: np.ones(2),
        linear_obj_1: 2.0 * np.ones(2),
    }

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        leq=aa.m.MockLEq(mapped_reconstructed_data_dict=mapped_reconstructed_data_dict),
        reconstruction=np.array([1.0, 1.0, 2.0, 2.0]),
        reconstruction_dict=[None, None],
    )

    assert (inversion.mapped_reconstructed_data_dict[linear_obj_0] == np.ones(2)).all()
    assert (
        inversion.mapped_reconstructed_data_dict[linear_obj_1] == 2.0 * np.ones(2)
    ).all()
    assert (inversion.mapped_reconstructed_data == 3.0 * np.ones(2)).all()


def test__mapped_reconstructed_image():

    linear_obj_0 = aa.m.MockLinearObjFunc()

    mapped_reconstructed_image_dict = {linear_obj_0: np.ones(3)}

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        leq=aa.m.MockLEq(
            mapped_reconstructed_image_dict=mapped_reconstructed_image_dict
        ),
        reconstruction=np.ones(3),
        reconstruction_dict=[None],
    )

    assert (inversion.mapped_reconstructed_image_dict[linear_obj_0] == np.ones(3)).all()
    assert (inversion.mapped_reconstructed_image == np.ones(3)).all()

    linear_obj_1 = aa.m.MockLinearObjFunc()

    mapped_reconstructed_image_dict = {
        linear_obj_0: np.ones(2),
        linear_obj_1: 2.0 * np.ones(2),
    }

    # noinspection PyTypeChecker
    inversion = aa.m.MockInversion(
        leq=aa.m.MockLEq(
            mapped_reconstructed_image_dict=mapped_reconstructed_image_dict
        ),
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
        inversion = aa.m.MockInversion(
            data_vector=np.ones(3), curvature_reg_matrix=np.ones((3, 3))
        )

        # noinspection PyStatementEffect
        inversion.reconstruction


def test__regularization_term():

    reconstruction = np.array([1.0, 1.0, 1.0])

    regularization_matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    leq = aa.m.MockLEq(linear_obj_list=[aa.m.MockMapper()])

    inversion = aa.m.MockInversion(
        leq=leq,
        reconstruction=reconstruction,
        regularization_matrix=regularization_matrix,
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

    leq = aa.m.MockLEq(linear_obj_list=[aa.m.MockMapper()])

    inversion = aa.m.MockInversion(
        leq=leq,
        reconstruction=reconstruction,
        regularization_matrix=regularization_matrix,
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

    leq = aa.m.MockLEq(linear_obj_list=[aa.m.MockMapper()])

    inversion = aa.m.MockInversion(
        leq=leq, preloads=aa.Preloads(log_det_regularization_matrix_term=1.0)
    )

    assert inversion.log_det_regularization_matrix_term == 1.0


def test__determinant_of_positive_definite_matrix_via_cholesky():

    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    leq = aa.m.MockInversion(curvature_reg_matrix=matrix)

    log_determinant = np.log(np.linalg.det(matrix))

    assert log_determinant == pytest.approx(leq.log_det_curvature_reg_matrix_term, 1e-4)

    matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

    leq = aa.m.MockInversion(curvature_reg_matrix=matrix)

    log_determinant = np.log(np.linalg.det(matrix))

    assert log_determinant == pytest.approx(leq.log_det_curvature_reg_matrix_term, 1e-4)


def test__brightest_reconstruction_pixel_and_centre():

    matrix_shape = (9, 3)

    mapper = aa.m.MockMapper(
        matrix_shape,
        source_pixelization_grid=aa.Grid2DVoronoi.manual_slim(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 0.0]]
        ),
    )

    inversion = aa.m.MockInversion(
        leq=aa.m.MockLEq(linear_obj_list=[mapper]),
        reconstruction=np.array([2.0, 3.0, 5.0, 0.0]),
    )

    assert inversion.brightest_reconstruction_pixel_list[0] == 2

    assert inversion.brightest_reconstruction_pixel_centre_list[0].in_list == [
        (5.0, 6.0)
    ]


def test__interpolated_reconstruction_list_from():

    interpolated_array = np.array([0.0, 1.0, 1.0, 1.0])

    mapper = aa.m.MockMapper(pixels=3, interpolated_array=interpolated_array)

    inversion = aa.m.MockInversion(
        leq=aa.m.MockLEq(linear_obj_list=[mapper]), reconstruction=interpolated_array
    )

    interpolated_reconstruction_list = inversion.interpolated_reconstruction_list_from(
        shape_native=(3, 3), extent=(-0.2, 0.2, -0.3, 0.3)
    )

    assert (interpolated_reconstruction_list[0] == np.array([0.0, 1.0, 1.0, 1.0])).all()


def test__interpolated_errors_list_from():

    interpolated_array = np.array([0.0, 1.0, 1.0, 1.0])

    mapper = aa.m.MockMapper(pixels=3, interpolated_array=interpolated_array)

    inversion = aa.m.MockInversion(
        leq=aa.m.MockLEq(linear_obj_list=[mapper]), errors=interpolated_array
    )

    interpolated_errors_list = inversion.interpolated_errors_list_from(
        shape_native=(3, 3), extent=(-0.2, 0.2, -0.3, 0.3)
    )

    assert (interpolated_errors_list[0] == np.array([0.0, 1.0, 1.0, 1.0])).all()
