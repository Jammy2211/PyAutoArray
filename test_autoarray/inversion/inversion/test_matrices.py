import numpy as np
from os import path
import pytest

import autoarray as aa

from autoarray.mock.mock import MockLinearEqn, MockInversion


directory = path.dirname(path.realpath(__file__))


def test__operated_mapping_matrix():

    linear_eqn_0 = MockLinearEqn(operated_mapping_matrix=np.ones((2, 2)))
    linear_eqn_1 = MockLinearEqn(operated_mapping_matrix=2.0 * np.ones((2, 3)))

    inversion = MockInversion(linear_eqn_list=[linear_eqn_0, linear_eqn_1])

    operated_mapping_matrix = np.array(
        [
            [1.0, 1.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 2.0],
        ]
    )

    assert inversion.operated_mapping_matrix == pytest.approx(operated_mapping_matrix)


def test__data_vector_from():

    linear_eqn_0 = MockLinearEqn(data_vector=np.ones(3))
    linear_eqn_1 = MockLinearEqn(data_vector=2.0 * np.ones(3))

    inversion = MockInversion(linear_eqn_list=[linear_eqn_0, linear_eqn_1])

    data_vector = np.array([3.0, 3.0, 3.0])

    assert inversion.data_vector == pytest.approx(data_vector)


def test__curvature_matrix():

    linear_eqn_0 = MockLinearEqn(curvature_matrix=np.ones((2, 2)))
    linear_eqn_1 = MockLinearEqn(curvature_matrix=2.0 * np.ones((3, 3)))

    inversion = MockInversion(linear_eqn_list=[linear_eqn_0, linear_eqn_1])

    curvature_matrix = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0, 2.0],
        ]
    )

    assert inversion.curvature_matrix == pytest.approx(curvature_matrix)


def test__errors_and_errors_with_covariance():

    curvature_reg_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]])

    inversion = MockInversion(curvature_reg_matrix=curvature_reg_matrix)

    assert inversion.errors_with_covariance == pytest.approx(
        np.array([[2.5, -1.0, -0.5], [-1.0, 1.0, 0.0], [-0.5, 0.0, 0.5]]), 1.0e-2
    )
    assert inversion.errors == pytest.approx(np.array([2.5, 1.0, 0.5]), 1.0e-3)


def test__integration__x2_mappers_give_same_result_via_mapping_and_w_tilde():
    mask = aa.Mask2D.manual(
        mask=[
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        pixel_scales=2.0,
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(
        mask=mask,
    )

    pix_0 = aa.pix.Rectangular(shape=(3, 3))
    pix_1 = aa.pix.Rectangular(shape=(4, 4))

    mapper_0 = pix_0.mapper_from(
        grid=grid, sparse_grid=None, settings=aa.SettingsPixelization(use_border=False)
    )

    mapper_1 = pix_1.mapper_from(
        grid=grid, sparse_grid=None, settings=aa.SettingsPixelization(use_border=False)
    )

    reg = aa.reg.Constant(coefficient=1.0)

    # image = aa.Array2D.manual_native(array=np.random.random(7, 7), pixel_scales=1.0)
    # noise_map = aa.Array2D.manual_native(array=np.random.random(7, 7), pixel_scales=1.0)
    # kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0],[0.0, 1.0, 0.0]])
    # psf = aa.Kernel2D.manual_native(array=kernel, pixel_scales=1.0)

    image = aa.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
    noise_map = aa.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
    psf = aa.Kernel2D.no_blur(pixel_scales=1.0)

    imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

    masked_imaging = imaging.apply_mask(mask=mask)

    inversion_w_tilde = aa.Inversion(
        dataset=masked_imaging,
        mapper_list=[mapper_0, mapper_1],
        regularization_list=[reg, reg],
        settings=aa.SettingsInversion(use_w_tilde=True, check_solution=False),
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging,
        mapper_list=[mapper_0, mapper_1],
        regularization_list=[reg, reg],
        settings=aa.SettingsInversion(use_w_tilde=False, check_solution=False),
    )

    assert inversion_w_tilde.reconstruction == pytest.approx(inversion_mapping.reconstruction, 1.0e-4)

    data_vector = aa.util.linear_eqn.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=inversion_w_tilde.operated_mapping_matrix,
        image=masked_imaging.image,
        noise_map=masked_imaging.noise_map
    )

    curvature_matrix = aa.util.linear_eqn.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=inversion_w_tilde.operated_mapping_matrix,
        noise_map=masked_imaging.noise_map
    )

    curvature_reg_matrix = curvature_matrix + inversion_w_tilde.regularization_matrix

    reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)

    assert inversion_w_tilde.reconstruction == pytest.approx(reconstruction, 1.0e-4)
