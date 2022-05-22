import numpy as np
from os import path
import pytest

import autoarray as aa


directory = path.dirname(path.realpath(__file__))


def test__curvature_matrix__via_w_tilde__identical_to_mapping():
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

    grid = aa.Grid2D.from_mask(mask=mask)

    pix_0 = aa.pix.Rectangular(shape=(3, 3))
    pix_1 = aa.pix.Rectangular(shape=(4, 4))

    mapper_0 = pix_0.mapper_from(
        source_grid_slim=grid,
        source_pixelization_grid=None,
        settings=aa.SettingsPixelization(use_border=False),
    )

    mapper_1 = pix_1.mapper_from(
        source_grid_slim=grid,
        source_pixelization_grid=None,
        settings=aa.SettingsPixelization(use_border=False),
    )

    reg = aa.reg.Constant(coefficient=1.0)

    image = aa.Array2D.manual_native(array=np.random.random((7, 7)), pixel_scales=1.0)
    noise_map = aa.Array2D.manual_native(
        array=np.random.random((7, 7)), pixel_scales=1.0
    )
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    psf = aa.Kernel2D.manual_native(array=kernel, pixel_scales=1.0)

    imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

    masked_imaging = imaging.apply_mask(mask=mask)

    inversion_w_tilde = aa.Inversion(
        dataset=masked_imaging,
        linear_obj_list=[mapper_0, mapper_1],
        regularization_list=[reg, reg],
        settings=aa.SettingsInversion(use_w_tilde=True, check_solution=False),
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging,
        linear_obj_list=[mapper_0, mapper_1],
        regularization_list=[reg, reg],
        settings=aa.SettingsInversion(use_w_tilde=False, check_solution=False),
    )

    assert inversion_w_tilde.curvature_matrix == pytest.approx(
        inversion_mapping.curvature_matrix, 1.0e-4
    )


def test__curvature_matrix_via_w_tilde__includes_source_interpolation__identical_to_mapping():
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

    grid = aa.Grid2D.from_mask(mask=mask)

    pix_0 = aa.pix.DelaunayMagnification(shape=(3, 3))
    pix_1 = aa.pix.DelaunayMagnification(shape=(4, 4))

    sparse_grid_0 = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        grid=grid, unmasked_sparse_shape=pix_0.shape
    )

    sparse_grid_1 = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        grid=grid, unmasked_sparse_shape=pix_1.shape
    )

    mapper_0 = pix_0.mapper_from(
        source_grid_slim=grid,
        source_pixelization_grid=sparse_grid_0,
        settings=aa.SettingsPixelization(use_border=False),
    )

    mapper_1 = pix_1.mapper_from(
        source_grid_slim=grid,
        source_pixelization_grid=sparse_grid_1,
        settings=aa.SettingsPixelization(use_border=False),
    )

    reg = aa.reg.Constant(coefficient=1.0)

    image = aa.Array2D.manual_native(array=np.random.random((7, 7)), pixel_scales=1.0)
    noise_map = aa.Array2D.manual_native(
        array=np.random.random((7, 7)), pixel_scales=1.0
    )
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    psf = aa.Kernel2D.manual_native(array=kernel, pixel_scales=1.0)

    imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

    masked_imaging = imaging.apply_mask(mask=mask)

    inversion_w_tilde = aa.Inversion(
        dataset=masked_imaging,
        linear_obj_list=[mapper_0, mapper_1],
        regularization_list=[reg, reg],
        settings=aa.SettingsInversion(use_w_tilde=True, check_solution=False),
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging,
        linear_obj_list=[mapper_0, mapper_1],
        regularization_list=[reg, reg],
        settings=aa.SettingsInversion(use_w_tilde=False, check_solution=False),
    )

    assert inversion_w_tilde.curvature_matrix == pytest.approx(
        inversion_mapping.curvature_matrix, 1.0e-4
    )


def test__curvature_reg_matrix_mapper():

    leq = aa.m.MockLEq(
        linear_obj_list=[aa.m.MockMapper(pixels=2), aa.m.MockLinearObjFunc()]
    )

    curvature_reg_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    inversion = aa.m.MockInversion(leq=leq, curvature_reg_matrix=curvature_reg_matrix)

    assert (
        inversion.curvature_reg_matrix_mapper == np.array([[1.0, 2.0], [4.0, 5.0]])
    ).all()


def test__errors_and_errors_with_covariance():

    curvature_reg_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]])

    inversion = aa.m.MockInversion(curvature_reg_matrix=curvature_reg_matrix)

    assert inversion.errors_with_covariance == pytest.approx(
        np.array([[2.5, -1.0, -0.5], [-1.0, 1.0, 0.0], [-0.5, 0.0, 0.5]]), 1.0e-2
    )
    assert inversion.errors == pytest.approx(np.array([2.5, 1.0, 0.5]), 1.0e-3)
