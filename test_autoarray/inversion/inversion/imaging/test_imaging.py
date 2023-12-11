import autoarray as aa
from autoarray.dataset.imaging.w_tilde import WTildeImaging
from autoarray.inversion.inversion.imaging.w_tilde import InversionImagingWTilde

from autoarray import exc

import numpy as np
from os import path
import pytest

directory = path.dirname(path.realpath(__file__))


def test__operated_mapping_matrix_property(convolver_7x7, rectangular_mapper_7x7_3x3):
    inversion = aa.m.MockInversionImaging(
        convolver=convolver_7x7, linear_obj_list=[rectangular_mapper_7x7_3x3]
    )

    assert inversion.operated_mapping_matrix_list[0][0, 0] == pytest.approx(1.0, 1e-4)
    assert inversion.operated_mapping_matrix[0, 0] == pytest.approx(1.0, 1e-4)

    convolver = aa.m.MockConvolver(operated_mapping_matrix=np.ones((2, 2)))

    inversion = aa.m.MockInversionImaging(
        convolver=convolver,
        linear_obj_list=[rectangular_mapper_7x7_3x3, rectangular_mapper_7x7_3x3],
    )

    operated_mapping_matrix_0 = np.array([[1.0, 1.0], [1.0, 1.0]])
    operated_mapping_matrix_1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    operated_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    assert inversion.operated_mapping_matrix_list[0] == pytest.approx(
        operated_mapping_matrix_0, 1.0e-4
    )
    assert inversion.operated_mapping_matrix_list[1] == pytest.approx(
        operated_mapping_matrix_1, 1.0e-4
    )
    assert inversion.operated_mapping_matrix == pytest.approx(
        operated_mapping_matrix, 1.0e-4
    )


def test__operated_mapping_matrix_property__with_operated_mapping_matrix_override(
    convolver_7x7, rectangular_mapper_7x7_3x3
):
    convolver = aa.m.MockConvolver(operated_mapping_matrix=np.ones((2, 2)))

    operated_mapping_matrix_override = np.array([[1.0, 2.0], [3.0, 4.0]])

    linear_obj = aa.m.MockLinearObjFuncList(
        mapping_matrix=None,
        operated_mapping_matrix_override=operated_mapping_matrix_override,
    )

    inversion = aa.m.MockInversionImaging(
        convolver=convolver, linear_obj_list=[rectangular_mapper_7x7_3x3, linear_obj]
    )

    operated_mapping_matrix_0 = np.array([[1.0, 1.0], [1.0, 1.0]])
    operated_mapping_matrix = np.array([[1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 3.0, 4.0]])

    assert inversion.operated_mapping_matrix_list[0] == pytest.approx(
        operated_mapping_matrix_0, 1.0e-4
    )
    assert inversion.operated_mapping_matrix_list[1] == pytest.approx(
        operated_mapping_matrix_override, 1.0e-4
    )
    assert inversion.operated_mapping_matrix == pytest.approx(
        operated_mapping_matrix, 1.0e-4
    )


def test__curvature_matrix(rectangular_mapper_7x7_3x3):
    noise_map = np.ones(2)
    convolver = aa.m.MockConvolver(operated_mapping_matrix=np.ones((2, 10)))

    operated_mapping_matrix_override = np.array([[1.0, 2.0], [3.0, 4.0]])

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=1,
        mapping_matrix=None,
        operated_mapping_matrix_override=operated_mapping_matrix_override,
        regularization=None,
    )

    inversion = aa.InversionImagingMapping(
        data=np.ones(2),
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        noise_map=noise_map,
        convolver=convolver,
        settings=aa.SettingsInversion(
            no_regularization_add_to_curvature_diag_value=False
        ),
    )

    assert inversion.curvature_matrix[0:2, 0:2] == pytest.approx(
        np.array([[10.0, 14.0], [14.0, 20.0]]), 1.0e-4
    )

    assert inversion.curvature_matrix[0, 0] - 10.0 < 1.0e-12
    assert inversion.curvature_matrix[3, 3] - 2.0 < 1.0e-12

    inversion = aa.InversionImagingMapping(
        data=np.ones(2),
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        noise_map=noise_map,
        convolver=convolver,
        settings=aa.SettingsInversion(
            no_regularization_add_to_curvature_diag_value=True
        ),
    )

    assert inversion.curvature_matrix[0, 0] - 10.0 > 0.0
    assert inversion.curvature_matrix[3, 3] - 2.0 < 1.0e-12


def test__w_tilde_checks_noise_map_and_raises_exception_if_preloads_dont_match_noise_map():
    matrix_shape = (9, 3)

    mask = aa.Mask2D(
        mask=np.array([[True, True, True], [False, False, False], [True, True, True]]),
        pixel_scales=1.0,
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    w_tilde = WTildeImaging(
        curvature_preload=None, indexes=None, lengths=None, noise_map_value=2.0
    )

    with pytest.raises(exc.InversionException):
        # noinspection PyTypeChecker
        InversionImagingWTilde(
            data=np.ones(9),
            noise_map=np.ones(9),
            convolver=aa.m.MockConvolver(matrix_shape),
            w_tilde=w_tilde,
            linear_obj_list=[
                aa.m.MockMapper(
                    mapping_matrix=np.ones(matrix_shape), source_plane_data_grid=grid
                )
            ],
        )
