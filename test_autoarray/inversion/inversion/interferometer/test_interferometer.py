import autoarray as aa

import numpy as np
from os import path
import pytest

directory = path.dirname(path.realpath(__file__))


def test__curvature_matrix(rectangular_mapper_7x7_3x3):

    operated_mapping_matrix = np.array(
        [[1.0 + 1j, 1.0 + 1j, 1.0 + 1j], [1.0 + 1j, 1.0 + 1j, 1.0 + 1j]]
    )
    noise_map = np.array([1.0 + 1j, 1.0 + 1j])

    inversion = aa.m.MockInversionInterferometerMapping(
        linear_obj_list=[aa.m.MockLinearObj(), rectangular_mapper_7x7_3x3],
        operated_mapping_matrix=operated_mapping_matrix,
        noise_map=noise_map,
        settings=aa.SettingsInversion(no_regularization_add_to_curvature_diag=False),
    )

    assert inversion.curvature_matrix[0:2, 0:2] == pytest.approx(
        np.array([[4.0, 4.0], [4.0, 4.0]]), 1.0e-4
    )

    assert inversion.curvature_matrix[0, 0] - 4.0 < 1.0e-12
    assert inversion.curvature_matrix[2, 2] - 4.0 < 1.0e-12

    inversion = aa.m.MockInversionInterferometerMapping(
        linear_obj_list=[aa.m.MockLinearObj(), rectangular_mapper_7x7_3x3],
        operated_mapping_matrix=operated_mapping_matrix,
        noise_map=noise_map,
        settings=aa.SettingsInversion(no_regularization_add_to_curvature_diag=True),
    )

    assert inversion.curvature_matrix[0, 0] - 4.0 > 0.0
    assert inversion.curvature_matrix[2, 2] - 4.0 < 1.0e-12
