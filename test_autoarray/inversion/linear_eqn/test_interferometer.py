import autoarray as aa

import numpy as np
from os import path
import pytest

directory = path.dirname(path.realpath(__file__))


def test__curvature_matrix():

    transformed_mapping_matrix = np.array([[1.0 + 1j, 1.0 + 1j], [1.0 + 1j, 1.0 + 1j]])
    noise_map = np.array([1.0 + 1j, 1.0 + 1j])

    leq = aa.m.MockLEqInterferometerMapping(
        linear_obj_list=[aa.m.MockLinearObjFunc()],
        transformed_mapping_matrix=transformed_mapping_matrix,
        noise_map=noise_map,
        settings=aa.SettingsInversion(linear_func_only_add_to_curvature_diag=False),
    )

    assert leq.curvature_matrix == pytest.approx(
        np.array([[4.0, 4.0], [4.0, 4.0]]), 1.0e-4
    )

    assert leq.curvature_matrix[0, 0] - 4.0 < 1.0e-8

    leq = aa.m.MockLEqInterferometerMapping(
        linear_obj_list=[aa.m.MockLinearObjFunc()],
        transformed_mapping_matrix=transformed_mapping_matrix,
        noise_map=noise_map,
        settings=aa.SettingsInversion(linear_func_only_add_to_curvature_diag=True),
    )

    assert leq.curvature_matrix[0, 0] - 4.0 > 0.0
