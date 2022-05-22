import numpy as np
from os import path
import pytest

import autoarray as aa

directory = path.dirname(path.realpath(__file__))


def test__linear_obj_func_index_list():

    leq = aa.m.MockLEq(
        linear_obj_list=[aa.m.MockLinearObjFunc(), aa.m.MockLinearObjFunc()]
    )

    assert leq.linear_obj_func_index_list == [0, 1]

    leq = aa.m.MockLEq(
        linear_obj_list=[
            aa.m.MockMapper(pixels=10),
            aa.m.MockLinearObjFunc(),
            aa.m.MockMapper(pixels=20),
            aa.m.MockLinearObjFunc(),
        ]
    )

    assert leq.linear_obj_func_index_list == [10, 31]


def test__add_to_curvature_diag():

    leq = aa.m.MockLEq(
        linear_obj_list=[aa.m.MockLinearObjFunc()],
        settings=aa.SettingsInversion(linear_func_only_add_to_curvature_diag=True),
    )

    assert leq.add_to_curvature_diag is True

    leq = aa.m.MockLEq(
        linear_obj_list=[aa.m.MockLinearObjFunc()],
        settings=aa.SettingsInversion(linear_func_only_add_to_curvature_diag=False),
    )

    assert leq.add_to_curvature_diag is False

    leq = aa.m.MockLEq(
        linear_obj_list=[aa.m.MockLinearObjFunc(), aa.m.MockMapper()],
        settings=aa.SettingsInversion(linear_func_only_add_to_curvature_diag=True),
    )

    assert leq.add_to_curvature_diag is False


def test__mapping_matrix():

    mapper_0 = aa.m.MockMapper(mapping_matrix=np.ones((2, 2)))
    mapper_1 = aa.m.MockMapper(mapping_matrix=2.0 * np.ones((2, 3)))

    leq = aa.m.MockLEq(linear_obj_list=[mapper_0, mapper_1])

    mapping_matrix = np.array([[1.0, 1.0, 2.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0, 2.0]])

    assert leq.mapping_matrix == pytest.approx(mapping_matrix, 1.0e-4)


def test__reconstruction_mapper():

    leq = aa.m.MockLEq(
        linear_obj_list=[aa.m.MockMapper(pixels=2), aa.m.MockLinearObjFunc()]
    )

    inversion = aa.m.MockInversion(leq=leq, reconstruction=np.array([1.0, 2.0, 3.0]))

    assert (inversion.reconstruction_mapper == np.array([1.0, 2.0])).all()
