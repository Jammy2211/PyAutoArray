import numpy as np
from os import path
import pytest

import autoarray as aa

directory = path.dirname(path.realpath(__file__))


def test__no_regularization_index_list():

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObjFunc(), aa.m.MockLinearObjFunc()]
    )

    assert inversion.no_regularization_index_list == [0, 1]

    inversion = aa.m.MockInversion(
        linear_obj_list=[
            aa.m.MockMapper(pixels=10),
            aa.m.MockLinearObjFunc(),
            aa.m.MockMapper(pixels=20),
            aa.m.MockLinearObjFunc(),
        ]
    )

    assert inversion.no_regularization_index_list == [10, 31]


def test__add_to_curvature_diag():

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObjFunc()],
        settings=aa.SettingsInversion(no_regularization_add_to_curvature_diag=True),
    )

    assert inversion.add_to_curvature_diag is True

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObjFunc()],
        settings=aa.SettingsInversion(no_regularization_add_to_curvature_diag=False),
    )

    assert inversion.add_to_curvature_diag is False

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObjFunc(), aa.m.MockMapper()],
        settings=aa.SettingsInversion(no_regularization_add_to_curvature_diag=True),
    )

    assert inversion.add_to_curvature_diag is True


def test__mapping_matrix():

    mapper_0 = aa.m.MockMapper(mapping_matrix=np.ones((2, 2)))
    mapper_1 = aa.m.MockMapper(mapping_matrix=2.0 * np.ones((2, 3)))

    inversion = aa.m.MockInversion(linear_obj_list=[mapper_0, mapper_1])

    mapping_matrix = np.array([[1.0, 1.0, 2.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0, 2.0]])

    assert inversion.mapping_matrix == pytest.approx(mapping_matrix, 1.0e-4)


def test__reconstruction_reduced():

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper(pixels=2), aa.m.MockLinearObjFunc()]
    )

    linear_obj_reg_list = [
        aa.m.MockLinearObjReg(pixels=2, regularization=1),
        aa.m.MockLinearObjReg(pixels=1),
    ]

    inversion = aa.m.MockInversion(
        inversion=inversion,
        linear_obj_reg_list=linear_obj_reg_list,
        reconstruction=np.array([1.0, 2.0, 3.0]),
    )

    assert (inversion.reconstruction_reduced == np.array([1.0, 2.0])).all()
