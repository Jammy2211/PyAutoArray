import numpy as np
from os import path
import pytest


import autoarray as aa
from autoarray.mock.mock import MockMapper, MockLinearEqn

directory = path.dirname(path.realpath(__file__))


def test__mapping_matrix():

    mapper_0 = MockMapper(mapping_matrix=np.ones((2, 2)))
    mapper_1 = MockMapper(mapping_matrix=2.0 * np.ones((2, 3)))

    linear_eqn = MockLinearEqn(mapper_list=[mapper_0, mapper_1])

    mapping_matrix = np.array([[1.0, 1.0, 2.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0, 2.0]])

    assert linear_eqn.mapping_matrix == pytest.approx(mapping_matrix, 1.0e-4)