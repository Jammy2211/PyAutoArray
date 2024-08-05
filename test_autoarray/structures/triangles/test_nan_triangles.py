import pytest

from autoarray.structures.triangles.array import ArrayTriangles
from jax import numpy as np


@pytest.fixture
def nan_triangles():
    return ArrayTriangles(
        indices=np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
                [-1, -1, -1],
            ]
        ),
        vertices=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )


def test_nan_triangles(nan_triangles, compare_with_nans):
    print(nan_triangles.triangles.tolist())
    assert compare_with_nans(
        nan_triangles.triangles,
        np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            ]
        ),
    ).all()


def test_up_sample_nan_triangles(nan_triangles):
    up_sampled = nan_triangles.up_sample()

    # print(up_sampled.indices.tolist())
    # print(up_sampled.vertices.tolist())

    for triangle in up_sampled.triangles:
        print(triangle.tolist())
