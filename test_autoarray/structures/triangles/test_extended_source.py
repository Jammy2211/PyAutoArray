import pytest
import numpy as np

from autoarray.structures.triangles.array import ArrayTriangles


@pytest.fixture
def triangles():
    return ArrayTriangles(
        indices=np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
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


@pytest.mark.parametrize(
    "point, indices",
    [
        (
            (0.1, 0.1),
            np.array([0]),
        ),
        (
            (0.6, 0.6),
            np.array([1]),
        ),
        (
            (0.5, 0.5),
            np.array([0, 1]),
        ),
    ],
)
def test_small_point(triangles, point, indices):
    containing_triangles = triangles.containing_indices_circle(
        point,
        radius=0.001,
    )
    print(containing_triangles)
    assert (containing_triangles == indices).all()
