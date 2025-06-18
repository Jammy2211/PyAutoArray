import pytest
import numpy as np

from autoarray.structures.triangles.array import ArrayTriangles
from autoarray.structures.triangles.shape import Circle


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
            [0],
        ),
        (
            (0.6, 0.6),
            [1],
        ),
        (
            (0.5, 0.5),
            [0, 1],
        ),
    ],
)
def test_small_point(triangles, point, indices):
    containing_triangles = triangles.containing_indices(
        Circle(
            *point,
            radius=0.001,
        )
    )
    assert [i for i in containing_triangles.tolist() if i != -1] == indices


@pytest.mark.parametrize(
    "radius, indices",
    [
        (0.1, []),
        (1.5, [0]),
        (2, [0, 1]),
    ],
)
def test_large_circle(
    triangles,
    radius,
    indices,
):
    containing_triangles = triangles.containing_indices(
        Circle(
            -1.0,
            0.0,
            radius=radius,
        )
    )
    assert [i for i in containing_triangles.tolist() if i != -1] == indices
