import numpy as np
import pytest

from autoarray.structures.triangles.array import ArrayTriangles
from autoarray.structures.triangles.shape import Triangle


def test_triangles(triangles):
    assert triangles.area == 1.0


@pytest.mark.parametrize(
    "vertices, area",
    [
        (
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            0.5,
        ),
        (
            [
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            0.5,
        ),
        (
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ],
            1.0,
        ),
    ],
)
def test_single_triangle(vertices, area):
    triangles = ArrayTriangles(
        indices=np.array(
            [
                [0, 1, 2],
            ]
        ),
        vertices=np.array(vertices),
    )
    assert triangles.area == area


def test_triangle():
    triangle = Triangle(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )
    assert triangle.area == 0.5
