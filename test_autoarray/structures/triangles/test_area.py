import numpy as np
import pytest

from autoarray.structures.triangles.array import ArrayTriangles


def test_area(triangles):
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
