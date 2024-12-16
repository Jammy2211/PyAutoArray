import numpy as np
import pytest

from autoarray.structures.triangles.array import ArrayTriangles
from autoarray.structures.triangles.shape import Triangle, Circle, Square, Polygon


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


def test_circle():
    circle = Circle(
        x=0.0,
        y=0.0,
        radius=1.0,
    )
    assert circle.area == np.pi


def test_square():
    square = Square(
        top=0.0,
        bottom=1.0,
        left=0.0,
        right=1.0,
    )
    assert square.area == 1.0


def test_polygon():
    polygon = Polygon(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ],
    )
    assert polygon.area == 1.0
