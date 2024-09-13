import pytest

from autoarray.structures.triangles.shape import Polygon, Triangle
import numpy as np


@pytest.mark.parametrize(
    "centroid, is_inside",
    [
        ((0.5, 0.5), True),
        ((0.0, 0.0), True),
        ((1.0, 0.0), True),
        ((0.0, 1.0), True),
        ((0.0, 2.0), False),
        ((2.0, 0.0), False),
        ((2.0, 2.0), False),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        Triangle(
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        Polygon(
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
            ]
        ),
    ],
)
def test_triangle(centroid, is_inside, shape):
    assert shape.mask(
        np.array(
            [3 * [centroid]],
        )
    ) == np.array([is_inside])


@pytest.mark.parametrize(
    "centroid, is_inside",
    [
        ((0.5, 0.5), True),
        ((0.0, 0.0), True),
        ((1.0, 0.0), True),
        ((0.0, 1.0), True),
        ((1.0, 1.0), True),
        ((0.0, 2.0), False),
        ((2.0, 0.0), False),
        ((2.0, 2.0), False),
    ],
)
def test_polygon(centroid, is_inside):
    polygon = Polygon(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
    )
    assert polygon.mask(
        np.array(
            [3 * [centroid]],
        )
    ) == np.array([is_inside])
