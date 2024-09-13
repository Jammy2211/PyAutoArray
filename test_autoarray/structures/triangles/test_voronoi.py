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
def test_triangle(centroid, is_inside):
    triangle = Triangle(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )
    assert triangle.mask(
        np.array(
            [3 * [centroid]],
        )
    ) == np.array([is_inside])
