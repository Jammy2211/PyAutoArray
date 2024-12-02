from autoarray.numpy_wrapper import jit
import pytest

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.shape import Point

try:
    from jax import numpy as np
    import jax

    jax.config.update("jax_log_compiles", True)
    from autoarray.structures.triangles.coordinate_array.jax_coordinate_array import (
        CoordinateArrayTriangles,
    )
except ImportError:
    import numpy as np
    from autoarray.structures.triangles.coordinate_array import CoordinateArrayTriangles


@pytest.fixture
def one_triangle():
    return CoordinateArrayTriangles(
        coordinates=np.array([[0, 0]]),
        side_length=1.0,
        mask=np.array([False]),
    )


@jit
def full_routine(triangles):
    neighborhood = triangles.neighborhood()
    up_sampled = neighborhood.up_sample()
    with_vertices = up_sampled.with_vertices(up_sampled.vertices)
    indexes = with_vertices.containing_indices(Point(0.1, 0.1))
    return up_sampled.for_indexes(indexes)


# def test_full_routine(one_triangle, compare_with_nans):
#     result = full_routine(one_triangle)
#
#     assert compare_with_nans(
#         result.triangles,
#         np.array(
#             [
#                 [
#                     [0.0, 0.4330126941204071],
#                     [0.25, 0.0],
#                     [-0.25, 0.0],
#                 ]
#             ]
#         ),
#     )


def test_neighborhood(one_triangle):
    assert np.allclose(
        np.array(jit(one_triangle.neighborhood)().triangles),
        np.array(
            [
                [
                    [-0.5, -0.4330126941204071],
                    [-1.0, 0.4330126941204071],
                    [0.0, 0.4330126941204071],
                ],
                [
                    [0.0, -1.299038052558899],
                    [-0.5, -0.4330126941204071],
                    [0.5, -0.4330126941204071],
                ],
                [
                    [0.0, 0.4330126941204071],
                    [0.5, -0.4330126941204071],
                    [-0.5, -0.4330126941204071],
                ],
                [
                    [0.5, -0.4330126941204071],
                    [0.0, 0.4330126941204071],
                    [1.0, 0.4330126941204071],
                ],
            ]
        ),
    )


def test_up_sample(one_triangle):
    up_sampled = jit(one_triangle.up_sample)()
    assert np.allclose(
        np.array(up_sampled.triangles),
        np.array(
            [
                [
                    [[0.0, -0.4330126941204071], [-0.25, 0.0], [0.25, 0.0]],
                    [
                        [0.25, 0.0],
                        [0.5, -0.4330126941204071],
                        [0.0, -0.4330126941204071],
                    ],
                    [
                        [-0.25, 0.0],
                        [0.0, -0.4330126941204071],
                        [-0.5, -0.4330126941204071],
                    ],
                    [[0.0, 0.4330126941204071], [0.25, 0.0], [-0.25, 0.0]],
                ]
            ]
        ),
    )


def test_means(one_triangle):
    assert len(one_triangle.means) == 1

    up_sampled = one_triangle.up_sample()
    neighborhood = up_sampled.neighborhood()
    assert np.count_nonzero(~np.isnan(neighborhood.means).any(axis=1)) == 10


ONE_TRIANGLE_AREA = HEIGHT_FACTOR * 0.5


def test_area(one_triangle):
    assert one_triangle.area == ONE_TRIANGLE_AREA
    assert one_triangle.up_sample().area == ONE_TRIANGLE_AREA

    neighborhood = one_triangle.neighborhood()
    assert neighborhood.area == 4 * ONE_TRIANGLE_AREA
    assert neighborhood.up_sample().area == 4 * ONE_TRIANGLE_AREA
    assert neighborhood.neighborhood().area == 10 * ONE_TRIANGLE_AREA
