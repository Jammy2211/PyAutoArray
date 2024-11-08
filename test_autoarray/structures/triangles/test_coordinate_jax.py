import jax
import pytest
import numpy as np

from autoarray.structures.triangles.jax_coordinate_array import CoordinateArrayTriangles
from autoarray.structures.triangles.shape import Point


@pytest.fixture
def one_triangle():
    return CoordinateArrayTriangles(
        coordinates=np.array([[0, 0]]),
        side_length=1.0,
    )


# @jax.jit
def full_routine(triangles):
    neighborhood = triangles.neighborhood()
    up_sampled = neighborhood.up_sample()
    with_vertices = up_sampled.with_vertices(up_sampled.vertices)
    indexes = with_vertices.containing_indices(Point(0.1, 0.1))
    return up_sampled.for_indexes(indexes)


def test_full_routine(one_triangle, plot):
    result = full_routine(one_triangle)

    plot(result)


def test_neighborhood(one_triangle):
    assert np.all(
        np.array(jax.jit(one_triangle.neighborhood)().triangles)
        == [
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
    )


def test_up_sample(one_triangle):
    up_sampled = jax.jit(one_triangle.up_sample)()
    assert np.all(
        np.array(up_sampled.triangles)
        == [
            [
                [[0.0, -0.4330126941204071], [-0.25, 0.0], [0.25, 0.0]],
                [[0.25, 0.0], [0.5, -0.4330126941204071], [0.0, -0.4330126941204071]],
                [[-0.25, 0.0], [0.0, -0.4330126941204071], [-0.5, -0.4330126941204071]],
                [[0.0, 0.4330126941204071], [0.25, 0.0], [-0.25, 0.0]],
            ]
        ]
    )
