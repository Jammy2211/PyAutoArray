from autoarray.numpy_wrapper import jit
import pytest

from autoarray.structures.triangles.shape import Point

try:
    from jax import numpy as np
    import jax

    jax.config.update("jax_log_compiles", True)
    from autoarray.structures.triangles.jax_coordinate_array import (
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
    )


@jit
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
        np.array(jit(one_triangle.neighborhood)().triangles)
        == np.array(
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
        )
    )


def test_up_sample(one_triangle):
    up_sampled = jit(one_triangle.up_sample)()
    assert np.all(
        np.array(up_sampled.triangles)
        == np.array(
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
        )
    )
