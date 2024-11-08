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


@jax.jit
def full_routine(triangles):
    neighborhood = triangles.neighborhood()
    up_sampled = neighborhood.up_sample()
    with_vertices = up_sampled.with_vertices(up_sampled.vertices)
    indexes = with_vertices.containing_indices(Point(0.1, 0.1))
    return up_sampled.for_indexes(indexes)


def test_full_routine(one_triangle, plot):
    full_routine(one_triangle)

    neighborhood = one_triangle.neighborhood()
    up_sampled = neighborhood.up_sample()
    with_vertices = up_sampled.with_vertices(up_sampled.vertices)

    selected = full_routine(one_triangle)

    plot(up_sampled.triangles, color="blue")
    plot(with_vertices.triangles, color="green")
    plot(neighborhood.triangles, color="red")
    plot(one_triangle.triangles, color="black")
    plot(selected.triangles, color="yellow")
