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


def test_full_routine(one_triangle, plot):
    neighborhood = one_triangle.neighborhood()
    up_sampled = one_triangle.up_sample()
    with_vertices = up_sampled.with_vertices(up_sampled.vertices)
    indexes = with_vertices.containing_indices(Point(0.1, 0.1))
    selected = up_sampled.for_indexes(indexes)

    plot(one_triangle.triangles, color="black")
    plot(neighborhood.triangles, color="red")
    plot(up_sampled.triangles, color="blue")
    plot(with_vertices.triangles, color="green")
    plot(selected.triangles, color="yellow")
