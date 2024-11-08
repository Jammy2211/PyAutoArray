import pytest

import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.coordinate_array import CoordinateArrayTriangles


@pytest.fixture
def two_triangles():
    return CoordinateArrayTriangles(
        coordinates=np.array([[0, 0], [1, 0]]),
        side_length=1.0,
    )


def test_two(two_triangles):
    assert np.all(two_triangles.centres == np.array([[0, 0], [0.5, 0]]))
    assert np.all(
        two_triangles.triangles
        == [
            [
                [0.0, HEIGHT_FACTOR / 2],
                [0.5, -HEIGHT_FACTOR / 2],
                [-0.5, -HEIGHT_FACTOR / 2],
            ],
            [
                [0.5, -HEIGHT_FACTOR / 2],
                [0.0, HEIGHT_FACTOR / 2],
                [1.0, HEIGHT_FACTOR / 2],
            ],
        ]
    )


@pytest.fixture
def one_triangle():
    return CoordinateArrayTriangles(
        coordinates=np.array([[0, 0]]),
        side_length=1.0,
    )


def test_trivial_triangles(one_triangle):
    assert one_triangle.flip_array == np.array([1])
    assert np.all(one_triangle.centres == np.array([[0, 0]]))
    assert np.all(
        one_triangle.triangles
        == [
            [
                [0.0, HEIGHT_FACTOR / 2],
                [0.5, -HEIGHT_FACTOR / 2],
                [-0.5, -HEIGHT_FACTOR / 2],
            ],
        ]
    )


def test_above():
    triangles = CoordinateArrayTriangles(
        coordinates=np.array([[0, 1]]),
        side_length=1.0,
    )
    assert np.all(
        triangles.up_sample().triangles
        == [
            [
                [0.0, 0.43301270189221935],
                [-0.25, 0.8660254037844386],
                [0.25, 0.8660254037844386],
            ],
            [
                [0.25, 0.8660254037844388],
                [0.0, 1.299038105676658],
                [0.5, 1.299038105676658],
            ],
            [
                [-0.25, 0.8660254037844388],
                [-0.5, 1.299038105676658],
                [0.0, 1.299038105676658],
            ],
            [
                [0.0, 1.299038105676658],
                [0.25, 0.8660254037844388],
                [-0.25, 0.8660254037844388],
            ],
        ]
    )


@pytest.fixture
def upside_down():
    return CoordinateArrayTriangles(
        coordinates=np.array([[1, 0]]),
        side_length=1.0,
    )


def test_upside_down(upside_down):
    assert np.all(upside_down.centres == np.array([[0.5, 0]]))
    assert np.all(
        upside_down.triangles
        == [
            [
                [0.5, -HEIGHT_FACTOR / 2],
                [0.0, HEIGHT_FACTOR / 2],
                [1.0, HEIGHT_FACTOR / 2],
            ],
        ]
    )


def test_up_sample(one_triangle):
    up_sampled = one_triangle.up_sample()
    assert up_sampled.side_length == 0.5
    assert np.all(
        up_sampled.triangles
        == [
            [[0.0, -0.4330127018922193], [-0.25, 0.0], [0.25, 0.0]],
            [[0.25, 0.0], [0.5, -0.4330127018922193], [0.0, -0.4330127018922193]],
            [[-0.25, 0.0], [0.0, -0.4330127018922193], [-0.5, -0.4330127018922193]],
            [[0.0, 0.4330127018922193], [0.25, 0.0], [-0.25, 0.0]],
        ]
    )


def test_up_sample_upside_down(upside_down):
    up_sampled = upside_down.up_sample()
    assert up_sampled.side_length == 0.5
    assert np.all(
        up_sampled.triangles
        == [
            [[0.5, -0.4330127018922193], [0.25, 0.0], [0.75, 0.0]],
            [[0.75, 0.0], [0.5, 0.4330127018922193], [1.0, 0.4330127018922193]],
            [[0.25, 0.0], [0.0, 0.4330127018922193], [0.5, 0.4330127018922193]],
            [[0.5, 0.4330127018922193], [0.75, 0.0], [0.25, 0.0]],
        ]
    )


def _test_up_sample_twice(one_triangle, plot):
    plot(one_triangle)
    one = one_triangle.up_sample()
    two = one.up_sample()
    three = two.up_sample()
    plot(three, color="blue")
    plot(two, color="green")
    plot(one, color="red")


def test_neighborhood(one_triangle):
    assert np.all(
        one_triangle.neighborhood().triangles
        == [
            [
                [-0.5, -0.4330127018922193],
                [-1.0, 0.4330127018922193],
                [0.0, 0.4330127018922193],
            ],
            [
                [0.0, -1.299038105676658],
                [-0.5, -0.4330127018922193],
                [0.5, -0.4330127018922193],
            ],
            [
                [0.0, 0.4330127018922193],
                [0.5, -0.4330127018922193],
                [-0.5, -0.4330127018922193],
            ],
            [
                [0.5, -0.4330127018922193],
                [0.0, 0.4330127018922193],
                [1.0, 0.4330127018922193],
            ],
        ]
    )


def test_upside_down_neighborhood(upside_down):
    assert np.all(
        upside_down.neighborhood().triangles
        == [
            [
                [0.0, 0.4330127018922193],
                [0.5, -0.4330127018922193],
                [-0.5, -0.4330127018922193],
            ],
            [
                [0.5, -0.4330127018922193],
                [0.0, 0.4330127018922193],
                [1.0, 0.4330127018922193],
            ],
            [
                [0.5, 1.299038105676658],
                [1.0, 0.4330127018922193],
                [0.0, 0.4330127018922193],
            ],
            [
                [1.0, 0.4330127018922193],
                [1.5, -0.4330127018922193],
                [0.5, -0.4330127018922193],
            ],
        ]
    )


def _test_complicated(plot, one_triangle):
    triangles = one_triangle.neighborhood().neighborhood()
    up_sampled = triangles.up_sample()


def test_vertices(one_triangle):
    assert np.all(
        one_triangle.vertices
        == [
            [-0.5, -0.4330127018922193],
            [0.0, 0.4330127018922193],
            [0.5, -0.4330127018922193],
        ]
    )


def test_up_sampled_vertices(one_triangle):
    assert np.all(
        one_triangle.up_sample().vertices
        == [
            [-0.5, -0.4330127018922193],
            [-0.25, 0.0],
            [0.0, -0.4330127018922193],
            [0.0, 0.4330127018922193],
            [0.25, 0.0],
            [0.5, -0.4330127018922193],
        ]
    )


def test_with_vertices(one_triangle):
    triangle = one_triangle.with_vertices(np.array([[0, 0], [1, 0], [0.5, 1]]))
    assert np.all(triangle.triangles == [[[1.0, 0.0], [0.5, 1.0], [0.0, 0.0]]])


def test_multiple_with_vertices(one_triangle, plot):
    up_sampled = one_triangle.up_sample()
    plot(up_sampled.with_vertices(2 * up_sampled.vertices).triangles.tolist())


def test_for_indexes(two_triangles):
    assert np.all(
        two_triangles.for_indexes(np.array([0])).triangles
        == [
            [
                [0.0, 0.4330127018922193],
                [0.5, -0.4330127018922193],
                [-0.5, -0.4330127018922193],
            ]
        ]
    )


def test_for_limits_and_scale():
    triangles = CoordinateArrayTriangles.for_limits_and_scale(
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
    )
    assert triangles.triangles.shape == (4, 3, 2)


def test_means(one_triangle):
    assert np.all(one_triangle.means == [[0.0, -0.14433756729740643]])
