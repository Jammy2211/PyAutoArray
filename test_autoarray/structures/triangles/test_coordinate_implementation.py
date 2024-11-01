import pytest
from matplotlib import pyplot as plt
import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.coordinate_array import CoordinateArrayTriangles


@pytest.fixture
def plot():
    plt.figure(figsize=(8, 8))

    def plot(triangles, color="black"):
        for triangle in triangles:
            triangle = np.append(triangle, [triangle[0]], axis=0)
            plt.plot(triangle[:, 0], triangle[:, 1], "o-", color=color)

    yield plot
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


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


def test_above(plot):
    triangles = CoordinateArrayTriangles(
        coordinates=np.array([[0, 1]]),
        side_length=1.0,
    )
    plot(triangles)
    plot(triangles.up_sample(), color="red")
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


def test_up_sample_twice(one_triangle, plot):
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


def test_complicated(plot, one_triangle):
    triangles = one_triangle.neighborhood().neighborhood()
    up_sampled = triangles.up_sample()
    plot(up_sampled, color="red")
    plot(triangles)


def test_vertices(one_triangle):
    assert np.all(
        one_triangle.vertices
        == [
            [0.0, HEIGHT_FACTOR / 2],
            [0.5, -HEIGHT_FACTOR / 2],
            [-0.5, -HEIGHT_FACTOR / 2],
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
