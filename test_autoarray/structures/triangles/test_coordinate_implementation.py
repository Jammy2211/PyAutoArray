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
    assert one_triangle.flip_mask == np.array([1])
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


def test_upside_down():
    array = CoordinateArrayTriangles(
        coordinates=np.array([[1, 0]]),
        side_length=1.0,
    )
    assert np.all(array.centres == np.array([[0.5, 0]]))
    assert np.all(
        array.triangles
        == [
            [
                [0.5, -HEIGHT_FACTOR / 2],
                [0.0, HEIGHT_FACTOR / 2],
                [1.0, HEIGHT_FACTOR / 2],
            ],
        ]
    )


def test_up_sample(one_triangle, plot):
    plot(one_triangle)

    up_sampled = one_triangle.up_sample()
    assert up_sampled.side_length == 0.5

    plot(up_sampled, color="red")
