from matplotlib import pyplot as plt
import numpy as np

from autoarray.structures.triangles.abstract import HEIGHT_FACTOR
from autoarray.structures.triangles.coordinate_array import CoordinateArrayTriangles


def plot(triangles):
    plt.figure(figsize=(8, 8))
    for triangle in triangles:
        triangle = np.append(triangle, [triangle[0]], axis=0)
        plt.plot(triangle[:, 0], triangle[:, 1], "o-", color="black")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def test_two():
    array = CoordinateArrayTriangles(
        coordinates=np.array([[0, 0], [1, 0]]),
        side_length=1.0,
    )
    assert np.all(array.centres == np.array([[0, 0], [0.5, 0]]))
    assert np.all(
        array.triangles
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
    plot(array)


def test_trivial_triangles():
    array = CoordinateArrayTriangles(
        coordinates=np.array([[0, 0]]),
        side_length=1.0,
    )
    assert array.flip_mask == np.array([1])
    assert np.all(array.centres == np.array([[0, 0]]))
    assert np.all(
        array.triangles
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
