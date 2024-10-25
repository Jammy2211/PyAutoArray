from matplotlib import pyplot as plt
import numpy as np

from autoarray.structures.triangles.coordinate_array import CoordinateArrayTriangles


def plot(triangles):
    plt.figure(figsize=(8, 8))
    for triangle in triangles:
        triangle = np.append(triangle, [triangle[0]], axis=0)
        plt.plot(triangle[:, 0], triangle[:, 1], "o-", color="black")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def test_trivial_triangles():
    array = CoordinateArrayTriangles(
        coordinates=np.array([[0, 0]]),
        side_length=1.0,
    )
    assert np.all(array.centres == np.array([[0, 0]]))
    assert np.all(
        array.triangles
        == [
            [
                [0.0, 0.4330127018922193],
                [0.5, -0.4330127018922193],
                [-0.5, -0.4330127018922193],
            ],
        ]
    )
