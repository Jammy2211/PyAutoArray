from autoarray.numpy_wrapper import np
from autoarray.structures.triangles.array import JAXArrayTriangles as ArrayTriangles

from matplotlib import pyplot as plt


import pytest


@pytest.fixture
def plot():
    plt.figure(figsize=(8, 8))

    def plot(triangles, color="black"):
        for triangle in triangles:
            triangle = np.array(triangle)
            triangle = np.append(triangle, np.array([triangle[0]]), axis=0)
            plt.plot(triangle[:, 0], triangle[:, 1], "o-", color=color)

    yield plot
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


@pytest.fixture
def compare_with_nans():
    def compare_with_nans_(arr1, arr2):
        nan_mask1 = np.isnan(arr1)
        nan_mask2 = np.isnan(arr2)

        arr1 = arr1[~nan_mask1]
        arr2 = arr2[~nan_mask2]

        return np.all(arr1 == arr2)

    return compare_with_nans_


@pytest.fixture
def triangles():
    return ArrayTriangles(
        indices=np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
            ]
        ),
        vertices=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )
