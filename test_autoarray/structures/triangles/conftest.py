import jax.numpy as jnp
from matplotlib import pyplot as plt
import pytest

from autoarray.structures.triangles.array import ArrayTriangles
from autoarray.structures.triangles.coordinate_array import CoordinateArrayTriangles


@pytest.fixture
def plot():
    plt.figure(figsize=(8, 8))

    def plot(triangles, color="black"):
        for triangle in triangles:
            triangle = jnp.array(triangle)
            triangle = jnp.append(triangle, jnp.array([triangle[0]]), axis=0)
            plt.plot(triangle[:, 0], triangle[:, 1], "o-", color=color)

    yield plot
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


@pytest.fixture
def compare_with_nans():
    def compare_with_nans_(arr1, arr2):
        nan_mask1 = jnp.isnan(arr1)
        nan_mask2 = jnp.isnan(arr2)

        arr1 = arr1[~nan_mask1]
        arr2 = arr2[~nan_mask2]

        return jnp.all(arr1 == arr2)

    return compare_with_nans_


@pytest.fixture
def triangles():
    return ArrayTriangles(
        indices=jnp.array(
            [
                [0, 1, 2],
                [1, 2, 3],
            ]
        ),
        vertices=jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )


@pytest.fixture
def one_triangle():
    return CoordinateArrayTriangles(
        coordinates=jnp.array([[0, 0]]),
        side_length=1.0,
    )


@pytest.fixture
def two_triangles():
    return CoordinateArrayTriangles(
        coordinates=jnp.array([[0, 0], [1, 0]]),
        side_length=1.0,
    )
