import numpy as np
import pytest

from autoarray.structures.triangles.array import ArrayTriangles


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


@pytest.mark.parametrize(
    "point, vertices, indices",
    [
        (
            (0.1, 0.1),
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ),
            np.array(
                [
                    [0, 2, 1],
                ]
            ),
        ),
        (
            (0.6, 0.6),
            np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1, 0, 2],
                ]
            ),
        ),
        (
            (0.5, 0.5),
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ]
            ),
            np.array(
                [
                    [0, 2, 1],
                    [2, 1, 3],
                ]
            ),
        ),
    ],
)
def test_contains_vertices(
    triangles,
    point,
    vertices,
    indices,
):
    containing = triangles.containing(point)

    assert (containing.indices == indices).all()
    assert (containing.vertices == vertices).all()


def test_up_sample(triangles):
    up_sampled = triangles.up_sample()

    assert (
        up_sampled.vertices
        == np.array(
            [
                [0.0, 0.0],
                [0.0, 0.5],
                [0.0, 1.0],
                [0.5, 0.0],
                [0.5, 0.5],
                [0.5, 1.0],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],
            ]
        )
    ).all()

    assert (
        up_sampled.indices
        == np.array(
            [
                [0, 3, 1],
                [6, 4, 7],
                [6, 4, 3],
                [2, 5, 4],
                [2, 1, 4],
                [8, 7, 5],
                [3, 4, 1],
                [4, 5, 7],
            ]
        )
    ).all()
