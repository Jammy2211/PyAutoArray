from jax import numpy as np
import pytest

from autoarray.structures.triangles.jax_array import ArrayTriangles


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
            np.array([0, -1, -1, -1, -1]),
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
            np.array([1, -1, -1, -1, -1]),
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
            np.array([0, 1, -1, -1, -1]),
        ),
    ],
)
def test_contains_vertices(
    triangles,
    point,
    vertices,
    indices,
):
    containing_indices = triangles.containing_indices(point)

    assert (containing_indices == indices).all()


@pytest.mark.parametrize(
    "indexes, vertices, indices",
    [
        (
            np.array([0]),
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
            np.array([1]),
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
            np.array([0, 1]),
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [np.nan, np.nan],
                    [np.nan, np.nan],
                ],
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
def test_for_indexes(
    triangles,
    indexes,
    vertices,
    indices,
    compare_with_nans,
):
    containing = triangles.for_indexes(indexes)

    assert (containing.indices == indices).all()
    assert compare_with_nans(
        containing.vertices,
        vertices,
    ).all()


def test_negative_index(
    triangles,
    compare_with_nans,
):
    indexes = np.array([0, -1])

    containing = triangles.for_indexes(indexes)

    assert (
        containing.indices
        == np.array(
            [
                [0, 2, 1],
                [-1, -1, -1],
            ],
        )
    ).all()
    assert compare_with_nans(
        containing.vertices,
        np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        ),
    )


def test_up_sample(
    triangles,
    compare_with_nans,
):
    up_sampled = triangles.up_sample()

    assert compare_with_nans(
        up_sampled.vertices,
        np.array(
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
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        ),
    )

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


@pytest.mark.parametrize(
    "offset",
    [-1, 0, 1],
)
def test_simple_neighborhood(offset):
    triangles = ArrayTriangles(
        indices=np.array(
            [
                [0, 1, 2],
            ]
        ),
        vertices=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        + offset,
    )
    assert (
        triangles.neighborhood().triangles
        == (
            np.array(
                [
                    [[-1.0, 1.0], [0.0, 0.0], [0.0, 1.0]],
                    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                    [[0.0, 0.0], [1.0, -1.0], [1.0, 0.0]],
                    [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                    [[-1.0, 1.0], [0.0, 0.0], [0.0, 1.0]],
                    [[-1.0, 1.0], [0.0, 0.0], [0.0, 1.0]],
                ]
            )
            + offset
        )
    ).all()


def test_neighborhood(triangles):
    neighborhood = triangles.neighborhood()

    assert (
        neighborhood.vertices
        == np.array(
            [
                [-1.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
            ]
        )
    ).all()

    assert (
        neighborhood.indices
        == np.array(
            [
                [0, 1, 2],
                [1, 2, 5],
                [1, 4, 5],
                [2, 3, 6],
                [2, 5, 6],
                [5, 6, 7],
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ]
        )
    ).all()


def test_means(triangles):
    means = triangles.means
    assert means == pytest.approx(
        np.array(
            [
                [0.33333333, 0.33333333],
                [0.66666667, 0.66666667],
            ]
        )
    )
