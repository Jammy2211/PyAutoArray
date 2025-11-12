import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

import pytest

from autoarray.structures.triangles.shape import Point
from autoarray.structures.triangles.array import ArrayTriangles

ArrayTriangles = register_pytree_node_class(ArrayTriangles)
Point = register_pytree_node_class(Point)

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
        max_containing_size=5,
    )


@pytest.mark.parametrize(
    "point, vertices, indices",
    [
        (
            Point(0.1, 0.1),
            jnp.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ),
            jnp.array([0, -1, -1, -1, -1]),
        ),
        (
            Point(0.6, 0.6),
            jnp.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ]
            ),
            jnp.array([1, -1, -1, -1, -1]),
        ),
        (
            Point(0.5, 0.5),
            jnp.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ]
            ),
            jnp.array([0, 1, -1, -1, -1]),
        ),
    ],
)
def test_contains_vertices(
    triangles,
    point,
    vertices,
    indices,
):
    containing_indices = jax.jit(triangles.containing_indices)(point)

    assert (containing_indices == indices).all()


@pytest.mark.parametrize(
    "indexes, vertices, indices",
    [
        (
            jnp.array([0]),
            jnp.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ),
            jnp.array(
                [
                    [0, 1, 2],
                ]
            ),
        ),
        (
            jnp.array([1]),
            jnp.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ]
            ),
            jnp.array(
                [
                    [0, 1, 2],
                ]
            ),
        ),
        (
            jnp.array([0, 1]),
            jnp.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ],
            ),
            jnp.array(
                [
                    [0, 1, 2],
                    [1, 2, 3],
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
    containing = jax.jit(triangles.for_indexes)(indexes)

    assert (containing.indices == indices).all()
    assert compare_with_nans(
        containing.vertices,
        vertices,
    ).all()


def test_negative_index(
    triangles,
    compare_with_nans,
):
    indexes = jnp.array([0, -1])

    containing = jax.jit(triangles.for_indexes)(indexes)

    assert (
        containing.indices
        == jnp.array(
            [
                [-1, -1, -1],
                [0, 1, 2],
            ],
        )
    ).all()
    assert compare_with_nans(
        containing.vertices,
        jnp.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
    )


def test_up_sample(
    triangles,
    compare_with_nans,
):
    up_sampled = jax.jit(triangles.up_sample)()

    assert compare_with_nans(
        up_sampled.vertices,
        jnp.array(
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
        ),
    )

    assert (
        up_sampled.indices
        == jnp.array(
            [
                [0, 1, 3],
                [1, 2, 4],
                [1, 3, 4],
                [2, 4, 5],
                [3, 4, 6],
                [4, 5, 7],
                [4, 6, 7],
                [5, 7, 8],
            ]
        )
    ).all()


@pytest.mark.parametrize(
    "offset",
    [-1, 0, 1],
)
def test_simple_neighborhood(offset, compare_with_nans):
    triangles = ArrayTriangles(
        indices=jnp.array(
            [
                [0, 1, 2],
            ]
        ),
        vertices=jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        + offset,
    )

    assert compare_with_nans(
        jax.jit(triangles.neighborhood)().triangles,
        (
            jnp.array(
                [
                    [[-1.0, 1.0], [0.0, 0.0], [0.0, 1.0]],
                    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                    [[0.0, 0.0], [1.0, -1.0], [1.0, 0.0]],
                    [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                ]
            )
            + offset
        ),
    )


def test_neighborhood(triangles, compare_with_nans):
    neighborhood = jax.jit(triangles.neighborhood)()

    assert compare_with_nans(
        neighborhood.vertices,
        jnp.array(
            [
                [-1.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ]
        ),
    )

    assert (
        neighborhood.indices
        == jnp.array(
            [
                [0, 1, 2],
                [1, 2, 5],
                [1, 4, 5],
                [2, 3, 6],
                [2, 5, 6],
                [5, 6, 7],
                [-1, -1, -1],
                [-1, -1, -1],
            ]
        )
    ).all()


def test_means(triangles):
    means = triangles.means
    assert means == pytest.approx(
        jnp.array(
            [
                [0.33333333, 0.33333333],
                [0.66666667, 0.66666667],
            ]
        )
    )
