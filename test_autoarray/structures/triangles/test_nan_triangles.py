import pytest

try:
    from jax import numpy as np
    from autoarray.structures.triangles.jax_array import ArrayTriangles
except ImportError:
    import numpy as np
    from autoarray.structures.triangles.array import ArrayTriangles


pytest.importorskip("jax")


@pytest.fixture
def nan_triangles():
    return ArrayTriangles(
        indices=np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
                [-1, -1, -1],
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


def test_nan_triangles(nan_triangles, compare_with_nans):
    assert compare_with_nans(
        nan_triangles.triangles,
        np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            ]
        ),
    ).all()


def test_up_sample_nan_triangles(nan_triangles, compare_with_nans):
    up_sampled = nan_triangles.up_sample()

    assert compare_with_nans(
        up_sampled.triangles,
        np.array(
            [
                [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0]],
                [[0.0, 0.5], [0.0, 1.0], [0.5, 0.5]],
                [[0.0, 0.5], [0.5, 0.0], [0.5, 0.5]],
                [[0.0, 1.0], [0.5, 0.5], [0.5, 1.0]],
                [[0.5, 0.0], [0.5, 0.5], [1.0, 0.0]],
                [[0.5, 0.5], [0.5, 1.0], [1.0, 0.5]],
                [[0.5, 0.5], [1.0, 0.0], [1.0, 0.5]],
                [[0.5, 1.0], [1.0, 0.5], [1.0, 1.0]],
            ]
        ),
    )


def test_neighborhood(nan_triangles, compare_with_nans):
    assert compare_with_nans(
        nan_triangles.neighborhood().triangles,
        np.array(
            [
                [[-1.0, 1.0], [0.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, -1.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 2.0], [1.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                [[1.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
            ]
        ),
    )
