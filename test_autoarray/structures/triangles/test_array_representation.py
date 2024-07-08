import numpy as np

from autoarray.structures.triangles.array import ArrayTriangles


def test_contains():
    triangles = ArrayTriangles(
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

    containing = triangles.containing((0.1, 0.1))

    assert (
        containing.indices
        == np.array(
            [
                [0, 2, 1],
            ]
        )
    ).all()
    assert (
        containing.vertices
        == np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
    ).all()
