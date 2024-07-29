from autoarray.structures.triangles.array import ArrayTriangles


def test_flatten(triangles):
    (indices, vertices), _ = triangles.tree_flatten()

    assert (indices == triangles.indices).all()
    assert (vertices == triangles.vertices).all()


def test_unflatten(triangles):
    new_triangles = ArrayTriangles.tree_unflatten(
        (),
        (
            triangles.indices,
            triangles.vertices,
        ),
    )

    assert (new_triangles.indices == triangles.indices).all()
    assert (new_triangles.vertices == triangles.vertices).all()
