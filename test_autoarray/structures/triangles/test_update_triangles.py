def test_update(triangles):
    transformed = 2 * triangles.grid_2d
    new = triangles.with_updated_grid(grid=transformed)

    assert len(new.rows) == len(triangles.rows)
    assert len(new.grid_2d) == len(triangles.grid_2d)
    assert (new.grid_2d == transformed).all()
    assert len(new.triangles) == 15
