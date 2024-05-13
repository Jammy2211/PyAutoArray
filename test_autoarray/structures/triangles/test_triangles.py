import pytest


def test_height(triangles):
    assert triangles.height == pytest.approx(0.4330127)


@pytest.fixture
def rows(triangles):
    return triangles.rows


def test_rows(triangles):
    rows = triangles.rows

    assert len(rows) == 4


def test_alternation(rows):
    first_row = rows[0]
    assert len(first_row) == 3

    second_row = rows[1]
    assert len(second_row) == 4

    third_row = rows[2]
    assert len(third_row) == 3


def test_extremes(rows, triangles):
    first_row = rows[0]
    assert first_row[0] == (-0.5, -0.5)
    assert first_row[-1] == (-0.5, 0.5)

    second_row = rows[1]
    assert second_row[0][1] == -0.75

    last_row = rows[-1]
    assert last_row[0][0] >= triangles.y_max
    assert last_row[0][1] <= triangles.x_min
    assert last_row[-1][0] >= triangles.y_max
    assert last_row[-1][1] >= triangles.x_max


def test_triangles(triangles):
    assert len(triangles.triangles) == 15


def test_grid_2d(triangles):
    assert triangles.grid_2d
