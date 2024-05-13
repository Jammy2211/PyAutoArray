import pytest

from autoarray import Grid2D
from autoarray.structures.triangles import Triangles


@pytest.fixture(name="triangles")
def make_triangles():
    grid = Grid2D.uniform(
        shape_native=(3, 3),
        pixel_scales=0.5,
    )
    return Triangles.for_grid(grid=grid)


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
