import pytest


def test_update(triangles):
    transformed = 2 * triangles.grid_2d
    new = triangles.with_updated_grid(grid=transformed)

    assert len(new) == len(triangles.triangles)


@pytest.mark.parametrize(
    "point, expected",
    [
        ((0.1, 0.1), True),
        ((0.5, 0.5), True),
        ((1.0, 1.0), False),
        ((0.0, 0.0), True),
        ((0.0, 2.0), False),
    ],
)
def test_contains(
    point,
    expected,
    right_triangle,
):
    assert right_triangle.contains(point) is expected


def test_subgrid(right_triangle):
    subsample = right_triangle.subdivide()
    assert len(subsample) == 4


def test_containing(triangles):
    assert len(triangles.containing((0.5, 0.5))) == 1
