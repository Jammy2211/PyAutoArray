import pytest

from autoarray.structures.triangles.triangles import Triangle


def test_update(triangles):
    transformed = 2 * triangles.grid_2d
    new = triangles.with_updated_grid(grid=transformed)

    assert len(new) == len(triangles.triangles)


@pytest.fixture
def triangle():
    return Triangle(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )


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
    triangle,
):
    assert triangle.contains(point) is expected


@pytest.mark.parametrize(
    "point, buffer, expected",
    [
        ((0.6, 0.5), 0.1, True),
        ((0.6, 0.5), 0.01, False),
        ((0.0, -0.1), 0.2, True),
        ((0.0, -0.3), 0.2, False),
    ],
)
def test_buffer(
    triangle,
    point,
    buffer,
    expected,
):
    assert triangle.contains(point, buffer=buffer) is expected


def test_subgrid(triangle):
    subsample = triangle.subdivide()
    assert len(subsample) == 4


def test_containing(triangles):
    assert len(triangles.containing((0.5, 0.5))) == 1