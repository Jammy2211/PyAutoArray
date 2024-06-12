import pytest

from autoarray import Grid2D
from autoarray.structures.triangles.triangle import Triangle
from autoarray.structures.triangles.triangles import Triangles


@pytest.fixture
def right_triangle():
    return Triangle(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )


@pytest.fixture(name="triangles")
def make_triangles():
    grid = Grid2D.uniform(
        shape_native=(3, 3),
        pixel_scales=0.5,
    )
    return Triangles.for_grid(grid=grid)


@pytest.fixture(name="triangle")
def make_triangle():
    return Triangle(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, 3**0.5 / 2),
    )
