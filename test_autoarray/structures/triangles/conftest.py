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
