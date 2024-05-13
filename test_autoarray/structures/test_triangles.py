import pytest

from autoarray import Grid2D
from autoarray.geometry.geometry_util import central_scaled_coordinate_2d_from
from autoarray.structures.triangles import Triangles


def test_triangles():
    grid = Grid2D.uniform(
        shape_native=(100, 100),
        pixel_scales=0.05,
    )
    triangles = Triangles.for_grid(grid=grid)

    assert triangles.height == pytest.approx(0.04330127)
