import pytest

from autoarray.structures.triangles.subsample_triangles import SubsampleTriangles


@pytest.fixture(name="subsample_triangles")
def make_subsample_triangles(triangle):
    return SubsampleTriangles([triangle])


def test_subsample_triangles(subsample_triangles):
    assert len(subsample_triangles.triangles) == 4


def test_grid_2d(subsample_triangles):
    assert len(subsample_triangles.grid_2d) == 6
