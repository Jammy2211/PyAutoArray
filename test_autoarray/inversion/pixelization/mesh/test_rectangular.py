import pytest

import autoarray as aa


def test__sparse_grid_from__returns_none_as_not_used(sub_grid_2d_7x7):

    pixelization = aa.mesh.Rectangular(shape=(3, 3))

    assert pixelization.data_mesh_grid_from(data_grid_slim=sub_grid_2d_7x7) == None


def test__preloads_used_for_relocated_grid(sub_grid_2d_7x7):

    pixelization = aa.mesh.Rectangular(shape=(3, 3))

    relocated_grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    mapper_grids = pixelization.mapper_grids_from(
        source_grid_slim=relocated_grid,
        source_mesh_grid=None,
        settings=aa.SettingsPixelization(use_border=True),
        preloads=aa.Preloads(relocated_grid=relocated_grid),
    )

    assert mapper_grids.source_grid_slim == pytest.approx(relocated_grid, 1.0e-4)
