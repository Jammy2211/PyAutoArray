import numpy as np
import pytest

import autoarray as aa


def test__rectangular_mapper():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        pixel_scales=2.0,
        sub_size=2,
    )

    # Assume a 2x2 sub-grid, so each of our 5 masked_image-pixels are split into 4.
    # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
    # happen for a real lens calculation. This is to make a mapping_matrix matrix which explicitly tests the
    # sub-grid.

    grid = aa.Grid2D(
        values=[
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        mask=mask,
    )

    mesh = aa.mesh.Rectangular(shape=(3, 3))

    mapper_grids = mesh.mapper_grids_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=None,
        settings=aa.SettingsPixelization(use_border=False),
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    assert isinstance(mapper, aa.MapperRectangularNoInterp)
    assert mapper.image_plane_mesh_grid == None
    assert mapper.source_plane_mesh_grid.geometry.shape_native_scaled == pytest.approx(
        (2.0, 2.0), 1.0e-4
    )
    assert mapper.source_plane_mesh_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

    assert (
        mapper.mapping_matrix
        == np.array(
            [
                [0.75, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.75],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()
    assert mapper.shape_native == (3, 3)


def test__delaunay_mapper():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, False, False, False, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
        ],
        pixel_scales=1.0,
        sub_size=2,
    )

    grid = np.array(
        [
            [1.01, 0.0],
            [1.01, 0.0],
            [1.01, 0.0],
            [0.01, 0.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.0, 1.01],
            [0.0, 1.01],
            [0.0, 1.01],
            [0.01, 0.0],
            [-1.01, 0.0],
            [-1.01, 0.0],
            [-1.01, 0.0],
            [0.01, 0.0],
        ]
    )

    grid = aa.Grid2D(values=grid, mask=mask)

    mesh = aa.mesh.Delaunay()
    image_mesh = aa.image_mesh.Overlay(shape_overlay=(3, 3))
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        grid=grid, adapt_data=None
    )

    mapper_grids = mesh.mapper_grids_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_plane_mesh_grid,
        settings=aa.SettingsPixelization(use_border=False),
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    assert isinstance(mapper, aa.MapperDelaunay)
    assert mapper.source_plane_data_grid.shape_native_scaled_interior == pytest.approx(
        (2.02, 2.01), 1.0e-4
    )
    assert (mapper.source_plane_mesh_grid == image_plane_mesh_grid).all()
    assert mapper.source_plane_mesh_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

    assert mapper.mapping_matrix == pytest.approx(
        np.array(
            [
                [0.7524, 0.0, 0.2475, 0.0, 0.0],
                [0.0025, 0.7475, 0.2500, 0.0, 0.0],
                [0.0099, 0.0, 0.9900, 0.0, 0.0],
                [0.0025, 0.0, 0.2475, 0.75, 0.0],
                [0.0025, 0.0, 0.2475, 0.0, 0.75],
            ]
        ),
        1.0e-2,
    )


def test__voronoi_mapper():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, False, False, False, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
        ],
        pixel_scales=1.0,
        sub_size=2,
    )

    grid = np.array(
        [
            [1.01, 0.0],
            [1.01, 0.0],
            [1.01, 0.0],
            [0.01, 0.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.0, 1.01],
            [0.0, 1.01],
            [0.0, 1.01],
            [0.01, 0.0],
            [-1.01, 0.0],
            [-1.01, 0.0],
            [-1.01, 0.0],
            [0.01, 0.0],
        ]
    )

    grid = aa.Grid2D(values=grid, mask=mask)

    mesh = aa.mesh.Voronoi()
    image_mesh = aa.image_mesh.Overlay(shape_overlay=(3, 3))
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        grid=grid, adapt_data=None
    )

    mapper_grids = mesh.mapper_grids_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_plane_mesh_grid,
        settings=aa.SettingsPixelization(use_border=False),
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    assert isinstance(mapper, aa.MapperVoronoiNoInterp)
    assert mapper.source_plane_data_grid.shape_native_scaled_interior == pytest.approx(
        (2.02, 2.01), 1.0e-4
    )
    assert (mapper.source_plane_mesh_grid == image_plane_mesh_grid).all()
    assert mapper.source_plane_mesh_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

    assert (
        mapper.mapping_matrix
        == np.array(
            [
                [0.75, 0.0, 0.25, 0.0, 0.0],
                [0.0, 0.75, 0.25, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.25, 0.75, 0.0],
                [0.0, 0.0, 0.25, 0.0, 0.75],
            ]
        )
    ).all()
