import numpy as np
import pytest

import autoarray as aa


def test__rectangular_mapper():

    mask = aa.Mask2D.manual(
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

    grid = aa.Grid2D.manual_mask(
        grid=[
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

    pix = aa.pix.Rectangular(shape=(3, 3))

    mapper = pix.mapper_from(
        source_grid_slim=grid,
        source_pixelization_grid=None,
        settings=aa.SettingsPixelization(use_border=False),
    )

    assert isinstance(mapper, aa.MapperRectangular)
    assert mapper.data_pixelization_grid == None
    assert mapper.source_pixelization_grid.shape_native_scaled == pytest.approx(
        (2.0, 2.0), 1.0e-4
    )
    assert mapper.source_pixelization_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

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

    mask = aa.Mask2D.manual(
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

    grid = aa.Grid2D.manual_mask(grid=grid, mask=mask)

    pix = aa.pix.DelaunayMagnification(shape=(3, 3))
    sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        grid=grid, unmasked_sparse_shape=pix.shape
    )

    mapper = pix.mapper_from(
        source_grid_slim=grid,
        source_pixelization_grid=sparse_grid,
        settings=aa.SettingsPixelization(use_border=False),
    )

    assert isinstance(mapper, aa.MapperDelaunay)
    assert mapper.source_grid_slim.shape_native_scaled == pytest.approx(
        (2.02, 2.01), 1.0e-4
    )
    assert (mapper.source_pixelization_grid == sparse_grid).all()
    assert mapper.source_pixelization_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

    print(mapper.mapping_matrix)

    assert mapper.mapping_matrix == pytest.approx(np.array(
            [
                [0.7524, 0.0, 0.2475, 0.0, 0.0],
                [0.0025, 0.7475, 0.2500, 0.0, 0.0],
                [0.0099, 0.0, 0.9900, 0.0, 0.0],
                [0.0025, 0.0, 0.2475, 0.75, 0.0],
                [0.0025, 0.0, 0.2475, 0.0, 0.75],
            ]
        ), 1.0e-2)


def test__voronoi_mapper():

    mask = aa.Mask2D.manual(
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

    grid = aa.Grid2D.manual_mask(grid=grid, mask=mask)

    pix = aa.pix.VoronoiMagnification(shape=(3, 3))
    sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        grid=grid, unmasked_sparse_shape=pix.shape
    )

    mapper = pix.mapper_from(
        source_grid_slim=grid,
        source_pixelization_grid=sparse_grid,
        settings=aa.SettingsPixelization(use_border=False),
    )

    assert isinstance(mapper, aa.MapperVoronoi)
    assert mapper.source_grid_slim.shape_native_scaled == pytest.approx(
        (2.02, 2.01), 1.0e-4
    )
    assert (mapper.source_pixelization_grid == sparse_grid).all()
    assert mapper.source_pixelization_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

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
