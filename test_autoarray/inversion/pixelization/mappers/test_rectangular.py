import numpy as np
import pytest

import autoarray as aa


def test__pix_indexes_for_sub_slim_index__matches_util():
    grid = aa.Grid2D.no_mask(
        values=[
            [1.5, -1.0],
            [1.3, 0.0],
            [1.0, 1.9],
            [-0.20, -1.0],
            [-5.0, 0.32],
            [6.5, 1.0],
            [-0.34, -7.34],
            [-0.34, 0.75],
            [-6.0, 8.0],
        ],
        pixel_scales=1.0,
        shape_native=(3, 3),
        over_sample_size=1,
    )

    mesh_grid = aa.Mesh2DRectangularUniform.overlay_grid(
        shape_native=(3, 3), grid=grid.over_sampled
    )

    mapper_grids = aa.MapperGrids(
        mask=grid.mask, source_plane_data_grid=grid, source_plane_mesh_grid=mesh_grid
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    mappings, weights = (
        aa.util.mapper.rectangular_mappings_weights_via_interpolation_from(
            shape_native=(3, 3),
            source_plane_mesh_grid=mesh_grid.array,
            source_plane_data_grid=aa.Grid2DIrregular(
                mapper_grids.source_plane_data_grid.over_sampled
            ).array,
        )
    )

    assert (mapper.pix_sub_weights.mappings == mappings).all()
    assert (mapper.pix_sub_weights.weights == weights).all()


def test__pixel_signals_from__matches_util(grid_2d_sub_1_7x7, image_7x7):
    mesh_grid = aa.Mesh2DRectangularUniform.overlay_grid(
        shape_native=(3, 3), grid=grid_2d_sub_1_7x7.over_sampled
    )

    mapper_grids = aa.MapperGrids(
        mask=grid_2d_sub_1_7x7.mask,
        source_plane_data_grid=grid_2d_sub_1_7x7,
        source_plane_mesh_grid=mesh_grid,
        adapt_data=image_7x7,
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    pixel_signals = mapper.pixel_signals_from(signal_scale=2.0)

    pixel_signals_util = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=9,
        signal_scale=2.0,
        pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
        pixel_weights=mapper.pix_weights_for_sub_slim_index,
        slim_index_for_sub_slim_index=grid_2d_sub_1_7x7.over_sampler.slim_for_sub_slim,
        adapt_data=image_7x7,
    )

    assert (pixel_signals == pixel_signals_util).all()


def test__areas_transformed(mask_2d_7x7):

    grid = aa.Grid2DIrregular(
        [
            [-1.5, -1.5],
            [-1.5, 0.0],
            [-1.5, 1.5],
            [0.0, -1.5],
            [0.0, 0.0],
            [0.0, 1.5],
            [1.5, -1.5],
            [1.5, 0.0],
            [1.5, 1.5],
        ],
    )

    mesh = aa.Mesh2DRectangularUniform.overlay_grid(
        shape_native=(3, 3), grid=grid, buffer=1e-8
    )

    mapper_grids = aa.MapperGrids(
        mask=mask_2d_7x7,
        source_plane_data_grid=grid,
        source_plane_mesh_grid=mesh,
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    assert mapper.areas_transformed[4] == pytest.approx(
        4.0,
        abs=1e-8,
    )


def test__edges_transformed(mask_2d_7x7):

    grid = aa.Grid2DIrregular(
        [
            [-1.5, -1.5],
            [-1.5, 0.0],
            [-1.5, 1.5],
            [0.0, -1.5],
            [0.0, 0.0],
            [0.0, 1.5],
            [1.5, -1.5],
            [1.5, 0.0],
            [1.5, 1.5],
        ],
    )

    mesh = aa.Mesh2DRectangularUniform.overlay_grid(
        shape_native=(3, 3), grid=grid, buffer=1e-8
    )

    mapper_grids = aa.MapperGrids(
        mask=mask_2d_7x7,
        source_plane_data_grid=grid,
        source_plane_mesh_grid=mesh,
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    assert mapper.edges_transformed[4] == pytest.approx(
        np.array(
            [1.5, 1.5],  # left
        ),
        abs=1e-8,
    )
