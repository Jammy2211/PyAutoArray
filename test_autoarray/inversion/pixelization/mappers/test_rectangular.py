import numpy as np

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
    )

    mesh_grid = aa.Mesh2DRectangular.overlay_grid(shape_native=(3, 3), grid=grid)

    mapper_grids = aa.MapperGrids(
        mask=grid.mask, source_plane_data_grid=grid, source_plane_mesh_grid=mesh_grid
    )

    over_sampler = aa.OverSampler(mask=grid.mask, sub_size=1)

    mapper = aa.Mapper(
        mapper_grids=mapper_grids, over_sampler=over_sampler, regularization=None
    )

    pix_indexes_for_sub_slim_index_util = np.array(
        [
            aa.util.geometry.grid_pixel_indexes_2d_slim_from(
                grid_scaled_2d_slim=np.array(grid),
                shape_native=mesh_grid.shape_native,
                pixel_scales=mesh_grid.pixel_scales,
                origin=mesh_grid.origin,
            ).astype("int")
        ]
    ).T

    assert (
        mapper.pix_indexes_for_sub_slim_index == pix_indexes_for_sub_slim_index_util
    ).all()


def test__pixel_signals_from__matches_util(grid_2d_7x7, image_7x7):
    mesh_grid = aa.Mesh2DRectangular.overlay_grid(shape_native=(3, 3), grid=grid_2d_7x7)

    over_sampler = aa.OverSampler(mask=grid_2d_7x7.mask, sub_size=1)

    mapper_grids = aa.MapperGrids(
        mask=grid_2d_7x7.mask,
        source_plane_data_grid=grid_2d_7x7,
        source_plane_mesh_grid=mesh_grid,
        adapt_data=image_7x7,
    )

    mapper = aa.Mapper(
        mapper_grids=mapper_grids, over_sampler=over_sampler, regularization=None
    )

    pixel_signals = mapper.pixel_signals_from(signal_scale=2.0)

    pixel_signals_util = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=9,
        signal_scale=2.0,
        pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
        pixel_weights=mapper.pix_weights_for_sub_slim_index,
        slim_index_for_sub_slim_index=over_sampler.slim_for_sub_slim,
        adapt_data=np.array(image_7x7),
    )

    assert (pixel_signals == pixel_signals_util).all()
