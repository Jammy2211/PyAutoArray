import numpy as np

import autoarray as aa


def test__pix_indexes_for_sub_slim_index__matches_util():

    grid = aa.Grid2D.manual_slim(
        [
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
        source_plane_data_grid=grid, source_plane_mesh_grid=mesh_grid
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    pix_indexes_for_sub_slim_index_util = np.array(
        [
            aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
                grid_scaled_2d_slim=grid,
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

    mapper_grids = aa.MapperGrids(
        source_plane_data_grid=grid_2d_7x7,
        source_plane_mesh_grid=mesh_grid,
        hyper_data=image_7x7,
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    pixel_signals = mapper.pixel_signals_from(signal_scale=2.0)

    pixel_signals_util = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=9,
        signal_scale=2.0,
        pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
        pixel_weights=mapper.pix_weights_for_sub_slim_index,
        slim_index_for_sub_slim_index=grid_2d_7x7.mask.slim_index_for_sub_slim_index,
        hyper_data=image_7x7,
    )

    assert (pixel_signals == pixel_signals_util).all()
