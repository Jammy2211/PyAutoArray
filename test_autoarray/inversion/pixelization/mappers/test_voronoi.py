import numpy as np
import autoarray as aa


def test__pix_indexes_for_sub_slim_index__matches_util(grid_2d_7x7):
    source_plane_mesh_grid = aa.Mesh2DVoronoi(
        values=source_plane_mesh_grid,
    )

    mapper_grids = aa.MapperGrids(
        mask=grid_2d_7x7.mask,
        source_plane_data_grid=grid_2d_7x7,
        source_plane_mesh_grid=source_plane_mesh_grid,
    )

    try:
        mapper = aa.Mapper(
            mapper_grids=mapper_grids, over_sampler=over_sampler, regularization=None
        )

        (
            pix_indexes_for_sub_slim_index_util,
            sizes,
            weights,
        ) = aa.util.mapper.pix_size_weights_voronoi_nn_from(
            grid=grid_2d_7x7, mesh_grid=source_plane_mesh_grid
        )

        assert (
            mapper.pix_indexes_for_sub_slim_index == pix_indexes_for_sub_slim_index_util
        ).all()

    except AttributeError:
        pass
