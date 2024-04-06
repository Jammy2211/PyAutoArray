import numpy as np
import autoarray as aa


def test__pix_indexes_for_sub_slim_index__matches_util(grid_2d_7x7):
    source_plane_mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    source_plane_mesh_grid = aa.Mesh2DVoronoi(
        values=source_plane_mesh_grid,
        uses_interpolation=False,
    )

    mapper_grids = aa.MapperGrids(
        source_plane_data_grid=grid_2d_7x7,
        source_plane_mesh_grid=source_plane_mesh_grid,
    )

    over_sampler = aa.OverSamplerUniform(mask=grid_2d_7x7.mask, sub_size=1)

    mapper = aa.Mapper(
        mapper_grids=mapper_grids,
        over_sampler=over_sampler,
        regularization=None,
    )

    pix_indexes_for_sub_slim_index_util = np.array(
        [
            aa.util.mapper.pix_indexes_for_sub_slim_index_voronoi_from(
                grid=np.array(grid_2d_7x7),
                slim_index_for_sub_slim_index=mapper.over_sampler.slim_for_sub_slim,
                mesh_grid=np.array(source_plane_mesh_grid),
                neighbors=source_plane_mesh_grid.neighbors,
                neighbors_sizes=source_plane_mesh_grid.neighbors.sizes,
            ).astype("int")
        ]
    ).T

    assert (
        mapper.pix_indexes_for_sub_slim_index == pix_indexes_for_sub_slim_index_util
    ).all()

    source_plane_mesh_grid = aa.Mesh2DVoronoi(
        values=source_plane_mesh_grid,
        uses_interpolation=True,
    )

    mapper_grids = aa.MapperGrids(
        source_plane_data_grid=grid_2d_7x7,
        source_plane_mesh_grid=source_plane_mesh_grid,
    )

    try:
        mapper = aa.Mapper(mapper_grids=mapper_grids, over_sampler=over_sampler, regularization=None)

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
