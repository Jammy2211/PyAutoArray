import numpy as np
import autoarray as aa


def test__pix_indexes_for_sub_slim_index__matches_util(grid_2d_7x7):

    source_plane_mesh_grid = aa.Grid2D.manual_slim(
        [[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    nearest_pixelization_index_for_slim_index = np.array([0, 0, 1, 0, 0, 1, 2, 2, 3])

    source_plane_mesh_grid = aa.Mesh2DVoronoi(
        grid=source_plane_mesh_grid,
        nearest_pixelization_index_for_slim_index=nearest_pixelization_index_for_slim_index,
        uses_interpolation=False,
    )

    mapper_grids = aa.MapperGrids(
        source_plane_data_grid=grid_2d_7x7,
        source_plane_mesh_grid=source_plane_mesh_grid,
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    pix_indexes_for_sub_slim_index_util = np.array(
        [
            aa.util.mapper.pix_indexes_for_sub_slim_index_voronoi_from(
                grid=grid_2d_7x7,
                nearest_pixelization_index_for_slim_index=source_plane_mesh_grid.nearest_pixelization_index_for_slim_index,
                slim_index_for_sub_slim_index=grid_2d_7x7.mask.indexes.slim_index_for_sub_slim_index,
                mesh_grid=source_plane_mesh_grid,
                neighbors=source_plane_mesh_grid.neighbors,
                neighbors_sizes=source_plane_mesh_grid.neighbors.sizes,
            ).astype("int")
        ]
    ).T

    assert (
        mapper.pix_indexes_for_sub_slim_index == pix_indexes_for_sub_slim_index_util
    ).all()

    source_plane_mesh_grid = aa.Mesh2DVoronoi(
        grid=source_plane_mesh_grid,
        nearest_pixelization_index_for_slim_index=nearest_pixelization_index_for_slim_index,
        uses_interpolation=True,
    )

    mapper_grids = aa.MapperGrids(
        source_plane_data_grid=grid_2d_7x7,
        source_plane_mesh_grid=source_plane_mesh_grid,
    )

    try:
        mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

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
