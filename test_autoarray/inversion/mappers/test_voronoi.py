import numpy as np
import autoarray as aa


def test__pix_indexes_for_sub_slim_index__matches_util(grid_2d_7x7):

    pixelization_grid = aa.Grid2D.manual_slim(
        [[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    nearest_pixelization_index_for_slim_index = np.array([0, 0, 1, 0, 0, 1, 2, 2, 3])

    pixelization_grid = aa.Grid2DVoronoi(
        grid=pixelization_grid,
        nearest_pixelization_index_for_slim_index=nearest_pixelization_index_for_slim_index,
        uses_interpolation=False,
    )

    mapper = aa.Mapper(source_grid_slim=grid_2d_7x7, source_mesh_grid=pixelization_grid)

    pix_indexes_for_sub_slim_index_util = np.array(
        [
            aa.util.mapper.pix_indexes_for_sub_slim_index_voronoi_from(
                grid=grid_2d_7x7,
                nearest_pixelization_index_for_slim_index=pixelization_grid.nearest_pixelization_index_for_slim_index,
                slim_index_for_sub_slim_index=grid_2d_7x7.mask.slim_index_for_sub_slim_index,
                pixelization_grid=pixelization_grid,
                neighbors=pixelization_grid.neighbors,
                neighbors_sizes=pixelization_grid.neighbors.sizes,
            ).astype("int")
        ]
    ).T

    assert (
        mapper.pix_indexes_for_sub_slim_index == pix_indexes_for_sub_slim_index_util
    ).all()

    pixelization_grid = aa.Grid2DVoronoi(
        grid=pixelization_grid,
        nearest_pixelization_index_for_slim_index=nearest_pixelization_index_for_slim_index,
        uses_interpolation=True,
    )

    mapper = aa.Mapper(source_grid_slim=grid_2d_7x7, source_mesh_grid=pixelization_grid)

    pix_indexes_for_sub_slim_index_util, sizes, weights = aa.util.mapper.pix_size_weights_voronoi_nn_from(
        grid=grid_2d_7x7, pixelization_grid=pixelization_grid
    )

    assert (
        mapper.pix_indexes_for_sub_slim_index == pix_indexes_for_sub_slim_index_util
    ).all()
