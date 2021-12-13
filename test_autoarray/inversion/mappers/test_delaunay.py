import numpy as np
import autoarray as aa


def test__pixelization_indexes_for_sub_slim_index__matches_util(grid_2d_7x7):
    pixelization_grid = aa.Grid2D.manual_slim(
        [[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    pixelization_grid = aa.Grid2DDelaunay(grid=pixelization_grid)

    mapper = aa.Mapper(
        source_grid_slim=grid_2d_7x7, source_pixelization_grid=pixelization_grid
    )

    pixelization_indexes_for_sub_slim_index_util, sizes = aa.util.mapper.pixelization_indexes_for_sub_slim_index_delaunay_from(
        delaunay=mapper.delaunay, source_grid_slim=mapper.source_grid_slim
    )
    pixelization_indexes_for_sub_slim_index_util = pixelization_indexes_for_sub_slim_index_util.astype(
        "int"
    )
    sizes = sizes.astype("int")

    assert (
        mapper.pixelization_indexes_for_sub_slim_index.mappings
        == pixelization_indexes_for_sub_slim_index_util
    ).all()
    assert (mapper.pixelization_indexes_for_sub_slim_index.sizes == sizes).all()

    assert (
        mapper.pixelization_indexes_for_sub_slim_index.mappings
        == np.array(
            [
                [0, -1, -1],
                [1, -1, -1],
                [1, 5, 3],
                [0, -1, -1],
                [0, -1, -1],
                [3, -1, -1],
                [0, -1, -1],
                [0, -1, -1],
                [3, -1, -1],
            ]
        )
    ).all()

    assert (
        mapper.pixelization_indexes_for_sub_slim_index.sizes
        == np.array([1, 1, 3, 1, 1, 1, 1, 1, 1])
    ).all()
