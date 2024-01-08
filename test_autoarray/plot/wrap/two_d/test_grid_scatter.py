import autoarray as aa
import autoarray.plot as aplt

import matplotlib.pyplot as plt
import numpy as np


def test__scatter_grid():
    scatter = aplt.GridScatter(s=2, marker="x", c="k")

    scatter.scatter_grid(grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0))


def test__scatter_colored_grid__lists_of_coordinates_or_equivalent_2d_grids__with_color_array():
    scatter = aplt.GridScatter(s=2, marker="x", c="k")

    cmap = plt.get_cmap("jet")

    scatter.scatter_grid_colored(
        grid=aa.Grid2DIrregular(
            [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]
        ),
        color_array=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        cmap=cmap,
    )
    scatter.scatter_grid_colored(
        grid=aa.Grid2D.uniform(shape_native=(3, 2), pixel_scales=1.0),
        color_array=np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        cmap=cmap,
    )


def test__scatter_grid_indexes_1d__input_grid_is_ndarray_and_indexes_are_valid():
    scatter = aplt.GridScatter(s=2, marker="x", c="k")

    scatter.scatter_grid_indexes(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
        indexes=[0, 1, 2],
    )

    scatter.scatter_grid_indexes(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
        indexes=[[0, 1, 2]],
    )

    scatter.scatter_grid_indexes(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
        indexes=[[0, 1], [2]],
    )


def test__scatter_grid_indexes_2d__input_grid_is_ndarray_and_indexes_are_valid():
    scatter = aplt.GridScatter(s=2, marker="x", c="k")

    scatter.scatter_grid_indexes(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
        indexes=[(0, 0), (0, 1), (0, 2)],
    )

    scatter.scatter_grid_indexes(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
        indexes=[[(0, 0), (0, 1), (0, 2)]],
    )

    scatter.scatter_grid_indexes(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
        indexes=[[(0, 0), (0, 1)], [(0, 2)]],
    )

    scatter.scatter_grid_indexes(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
        indexes=[[[0, 0], [0, 1]], [[0, 2]]],
    )


def test__scatter_coordinates():
    scatter = aplt.GridScatter(s=2, marker="x", c="k")

    scatter.scatter_grid_list(grid_list=[aa.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])])
