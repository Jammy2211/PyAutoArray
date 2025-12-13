import numpy as np
import pytest

import autoarray as aa


def test__interpolated_array_from():
    grid = aa.Grid2D.no_mask(
        values=[[0.0, 0.0], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    mesh = aa.Mesh2DDelaunay(values=grid)

    interpolated_array = mesh.interpolated_array_from(
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), shape_native=(3, 2)
    )

    assert interpolated_array.native == pytest.approx(
        np.array([[3.0, 5.0], [2.0, 5.0], [1.0, 5.0]]), 1.0e-4
    )

    interpolated_array = mesh.interpolated_array_from(
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), shape_native=(2, 3)
    )

    assert interpolated_array.native == pytest.approx(
        np.array([[3.0, 6.0, 5.0], [1.0, 4.0, 5.0]]), 1.0e-4
    )

    interpolated_array = mesh.interpolated_array_from(
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        shape_native=(3, 2),
        extent=(-0.4, 0.4, -0.4, 0.4),
    )

    assert interpolated_array.native == pytest.approx(
        np.array([[1.0, 1.907216], [1.0, 1.0], [1.0, 1.0]]), 1.0e-4
    )


def test__neighbors(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_grid = aa.Mesh2DDelaunay(
        values=mesh_grid, source_plane_data_grid_over_sampled=grid_2d_sub_1_7x7
    )

    neighbors = mesh_grid.neighbors

    expected = np.array(
        [
            [1, 2, 3, 4],
            [0, 2, 3, 5],
            [0, 1, 5, -1],
            [0, 1, 4, 5],
            [0, 3, 5, -1],
            [1, 2, 3, 4],
        ]
    )

    assert all(
        set(neighbors[i]) - {-1} == set(expected[i]) - {-1}
        for i in range(neighbors.shape[0])
    )


def test__voronoi_areas_via_delaunay_from(grid_2d_sub_1_7x7):

    mesh_grid = np.array(
        [[0.0, 0.0], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]]
    )

    mesh = aa.Mesh2DDelaunay(
        values=mesh_grid,
        source_plane_data_grid_over_sampled=grid_2d_sub_1_7x7.over_sampled,
    )

    voronoi_areas = mesh.voronoi_areas

    assert voronoi_areas[1] == pytest.approx(1.39137102, 1.0e-4)
    assert voronoi_areas[3] == pytest.approx(29.836324, 1.0e-4)
    assert voronoi_areas[4] == pytest.approx(-1.0, 1.0e-4)
