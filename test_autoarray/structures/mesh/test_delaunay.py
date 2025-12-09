import numpy as np
import pytest

import autoarray as aa


def test__edge_pixel_list():

    grid = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
        ]
    )

    mesh = aa.Mesh2DDelaunay(values=grid)

    assert mesh.edge_pixel_list == [0, 1, 2, 3, 5, 6, 7, 8]



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


def test__voronoi_areas_via_delaunay_from():

    mesh_grid = np.array(
        [[0.0, 0.0], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]]
    )

    mesh = aa.Mesh2DDelaunay(mesh_grid)

    voronoi_areas = mesh.delaunay.voronoi_areas

    assert voronoi_areas[1] == pytest.approx(1.39137102, 1.0e-4)
    assert voronoi_areas[3] == pytest.approx(29.836324, 1.0e-4)
    assert voronoi_areas[4] == pytest.approx(-1.0, 1.0e-4)

def test__voronoi_pixel_areas_for_split():

    grid = np.array(
        [
            [-2.0, 0.0],
            [-np.sqrt(2), np.sqrt(2)],
            [0.0, 0.0],
            [0.0, 2.0],
            [np.sqrt(2), np.sqrt(2)],
            [2.0, 0.0],
            [np.sqrt(2), -np.sqrt(2)],
            [0.0, -2.0],
            [-np.sqrt(2), -np.sqrt(2)],
        ]
    )

    mesh = aa.Mesh2DDelaunay(grid)

    assert mesh.voronoi_pixel_areas_for_split == pytest.approx(
        np.array(
            [
                -0.1372583,
                -0.1372583,
                -0.1372583,
                -0.1372583,
                -0.1372583,
                -0.1372583,
                -0.1372583,
                -0.1372583,
                -0.1372583,
            ]
        ),
        1e-6,
    )