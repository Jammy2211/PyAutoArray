import numpy as np
import pytest

import autoarray as aa

from autoarray.structures.mesh.delaunay_2d import voronoi_areas_from


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


def test__mesh_areas():
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

    mesh = aa.Mesh2DVoronoi(values=grid)

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

    import scipy.spatial

    mesh_grid = np.array(
        [[0.0, 0.0], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]]
    )

    delaunay = scipy.spatial.Delaunay(mesh_grid)

    voronoi_areas = voronoi_areas_from(
        mesh_grid,
    )

    voronoi = scipy.spatial.Voronoi(
        mesh_grid,
        qhull_options="Qbb Qc Qx Qm",
    )

    voronoi_vertices = voronoi.vertices
    voronoi_regions = voronoi.regions
    voronoi_point_region = voronoi.point_region

    pixels = mesh_grid.shape[0]

    region_areas = np.zeros(pixels)

    for i in range(pixels):
        region_vertices_indexes = voronoi_regions[voronoi_point_region[i]]
        if -1 in region_vertices_indexes:
            region_areas[i] = -1
        else:
            region_areas[i] = aa.util.grid_2d.compute_polygon_area(
                voronoi_vertices[region_vertices_indexes]
            )

    assert voronoi_areas[1] == pytest.approx(region_areas[1], 1.0e-4)
    assert voronoi_areas[3] == pytest.approx(region_areas[3], 1.0e-4)

    # Old Voronoi cell code put -1 in edge pixels, new code puts large area

    assert voronoi_areas[4] == pytest.approx(32.83847776, 1.0e-4)
