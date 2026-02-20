import numpy as np
import pytest

import autoarray as aa


def test__neighbors(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_grid = aa.InterpolatorDelaunay(
        mesh=aa.mesh.Delaunay(),
        mesh_grid=mesh_grid,
        data_grid_over_sampled=grid_2d_sub_1_7x7,
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

    mesh_grid = aa.Grid2DIrregular(
        [[0.0, 0.0], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]]
    )

    mesh = aa.InterpolatorDelaunay(
        mesh=aa.mesh.Delaunay(),
        mesh_grid=mesh_grid,
        data_grid_over_sampled=grid_2d_sub_1_7x7.over_sampled,
    )

    voronoi_areas = mesh.voronoi_areas

    assert voronoi_areas[1] == pytest.approx(1.39137102, 1.0e-4)
    assert voronoi_areas[3] == pytest.approx(29.836324, 1.0e-4)
    assert voronoi_areas[4] == pytest.approx(-1.0, 1.0e-4)


def test__scipy_delaunay__simplices(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_grid = aa.InterpolatorDelaunay(
        mesh=aa.mesh.Delaunay(),
        mesh_grid=mesh_grid,
        data_grid_over_sampled=grid_2d_sub_1_7x7,
    )

    assert (mesh_grid.delaunay.simplices[0, :] == np.array([3, 4, 0])).all()
    assert (mesh_grid.delaunay.simplices[1, :] == np.array([3, 5, 4])).all()
    assert (mesh_grid.delaunay.simplices[-1, :] == np.array([-1, -1, -1])).all()


def test__scipy_delaunay__split(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_grid = aa.InterpolatorDelaunay(
        mesh=aa.mesh.Delaunay(),
        mesh_grid=mesh_grid,
        data_grid_over_sampled=grid_2d_sub_1_7x7,
    )

    assert mesh_grid.delaunay.split_points[0, :] == pytest.approx(
        [2.30929334, 0.1], 1.0e-4
    )
    assert mesh_grid.delaunay.split_points[1, :] == pytest.approx(
        [-2.10929334, 0.1], 1.0e-4
    )
    assert mesh_grid.delaunay.split_points[-1, :] == pytest.approx(
        [2.1, -1.10929334], 1.0e-4
    )

    assert mesh_grid.delaunay.splitted_mappings[0, :] == pytest.approx(
        [2, -1, -1], 1.0e-4
    )
    assert mesh_grid.delaunay.splitted_mappings[1, :] == pytest.approx(
        [0, -1, -1], 1.0e-4
    )
    assert mesh_grid.delaunay.splitted_mappings[-1, :] == pytest.approx(
        [2, -1, -1], 1.0e-4
    )


def test__scipy_delaunay__split__uses_barycentric_dual_area_from(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_grid = aa.InterpolatorDelaunay(
        mesh=aa.mesh.Delaunay(),
        mesh_grid=mesh_grid,
        data_grid_over_sampled=grid_2d_sub_1_7x7,
        preloads=aa.Preloads(use_voronoi_areas=False),
    )

    assert mesh_grid.delaunay.split_points[0, :] == pytest.approx(
        [0.45059473, 0.1], 1.0e-4
    )
    assert mesh_grid.delaunay.split_points[1, :] == pytest.approx(
        [-0.25059473, 0.1], 1.0e-4
    )
    assert mesh_grid.delaunay.split_points[-1, :] == pytest.approx(
        [2.1, 0.39142161], 1.0e-4
    )