import numpy as np
import scipy.spatial
import pytest

import autoarray as aa

from autoarray.inversion.pixelization.mesh_grid.delaunay_2d import (
    pix_indexes_for_sub_slim_index_delaunay_from,
)


def test__pix_indexes_for_sub_slim_index__matches_util(grid_2d_sub_1_7x7):
    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh = aa.mesh.Delaunay()

    mapper = mesh.mapper_from(
        mask=grid_2d_sub_1_7x7.mask,
        source_plane_data_grid=grid_2d_sub_1_7x7,
        source_plane_mesh_grid=mesh_grid,
        regularization=None
    )

    delaunay = scipy.spatial.Delaunay(mapper.mesh_geometry.mesh_grid_xy)

    simplex_index_for_sub_slim_index = delaunay.find_simplex(
        mapper.source_plane_data_grid
    )
    pix_indexes_for_simplex_index = mapper.delaunay.simplices

    pix_indexes_for_sub_slim_index_util = pix_indexes_for_sub_slim_index_delaunay_from(
        source_plane_data_grid=mapper.source_plane_data_grid.array,
        simplex_index_for_sub_slim_index=simplex_index_for_sub_slim_index,
        pix_indexes_for_simplex_index=pix_indexes_for_simplex_index,
        delaunay_points=mapper.delaunay.points,
    )
    sizes = (
        np.sum(pix_indexes_for_sub_slim_index_util >= 0, axis=1)
        .astype(np.int32)
        .astype("int")
    )

    assert (
        mapper.pix_indexes_for_sub_slim_index == pix_indexes_for_sub_slim_index_util
    ).all()
    assert (mapper.pix_sizes_for_sub_slim_index == sizes).all()

    assert (
        mapper.pix_indexes_for_sub_slim_index
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
        mapper.pix_sizes_for_sub_slim_index == np.array([1, 1, 3, 1, 1, 1, 1, 1, 1])
    ).all()


def test__scipy_delaunay__simplices(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_grid = aa.Mesh2DDelaunay(
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

    mesh_grid = aa.Mesh2DDelaunay(
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

    mesh_grid = aa.Mesh2DDelaunay(
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
