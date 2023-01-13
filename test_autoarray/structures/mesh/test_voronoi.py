import numpy as np
import pytest
import scipy.spatial

from autoarray import exc
import autoarray as aa


def test__neighbors__compare_to_mesh_util():

    # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

    grid = np.array(
        [
            [3.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [8.0, 3.0],
            [1.0, 3.0],
            [1.0, 9.0],
            [6.0, 31.0],
            [0.0, 2.0],
            [3.0, 5.0],
        ]
    )

    mesh = aa.Mesh2DVoronoi(values=grid)

    voronoi = scipy.spatial.Voronoi(
        np.asarray([grid[:, 1], grid[:, 0]]).T, qhull_options="Qbb Qc Qx Qm"
    )

    (neighbors_util, neighbors_sizes_util) = aa.util.mesh.voronoi_neighbors_from(
        pixels=9, ridge_points=np.array(voronoi.ridge_points)
    )

    assert (mesh.neighbors == neighbors_util).all()
    assert (mesh.neighbors.sizes == neighbors_sizes_util).all()


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

    assert mesh.voronoi_pixel_areas == pytest.approx(
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


def test__mesh_grid__attributes():

    mesh = aa.Mesh2DVoronoi(
        values=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 4.0]]),
        nearest_pixelization_index_for_slim_index=np.array([0, 1, 2, 3]),
    )

    assert type(mesh) == aa.Mesh2DVoronoi
    assert (mesh == np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 4.0]])).all()
    assert (
        mesh.nearest_pixelization_index_for_slim_index == np.array([0, 1, 2, 3])
    ).all()


def test__from_unmasked_sparse_shape_and_grid():

    mask = aa.Mask2D(
        mask=np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        ),
        pixel_scales=(0.5, 0.5),
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        unmasked_sparse_shape=(10, 10), grid=grid
    )

    mesh = aa.Mesh2DVoronoi(
        values=sparse_grid,
        nearest_pixelization_index_for_slim_index=sparse_grid.sparse_index_for_slim_index,
    )

    assert (sparse_grid == mesh).all()
    assert (
        sparse_grid.sparse_index_for_slim_index
        == mesh.nearest_pixelization_index_for_slim_index
    ).all()


def test__voronoi_grid__simple_shapes_make_voronoi_grid_correctly():

    # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

    grid = np.array(
        [
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ]
    )

    mesh = aa.Mesh2DVoronoi(values=grid)

    assert (
        mesh.voronoi.points
        == np.array(
            [
                [0.0, 2.0],
                [1.0, 2.0],
                [2.0, 2.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        )
    ).all()

    # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

    grid = np.array([[-1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]])

    mesh = aa.Mesh2DVoronoi(values=grid)

    mesh.voronoi.vertices = list(map(lambda x: list(x), mesh.voronoi.vertices))

    assert [0, 1.0] in mesh.voronoi.vertices
    assert [-1.0, 0.0] in mesh.voronoi.vertices
    assert [1.0, 0.0] in mesh.voronoi.vertices
    assert [0.0, -1.0] in mesh.voronoi.vertices

    # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

    grid = np.array(
        [
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ]
    )

    mesh = aa.Mesh2DVoronoi(values=grid)

    # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
    # to look for each list

    mesh.voronoi.vertices = list(map(lambda x: list(x), mesh.voronoi.vertices))

    assert [0.5, 1.5] in mesh.voronoi.vertices
    assert [1.5, 0.5] in mesh.voronoi.vertices
    assert [0.5, 0.5] in mesh.voronoi.vertices
    assert [1.5, 1.5] in mesh.voronoi.vertices

    # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

    grid = np.array(
        [
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ]
    )

    mesh = aa.Mesh2DVoronoi(values=grid)

    # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
    # to look for each list

    mesh.voronoi.ridge_grid = list(map(lambda x: list(x), mesh.voronoi.ridge_points))

    assert len(mesh.voronoi.ridge_points) == 12

    assert [0, 1] in mesh.voronoi.ridge_points or [1, 0] in mesh.voronoi.ridge_points
    assert [1, 2] in mesh.voronoi.ridge_points or [2, 1] in mesh.voronoi.ridge_points
    assert [3, 4] in mesh.voronoi.ridge_points or [4, 3] in mesh.voronoi.ridge_points
    assert [4, 5] in mesh.voronoi.ridge_points or [5, 4] in mesh.voronoi.ridge_points
    assert [6, 7] in mesh.voronoi.ridge_points or [7, 6] in mesh.voronoi.ridge_points
    assert [7, 8] in mesh.voronoi.ridge_points or [8, 7] in mesh.voronoi.ridge_points

    assert [0, 3] in mesh.voronoi.ridge_points or [3, 0] in mesh.voronoi.ridge_points
    assert [1, 4] in mesh.voronoi.ridge_points or [4, 1] in mesh.voronoi.ridge_points
    assert [4, 7] in mesh.voronoi.ridge_points or [7, 4] in mesh.voronoi.ridge_points
    assert [2, 5] in mesh.voronoi.ridge_points or [5, 2] in mesh.voronoi.ridge_points
    assert [5, 8] in mesh.voronoi.ridge_points or [8, 5] in mesh.voronoi.ridge_points
    assert [3, 6] in mesh.voronoi.ridge_points or [6, 3] in mesh.voronoi.ridge_points


def test__qhull_error_is_caught():

    grid = np.array([[3.0, 0.0]])
    mesh = aa.Mesh2DVoronoi(values=grid)

    with pytest.raises(exc.MeshException):
        mesh.voronoi


def _test__interpolated_array_from():

    grid = aa.Grid2D(
        [[0.0, 0.0], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    mesh = aa.Mesh2DVoronoi(values=grid)

    interpolated_array = mesh.interpolated_array_from(
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        shape_native=(3, 2),
        use_nn=True,
    )

    assert interpolated_array.native == pytest.approx(
        np.array([[3.0, 5.0], [2.0, 5.0], [1.0, 5.0]]), 1.0e-4
    )

    interpolated_array = mesh.interpolated_array_from(
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        shape_native=(2, 3),
        use_nn=True,
    )

    assert interpolated_array.native == pytest.approx(
        np.array([[3.0, 6.0, 5.0], [1.0, 4.0, 5.0]]), 1.0e-4
    )

    interpolated_array = mesh.interpolated_array_from(
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        extent=(-0.4, 0.4, -0.4, 0.4),
        shape_native=(2, 3),
        use_nn=True,
    )

    assert interpolated_array.native == pytest.approx(
        np.array([[1.0, 1.0, 1.907233], [1.0, 1.0, 1.0]]), 1.0e-4
    )
