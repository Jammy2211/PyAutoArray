import numpy as np
import pytest
import scipy.spatial

import autoarray as aa


def test__rectangular_neighbors_from():

    # I0I1I2I
    # I3I4I5I
    # I6I7I8I

    (neighbors, neighbors_sizes) = aa.util.pixelization.rectangular_neighbors_from(
        shape_native=(3, 3)
    )

    assert (neighbors[0] == [1, 3, -1, -1]).all()
    assert (neighbors[1] == [0, 2, 4, -1]).all()
    assert (neighbors[2] == [1, 5, -1, -1]).all()
    assert (neighbors[3] == [0, 4, 6, -1]).all()
    assert (neighbors[4] == [1, 3, 5, 7]).all()
    assert (neighbors[5] == [2, 4, 8, -1]).all()
    assert (neighbors[6] == [3, 7, -1, -1]).all()
    assert (neighbors[7] == [4, 6, 8, -1]).all()
    assert (neighbors[8] == [5, 7, -1, -1]).all()

    assert (neighbors_sizes == np.array([2, 3, 2, 3, 4, 3, 2, 3, 2])).all()

    # I0I1I 2I 3I
    # I4I5I 6I 7I
    # I8I9I10I11I

    (neighbors, neighbors_sizes) = aa.util.pixelization.rectangular_neighbors_from(
        shape_native=(3, 4)
    )

    assert (neighbors[0] == [1, 4, -1, -1]).all()
    assert (neighbors[1] == [0, 2, 5, -1]).all()
    assert (neighbors[2] == [1, 3, 6, -1]).all()
    assert (neighbors[3] == [2, 7, -1, -1]).all()
    assert (neighbors[4] == [0, 5, 8, -1]).all()
    assert (neighbors[5] == [1, 4, 6, 9]).all()
    assert (neighbors[6] == [2, 5, 7, 10]).all()
    assert (neighbors[7] == [3, 6, 11, -1]).all()
    assert (neighbors[8] == [4, 9, -1, -1]).all()
    assert (neighbors[9] == [5, 8, 10, -1]).all()
    assert (neighbors[10] == [6, 9, 11, -1]).all()
    assert (neighbors[11] == [7, 10, -1, -1]).all()

    assert (neighbors_sizes == np.array([2, 3, 3, 2, 3, 4, 4, 3, 2, 3, 3, 2])).all()

    # I0I 1I 2I
    # I3I 4I 5I
    # I6I 7I 8I
    # I9I10I11I

    (neighbors, neighbors_sizes) = aa.util.pixelization.rectangular_neighbors_from(
        shape_native=(4, 3)
    )

    assert (neighbors[0] == [1, 3, -1, -1]).all()
    assert (neighbors[1] == [0, 2, 4, -1]).all()
    assert (neighbors[2] == [1, 5, -1, -1]).all()
    assert (neighbors[3] == [0, 4, 6, -1]).all()
    assert (neighbors[4] == [1, 3, 5, 7]).all()
    assert (neighbors[5] == [2, 4, 8, -1]).all()
    assert (neighbors[6] == [3, 7, 9, -1]).all()
    assert (neighbors[7] == [4, 6, 8, 10]).all()
    assert (neighbors[8] == [5, 7, 11, -1]).all()
    assert (neighbors[9] == [6, 10, -1, -1]).all()
    assert (neighbors[10] == [7, 9, 11, -1]).all()
    assert (neighbors[11] == [8, 10, -1, -1]).all()

    assert (neighbors_sizes == np.array([2, 3, 2, 3, 4, 3, 3, 4, 3, 2, 3, 2])).all()

    # I0 I 1I 2I 3I
    # I4 I 5I 6I 7I
    # I8 I 9I10I11I
    # I12I13I14I15I

    (neighbors, neighbors_sizes) = aa.util.pixelization.rectangular_neighbors_from(
        shape_native=(4, 4)
    )

    assert (neighbors[0] == [1, 4, -1, -1]).all()
    assert (neighbors[1] == [0, 2, 5, -1]).all()
    assert (neighbors[2] == [1, 3, 6, -1]).all()
    assert (neighbors[3] == [2, 7, -1, -1]).all()
    assert (neighbors[4] == [0, 5, 8, -1]).all()
    assert (neighbors[5] == [1, 4, 6, 9]).all()
    assert (neighbors[6] == [2, 5, 7, 10]).all()
    assert (neighbors[7] == [3, 6, 11, -1]).all()
    assert (neighbors[8] == [4, 9, 12, -1]).all()
    assert (neighbors[9] == [5, 8, 10, 13]).all()
    assert (neighbors[10] == [6, 9, 11, 14]).all()
    assert (neighbors[11] == [7, 10, 15, -1]).all()
    assert (neighbors[12] == [8, 13, -1, -1]).all()
    assert (neighbors[13] == [9, 12, 14, -1]).all()
    assert (neighbors[14] == [10, 13, 15, -1]).all()
    assert (neighbors[15] == [11, 14, -1, -1]).all()

    assert (
        neighbors_sizes == np.array([2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 2, 3, 3, 2])
    ).all()


def test__voronoi_neighbors_from():

    points = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]])

    voronoi = scipy.spatial.Voronoi(points, qhull_options="Qbb Qc Qx Qm")
    (neighbors, neighbors_sizes) = aa.util.pixelization.voronoi_neighbors_from(
        pixels=5, ridge_points=np.array(voronoi.ridge_points)
    )

    assert set(neighbors[0]) == {1, 2, 3, -1}
    assert set(neighbors[1]) == {0, 2, 4, -1}
    assert set(neighbors[2]) == {0, 1, 3, 4}
    assert set(neighbors[3]) == {0, 2, 4, -1}
    assert set(neighbors[4]) == {1, 2, 3, -1}

    assert (neighbors_sizes == np.array([3, 3, 4, 3, 3])).all()

    # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

    points = np.array(
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

    voronoi = scipy.spatial.Voronoi(points, qhull_options="Qbb Qc Qx Qm")
    (neighbors, neighbors_sizes) = aa.util.pixelization.voronoi_neighbors_from(
        pixels=9, ridge_points=np.array(voronoi.ridge_points)
    )

    assert set(neighbors[0]) == {1, 3, -1, -1}
    assert set(neighbors[1]) == {0, 2, 4, -1}
    assert set(neighbors[2]) == {1, 5, -1, -1}
    assert set(neighbors[3]) == {0, 4, 6, -1}
    assert set(neighbors[4]) == {1, 3, 5, 7}
    assert set(neighbors[5]) == {2, 4, 8, -1}
    assert set(neighbors[6]) == {3, 7, -1, -1}
    assert set(neighbors[7]) == {4, 6, 8, -1}
    assert set(neighbors[8]) == {5, 7, -1, -1}

    assert (neighbors_sizes == np.array([2, 3, 2, 3, 4, 3, 2, 3, 2])).all()


def test__delaunay_interpolated_grid_from():

    shape_native = (3, 3)

    grid_interpolate_slim = aa.Grid2D.uniform(
        shape_native=shape_native, pixel_scales=1.0
    ).slim

    delaunay_grid = np.array(
        [[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]]
    )

    delaunay = scipy.spatial.Delaunay(delaunay_grid)

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    interpolated_grid = aa.util.pixelization.delaunay_interpolated_array_from(
        shape_native=shape_native,
        interpolation_grid_slim=grid_interpolate_slim,
        pixel_values=values,
        delaunay=delaunay,
    )

    assert interpolated_grid == pytest.approx(
        np.array([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5], [4.0, 4.5, 5.0]]), 1.0e-4
    )
