import numpy as np
import pytest
import scipy.spatial

from autoarray import exc
import autoarray as aa

from autoarray.inversion.pixelization.mesh.rectangular_adapt_density import (
    overlay_grid_from,
)
from autoarray.inversion.pixelization.mesh_grid.rectangular import rectangular_neighbors_from



def test__rectangular_neighbors_from():
    # I0I1I2I
    # I3I4I5I
    # I6I7I8I

    (neighbors, neighbors_sizes) = rectangular_neighbors_from(
        shape_native=(3, 3)
    )

    # TODO : Use pytest.parameterize

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

    (neighbors, neighbors_sizes) = rectangular_neighbors_from(
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

    (neighbors, neighbors_sizes) = rectangular_neighbors_from(
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

    (neighbors, neighbors_sizes) = rectangular_neighbors_from(
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


def test__neighbors__compare_to_mesh_util():
    # I0 I 1I 2I 3I
    # I4 I 5I 6I 7I
    # I8 I 9I10I11I
    # I12I13I14I15I

    mesh = aa.mesh.RectangularUniform(shape=(7, 5))

    mesh_grid = overlay_grid_from(
        shape_native=mesh.shape, grid=aa.Grid2DIrregular(np.zeros((2, 2))), buffer=1e-8
    )

    mesh = aa.Mesh2DRectangular(
        mesh=mesh, mesh_grid=mesh_grid, data_grid_over_sampled=None
    )

    (neighbors_util, neighbors_sizes_util) = rectangular_neighbors_from(
        shape_native=(7, 5)
    )

    assert (mesh.neighbors == neighbors_util).all()
    assert (mesh.neighbors.sizes == neighbors_sizes_util).all()


def test__shape_native_and_pixel_scales():
    grid = aa.Grid2DIrregular(
        [
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    mesh = aa.mesh.RectangularUniform(shape=(3, 3))

    mesh_grid = overlay_grid_from(shape_native=mesh.shape, grid=grid, buffer=1e-8)

    mesh = aa.Mesh2DRectangular(
        mesh=mesh, mesh_grid=mesh_grid, data_grid_over_sampled=None
    )

    assert mesh.shape_native == (3, 3)
    assert mesh.pixel_scales == pytest.approx((2.0 / 3.0, 2.0 / 3.0), 1e-2)

    grid = aa.Grid2DIrregular(
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

    mesh = aa.mesh.RectangularUniform(shape=(5, 4))

    mesh_grid = overlay_grid_from(shape_native=mesh.shape, grid=grid, buffer=1e-8)

    mesh = aa.Mesh2DRectangular(
        mesh=mesh, mesh_grid=mesh_grid, data_grid_over_sampled=None
    )

    assert mesh.shape_native == (5, 4)
    assert mesh.pixel_scales == pytest.approx((2.0 / 5.0, 2.0 / 4.0), 1e-2)

    grid = aa.Grid2DIrregular([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])

    mesh = aa.mesh.RectangularUniform(shape=(3, 3))

    mesh_grid = overlay_grid_from(shape_native=mesh.shape, grid=grid, buffer=1e-8)

    mesh = aa.Mesh2DRectangular(
        mesh=mesh, mesh_grid=mesh_grid, data_grid_over_sampled=None
    )

    assert mesh.shape_native == (3, 3)
    assert mesh.pixel_scales == pytest.approx((6.0 / 3.0, 6.0 / 3.0), 1e-2)


def test__pixel_centres__3x3_grid__pixel_centres():
    grid = aa.Grid2DIrregular(
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

    mesh_grid = overlay_grid_from(shape_native=(3, 3), grid=grid, buffer=1e-8)

    assert mesh_grid == pytest.approx(
        np.array(
            [
                [2.0 / 3.0, -2.0 / 3.0],
                [2.0 / 3.0, 0.0],
                [2.0 / 3.0, 2.0 / 3.0],
                [0.0, -2.0 / 3.0],
                [0.0, 0.0],
                [0.0, 2.0 / 3.0],
                [-2.0 / 3.0, -2.0 / 3.0],
                [-2.0 / 3.0, 0.0],
                [-2.0 / 3.0, 2.0 / 3.0],
            ]
        )
    )

    grid = aa.Grid2DIrregular(
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

    mesh_grid = overlay_grid_from(shape_native=(4, 3), grid=grid, buffer=1e-8)

    assert mesh_grid == pytest.approx(
        np.array(
            [
                [0.75, -2.0 / 3.0],
                [0.75, 0.0],
                [0.75, 2.0 / 3.0],
                [0.25, -2.0 / 3.0],
                [0.25, 0.0],
                [0.25, 2.0 / 3.0],
                [-0.25, -2.0 / 3.0],
                [-0.25, 0.0],
                [-0.25, 2.0 / 3.0],
                [-0.75, -2.0 / 3.0],
                [-0.75, 0.0],
                [-0.75, 2.0 / 3.0],
            ]
        )
    )
