import numpy as np
import pytest
import scipy.spatial

from autoarray import exc
import autoarray as aa

from autoarray.inversion.pixelization.mesh.rectangular_adapt_density import (
    overlay_grid_from,
)


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

    (neighbors_util, neighbors_sizes_util) = aa.util.mesh.rectangular_neighbors_from(
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
