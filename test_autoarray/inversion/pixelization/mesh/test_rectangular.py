import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.mesh.mesh.rectangular_adapt_density import (
    overlay_grid_from,
)


def test__overlay_grid_from__shape_native_and_pixel_scales():
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

    mesh = aa.MeshGeometryRectangular(mesh=mesh, mesh_grid=mesh_grid, data_grid=None)

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

    mesh = aa.MeshGeometryRectangular(mesh=mesh, mesh_grid=mesh_grid, data_grid=None)

    assert mesh.shape_native == (5, 4)
    assert mesh.pixel_scales == pytest.approx((2.0 / 5.0, 2.0 / 4.0), 1e-2)

    grid = aa.Grid2DIrregular([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])

    mesh = aa.mesh.RectangularUniform(shape=(3, 3))

    mesh_grid = overlay_grid_from(shape_native=mesh.shape, grid=grid, buffer=1e-8)

    mesh = aa.MeshGeometryRectangular(mesh=mesh, mesh_grid=mesh_grid, data_grid=None)

    assert mesh.shape_native == (3, 3)
    assert mesh.pixel_scales == pytest.approx((6.0 / 3.0, 6.0 / 3.0), 1e-2)


def test__overlay_grid_from__pixel_centres__3x3_grid__pixel_centres():
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
