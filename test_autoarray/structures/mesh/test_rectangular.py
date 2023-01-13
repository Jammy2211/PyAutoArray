import numpy as np
import pytest
import scipy.spatial

from autoarray import exc
import autoarray as aa


def test__neighbors__compare_to_mesh_util():
    # I0 I 1I 2I 3I
    # I4 I 5I 6I 7I
    # I8 I 9I10I11I
    # I12I13I14I15I

    mesh = aa.Mesh2DRectangular.overlay_grid(
        shape_native=(7, 5), grid=np.zeros((2, 2)), buffer=1e-8
    )

    (neighbors_util, neighbors_sizes_util) = aa.util.mesh.rectangular_neighbors_from(
        shape_native=(7, 5)
    )

    assert (mesh.neighbors == neighbors_util).all()
    assert (mesh.neighbors.sizes == neighbors_sizes_util).all()


def test__shape_native_and_pixel_scales():
    grid = np.array(
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

    mesh = aa.Mesh2DRectangular.overlay_grid(
        shape_native=(3, 3), grid=grid, buffer=1e-8
    )

    assert mesh.shape_native == (3, 3)
    assert mesh.pixel_scales == pytest.approx((2.0 / 3.0, 2.0 / 3.0), 1e-2)

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

    mesh = aa.Mesh2DRectangular.overlay_grid(
        shape_native=(5, 4), grid=grid, buffer=1e-8
    )

    assert mesh.shape_native == (5, 4)
    assert mesh.pixel_scales == pytest.approx((2.0 / 5.0, 2.0 / 4.0), 1e-2)

    grid = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])

    mesh = aa.Mesh2DRectangular.overlay_grid(
        shape_native=(3, 3), grid=grid, buffer=1e-8
    )

    assert mesh.shape_native == (3, 3)
    assert mesh.pixel_scales == pytest.approx((6.0 / 3.0, 6.0 / 3.0), 1e-2)


def test__pixel_centres__3x3_grid__pixel_centres():

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

    mesh = aa.Mesh2DRectangular.overlay_grid(
        shape_native=(3, 3), grid=grid, buffer=1e-8
    )

    assert mesh == pytest.approx(
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

    mesh = aa.Mesh2DRectangular.overlay_grid(
        shape_native=(4, 3), grid=grid, buffer=1e-8
    )

    assert mesh == pytest.approx(
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


def test__interpolated_array_from():

    grid = aa.Grid2D.no_mask(
        values=[[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]],
        shape_native=(2, 2),
        pixel_scales=1.0,
    )

    grid_rectangular = aa.Mesh2DRectangular(
        values=grid, shape_native=grid.shape_native, pixel_scales=grid.pixel_scales
    )

    interpolated_array = grid_rectangular.interpolated_array_from(
        values=np.array([1.0, 2.0, 3.0, 4.0]), shape_native=(3, 2)
    )

    assert interpolated_array.native == pytest.approx(
        np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]), 1.0e-4
    )

    interpolated_array = grid_rectangular.interpolated_array_from(
        values=np.array([1.0, 2.0, 3.0, 4.0]), shape_native=(2, 3)
    )

    assert interpolated_array.native == pytest.approx(
        np.array([[1.0, 1.5, 2.0], [3.0, 3.5, 4.0]]), 1.0e-4
    )

    interpolated_array = grid_rectangular.interpolated_array_from(
        values=np.array([1.0, 2.0, 3.0, 4.0]),
        shape_native=(3, 2),
        extent=(-0.4, 0.4, -0.4, 0.4),
    )

    assert interpolated_array.native == pytest.approx(
        np.array([[1.9, 2.3], [2.3, 2.7], [2.7, 3.1]]), 1.0e-4
    )
