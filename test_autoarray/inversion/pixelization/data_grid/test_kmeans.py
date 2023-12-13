import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.pixelization.image_mesh import kmeans as data_grid


def test__from_total_pixels_grid_and_weight_map():
    mask = aa.Mask2D(
        mask=np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        ),
        pixel_scales=(0.5, 0.5),
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    weight_map = np.ones(mask.pixels_in_mask)

    sparse_grid = data_grid.via_kmeans_from(
        total_pixels=8, grid=grid, weight_map=weight_map, n_iter=10, max_iter=20, seed=1
    )

    assert (
        sparse_grid
        == np.array(
            [
                [-0.25, 0.25],
                [0.5, -0.5],
                [0.75, 0.5],
                [0.25, 0.5],
                [-0.5, -0.25],
                [-0.5, -0.75],
                [-0.75, 0.5],
                [-0.25, 0.75],
            ]
        )
    ).all()

    mask = aa.Mask2D(
        mask=np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        ),
        pixel_scales=(0.5, 0.5),
        sub_size=2,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    weight_map = np.ones(mask.pixels_in_mask)
    weight_map[0:15] = 0.00000001

    sparse_grid = data_grid.via_kmeans_from(
        total_pixels=8, grid=grid, weight_map=weight_map, n_iter=10, max_iter=30, seed=1
    )

    assert sparse_grid[1] == pytest.approx(np.array([0.4166666, -0.0833333]), 1.0e-4)


def test__from_total_pixels_grid_and_weight_map__stochastic_true():
    mask = aa.Mask2D(
        mask=np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        ),
        pixel_scales=(0.5, 0.5),
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    weight_map = np.ones(mask.pixels_in_mask)

    sparse_grid_weight_0 = data_grid.via_kmeans_from(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=False,
    )

    sparse_grid_weight_1 = data_grid.via_kmeans_from(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=False,
    )

    assert (sparse_grid_weight_0 == sparse_grid_weight_1).all()

    sparse_grid_weight_0 = data_grid.via_kmeans_from(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=True,
    )

    sparse_grid_weight_1 = data_grid.via_kmeans_from(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=True,
    )

    assert (sparse_grid_weight_0 != sparse_grid_weight_1).any()
