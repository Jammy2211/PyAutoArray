import numpy as np
import pytest

import autoarray as aa



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

    sparse_grid = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
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

    sparse_grid = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
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

    sparse_grid_weight_0 = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=False,
    )

    sparse_grid_weight_1 = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=False,
    )

    assert (sparse_grid_weight_0 == sparse_grid_weight_1).all()

    sparse_grid_weight_0 = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=True,
    )

    sparse_grid_weight_1 = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=True,
    )

    assert (sparse_grid_weight_0 != sparse_grid_weight_1).any()


def test__from_snr_split():
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

    snr_map = aa.Array2D(
        values=[
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 4.0, 4.0, 1.0],
            [1.0, 4.0, 4.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        mask=mask,
    )

    sparse_grid = aa.Grid2DSparse.from_snr_split(
        pixels=8,
        fraction_high_snr=0.5,
        snr_cut=3.0,
        grid=grid,
        snr_map=snr_map,
        n_iter=10,
        max_iter=20,
        seed=1,
    )

    assert sparse_grid == pytest.approx(
        np.array(
            [
                [0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, -0.25],
                [0.25, -0.25],
                [0.58333, 0.58333],
                [0.58333, -0.58333],
                [-0.58333, -0.58333],
                [-0.58333, 0.58333],
            ]
        ),
        1.0e-4,
    )
