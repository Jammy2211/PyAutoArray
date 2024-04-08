import autoarray as aa
from autoarray import util

import numpy as np
import pytest


def test__total_sub_pixels_2d_from():
    mask_2d = np.array(
        [[True, False, True], [False, False, False], [True, False, True]]
    )

    assert util.over_sample.total_sub_pixels_2d_from(mask_2d, sub_size=2) == 20


def test__native_sub_index_for_slim_sub_index_2d_from():
    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    sub_mask_index_for_sub_mask_1d_index = (
        util.over_sample.native_sub_index_for_slim_sub_index_2d_from(
            mask_2d=mask, sub_size=2
        )
    )

    assert (
        sub_mask_index_for_sub_mask_1d_index
        == np.array([[2, 2], [2, 3], [3, 2], [3, 3]])
    ).all()

    mask = np.array([[True, False, True], [False, False, False], [True, False, True]])

    sub_mask_index_for_sub_mask_1d_index = (
        util.over_sample.native_sub_index_for_slim_sub_index_2d_from(
            mask_2d=mask, sub_size=2
        )
    )

    assert (
        sub_mask_index_for_sub_mask_1d_index
        == np.array(
            [
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                [2, 0],
                [2, 1],
                [3, 0],
                [3, 1],
                [2, 2],
                [2, 3],
                [3, 2],
                [3, 3],
                [2, 4],
                [2, 5],
                [3, 4],
                [3, 5],
                [4, 2],
                [4, 3],
                [5, 2],
                [5, 3],
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True],
            [True, False, True],
            [True, True, True],
            [True, True, False],
        ]
    )

    sub_mask_index_for_sub_mask_1d_index = (
        util.over_sample.native_sub_index_for_slim_sub_index_2d_from(
            mask_2d=mask, sub_size=2
        )
    )

    assert (
        sub_mask_index_for_sub_mask_1d_index
        == np.array([[2, 2], [2, 3], [3, 2], [3, 3], [6, 4], [6, 5], [7, 4], [7, 5]])
    ).all()


def test__slim_index_for_sub_slim_index_via_mask_2d_from():
    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    slim_index_for_sub_slim_index = (
        util.over_sample.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask, sub_size=2
        )
    )

    assert (slim_index_for_sub_slim_index == np.array([0, 0, 0, 0])).all()

    mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

    slim_index_for_sub_slim_index = (
        util.over_sample.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask, sub_size=2
        )
    )

    assert (
        slim_index_for_sub_slim_index == np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    ).all()

    mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

    slim_index_for_sub_slim_index = (
        util.over_sample.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask, sub_size=3
        )
    )

    assert (
        slim_index_for_sub_slim_index
        == np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ]
        )
    ).all()


def test__sub_slim_index_for_sub_native_index_from():
    mask = np.full(fill_value=False, shape=(3, 3))

    sub_mask_1d_index_for_sub_mask_index = (
        util.over_sample.sub_slim_index_for_sub_native_index_from(sub_mask_2d=mask)
    )

    assert (
        sub_mask_1d_index_for_sub_mask_index
        == np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    ).all()

    mask = np.full(fill_value=False, shape=(2, 3))

    sub_mask_1d_index_for_sub_mask_index = (
        util.over_sample.sub_slim_index_for_sub_native_index_from(sub_mask_2d=mask)
    )

    assert (
        sub_mask_1d_index_for_sub_mask_index == np.array([[0, 1, 2], [3, 4, 5]])
    ).all()

    mask = np.full(fill_value=False, shape=(3, 2))

    sub_mask_1d_index_for_sub_mask_index = (
        util.over_sample.sub_slim_index_for_sub_native_index_from(sub_mask_2d=mask)
    )

    assert (
        sub_mask_1d_index_for_sub_mask_index == np.array([[0, 1], [2, 3], [4, 5]])
    ).all()

    mask = np.array([[False, True, False], [True, True, False], [False, False, True]])

    sub_mask_1d_index_for_sub_mask_index = (
        util.over_sample.sub_slim_index_for_sub_native_index_from(sub_mask_2d=mask)
    )

    assert (
        sub_mask_1d_index_for_sub_mask_index
        == np.array([[0, -1, 1], [-1, -1, 2], [3, 4, -1]])
    ).all()

    mask = np.array(
        [
            [False, True, True, False],
            [True, True, False, False],
            [False, False, True, False],
        ]
    )

    sub_mask_1d_index_for_sub_mask_index = (
        util.over_sample.sub_slim_index_for_sub_native_index_from(sub_mask_2d=mask)
    )

    assert (
        sub_mask_1d_index_for_sub_mask_index
        == np.array([[0, -1, -1, 1], [-1, -1, 2, 3], [4, 5, -1, 6]])
    ).all()

    mask = np.array(
        [
            [False, True, False],
            [True, True, False],
            [False, False, True],
            [False, False, True],
        ]
    )

    sub_mask_1d_index_for_sub_mask_index = (
        util.over_sample.sub_slim_index_for_sub_native_index_from(sub_mask_2d=mask)
    )

    assert (
        sub_mask_1d_index_for_sub_mask_index
        == np.array([[0, -1, 1], [-1, -1, 2], [3, 4, -1], [5, 6, -1]])
    ).all()


def test__oversample_mask_from():
    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    oversample_mask = util.over_sample.oversample_mask_2d_from(mask=mask, sub_size=2)

    assert (
        oversample_mask
        == np.array(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        )
    ).all()


def test__grid_2d_slim_over_sampled_via_mask_from():
    mask = np.array([[True, True, False], [False, False, False], [True, True, False]])

    grid = aa.util.over_sample.grid_2d_slim_over_sampled_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2
    )

    assert (
        grid
        == np.array(
            [
                [3.75, 2.25],
                [3.75, 3.75],
                [2.25, 2.25],
                [2.25, 3.75],
                [0.75, -3.75],
                [0.75, -2.25],
                [-0.75, -3.75],
                [-0.75, -2.25],
                [0.75, -0.75],
                [0.75, 0.75],
                [-0.75, -0.75],
                [-0.75, 0.75],
                [0.75, 2.25],
                [0.75, 3.75],
                [-0.75, 2.25],
                [-0.75, 3.75],
                [-2.25, 2.25],
                [-2.25, 3.75],
                [-3.75, 2.25],
                [-3.75, 3.75],
            ]
        )
    ).all()

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    grid = aa.util.over_sample.grid_2d_slim_over_sampled_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=3
    )

    assert (
        grid
        == np.array(
            [
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
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, False],
            [True, False, False, True],
            [False, True, False, True],
        ]
    )

    grid = aa.util.over_sample.grid_2d_slim_over_sampled_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2
    )

    assert (
        grid
        == np.array(
            [
                [3.75, 3.75],
                [3.75, 5.25],
                [2.25, 3.75],
                [2.25, 5.25],
                [0.75, -2.25],
                [0.75, -0.75],
                [-0.75, -2.25],
                [-0.75, -0.75],
                [0.75, 0.75],
                [0.75, 2.25],
                [-0.75, 0.75],
                [-0.75, 2.25],
                [-2.25, -5.25],
                [-2.25, -3.75],
                [-3.75, -5.25],
                [-3.75, -3.75],
                [-2.25, 0.75],
                [-2.25, 2.25],
                [-3.75, 0.75],
                [-3.75, 2.25],
            ]
        )
    ).all()

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    grid = aa.util.over_sample.grid_2d_slim_over_sampled_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 6.0), sub_size=2, origin=(1.0, 1.0)
    )

    assert grid[0:4] == pytest.approx(
        np.array([[1.75, -0.5], [1.75, 2.5], [0.25, -0.5], [0.25, 2.5]]), 1e-4
    )
