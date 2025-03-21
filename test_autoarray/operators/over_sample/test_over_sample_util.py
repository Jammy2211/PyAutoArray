import autoarray as aa
from autoarray import util

import numpy as np
import pytest


def test__total_sub_pixels_2d_from():
    assert (
        util.over_sample.total_sub_pixels_2d_from(sub_size=np.array([2, 2, 2, 2, 2]))
        == 20
    )


def test__native_sub_index_for_slim_sub_index_2d_from():
    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    sub_mask_index_for_sub_mask_1d_index = (
        util.over_sample.native_sub_index_for_slim_sub_index_2d_from(
            mask_2d=mask, sub_size=np.array([2])
        )
    )

    assert (
        sub_mask_index_for_sub_mask_1d_index
        == np.array([[2, 2], [2, 3], [3, 2], [3, 3]])
    ).all()

    mask = np.array([[True, False, True], [False, False, False], [True, False, True]])

    sub_mask_index_for_sub_mask_1d_index = (
        util.over_sample.native_sub_index_for_slim_sub_index_2d_from(
            mask_2d=mask, sub_size=np.array([2, 2, 2, 2, 2])
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
            mask_2d=mask, sub_size=np.array([2, 2])
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
            mask, sub_size=np.array([2])
        )
    )

    assert (slim_index_for_sub_slim_index == np.array([0, 0, 0, 0])).all()

    mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

    slim_index_for_sub_slim_index = (
        util.over_sample.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask, sub_size=np.array([2, 2, 2])
        )
    )

    assert (
        slim_index_for_sub_slim_index == np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    ).all()

    mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

    slim_index_for_sub_slim_index = (
        util.over_sample.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask, sub_size=np.array([3, 3, 3])
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
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=np.array([2, 2, 2, 2, 2])
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
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=np.array([3])
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
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=np.array([2, 2, 2, 2, 2])
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
        mask_2d=mask, pixel_scales=(3.0, 6.0), sub_size=np.array([2]), origin=(1.0, 1.0)
    )

    assert grid[0:4] == pytest.approx(
        np.array([[1.75, -0.5], [1.75, 2.5], [0.25, -0.5], [0.25, 2.5]]), 1e-4
    )


def test__from_manual_adapt_radial_bin():
    mask = aa.Mask2D.circular(shape_native=(5, 5), pixel_scales=2.0, radius=3.0)

    grid = aa.Grid2D.from_mask(mask=mask)

    sub_size = aa.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid, sub_size_list=[8, 4, 2], radial_list=[1.5, 2.5]
    )
    assert sub_size.native == pytest.approx(
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 4, 2, 0],
                [0, 4, 8, 4, 0],
                [0, 2, 4, 2, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
        1.0e-4,
    )


def test__from_manual_adapt_radial_bin__centre_list_input():
    mask = aa.Mask2D.circular(shape_native=(5, 5), pixel_scales=2.0, radius=3.0)

    grid = aa.Grid2D.from_mask(mask=mask)

    sub_size = aa.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=[8, 4, 2],
        radial_list=[1.5, 2.5],
        centre_list=[(0.0, -2.0), (0.0, 2.0)],
    )

    assert sub_size.native == pytest.approx(
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 4, 2, 4, 0],
                [0, 8, 4, 8, 0],
                [0, 4, 2, 4, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
        1.0e-4,
    )


def test__from_adapt():
    mask = aa.Mask2D(
        mask=[[True, True, True], [True, False, False], [True, True, False]],
        pixel_scales=1.0,
    )

    data = aa.Array2D(values=[1.0, 2.0, 3.0], mask=mask)
    noise_map = aa.Array2D(values=[1.0, 2.0, 1.0], mask=mask)

    sub_size = aa.util.over_sample.over_sample_size_via_adapt_from(
        data=data,
        noise_map=noise_map,
        signal_to_noise_cut=1.5,
        sub_size_lower=2,
        sub_size_upper=4,
    )

    assert sub_size == pytest.approx([2, 2, 4], 1.0e-4)

    sub_size = aa.util.over_sample.over_sample_size_via_adapt_from(
        data=data,
        noise_map=noise_map,
        signal_to_noise_cut=0.5,
        sub_size_lower=2,
        sub_size_upper=4,
    )

    assert sub_size == pytest.approx([4, 4, 4], 1.0e-4)
