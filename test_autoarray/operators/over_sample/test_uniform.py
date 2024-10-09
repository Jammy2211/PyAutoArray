import numpy as np
import pytest

import autoarray as aa


@pytest.fixture(name="indexes_2d_9x9")
def make_indexes_2d_9x9():
    mask_2d = aa.Mask2D(
        mask=[
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    return aa.DeriveIndexes2D(mask=mask_2d)


def test__from_sub_size_int():
    mask = aa.Mask2D(
        mask=[[True, True, True], [True, False, False], [True, True, False]],
        pixel_scales=1.0,
    )

    over_sampling = aa.OverSamplerUniform(mask=mask, sub_size=2)

    assert over_sampling.sub_size.slim == pytest.approx([2, 2, 2], 1.0e-4)
    assert over_sampling.sub_size.native == pytest.approx(
        np.array([[0, 0, 0], [0, 2, 2], [0, 0, 2]]), 1.0e-4
    )


def test__sub_pixel_areas():
    mask = aa.Mask2D(
        mask=[[True, True, True], [True, False, False], [True, True, True]],
        pixel_scales=1.0,
    )

    over_sampling = aa.OverSamplerUniform(mask=mask, sub_size=np.array([1, 2]))

    areas = over_sampling.sub_pixel_areas

    assert areas == pytest.approx([1.0, 0.25, 0.25, 0.25, 0.25], 1.0e-4)


def test__from_manual_adapt_radial_bin():
    mask = aa.Mask2D.circular(shape_native=(5, 5), pixel_scales=2.0, radius=3.0)

    grid = aa.Grid2D.from_mask(mask=mask)

    over_sampling = aa.OverSamplingUniform.from_radial_bins(
        grid=grid, sub_size_list=[8, 4, 2], radial_list=[1.5, 2.5]
    )
    assert over_sampling.sub_size.native == pytest.approx(
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

    over_sampling = aa.OverSamplingUniform.from_radial_bins(
        grid=grid,
        sub_size_list=[8, 4, 2],
        radial_list=[1.5, 2.5],
        centre_list=[(0.0, -2.0), (0.0, 2.0)],
    )

    assert over_sampling.sub_size.native == pytest.approx(
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

    over_sampling = aa.OverSamplingUniform.from_adapt(
        data=data,
        noise_map=noise_map,
        signal_to_noise_cut=1.5,
        sub_size_lower=2,
        sub_size_upper=4,
    )

    assert over_sampling.sub_size == pytest.approx([2, 2, 4], 1.0e-4)

    over_sampling = aa.OverSamplingUniform.from_adapt(
        data=data,
        noise_map=noise_map,
        signal_to_noise_cut=0.5,
        sub_size_lower=2,
        sub_size_upper=4,
    )

    assert over_sampling.sub_size == pytest.approx([4, 4, 4], 1.0e-4)


def test__sub_fraction():
    mask = aa.Mask2D(
        mask=[[False, False], [True, True]],
        pixel_scales=1.0,
    )

    over_sampling = aa.OverSamplerUniform(
        mask=mask, sub_size=aa.Array2D(values=[1, 2], mask=mask)
    )

    assert over_sampling.sub_fraction.slim == pytest.approx([1.0, 0.25], 1.0e-4)


def test__over_sampled_grid():
    mask = aa.Mask2D(
        mask=[[False, False], [True, True]],
        pixel_scales=1.0,
    )

    over_sampling = aa.OverSamplerUniform(
        mask=mask, sub_size=aa.Array2D(values=[1, 2], mask=mask)
    )

    assert over_sampling.over_sampled_grid.native == pytest.approx(
        np.array([[0.5, -0.5], [0.75, 0.25], [0.75, 0.75], [0.25, 0.25], [0.25, 0.75]]),
        1.0e-4,
    )


def test__binned_array_2d_from():
    mask = aa.Mask2D(
        mask=[[False, False], [True, True]],
        pixel_scales=1.0,
    )

    over_sampling = aa.OverSamplerUniform(
        mask=mask, sub_size=aa.Array2D(values=[1, 2], mask=mask)
    )

    arr = np.array([1.0, 5.0, 7.0, 10.0, 10.0])

    binned_array_2d = over_sampling.binned_array_2d_from(array=arr)

    assert binned_array_2d.slim == pytest.approx(np.array([1.0, 8.0]), 1.0e-4)


def test__sub_mask_index_for_sub_mask_1d_index():
    mask = aa.Mask2D(
        mask=[[True, True, True], [True, False, False], [True, True, False]],
        pixel_scales=1.0,
        sub_size=2,
    )

    over_sampling = aa.OverSamplerUniform(mask=mask, sub_size=2)

    sub_mask_index_for_sub_mask_1d_index = (
        aa.util.over_sample.native_sub_index_for_slim_sub_index_2d_from(
            mask_2d=np.array(mask), sub_size=np.array([2, 2, 2])
        )
    )

    assert over_sampling.sub_mask_native_for_sub_mask_slim == pytest.approx(
        sub_mask_index_for_sub_mask_1d_index, 1e-4
    )


def test__slim_index_for_sub_slim_index():
    mask = aa.Mask2D(
        mask=[[True, False, True], [False, False, False], [True, False, False]],
        pixel_scales=1.0,
        sub_size=2,
    )

    over_sampling = aa.OverSamplerUniform(mask=mask, sub_size=2)

    slim_index_for_sub_slim_index_util = (
        aa.util.over_sample.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask_2d=np.array(mask), sub_size=np.array([2, 2, 2, 2, 2, 2])
        )
    )

    assert (over_sampling.slim_for_sub_slim == slim_index_for_sub_slim_index_util).all()
