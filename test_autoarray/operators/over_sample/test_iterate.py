import numpy as np

import autoarray as aa

from autoarray.structures.mock.mock_decorators import (
    ndarray_1d_from,
    ndarray_1d_zeros_from,
    ndarray_2d_from,
)


def test__threshold_mask_from():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    arr = aa.Array2D(
        values=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask,
    )

    over_sample = aa.OverSampleIterateFunc(mask=mask, fractional_accuracy=0.9999)

    threshold_mask = over_sample.threshold_mask_from(
        array_lower_sub_2d=arr.native, array_higher_sub_2d=arr.native
    )

    assert (
        threshold_mask
        == np.array(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
            ]
        )
    ).all()

    mask_lower_sub = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    mask_higher_sub = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    over_sample = aa.OverSampleIterateFunc(mask=mask, fractional_accuracy=0.5)

    array_lower_sub = aa.Array2D(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask_lower_sub,
    )

    array_higher_sub = aa.Array2D(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 5.0, 0.0],
            [0.0, 5.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask_higher_sub,
    )

    threshold_mask = over_sample.threshold_mask_from(
        array_lower_sub_2d=array_lower_sub.native,
        array_higher_sub_2d=array_higher_sub.native,
    )

    assert (
        threshold_mask
        == np.array(
            [
                [True, True, True, True],
                [True, False, True, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )
    ).all()


def test__array_via_func_from__extreme_fractional_accuracies_uses_last_or_first_sub():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
        origin=(0.001, 0.001),
    )

    over_sample = aa.OverSampleIterateFunc(
        mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
    )

    values = over_sample.array_via_func_from(
        func=ndarray_1d_from,
        cls=None,
    )

    over_sample_uniform = aa.OverSampleUniformFunc(mask=mask, sub_size=3)

    values_sub_3 = over_sample_uniform.array_via_func_from(
        func=ndarray_1d_from, cls=object
    )

    assert (values == values_sub_3).all()

    # This test ensures that if the fractional accuracy is met on the last sub_size jump (e.g. 2 doesnt meet it,
    # but 3 does) that the sub_size of 3 is used. There was a bug where the mask was not updated correctly and the
    # iterated array double counted the values.

    values = over_sample.array_via_func_from(
        func=ndarray_1d_from,
        cls=None,
    )

    assert (values == values_sub_3).all()

    over_sample = aa.OverSampleIterateFunc(
        mask=mask, fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
    )

    values = over_sample.array_via_func_from(
        func=ndarray_1d_from,
        cls=None,
    )

    over_sample_uniform = aa.OverSampleUniformFunc(mask=mask, sub_size=2)

    values_sub_2 = over_sample_uniform.array_via_func_from(
        func=ndarray_1d_from, cls=object
    )

    assert (values == values_sub_2).all()


def test__array_via_func_from__check_values_computed_to_fractional_accuracy():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
        origin=(0.001, 0.001),
    )

    over_sample = aa.OverSampleIterateFunc(
        mask=mask, fractional_accuracy=0.5, sub_steps=[2, 4]
    )

    values = over_sample.array_via_func_from(
        func=ndarray_1d_from,
        cls=None,
    )

    over_sample_uniform = aa.OverSampleUniformFunc(mask=mask, sub_size=2)
    values_sub_2 = over_sample_uniform.array_via_func_from(
        func=ndarray_1d_from, cls=object
    )
    over_sample_uniform = aa.OverSampleUniformFunc(mask=mask, sub_size=4)
    values_sub_4 = over_sample_uniform.array_via_func_from(
        func=ndarray_1d_from, cls=object
    )

    assert values.native[1, 1] == values_sub_2.native[1, 1]
    assert values.native[2, 2] != values_sub_2.native[2, 2]

    assert values.native[1, 1] != values_sub_4.native[1, 1]
    assert values.native[2, 2] == values_sub_4.native[2, 2]


def test__array_via_func_from__func_returns_all_zeros__iteration_terminated():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
        origin=(0.001, 0.001),
    )

    over_sample = aa.OverSampleIterateFunc(
        mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
    )

    values = over_sample.array_via_func_from(
        func=ndarray_1d_zeros_from,
        cls=None,
    )

    assert (values == np.zeros((9,))).all()
