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


def test__sub_pixels_in_mask():
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)

    over_sample = aa.OverSamplerUniform(mask=mask, sub_size=1)

    assert over_sample.sub_pixels_in_mask == 25

    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)

    over_sample = aa.OverSamplerUniform(mask=mask, sub_size=2)

    assert over_sample.sub_pixels_in_mask == 100

    mask = aa.Mask2D.all_false(shape_native=(10, 10), pixel_scales=1.0)

    over_sample = aa.OverSamplerUniform(mask=mask, sub_size=3)

    assert over_sample.sub_pixels_in_mask == 900


def test__sub_mask_index_for_sub_mask_1d_index():
    mask = aa.Mask2D(
        mask=[[True, True, True], [True, False, False], [True, True, False]],
        pixel_scales=1.0,
        sub_size=2,
    )

    over_sample = aa.OverSamplerUniform(mask=mask, sub_size=2)

    sub_mask_index_for_sub_mask_1d_index = (
        aa.util.mask_2d.native_index_for_slim_index_2d_from(
            mask_2d=np.array(mask), sub_size=2
        )
    )

    assert over_sample.sub_mask_native_for_sub_mask_slim == pytest.approx(
        sub_mask_index_for_sub_mask_1d_index, 1e-4
    )


def test__slim_index_for_sub_slim_index():
    mask = aa.Mask2D(
        mask=[[True, False, True], [False, False, False], [True, False, False]],
        pixel_scales=1.0,
        sub_size=2,
    )

    over_sample = aa.OverSamplerUniform(mask=mask, sub_size=2)

    slim_index_for_sub_slim_index_util = (
        aa.util.mask_2d.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask_2d=np.array(mask), sub_size=2
        )
    )

    assert (over_sample.slim_for_sub_slim == slim_index_for_sub_slim_index_util).all()
