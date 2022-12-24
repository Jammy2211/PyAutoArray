import numpy as np
import pytest

import autoarray as aa


@pytest.fixture(name="derived_masks_2d_9x9")
def make_derived_masks_2d_9x9():

    mask_2d = aa.Mask2D.manual(
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

    return aa.DerivedMasks2D(mask=mask_2d)


def test__unmasked_mask(derived_masks_2d_9x9):

    assert (
        derived_masks_2d_9x9.unmasked_mask == np.full(fill_value=False, shape=(9, 9))
    ).all()


def test__blurring_mask_from(derived_masks_2d_9x9):

    blurring_mask_via_util = aa.util.mask_2d.blurring_mask_2d_from(
        mask_2d=derived_masks_2d_9x9.mask, kernel_shape_native=(3, 3)
    )

    blurring_mask = derived_masks_2d_9x9.blurring_mask_from(kernel_shape_native=(3, 3))

    assert (blurring_mask == blurring_mask_via_util).all()


def test__edge_mask(derived_masks_2d_9x9):

    assert (
        derived_masks_2d_9x9.edge_mask
        == np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, False, True, False, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )
    ).all()


def test__border_mask(derived_masks_2d_9x9):

    assert (
        derived_masks_2d_9x9.border_mask
        == np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )
    ).all()