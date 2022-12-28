import numpy as np
import pytest

import autoarray as aa


@pytest.fixture(name="derive_mask_2d_9x9")
def make_derive_mask_2d_9x9():

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

    return aa.DeriveMask2D(mask=mask_2d)


def test__sub_mask():

    mask = aa.Mask2D.manual(
        mask=[[False, False, True], [False, True, False]],
        pixel_scales=1.0,
        sub_size=2,
    )

    derive_mask_2d = aa.DeriveMask2D(mask=mask)

    assert (
        derive_mask_2d.sub
        == np.array(
            [
                [False, False, False, False, True, True],
                [False, False, False, False, True, True],
                [False, False, True, True, False, False],
                [False, False, True, True, False, False],
            ]
        )
    ).all()


def test__rescaled_from():

    mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True

    derive_mask_2d = aa.DeriveMask2D(mask=mask)

    mask_rescaled = derive_mask_2d.rescaled_from(rescale_factor=2.0)

    mask_rescaled_manual = np.full(fill_value=False, shape=(3, 3))
    mask_rescaled_manual[1, 1] = True

    mask_rescaled_manual = aa.util.mask_2d.rescaled_mask_2d_from(
        mask_2d=mask, rescale_factor=2.0
    )

    assert (mask_rescaled == mask_rescaled_manual).all()


def test__resized_from():

    mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True

    derive_mask_2d = aa.DeriveMask2D(mask=mask)

    mask_resized = derive_mask_2d.resized_from(new_shape=(7, 7))

    mask_resized_manual = np.full(fill_value=False, shape=(7, 7))
    mask_resized_manual[3, 3] = True

    assert (mask_resized == mask_resized_manual).all()

    mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True
    derive_mask_2d = aa.DeriveMask2D(mask=mask)

    mask_resized = derive_mask_2d.resized_from(new_shape=(3, 3))

    mask_resized_manual = np.full(fill_value=False, shape=(3, 3))
    mask_resized_manual[1, 1] = True

    assert (mask_resized == mask_resized_manual).all()


def test__unmasked_mask(derive_mask_2d_9x9):

    assert (
        derive_mask_2d_9x9.unmasked == np.full(fill_value=False, shape=(9, 9))
    ).all()


def test__blurring_mask_from(derive_mask_2d_9x9):

    blurring_mask_via_util = aa.util.mask_2d.blurring_mask_2d_from(
        mask_2d=derive_mask_2d_9x9.mask, kernel_shape_native=(3, 3)
    )

    blurring_mask = derive_mask_2d_9x9.blurring_from(kernel_shape_native=(3, 3))

    assert (blurring_mask == blurring_mask_via_util).all()


def test__edge_mask(derive_mask_2d_9x9):

    assert (
        derive_mask_2d_9x9.edge
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


def test__edge_buffed_mask():

    mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True

    derive_mask_2d = aa.DeriveMask2D(mask=mask)

    edge_buffed_mask_manual = aa.util.mask_2d.buffed_mask_2d_from(mask_2d=mask).astype(
        "bool"
    )

    assert (derive_mask_2d.edge_buffed == edge_buffed_mask_manual).all()


def test__border_mask(derive_mask_2d_9x9):

    assert (
        derive_mask_2d_9x9.border
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
