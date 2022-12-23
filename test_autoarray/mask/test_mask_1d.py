import numpy as np
import pytest

import autoarray as aa
from autoarray import exc


def test__manual():

    mask = aa.Mask1D.manual(mask=[False, True], pixel_scales=1.0)

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, True])).all()
    assert mask.pixel_scale == 1.0
    assert mask.pixel_scales == (1.0,)
    assert mask.origin == (0.0,)
    assert (mask.extent == np.array([-1.0, 1.0])).all()

    mask = aa.Mask1D.manual(mask=[False, False, True], pixel_scales=3.0, origin=(1.0,))

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, False, True])).all()
    assert mask.pixel_scale == 3.0
    assert mask.origin == (1.0,)
    assert (mask.extent == np.array([-3.5, 5.5])).all()

    mask = aa.Mask1D.manual(
        mask=[False, False, True, True], pixel_scales=3.0, sub_size=2, origin=(1.0,)
    )

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, False, True, True])).all()
    assert mask.pixel_scale == 3.0
    assert mask.origin == (1.0,)
    assert mask.sub_size == 2


def test__manual__invert_is_true():

    mask = aa.Mask1D.manual(mask=[True, True, False], pixel_scales=1.0, invert=True)

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, False, True])).all()


def test__manual__input_is_2d_mask__no_shape_native__raises_exception():

    with pytest.raises(exc.MaskException):

        aa.Mask1D.manual(mask=[[False, False, True]], pixel_scales=1.0)


def test__is_all_true():

    mask = aa.Mask1D.manual(mask=[False, False, False, False], pixel_scales=1.0)

    assert mask.is_all_true is False

    mask = aa.Mask1D.manual(mask=[False, False], pixel_scales=1.0)

    assert mask.is_all_true is False

    mask = aa.Mask1D.manual(mask=[False, True, False, False], pixel_scales=1.0)

    assert mask.is_all_true is False

    mask = aa.Mask1D.manual(mask=[True, True, True, True], pixel_scales=1.0)

    assert mask.is_all_true is True


def test__is_all_false():

    mask = aa.Mask1D.manual(mask=[False, False, False, False], pixel_scales=1.0)

    assert mask.is_all_false is True

    mask = aa.Mask1D.manual(mask=[False, False], pixel_scales=1.0)

    assert mask.is_all_false is True

    mask = aa.Mask1D.manual(mask=[False, True, False, False], pixel_scales=1.0)

    assert mask.is_all_false is False

    mask = aa.Mask1D.manual(mask=[True, True, False, False], pixel_scales=1.0)

    assert mask.is_all_false is False


def test__to_mask_2d():

    mask_1d = aa.Mask1D.manual(mask=[False, True], pixel_scales=1.0, sub_size=2)

    mask_2d = mask_1d.to_mask_2d

    assert (mask_2d == np.array([[False, True]])).all()
    assert mask_2d.pixel_scales == (1.0, 1.0)
    assert mask_2d.sub_size == 2
    assert mask_2d.origin == (0.0, 0.0)


def test__unmasked_mask():

    mask = aa.Mask1D.manual(
        mask=[True, False, True, False, False, False, True, False, True],
        pixel_scales=1.0,
    )

    assert (mask.unmasked_mask == np.full(fill_value=False, shape=(9,))).all()
