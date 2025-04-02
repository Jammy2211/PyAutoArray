import numpy as np
from os import path
import pytest

import autoarray as aa
from autoarray import exc

test_data_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "mask"
)


def test__constructor():
    mask = aa.Mask1D(mask=[False, True], pixel_scales=1.0)

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, True])).all()
    assert mask.pixel_scale == 1.0
    assert mask.pixel_scales == (1.0,)
    assert mask.origin == (0.0,)
    assert (mask.geometry.extent == np.array([-1.0, 1.0])).all()

    mask = aa.Mask1D(mask=[False, False, True], pixel_scales=3.0, origin=(1.0,))

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, False, True])).all()
    assert mask.pixel_scale == 3.0
    assert mask.origin == (1.0,)
    assert (mask.geometry.extent == np.array([-3.5, 5.5])).all()

    mask = aa.Mask1D(mask=[False, False, True, True], pixel_scales=3.0, origin=(1.0,))

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, False, True, True])).all()
    assert mask.pixel_scale == 3.0
    assert mask.origin == (1.0,)


def test__constructor__invert_is_true():
    mask = aa.Mask1D(mask=[True, True, False], pixel_scales=1.0, invert=True)

    assert type(mask) == aa.Mask1D
    assert (mask == np.array([False, False, True])).all()


def test__constructor__input_is_2d_mask__raises_exception():
    with pytest.raises(exc.MaskException):
        aa.Mask1D(mask=[[False, False, True]], pixel_scales=1.0)


def test__is_all_true():
    mask = aa.Mask1D(mask=[False, False, False, False], pixel_scales=1.0)

    assert mask.is_all_true == False

    mask = aa.Mask1D(mask=[False, False], pixel_scales=1.0)

    assert mask.is_all_true == False

    mask = aa.Mask1D(mask=[False, True, False, False], pixel_scales=1.0)

    assert mask.is_all_true == False

    mask = aa.Mask1D(mask=[True, True, True, True], pixel_scales=1.0)

    assert mask.is_all_true == True


def test__is_all_false():
    mask = aa.Mask1D(mask=[False, False, False, False], pixel_scales=1.0)

    assert mask.is_all_false == True

    mask = aa.Mask1D(mask=[False, False], pixel_scales=1.0)

    assert mask.is_all_false == True

    mask = aa.Mask1D(mask=[False, True, False, False], pixel_scales=1.0)

    assert mask.is_all_false == False

    mask = aa.Mask1D(mask=[True, True, False, False], pixel_scales=1.0)

    assert mask.is_all_false == False
