from astropy.io import fits
import numpy as np
import os
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

    assert mask.is_all_true is False

    mask = aa.Mask1D(mask=[False, False], pixel_scales=1.0)

    assert mask.is_all_true is False

    mask = aa.Mask1D(mask=[False, True, False, False], pixel_scales=1.0)

    assert mask.is_all_true is False

    mask = aa.Mask1D(mask=[True, True, True, True], pixel_scales=1.0)

    assert mask.is_all_true is True


def test__is_all_false():
    mask = aa.Mask1D(mask=[False, False, False, False], pixel_scales=1.0)

    assert mask.is_all_false is True

    mask = aa.Mask1D(mask=[False, False], pixel_scales=1.0)

    assert mask.is_all_false is True

    mask = aa.Mask1D(mask=[False, True, False, False], pixel_scales=1.0)

    assert mask.is_all_false is False

    mask = aa.Mask1D(mask=[True, True, False, False], pixel_scales=1.0)

    assert mask.is_all_false is False


def test__from_primary_hdu():
    file_path = os.path.join(test_data_path, "mask_out.fits")

    if os.path.exists(file_path):
        os.remove(file_path)

    mask = np.array([True, False, True, False, False, True]).astype("int")

    aa.util.array_1d.numpy_array_1d_to_fits(
        mask, file_path=file_path, header_dict={"PIXSCALE": 0.1}
    )

    primary_hdu = fits.open(file_path)

    mask_via_hdu = aa.Mask1D.from_primary_hdu(
        primary_hdu=primary_hdu[0],
    )

    assert type(mask_via_hdu) == aa.Mask1D
    assert (mask_via_hdu == mask).all()
    assert mask_via_hdu.pixel_scales == (0.1,)
