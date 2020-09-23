import os

import numpy as np
import pytest

import autoarray as aa
from autoarray import exc

test_data_dir = "{}/files/mask/".format(os.path.dirname(os.path.realpath(__file__)))


class TestMask1D:
    def test__mask__makes_mask_without_other_inputs(self):

        mask = aa.Mask1D.manual(mask=[False, False])

        assert type(mask) == aa.Mask1D
        assert (mask == np.array([False, False])).all()

        mask = aa.Mask1D.manual(mask=[True, True, False])

        assert type(mask) == aa.Mask1D
        assert (mask == np.array([True, True, False])).all()

    def test__mask__makes_mask_with_pixel_scale(self):

        mask = aa.Mask1D.manual(mask=[False, True], pixel_scales=1.0)

        print(mask.pixel_scales)

        assert type(mask) == aa.Mask1D
        assert (mask == np.array([False, True])).all()
        assert mask.pixel_scale == 1.0
        assert mask.pixel_scales == (1.0,)
        assert mask.origin == 0.0

        mask = aa.Mask1D.manual(mask=[False, False, True], pixel_scales=3.0, origin=1.0)

        assert type(mask) == aa.Mask1D
        assert (mask == np.array([False, False, True])).all()
        assert mask.pixel_scale == 3.0
        assert mask.origin == 1.0

    def test__mask__makes_mask_with_pixel_scale_and_sub_size(self):

        mask = aa.Mask1D.manual(
            mask=[False, False, True, True], pixel_scales=1.0, sub_size=1
        )

        assert type(mask) == aa.Mask1D
        assert (mask == np.array([False, False, True, True])).all()
        assert mask.pixel_scale == 1.0
        assert mask.origin == 0.0
        assert mask.sub_size == 1

        mask = aa.Mask1D.manual(
            mask=[False, False, True, True], pixel_scales=3.0, sub_size=2, origin=1.0
        )

        assert type(mask) == aa.Mask1D
        assert (mask == np.array([False, False, True, True])).all()
        assert mask.pixel_scale == 3.0
        assert mask.origin == 1.0
        assert mask.sub_size == 2

        mask = aa.Mask1D.manual(
            mask=[False, False, True, True, True, False, False, True],
            pixel_scales=1.0,
            sub_size=2,
        )

        assert type(mask) == aa.Mask1D
        assert (
            mask == np.array([False, False, True, True, True, False, False, True])
        ).all()
        assert mask.pixel_scale == 1.0
        assert mask.origin == 0.0
        assert mask.sub_size == 2

    def test__mask__invert_is_true_inverts_the_mask(self):

        mask = aa.Mask1D.manual(mask=[True, True, False], invert=True)

        assert type(mask) == aa.Mask1D
        assert (mask == np.array([False, False, True])).all()

    def test__mask__input_is_2d_mask__no_shape_2d__raises_exception(self):

        with pytest.raises(exc.MaskException):

            aa.Mask1D.manual(mask=[[False, False, True]])

    def test__is_all_true(self):

        mask = aa.Mask1D.manual(mask=[False, False, False, False])

        assert mask.is_all_true == False

        mask = aa.Mask1D.manual(mask=[False, False])

        assert mask.is_all_true == False

        mask = aa.Mask1D.manual(mask=[False, True, False, False])

        assert mask.is_all_true == False

        mask = aa.Mask1D.manual(mask=[True, True, True, True])

        assert mask.is_all_true == True

    def test__is_all_false(self):

        mask = aa.Mask1D.manual(mask=[False, False, False, False])

        assert mask.is_all_false == True

        mask = aa.Mask1D.manual(mask=[False, False])

        assert mask.is_all_false == True

        mask = aa.Mask1D.manual(mask=[False, True, False, False])

        assert mask.is_all_false == False

        mask = aa.Mask1D.manual(mask=[True, True, False, False])

        assert mask.is_all_false == False
