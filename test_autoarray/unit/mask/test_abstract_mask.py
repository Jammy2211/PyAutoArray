import os

import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray import exc

test_data_dir = "{}/files/mask/".format(os.path.dirname(os.path.realpath(__file__)))


class TestSubQuantities:
    def test__sub_pixels_in_mask_is_pixels_in_mask_times_sub_size_squared(self):

        mask = aa.Mask2D.unmasked(shape_2d=(5, 5), sub_size=1)

        assert mask.sub_pixels_in_mask == 25

        mask = aa.Mask2D.unmasked(shape_2d=(5, 5), sub_size=2)

        assert mask.sub_pixels_in_mask == 100

        mask = aa.Mask2D.unmasked(shape_2d=(10, 10), sub_size=3)

        assert mask.sub_pixels_in_mask == 900


class TestNewMask:
    def test__new_mask_with_new_sub_size(self):

        mask = aa.Mask2D.unmasked(shape_2d=(3, 3), sub_size=4)

        mask_new = mask.mask_new_sub_size_from_mask(mask=mask)

        assert (mask_new == np.full(fill_value=False, shape=(3, 3))).all()
        assert mask_new.sub_size == 1

        mask_new = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)

        assert (mask_new == np.full(fill_value=False, shape=(3, 3))).all()
        assert mask_new.sub_size == 8
