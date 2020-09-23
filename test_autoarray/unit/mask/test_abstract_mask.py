import os

import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray import exc

test_data_dir = "{}/files/mask/".format(os.path.dirname(os.path.realpath(__file__)))


class TestMask:
    def test__mask__makes_mask_without_other_inputs(self):

        mask = aa.Mask.manual(mask=[[False, False], [False, False]])

        assert type(mask) == aa.Mask
        assert (mask == np.array([[False, False], [False, False]])).all()

        mask = aa.Mask.manual(mask=[[False, False, True], [True, True, False]])

        assert type(mask) == aa.Mask
        assert (mask == np.array([[False, False, True], [True, True, False]])).all()

    def test__mask__makes_mask_with_pixel_scale(self):

        mask = aa.Mask.manual(mask=[[False, False], [True, True]], pixel_scales=1.0)

        assert type(mask) == aa.Mask
        assert (mask == np.array([[False, False], [True, True]])).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)

        mask = aa.Mask.manual(
            mask=[[False, False, True], [True, True, False]],
            pixel_scales=(2.0, 3.0),
            origin=(0.0, 1.0),
        )

        assert type(mask) == aa.Mask
        assert (mask == np.array([[False, False, True], [True, True, False]])).all()
        assert mask.pixel_scales == (2.0, 3.0)
        assert mask.origin == (0.0, 1.0)

    def test__mask__makes_mask_with_pixel_scale_and_sub_size(self):

        mask = aa.Mask.manual(
            mask=[[False, False], [True, True]], pixel_scales=1.0, sub_size=1
        )

        assert type(mask) == aa.Mask
        assert (mask == np.array([[False, False], [True, True]])).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)
        assert mask.sub_size == 1

        mask = aa.Mask.manual(
            mask=[[False, False], [True, True]],
            pixel_scales=(2.0, 3.0),
            sub_size=2,
            origin=(0.0, 1.0),
        )

        assert type(mask) == aa.Mask
        assert (mask == np.array([[False, False], [True, True]])).all()
        assert mask.pixel_scales == (2.0, 3.0)
        assert mask.origin == (0.0, 1.0)
        assert mask.sub_size == 2

        mask = aa.Mask.manual(
            mask=[[False, False], [True, True], [True, False], [False, True]],
            pixel_scales=1.0,
            sub_size=2,
        )

        assert type(mask) == aa.Mask
        assert (
            mask
            == np.array([[False, False], [True, True], [True, False], [False, True]])
        ).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)
        assert mask.sub_size == 2

    def test__mask__invert_is_true_inverts_the_mask(self):

        mask = aa.Mask.manual(
            mask=[[False, False, True], [True, True, False]], invert=True
        )

        assert type(mask) == aa.Mask
        assert (mask == np.array([[True, True, False], [False, False, True]])).all()

    def test__mask__input_is_1d_mask__no_shape_2d__raises_exception(self):

        with pytest.raises(exc.MaskException):

            aa.Mask.manual(mask=[False, False, True])

        with pytest.raises(exc.MaskException):

            aa.Mask.manual(mask=[False, False, True], pixel_scales=False)

        with pytest.raises(exc.MaskException):

            aa.Mask.manual(mask=[False, False, True], sub_size=1)

        with pytest.raises(exc.MaskException):

            aa.Mask.manual(mask=[False, False, True], pixel_scales=False, sub_size=1)

    def test__is_all_true(self):

        mask = aa.Mask.manual(mask=[[False, False], [False, False]])

        assert mask.is_all_true == False

        mask = aa.Mask.manual(mask=[[False, False]])

        assert mask.is_all_true == False

        mask = aa.Mask.manual(mask=[[False, True], [False, False]])

        assert mask.is_all_true == False

        mask = aa.Mask.manual(mask=[[True, True], [True, True]])

        assert mask.is_all_true == True

    def test__is_all_false(self):

        mask = aa.Mask.manual(mask=[[False, False], [False, False]])

        assert mask.is_all_false == True

        mask = aa.Mask.manual(mask=[[False, False]])

        assert mask.is_all_false == True

        mask = aa.Mask.manual(mask=[[False, True], [False, False]])

        assert mask.is_all_false == False

        mask = aa.Mask.manual(mask=[[True, True], [False, False]])

        assert mask.is_all_false == False


class TestToFits:
    def test__load_and_output_mask_to_fits(self):

        mask = aa.Mask.from_fits(
            file_path=test_data_dir + "3x3_ones.fits",
            hdu=0,
            sub_size=1,
            pixel_scales=(1.0, 1.0),
        )

        output_data_dir = "{}/files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )

        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        mask.output_to_fits(file_path=output_data_dir + "mask.fits")

        mask = aa.Mask.from_fits(
            file_path=output_data_dir + "mask.fits",
            hdu=0,
            sub_size=1,
            pixel_scales=(1.0, 1.0),
            origin=(2.0, 2.0),
        )

        assert (mask == np.ones((3, 3))).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (2.0, 2.0)

    def test__load_from_fits_with_resized_mask_shape(self):

        mask = aa.Mask.from_fits(
            file_path=test_data_dir + "3x3_ones.fits",
            hdu=0,
            sub_size=1,
            pixel_scales=(1.0, 1.0),
            resized_mask_shape=(1, 1),
        )

        assert mask.shape_2d == (1, 1)

        mask = aa.Mask.from_fits(
            file_path=test_data_dir + "3x3_ones.fits",
            hdu=0,
            sub_size=1,
            pixel_scales=(1.0, 1.0),
            resized_mask_shape=(5, 5),
        )

        assert mask.shape_2d == (5, 5)


class TestSubQuantities:
    def test__sub_shape_is_shape_times_sub_size(self):

        mask = aa.Mask.unmasked(shape_2d=(5, 5), sub_size=1)

        assert mask.sub_shape_2d == (5, 5)

        mask = aa.Mask.unmasked(shape_2d=(5, 5), sub_size=2)

        assert mask.sub_shape_2d == (10, 10)

        mask = aa.Mask.unmasked(shape_2d=(10, 5), sub_size=3)

        assert mask.sub_shape_2d == (30, 15)

    def test__sub_pixels_in_mask_is_pixels_in_mask_times_sub_size_squared(self):

        mask = aa.Mask.unmasked(shape_2d=(5, 5), sub_size=1)

        assert mask.sub_pixels_in_mask == 25

        mask = aa.Mask.unmasked(shape_2d=(5, 5), sub_size=2)

        assert mask.sub_pixels_in_mask == 100

        mask = aa.Mask.unmasked(shape_2d=(10, 10), sub_size=3)

        assert mask.sub_pixels_in_mask == 900

    def test__sub_mask__is_mask_at_sub_grid_resolution(self):

        mask = aa.Mask.manual([[False, True], [False, False]], sub_size=2)

        assert (
            mask.sub_mask
            == np.array(
                [
                    [False, False, True, True],
                    [False, False, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            )
        ).all()

        mask = aa.Mask.manual([[False, False, True], [False, True, False]], sub_size=2)

        assert (
            mask.sub_mask
            == np.array(
                [
                    [False, False, False, False, True, True],
                    [False, False, False, False, True, True],
                    [False, False, True, True, False, False],
                    [False, False, True, True, False, False],
                ]
            )
        ).all()


class TestNewMask:
    def test__new_mask_with_new_sub_size(self):

        mask = aa.Mask.unmasked(shape_2d=(3, 3), sub_size=4)

        mask_new = mask.mask_new_sub_size_from_mask(mask=mask)

        assert (mask_new == np.full(fill_value=False, shape=(3, 3))).all()
        assert mask_new.sub_size == 1

        mask_new = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)

        assert (mask_new == np.full(fill_value=False, shape=(3, 3))).all()
        assert mask_new.sub_size == 8


class TestBinnedMask:
    def test__compare_to_mask_via_util(self):

        mask = aa.Mask.unmasked(shape_2d=(14, 19), pixel_scales=(1.0, 1.0))
        mask[1, 5] = False
        mask[6, 5] = False
        mask[4, 9] = False
        mask[11, 10] = False

        binned_up_mask_via_util = aa.util.binning.bin_mask(mask=mask, bin_up_factor=2)

        mask = mask.binned_mask_from_bin_up_factor(bin_up_factor=2)
        assert (mask == binned_up_mask_via_util).all()
        assert mask.pixel_scales == (2.0, 2.0)

        mask = aa.Mask.unmasked(shape_2d=(14, 19), pixel_scales=(2.0, 2.0))
        mask[1, 5] = False
        mask[6, 5] = False
        mask[4, 9] = False
        mask[11, 10] = False

        binned_up_mask_via_util = aa.util.binning.bin_mask(mask=mask, bin_up_factor=3)

        mask = mask.binned_mask_from_bin_up_factor(bin_up_factor=3)
        assert (mask == binned_up_mask_via_util).all()
        assert mask.pixel_scales == (6.0, 6.0)


class TestResizedMask:
    def test__resized_mask__pad__compare_to_manual_mask(self):

        mask = aa.Mask.unmasked(shape_2d=(5, 5))
        mask[2, 2] = True

        mask_resized = mask.resized_mask_from_new_shape(new_shape=(7, 7))

        mask_resized_manual = np.full(fill_value=False, shape=(7, 7))
        mask_resized_manual[3, 3] = True

        assert (mask_resized == mask_resized_manual).all()

    def test__resized_mask__trim__compare_to_manual_mask(self):

        mask = aa.Mask.unmasked(shape_2d=(5, 5))
        mask[2, 2] = True

        mask_resized = mask.resized_mask_from_new_shape(new_shape=(3, 3))

        mask_resized_manual = np.full(fill_value=False, shape=(3, 3))
        mask_resized_manual[1, 1] = True

        assert (mask_resized == mask_resized_manual).all()

    def test__rescaled_mask_from_rescale_factor__compare_to_manual_mask(self):

        mask = aa.Mask.unmasked(shape_2d=(5, 5))
        mask[2, 2] = True

        mask_rescaled = mask.rescaled_mask_from_rescale_factor(rescale_factor=2.0)

        mask_rescaled_manual = np.full(fill_value=False, shape=(3, 3))
        mask_rescaled_manual[1, 1] = True

        mask_rescaled_manual = aa.util.mask.rescaled_mask_from(
            mask=mask, rescale_factor=2.0
        )

        assert (mask_rescaled == mask_rescaled_manual).all()

    def test__edged_buffed_mask__compare_to_manual_mask(self):

        mask = aa.Mask.unmasked(shape_2d=(5, 5))
        mask[2, 2] = True

        edge_buffed_mask_manual = aa.util.mask.buffed_mask_from(mask=mask).astype(
            "bool"
        )

        assert (mask.edge_buffed_mask == edge_buffed_mask_manual).all()
