import os
from os import path
import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray import exc

test_data_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "mask"
)


class TestMask:
    def test__mask__makes_mask_with_pixel_scale(self):

        mask = aa.Mask2D.manual(mask=[[False, False], [True, True]], pixel_scales=1.0)

        assert type(mask) == aa.Mask2D
        assert (mask == np.array([[False, False], [True, True]])).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)

        mask = aa.Mask2D.manual(
            mask=[[False, False, True], [True, True, False]],
            pixel_scales=(2.0, 3.0),
            origin=(0.0, 1.0),
        )

        assert type(mask) == aa.Mask2D
        assert (mask == np.array([[False, False, True], [True, True, False]])).all()
        assert mask.pixel_scales == (2.0, 3.0)
        assert mask.origin == (0.0, 1.0)

    def test__mask__makes_mask_with_pixel_scale_and_sub_size(self):

        mask = aa.Mask2D.manual(
            mask=[[False, False], [True, True]], pixel_scales=1.0, sub_size=1
        )

        assert type(mask) == aa.Mask2D
        assert (mask == np.array([[False, False], [True, True]])).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)
        assert mask.sub_size == 1

        mask = aa.Mask2D.manual(
            mask=[[False, False], [True, True]],
            pixel_scales=(2.0, 3.0),
            sub_size=2,
            origin=(0.0, 1.0),
        )

        assert type(mask) == aa.Mask2D
        assert (mask == np.array([[False, False], [True, True]])).all()
        assert mask.pixel_scales == (2.0, 3.0)
        assert mask.origin == (0.0, 1.0)
        assert mask.sub_size == 2

        mask = aa.Mask2D.manual(
            mask=[[False, False], [True, True], [True, False], [False, True]],
            pixel_scales=1.0,
            sub_size=2,
        )

        assert type(mask) == aa.Mask2D
        assert (
            mask
            == np.array([[False, False], [True, True], [True, False], [False, True]])
        ).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)
        assert mask.sub_size == 2

    def test__mask__invert_is_true_inverts_the_mask(self):

        mask = aa.Mask2D.manual(
            mask=[[False, False, True], [True, True, False]],
            pixel_scales=1.0,
            invert=True,
        )

        assert type(mask) == aa.Mask2D
        assert (mask == np.array([[True, True, False], [False, False, True]])).all()

    def test__mask__input_is_1d_mask__no_shape_2d__raises_exception(self):

        with pytest.raises(exc.MaskException):

            aa.Mask2D.manual(mask=[False, False, True], pixel_scales=1.0)

        with pytest.raises(exc.MaskException):

            aa.Mask2D.manual(mask=[False, False, True], pixel_scales=False)

        with pytest.raises(exc.MaskException):

            aa.Mask2D.manual(mask=[False, False, True], pixel_scales=1.0, sub_size=1)

        with pytest.raises(exc.MaskException):

            aa.Mask2D.manual(mask=[False, False, True], pixel_scales=False, sub_size=1)

    def test__is_all_true(self):

        mask = aa.Mask2D.manual(mask=[[False, False], [False, False]], pixel_scales=1.0)

        assert mask.is_all_true == False

        mask = aa.Mask2D.manual(mask=[[False, False]], pixel_scales=1.0)

        assert mask.is_all_true == False

        mask = aa.Mask2D.manual(mask=[[False, True], [False, False]], pixel_scales=1.0)

        assert mask.is_all_true == False

        mask = aa.Mask2D.manual(mask=[[True, True], [True, True]], pixel_scales=1.0)

        assert mask.is_all_true == True

    def test__is_all_false(self):

        mask = aa.Mask2D.manual(mask=[[False, False], [False, False]], pixel_scales=1.0)

        assert mask.is_all_false == True

        mask = aa.Mask2D.manual(mask=[[False, False]], pixel_scales=1.0)

        assert mask.is_all_false == True

        mask = aa.Mask2D.manual(mask=[[False, True], [False, False]], pixel_scales=1.0)

        assert mask.is_all_false == False

        mask = aa.Mask2D.manual(mask=[[True, True], [False, False]], pixel_scales=1.0)

        assert mask.is_all_false == False


class TestClassMethods:
    def test__mask_all_unmasked__5x5__input__all_are_false(self):

        mask = aa.Mask2D.unmasked(shape_2d=(5, 5), pixel_scales=1.0, invert=False)

        assert mask.shape == (5, 5)
        assert (
            mask
            == np.array(
                [
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ]
            )
        ).all()

        mask = aa.Mask2D.unmasked(
            shape_2d=(3, 3), pixel_scales=(1.5, 1.5), invert=False, sub_size=2
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        assert mask.sub_size == 2
        assert mask.pixel_scales == (1.5, 1.5)
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)

        mask = aa.Mask2D.unmasked(
            shape_2d=(3, 3),
            pixel_scales=(2.0, 2.5),
            invert=True,
            sub_size=4,
            origin=(1.0, 2.0),
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array([[True, True, True], [True, True, True], [True, True, True]])
        ).all()

        assert mask.sub_size == 4
        assert mask.pixel_scales == (2.0, 2.5)
        assert mask.origin == (1.0, 2.0)

    def test__mask_circular__compare_to_array_util(self):

        mask_via_util = aa.util.mask.mask_circular_from(
            shape_2d=(5, 4), pixel_scales=(2.7, 2.7), radius=3.5, centre=(0.0, 0.0)
        )

        mask = aa.Mask2D.circular(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            radius=3.5,
            centre=(0.0, 0.0),
        )

        assert (mask == mask_via_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    def test__mask_circular__inverted__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_circular_from(
            shape_2d=(5, 4), pixel_scales=(2.7, 2.7), radius=3.5, centre=(0.0, 0.0)
        )

        mask = aa.Mask2D.circular(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            radius=3.5,
            centre=(0.0, 0.0),
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)

    def test__mask_annulus__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_circular_annular_from(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            inner_radius=0.8,
            outer_radius=3.5,
            centre=(0.0, 0.0),
        )

        mask = aa.Mask2D.circular_annular(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            inner_radius=0.8,
            outer_radius=3.5,
            centre=(0.0, 0.0),
        )

        assert (mask == mask_via_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    def test__mask_annulus_inverted__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_circular_annular_from(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            inner_radius=0.8,
            outer_radius=3.5,
            centre=(0.0, 0.0),
        )

        mask = aa.Mask2D.circular_annular(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            inner_radius=0.8,
            outer_radius=3.5,
            centre=(0.0, 0.0),
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)

    def test__mask_anti_annulus__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_circular_anti_annular_from(
            shape_2d=(9, 9),
            pixel_scales=(1.2, 1.2),
            inner_radius=0.8,
            outer_radius=2.2,
            outer_radius_2_scaled=3.0,
            centre=(0.0, 0.0),
        )

        mask = aa.Mask2D.circular_anti_annular(
            shape_2d=(9, 9),
            pixel_scales=(1.2, 1.2),
            sub_size=1,
            inner_radius=0.8,
            outer_radius=2.2,
            outer_radius_2=3.0,
            centre=(0.0, 0.0),
        )

        assert (mask == mask_via_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)

    def test__mask_anti_annulus_inverted__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_circular_anti_annular_from(
            shape_2d=(9, 9),
            pixel_scales=(1.2, 1.2),
            inner_radius=0.8,
            outer_radius=2.2,
            outer_radius_2_scaled=3.0,
            centre=(0.0, 0.0),
        )

        mask = aa.Mask2D.circular_anti_annular(
            shape_2d=(9, 9),
            pixel_scales=(1.2, 1.2),
            sub_size=1,
            inner_radius=0.8,
            outer_radius=2.2,
            outer_radius_2=3.0,
            centre=(0.0, 0.0),
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)

    def test__mask_elliptical__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_elliptical_from(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            major_axis_radius=5.7,
            axis_ratio=0.4,
            phi=40.0,
            centre=(0.0, 0.0),
        )

        mask = aa.Mask2D.elliptical(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            major_axis_radius=5.7,
            axis_ratio=0.4,
            phi=40.0,
            centre=(0.0, 0.0),
        )

        assert (mask == mask_via_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    def test__mask_elliptical_inverted__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_elliptical_from(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            major_axis_radius=5.7,
            axis_ratio=0.4,
            phi=40.0,
            centre=(0.0, 0.0),
        )

        mask = aa.Mask2D.elliptical(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            major_axis_radius=5.7,
            axis_ratio=0.4,
            phi=40.0,
            centre=(0.0, 0.0),
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)

    def test__mask_elliptical_annular__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_elliptical_annular_from(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            inner_major_axis_radius=2.1,
            inner_axis_ratio=0.6,
            inner_phi=20.0,
            outer_major_axis_radius=5.7,
            outer_axis_ratio=0.4,
            outer_phi=40.0,
            centre=(0.0, 0.0),
        )

        mask = aa.Mask2D.elliptical_annular(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            inner_major_axis_radius=2.1,
            inner_axis_ratio=0.6,
            inner_phi=20.0,
            outer_major_axis_radius=5.7,
            outer_axis_ratio=0.4,
            outer_phi=40.0,
            centre=(0.0, 0.0),
        )

        assert (mask == mask_via_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    def test__mask_elliptical_annular_inverted__compare_to_array_util(self):

        mask_via_util = aa.util.mask.mask_elliptical_annular_from(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            inner_major_axis_radius=2.1,
            inner_axis_ratio=0.6,
            inner_phi=20.0,
            outer_major_axis_radius=5.7,
            outer_axis_ratio=0.4,
            outer_phi=40.0,
            centre=(0.0, 0.0),
        )

        mask = aa.Mask2D.elliptical_annular(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            inner_major_axis_radius=2.1,
            inner_axis_ratio=0.6,
            inner_phi=20.0,
            outer_major_axis_radius=5.7,
            outer_axis_ratio=0.4,
            outer_phi=40.0,
            centre=(0.0, 0.0),
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)

    def test__from_pixel_coordinates__mask_with_or_without_buffer__false_at_buffed_coordinates(
        self,
    ):

        mask = aa.Mask2D.from_pixel_coordinates(
            shape_2d=(5, 5), pixel_coordinates=[[2, 2]], pixel_scales=1.0, buffer=0
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, False, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                ]
            )
        ).all()

        mask = aa.Mask2D.from_pixel_coordinates(
            shape_2d=(5, 5), pixel_coordinates=[[2, 2]], pixel_scales=1.0, buffer=1
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, True, True, True, True],
                ]
            )
        ).all()

        mask = aa.Mask2D.from_pixel_coordinates(
            shape_2d=(7, 7),
            pixel_coordinates=[[2, 2], [5, 5]],
            pixel_scales=1.0,
            buffer=1,
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, False, False, False, True, True, True],
                    [True, False, False, False, True, True, True],
                    [True, False, False, False, True, True, True],
                    [True, True, True, True, False, False, False],
                    [True, True, True, True, False, False, False],
                    [True, True, True, True, False, False, False],
                ]
            )
        ).all()


class TestToFromFits:
    def test__load_and_output_mask_to_fits(self):

        mask = aa.Mask2D.from_fits(
            file_path=path.join(test_data_dir, "3x3_ones.fits"),
            hdu=0,
            sub_size=1,
            pixel_scales=(1.0, 1.0),
        )

        output_data_dir = path.join(
            "{}".format(path.dirname(path.realpath(__file__))),
            "files",
            "array",
            "output_test",
        )

        if path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        mask.output_to_fits(file_path=path.join(output_data_dir, "mask.fits"))

        mask = aa.Mask2D.from_fits(
            file_path=path.join(output_data_dir, "mask.fits"),
            hdu=0,
            sub_size=1,
            pixel_scales=(1.0, 1.0),
            origin=(2.0, 2.0),
        )

        assert (mask == np.ones((3, 3))).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (2.0, 2.0)

    def test__load_from_fits_with_resized_mask_shape(self):

        mask = aa.Mask2D.from_fits(
            file_path=path.join(test_data_dir, "3x3_ones.fits"),
            hdu=0,
            sub_size=1,
            pixel_scales=(1.0, 1.0),
            resized_mask_shape=(1, 1),
        )

        assert mask.shape_2d == (1, 1)

        mask = aa.Mask2D.from_fits(
            file_path=path.join(test_data_dir, "3x3_ones.fits"),
            hdu=0,
            sub_size=1,
            pixel_scales=(1.0, 1.0),
            resized_mask_shape=(5, 5),
        )

        assert mask.shape_2d == (5, 5)


class TestSubQuantities:
    def test__sub_shape_is_shape_times_sub_size(self):

        mask = aa.Mask2D.unmasked(shape_2d=(5, 5), pixel_scales=1.0, sub_size=1)

        assert mask.sub_shape_2d == (5, 5)

        mask = aa.Mask2D.unmasked(shape_2d=(5, 5), pixel_scales=1.0, sub_size=2)

        assert mask.sub_shape_2d == (10, 10)

        mask = aa.Mask2D.unmasked(shape_2d=(10, 5), pixel_scales=1.0, sub_size=3)

        assert mask.sub_shape_2d == (30, 15)


class TestNewMasksFromMask:
    def test__sub_mask__is_mask_at_sub_grid_resolution(self):

        mask = aa.Mask2D.manual(
            mask=[[False, True], [False, False]], pixel_scales=1.0, sub_size=2
        )

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

        mask = aa.Mask2D.manual(
            mask=[[False, False, True], [False, True, False]],
            pixel_scales=1.0,
            sub_size=2,
        )

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

    def test__binned_mask__compare_to_mask_via_util(self):

        mask = aa.Mask2D.unmasked(shape_2d=(14, 19), pixel_scales=(1.0, 1.0))
        mask[1, 5] = False
        mask[6, 5] = False
        mask[4, 9] = False
        mask[11, 10] = False

        binned_up_mask_via_util = aa.util.binning.bin_mask(mask=mask, bin_up_factor=2)

        mask = mask.binned_mask_from_bin_up_factor(bin_up_factor=2)
        assert (mask == binned_up_mask_via_util).all()
        assert mask.pixel_scales == (2.0, 2.0)

        mask = aa.Mask2D.unmasked(shape_2d=(14, 19), pixel_scales=(2.0, 2.0))
        mask[1, 5] = False
        mask[6, 5] = False
        mask[4, 9] = False
        mask[11, 10] = False

        binned_up_mask_via_util = aa.util.binning.bin_mask(mask=mask, bin_up_factor=3)

        mask = mask.binned_mask_from_bin_up_factor(bin_up_factor=3)
        assert (mask == binned_up_mask_via_util).all()
        assert mask.pixel_scales == (6.0, 6.0)

    def test__resized_mask__pad__compare_to_manual_mask(self):

        mask = aa.Mask2D.unmasked(shape_2d=(5, 5), pixel_scales=1.0)
        mask[2, 2] = True

        mask_resized = mask.resized_mask_from_new_shape(new_shape=(7, 7))

        mask_resized_manual = np.full(fill_value=False, shape=(7, 7))
        mask_resized_manual[3, 3] = True

        assert (mask_resized == mask_resized_manual).all()

    def test__resized_mask__trim__compare_to_manual_mask(self):

        mask = aa.Mask2D.unmasked(shape_2d=(5, 5), pixel_scales=1.0)
        mask[2, 2] = True

        mask_resized = mask.resized_mask_from_new_shape(new_shape=(3, 3))

        mask_resized_manual = np.full(fill_value=False, shape=(3, 3))
        mask_resized_manual[1, 1] = True

        assert (mask_resized == mask_resized_manual).all()

    def test__rescaled_mask_from_rescale_factor__compare_to_manual_mask(self):

        mask = aa.Mask2D.unmasked(shape_2d=(5, 5), pixel_scales=1.0)
        mask[2, 2] = True

        mask_rescaled = mask.rescaled_mask_from_rescale_factor(rescale_factor=2.0)

        mask_rescaled_manual = np.full(fill_value=False, shape=(3, 3))
        mask_rescaled_manual[1, 1] = True

        mask_rescaled_manual = aa.util.mask.rescaled_mask_from(
            mask=mask, rescale_factor=2.0
        )

        assert (mask_rescaled == mask_rescaled_manual).all()

    def test__edged_buffed_mask__compare_to_manual_mask(self):

        mask = aa.Mask2D.unmasked(shape_2d=(5, 5), pixel_scales=1.0)
        mask[2, 2] = True

        edge_buffed_mask_manual = aa.util.mask.buffed_mask_from(mask=mask).astype(
            "bool"
        )

        assert (mask.edge_buffed_mask == edge_buffed_mask_manual).all()
