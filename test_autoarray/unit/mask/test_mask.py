import os

import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray.mask import mask as msk
from autoarray import exc

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)

    
class TestMask:

    def test__mask__makes_mask_without_other_inputs(self):

        mask = aa.mask.manual(mask_2d=[[False, False], [False, False]])

        assert type(mask) == msk.Mask
        assert (mask == np.array([[False, False], [False, False]])).all()

        mask = aa.mask.manual(mask_2d=[[False, False, True], [True, True, False]])

        assert type(mask) == msk.Mask
        assert (mask == np.array([[False, False, True], [True, True, False]])).all()

    def test__mask__makes_mask_with_pixel_scale(self):

        mask = aa.mask.manual(mask_2d=[[False, False], [True, True]], pixel_scales=1.0)

        assert type(mask) == msk.Mask
        assert (mask == np.array([[False, False], [True, True]])).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)

        mask = aa.mask.manual(mask_2d=[[False, False, True], [True, True, False]], pixel_scales=(2.0, 3.0), origin=(0.0, 1.0))

        assert type(mask) == msk.Mask
        assert (mask == np.array([[False, False, True], [True, True, False]])).all()
        assert mask.pixel_scales == (2.0, 3.0)
        assert mask.origin == (0.0, 1.0)

    def test__mask__makes_mask_with_pixel_scale_and_sub_size(self):

        mask = aa.mask.manual(mask_2d=[[False, False], [True, True]], pixel_scales=1.0, sub_size=1)

        assert type(mask) == msk.Mask
        assert (mask == np.array([[False, False], [True, True]])).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)
        assert mask.sub_size == 1

        mask = aa.mask.manual(mask_2d=[[False, False], [True, True]], pixel_scales=(2.0, 3.0), sub_size=2, origin=(0.0, 1.0))

        assert type(mask) == msk.Mask
        assert (mask == np.array([[False, False], [True, True]])).all()
        assert mask.pixel_scales == (2.0, 3.0)
        assert mask.origin == (0.0, 1.0)
        assert mask.sub_size == 2

        mask = aa.mask.manual(mask_2d=[[False, False], [True, True], [True, False], [False, True]], pixel_scales=1.0, sub_size=2)

        assert type(mask) == msk.Mask
        assert (mask == np.array([[False, False], [True, True], [True, False], [False, True]])).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)
        assert mask.sub_size == 2

    def test__mask__invert_is_true_inverts_the_mask(self):

        mask = aa.mask.manual(mask_2d=[[False, False, True], [True, True, False]], invert=True)

        assert type(mask) == msk.Mask
        assert (mask == np.array([[True, True, False], [False, False, True]])).all()

    def test__mask__input_is_1d_mask__no_shape_2d__raises_exception(self):

        with pytest.raises(exc.MaskException):

            aa.mask.manual(mask_2d=[False, False, True])

        with pytest.raises(exc.MaskException):

            aa.mask.manual(mask_2d=[False, False, True], pixel_scales=False)

        with pytest.raises(exc.MaskException):

            aa.mask.manual(mask_2d=[False, False, True], sub_size=1)

        with pytest.raises(exc.MaskException):

            aa.mask.manual(mask_2d=[False, False, True], pixel_scales=False, sub_size=1)


class TestUnmasked:
    
    def test__mask_all_unmasked__5x5__input__all_are_false(self):

        mask = aa.mask.unmasked(
            shape_2d=(5, 5), invert=False,
        )

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

        mask = aa.mask.unmasked(
            shape_2d=(3, 3), pixel_scales=(1.5, 1.5), invert=False, sub_size=2
        )

        assert mask.shape == (3, 3)
        assert (
                mask
                == np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False]
            ]
        )
        ).all()

        assert mask.sub_size == 2
        assert mask.pixel_scales == (1.5, 1.5)
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)

        mask = aa.mask.unmasked(
            shape_2d=(3, 3), pixel_scales=(2.0, 2.5), invert=True, sub_size=4, origin=(1.0, 2.0)
        )

        assert mask.shape == (3, 3)
        assert (
                mask
                == np.array(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True]
            ]
        )
        ).all()

        assert mask.sub_size == 4
        assert mask.pixel_scales == (2.0, 2.5)
        assert mask.origin == (1.0, 2.0)


class TestCircular:

    def test__mask_circular__compare_to_array_util(self):
        
        mask_via_util = aa.util.mask.mask_2d_circular_from_shape_2d_pixel_scales_and_radius(
            shape_2d=(5, 4), pixel_scales=(2.7, 2.7), radius_arcsec=3.5, centre=(0.0, 0.0)
        )

        mask = aa.mask.circular(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            radius_arcsec=3.5,
            centre=(0.0, 0.0),
        )

        assert (mask == mask_via_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    def test__mask_circular__inverted__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_2d_circular_from_shape_2d_pixel_scales_and_radius(
            shape_2d=(5, 4), pixel_scales=(2.7, 2.7), radius_arcsec=3.5, centre=(0.0, 0.0)
        )

        mask = aa.mask.circular(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            radius_arcsec=3.5,
            centre=(0.0, 0.0),
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)


class TestAnnular:

    def test__mask_annulus__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_2d_circular_annular_from_shape_2d_pixel_scales_and_radii(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            inner_radius_arcsec=0.8,
            outer_radius_arcsec=3.5,
            centre=(0.0, 0.0),
        )

        mask = aa.mask.circular_annular(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            inner_radius_arcsec=0.8,
            outer_radius_arcsec=3.5,
            centre=(0.0, 0.0),
        )

        assert (mask == mask_via_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    def test__mask_annulus_inverted__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_2d_circular_annular_from_shape_2d_pixel_scales_and_radii(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            inner_radius_arcsec=0.8,
            outer_radius_arcsec=3.5,
            centre=(0.0, 0.0),
        )

        mask = aa.mask.circular_annular(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            inner_radius_arcsec=0.8,
            outer_radius_arcsec=3.5,
            centre=(0.0, 0.0),
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)


class TestAntiAnnular:

    def test__mask_anti_annulus__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_2d_circular_anti_annular_from_shape_2d_pixel_scales_and_radii(
            shape_2d=(9, 9),
            pixel_scales=(1.2, 1.2),
            inner_radius_arcsec=0.8,
            outer_radius_arcsec=2.2,
            outer_radius_2_arcsec=3.0,
            centre=(0.0, 0.0),
        )

        mask = aa.mask.circular_anti_annular(
            shape_2d=(9, 9),
            pixel_scales=(1.2, 1.2),
            sub_size=1,
            inner_radius_arcsec=0.8,
            outer_radius_arcsec=2.2,
            outer_radius_2_arcsec=3.0,
            centre=(0.0, 0.0),
        )

        assert (mask == mask_via_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)

    def test__mask_anti_annulus_inverted__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_2d_circular_anti_annular_from_shape_2d_pixel_scales_and_radii(
            shape_2d=(9, 9),
            pixel_scales=(1.2, 1.2),
            inner_radius_arcsec=0.8,
            outer_radius_arcsec=2.2,
            outer_radius_2_arcsec=3.0,
            centre=(0.0, 0.0),
        )

        mask = aa.mask.circular_anti_annular(
            shape_2d=(9, 9),
            pixel_scales=(1.2, 1.2),
            sub_size=1,
            inner_radius_arcsec=0.8,
            outer_radius_arcsec=2.2,
            outer_radius_2_arcsec=3.0,
            centre=(0.0, 0.0),
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)


class TestElliptical:

    def test__mask_elliptical__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_2d_elliptical_from_shape_2d_pixel_scales_and_radius(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            major_axis_radius_arcsec=5.7,
            axis_ratio=0.4,
            phi=40.0,
            centre=(0.0, 0.0),
        )

        mask = aa.mask.elliptical(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            major_axis_radius_arcsec=5.7,
            axis_ratio=0.4,
            phi=40.0,
            centre=(0.0, 0.0),
        )

        assert (mask == mask_via_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    def test__mask_elliptical_inverted__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_2d_elliptical_from_shape_2d_pixel_scales_and_radius(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            major_axis_radius_arcsec=5.7,
            axis_ratio=0.4,
            phi=40.0,
            centre=(0.0, 0.0),
        )

        mask = aa.mask.elliptical(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            major_axis_radius_arcsec=5.7,
            axis_ratio=0.4,
            phi=40.0,
            centre=(0.0, 0.0),
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)


class TestEllipiticalAnnular:

    def test__mask_elliptical_annular__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_2d_elliptical_annular_from_shape_2d_pixel_scales_and_radius(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            inner_major_axis_radius_arcsec=2.1,
            inner_axis_ratio=0.6,
            inner_phi=20.0,
            outer_major_axis_radius_arcsec=5.7,
            outer_axis_ratio=0.4,
            outer_phi=40.0,
            centre=(0.0, 0.0),
        )

        mask = aa.mask.elliptical_annular(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            inner_major_axis_radius_arcsec=2.1,
            inner_axis_ratio=0.6,
            inner_phi=20.0,
            outer_major_axis_radius_arcsec=5.7,
            outer_axis_ratio=0.4,
            outer_phi=40.0,
            centre=(0.0, 0.0),
        )

        assert (mask == mask_via_util).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    def test__mask_elliptical_annular_inverted__compare_to_array_util(self):
        
        mask_via_util = aa.util.mask.mask_2d_elliptical_annular_from_shape_2d_pixel_scales_and_radius(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            inner_major_axis_radius_arcsec=2.1,
            inner_axis_ratio=0.6,
            inner_phi=20.0,
            outer_major_axis_radius_arcsec=5.7,
            outer_axis_ratio=0.4,
            outer_phi=40.0,
            centre=(0.0, 0.0),
        )

        mask = aa.mask.elliptical_annular(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            sub_size=1,
            inner_major_axis_radius_arcsec=2.1,
            inner_axis_ratio=0.6,
            inner_phi=20.0,
            outer_major_axis_radius_arcsec=5.7,
            outer_axis_ratio=0.4,
            outer_phi=40.0,
            centre=(0.0, 0.0),
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.geometry.mask_centre == (0.0, 0.0)


class TestFromAndToFits:

    def test__load_and_output_mask_to_fits(self):

        mask = msk.Mask.from_fits(
            file_path=test_data_dir + "3x3_ones.fits", hdu=0, sub_size=1, pixel_scales=(1.0, 1.0),
        )

        output_data_dir = "{}/../../test_files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )

        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        mask.output_fits(file_path=output_data_dir + "mask.fits")

        mask = msk.Mask.from_fits(
            file_path=output_data_dir + "mask.fits", hdu=0, sub_size=1, pixel_scales=(1.0, 1.0), origin=(2.0, 2.0))

        assert (mask == np.ones((3, 3))).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (2.0, 2.0)


class TestSubQuantities:

    def test__sub_shape_is_shape_times_sub_size(self):

        mask = aa.mask.unmasked(
            shape_2d=(5, 5), sub_size=1,
        )

        assert mask.sub_shape_2d == (5, 5)

        mask = aa.mask.unmasked(
            shape_2d=(5, 5), sub_size=2,
        )

        assert mask.sub_shape_2d == (10, 10)

        mask = aa.mask.unmasked(
            shape_2d=(10, 5), sub_size=3,
        )

        assert mask.sub_shape_2d == (30, 15)

    def test__sub_pixels_in_mask_is_pixels_in_mask_times_sub_size_squared(self):

        mask = aa.mask.unmasked(
            shape_2d=(5, 5), sub_size=1,
        )

        assert mask.sub_pixels_in_mask == 25

        mask = aa.mask.unmasked(
            shape_2d=(5, 5), sub_size=2,
        )

        assert mask.sub_pixels_in_mask == 100

        mask = aa.mask.unmasked(
            shape_2d=(10, 10), sub_size=3,
        )

        assert mask.sub_pixels_in_mask == 900

    def test__sub_mask__is_mask_at_sub_grid_resolution(self):

        mask = aa.mask.manual([[False, True], [False, False]], sub_size=2)

        assert (
                mask.sub_mask_2d
                == np.array(
            [
                [False, False, True, True],
                [False, False, True, True],
                [False, False, False, False],
                [False, False, False, False],
            ]
        )
        ).all()

        mask = aa.mask.manual([[False, False, True], [False, True, False]], sub_size=2)

        assert (
                mask.sub_mask_2d
                == np.array(
            [
                [False, False, False, False, True, True],
                [False, False, False, False, True, True],
                [False, False, True, True, False, False],
                [False, False, True, True, False, False],
            ]
        )
        ).all()

class TestBinnedMask:

    def test__compare_to_mask_via_util(self):

        mask = aa.mask.unmasked(shape_2d=(14, 19), pixel_scales=(1.0, 1.0))
        mask[1, 5] = False
        mask[6, 5] = False
        mask[4, 9] = False
        mask[11, 10] = False

        binned_up_mask_via_util = aa.util.binning.bin_mask_2d(
            mask_2d=mask, bin_up_factor=2
        )

        mask = mask.mapping.binned_mask_from_bin_up_factor(bin_up_factor=2)
        assert (mask == binned_up_mask_via_util).all()
        assert mask.pixel_scales == (2.0, 2.0)

        mask = aa.mask.unmasked(shape_2d=(14, 19), pixel_scales=(2.0, 2.0))
        mask[1, 5] = False
        mask[6, 5] = False
        mask[4, 9] = False
        mask[11, 10] = False

        binned_up_mask_via_util = aa.util.binning.bin_mask_2d(
            mask_2d=mask, bin_up_factor=3
        )

        mask = mask.mapping.binned_mask_from_bin_up_factor(bin_up_factor=3)
        assert (mask == binned_up_mask_via_util).all()
        assert mask.pixel_scales == (6.0, 6.0)