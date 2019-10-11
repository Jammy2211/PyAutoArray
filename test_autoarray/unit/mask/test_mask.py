import os

import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray import exc

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)



class TestMask:
    class TestConstructors:

        def test__mask_all_unmasked__5x5__input__all_are_false(self):
            mask = aa.Mask.unmasked_from_shape(
                shape=(5, 5), invert=False,
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

        def test__mask_all_unmasked_inverted__5x5__input__all_are_true(self):
            mask = aa.Mask.unmasked_from_shape(
                shape=(5, 5), invert=True, 
            )

            assert mask.shape == (5, 5)
            assert (
                    mask
                    == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                ]
            )
            ).all()

    class TestParse:

        def test__load_and_output_mask_to_fits(self):
            mask = aa.Mask.from_fits(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0,
            )

            output_data_dir = "{}/../../test_files/array/output_test/".format(
                os.path.dirname(os.path.realpath(__file__))
            )

            if os.path.exists(output_data_dir):
                shutil.rmtree(output_data_dir)

            os.makedirs(output_data_dir)

            mask.output_mask_to_fits(file_path=output_data_dir + "mask.fits")

            mask = aa.Mask.from_fits(
                file_path=output_data_dir + "mask.fits", hdu=0)

            assert (mask == np.ones((3, 3))).all()


class TestScaledMask:

    class TestConstructors:

        def test__mask_all_unmasked__5x5__input__all_are_false(self):
            mask = aa.ScaledMask.unmasked_from_shape(
                shape=(5, 5), invert=False, pixel_scales=(1.0, 1.0),
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
            assert mask.geometry.pixel_scales == (1.0, 1.0)

        def test__mask_all_unmasked_inverted__5x5__input__all_are_true(self):
            mask = aa.ScaledMask.unmasked_from_shape(
                shape=(5, 5), invert=True, pixel_scales=(1.0, 1.0)
            )

            assert mask.shape == (5, 5)
            assert (
                    mask
                    == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                ]
            )
            ).all()
            assert mask.geometry.pixel_scales == (1.0, 1.0)

    class TestParse:

        def test__load_and_output_mask_to_fits(self):
            mask = aa.ScaledMask.from_fits(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0, pixel_scales=(1.0, 1.0)
            )

            output_data_dir = "{}/../../test_files/array/output_test/".format(
                os.path.dirname(os.path.realpath(__file__))
            )

            if os.path.exists(output_data_dir):
                shutil.rmtree(output_data_dir)

            os.makedirs(output_data_dir)

            mask.output_mask_to_fits(file_path=output_data_dir + "mask.fits")

            mask = aa.ScaledMask.from_fits(
                file_path=output_data_dir + "mask.fits", hdu=0, pixel_scales=(1.0, 1.0))

            assert (mask == np.ones((3, 3))).all()
            assert mask.geometry.pixel_scales == (1.0, 1.0)


class TestScaledSubMask:
    
    class TestConstructorsViaShapes:

        def test__mask_all_unmasked__5x5__input__all_are_false(self):
            mask = aa.ScaledSubMask.unmasked_from_shape(
                shape=(5, 5), pixel_scales=(1.5, 1.5), invert=False, sub_size=1
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

            assert mask.geometry.origin == (0.0, 0.0)
            assert mask.geometry.mask_centre == (0.0, 0.0)

        def test__mask_all_unmasked_inverted__5x5__input__all_are_true(self):
            mask = aa.ScaledSubMask.unmasked_from_shape(
                shape=(5, 5), pixel_scales=(1.0, 1.0), invert=True, sub_size=1
            )

            assert mask.shape == (5, 5)
            assert (
                    mask
                    == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                ]
            )
            ).all()

            assert mask.geometry.origin == (0.0, 0.0)

        def test__mask_circular__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
                shape=(5, 4), pixel_scales=(2.7, 2.7), radius_arcsec=3.5, centre=(0.0, 0.0)
            )

            mask = aa.ScaledSubMask.circular(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                sub_size=1,
                radius_arcsec=3.5,
                centre=(0.0, 0.0),
            )

            assert (mask == mask_via_util).all()
            assert mask.geometry.origin == (0.0, 0.0)
            assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

        def test__mask_circular__inverted__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
                shape=(5, 4), pixel_scales=(2.7, 2.7), radius_arcsec=3.5, centre=(0.0, 0.0)
            )

            mask = aa.ScaledSubMask.circular(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                sub_size=1,
                radius_arcsec=3.5,
                centre=(0.0, 0.0),
                invert=True,
            )

            assert (mask == np.invert(mask_via_util)).all()
            assert mask.geometry.origin == (0.0, 0.0)
            assert mask.geometry.mask_centre == (0.0, 0.0)

        def test__mask_annulus__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=3.5,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledSubMask.circular_annular(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                sub_size=1,
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=3.5,
                centre=(0.0, 0.0),
            )

            assert (mask == mask_via_util).all()
            assert mask.geometry.origin == (0.0, 0.0)
            assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

        def test__mask_annulus_inverted__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=3.5,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledSubMask.circular_annular(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                sub_size=1,
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=3.5,
                centre=(0.0, 0.0),
                invert=True,
            )

            assert (mask == np.invert(mask_via_util)).all()
            assert mask.geometry.origin == (0.0, 0.0)
            assert mask.geometry.mask_centre == (0.0, 0.0)

        def test__mask_anti_annulus__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
                shape=(9, 9),
                pixel_scales=(1.2, 1.2),
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=2.2,
                outer_radius_2_arcsec=3.0,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledSubMask.circular_anti_annular(
                shape=(9, 9),
                pixel_scales=(1.2, 1.2),
                sub_size=1,
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=2.2,
                outer_radius_2_arcsec=3.0,
                centre=(0.0, 0.0),
            )

            assert (mask == mask_via_util).all()
            assert mask.geometry.origin == (0.0, 0.0)
            assert mask.geometry.mask_centre == (0.0, 0.0)

        def test__mask_anti_annulus_inverted__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
                shape=(9, 9),
                pixel_scales=(1.2, 1.2),
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=2.2,
                outer_radius_2_arcsec=3.0,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledSubMask.circular_anti_annular(
                shape=(9, 9),
                pixel_scales=(1.2, 1.2),
                sub_size=1,
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=2.2,
                outer_radius_2_arcsec=3.0,
                centre=(0.0, 0.0),
                invert=True,
            )

            assert (mask == np.invert(mask_via_util)).all()
            assert mask.geometry.origin == (0.0, 0.0)
            assert mask.geometry.mask_centre == (0.0, 0.0)

        def test__mask_elliptical__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
                shape=(8, 5),
                pixel_scales=(2.7, 2.7),
                major_axis_radius_arcsec=5.7,
                axis_ratio=0.4,
                phi=40.0,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledSubMask.elliptical(
                shape=(8, 5),
                pixel_scales=(2.7, 2.7),
                sub_size=1,
                major_axis_radius_arcsec=5.7,
                axis_ratio=0.4,
                phi=40.0,
                centre=(0.0, 0.0),
            )

            assert (mask == mask_via_util).all()
            assert mask.geometry.origin == (0.0, 0.0)
            assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

        def test__mask_elliptical_inverted__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
                shape=(8, 5),
                pixel_scales=(2.7, 2.7),
                major_axis_radius_arcsec=5.7,
                axis_ratio=0.4,
                phi=40.0,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledSubMask.elliptical(
                shape=(8, 5),
                pixel_scales=(2.7, 2.7),
                sub_size=1,
                major_axis_radius_arcsec=5.7,
                axis_ratio=0.4,
                phi=40.0,
                centre=(0.0, 0.0),
                invert=True,
            )

            assert (mask == np.invert(mask_via_util)).all()
            assert mask.geometry.origin == (0.0, 0.0)
            assert mask.geometry.mask_centre == (0.0, 0.0)

        def test__mask_elliptical_annular__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
                shape=(8, 5),
                pixel_scales=(2.7, 2.7),
                inner_major_axis_radius_arcsec=2.1,
                inner_axis_ratio=0.6,
                inner_phi=20.0,
                outer_major_axis_radius_arcsec=5.7,
                outer_axis_ratio=0.4,
                outer_phi=40.0,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledSubMask.elliptical_annular(
                shape=(8, 5),
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
            assert mask.geometry.origin == (0.0, 0.0)
            assert mask.geometry.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

        def test__mask_elliptical_annular_inverted__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
                shape=(8, 5),
                pixel_scales=(2.7, 2.7),
                inner_major_axis_radius_arcsec=2.1,
                inner_axis_ratio=0.6,
                inner_phi=20.0,
                outer_major_axis_radius_arcsec=5.7,
                outer_axis_ratio=0.4,
                outer_phi=40.0,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledSubMask.elliptical_annular(
                shape=(8, 5),
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
            assert mask.geometry.origin == (0.0, 0.0)
            assert mask.geometry.mask_centre == (0.0, 0.0)

    class TestParse:

        def test__load_and_output_mask_to_fits(self):

            mask = aa.ScaledSubMask.from_fits(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0, sub_size=1, pixel_scales=(1.0, 1.0),
            )

            output_data_dir = "{}/../../test_files/array/output_test/".format(
                os.path.dirname(os.path.realpath(__file__))
            )

            if os.path.exists(output_data_dir):
                shutil.rmtree(output_data_dir)

            os.makedirs(output_data_dir)

            mask.output_mask_to_fits(file_path=output_data_dir + "mask.fits")

            mask = aa.ScaledSubMask.from_fits(
                file_path=output_data_dir + "mask.fits", hdu=0, sub_size=1, pixel_scales=(1.0, 1.0), origin=(2.0, 2.0))

            assert (mask == np.ones((3, 3))).all()
            assert mask.geometry.pixel_scales == (1.0, 1.0)
            assert mask.geometry.origin == (2.0, 2.0)