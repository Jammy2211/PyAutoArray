import os

import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray import exc

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestAbstractMask:

    class MaskSetup:

        def test__array_finalize__sets_non_inputs_to_none(self):
            mask = np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )

            mask = aa.AbstractMask(array_2d=mask, sub_size=1)

            mask_new = mask + mask

            assert mask_new.pixel_scales == None
            assert mask_new.origin == None

        def test__new_mask_with_new_sub_size(self):

            mask = np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )

            mask = aa.AbstractMask(mask, sub_size=1)

            assert (
                mask
                == np.array(
                    [
                        [True, True, True, True],
                        [True, False, False, True],
                        [True, True, True, True],
                    ]
                )
            ).all()

            assert mask.sub_size == 1
            assert mask.pixel_scales == None
            assert mask.shape == (3, 4)

            mask = mask.new_mask_with_new_sub_size(sub_size=2)

            assert (
                mask
                == np.array(
                    [
                        [True, True, True, True],
                        [True, False, False, True],
                        [True, True, True, True],
                    ]
                )
            ).all()

            assert mask.sub_size == 2
            assert mask.pixel_scales == None
            assert mask.shape == (3, 4)

        def test__sub_mask__is_mask_at_sub_grid_resolution(self):

            mask = np.array([[False, True], [False, False]])

            mask = aa.AbstractMask(array_2d=mask, sub_size=2)

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

            mask = np.array([[False, False, True], [False, True, False]])

            mask = aa.AbstractMask(array_2d=mask, sub_size=2)

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

    class TestMaskRegions:
        def test__blurring_mask_for_psf_shape__compare_to_array_util(self):
            mask = np.array(
                [
                    [True, True, True, True, True, True, True, True],
                    [True, False, True, True, True, False, True, True],
                    [True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True],
                    [True, False, True, True, True, False, True, True],
                    [True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True],
                ]
            )

            blurring_mask_via_util = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
                mask=mask, kernel_shape=(3, 3)
            )

            mask = aa.AbstractMask(mask, sub_size=1)
            blurring_mask = mask.blurring_mask_from_kernel_shape(kernel_shape=(3, 3))

            assert (blurring_mask == blurring_mask_via_util).all()

        def test__edge_image_pixels__compare_to_array_util(self):
            mask = np.array(
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

            edge_pixels_util = aa.mask_util.edge_1d_indexes_from_mask(mask=mask)

            mask = aa.AbstractMask(array_2d=mask, sub_size=1)

            assert mask._edge_1d_indexes == pytest.approx(edge_pixels_util, 1e-4)
            assert mask._edge_2d_indexes[0] == pytest.approx(np.array([1, 1]), 1e-4)
            assert mask._edge_2d_indexes[10] == pytest.approx(np.array([3, 3]), 1e-4)
            assert mask._edge_1d_indexes.shape[0] == mask._edge_2d_indexes.shape[0]

        def test__edge_mask(self):
            mask = np.array(
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

            mask = aa.AbstractMask(array_2d=mask, sub_size=1)

            assert (
                    mask.edge_mask
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

        def test__border_image_pixels__compare_to_array_util(self):
            mask = np.array(
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

            border_pixels_util = aa.mask_util.border_1d_indexes_from_mask(mask=mask)

            mask = aa.AbstractMask(mask, sub_size=1)

            assert mask._border_1d_indexes == pytest.approx(border_pixels_util, 1e-4)
            assert mask._border_2d_indexes[0] == pytest.approx(np.array([1, 1]), 1e-4)
            assert mask._border_2d_indexes[10] == pytest.approx(np.array([3, 7]), 1e-4)
            assert mask._border_1d_indexes.shape[0] == mask._border_2d_indexes.shape[0]

        def test__border_mask(self):
            mask = np.array(
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

            mask = aa.AbstractMask(array_2d=mask, sub_size=1)

            assert (
                    mask.border_mask
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

        def test__sub_border_1d_indexes__compare_to_array_util_and_numerics(self):
            mask = np.array(
                [
                    [False, False, False, False, False, False, False, True],
                    [False, True, True, True, True, True, False, True],
                    [False, True, False, False, False, True, False, True],
                    [False, True, False, True, False, True, False, True],
                    [False, True, False, False, False, True, False, True],
                    [False, True, True, True, True, True, False, True],
                    [False, False, False, False, False, False, False, True],
                ]
            )

            sub_border_pixels_util = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
                mask=mask, sub_size=2
            )

            mask = aa.AbstractMask(array_2d=mask, sub_size=2)

            assert mask._sub_border_1d_indexes == pytest.approx(
                sub_border_pixels_util, 1e-4
            )

            mask = np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )

            mask = aa.AbstractMask(mask, sub_size=2)

            assert (
                    mask._sub_border_1d_indexes == np.array([0, 5, 9, 14, 23, 26, 31, 35])
            ).all()


class TestPixelMask:
    
    class TestConstructors:

        def test__mask_all_unmasked__5x5__input__all_are_false(self):

            mask = aa.PixelMask.unmasked_from_shape_and_sub_size(
                shape=(5, 5), invert=False, sub_size=1
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
            mask = aa.PixelMask.unmasked_from_shape_and_sub_size(
                shape=(5, 5), invert=True, sub_size=1
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

            mask = aa.PixelMask.from_fits(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0, sub_size=1
            )

            output_data_dir = "{}/../../test_files/array/output_test/".format(
                os.path.dirname(os.path.realpath(__file__))
            )

            if os.path.exists(output_data_dir):
                shutil.rmtree(output_data_dir)

            os.makedirs(output_data_dir)

            mask.output_mask_to_fits(file_path=output_data_dir + "mask.fits")

            mask = aa.PixelMask.from_fits(
                file_path=output_data_dir + "mask.fits", hdu=0, sub_size=1)

            assert (mask == np.ones((3, 3))).all()

    class TestResizing:

        def test__pad__compare_to_manual_mask(self):

            mask_2d = np.full(fill_value=False, shape=(5, 5))
            mask_2d[2, 2] = True

            mask = aa.PixelMask(array_2d=mask_2d, sub_size=1)

            mask_resized = mask.resized_mask_from_new_shape(
                new_shape=(7, 7), new_centre_pixels=(1, 1)
            )

            mask_resized_manual = np.full(fill_value=False, shape=(7, 7))
            mask_resized_manual[4, 4] = True

            assert type(mask_resized) == aa.PixelMask
            assert (mask_resized == mask_resized_manual).all()

        def test__trim__compare_to_manual_mask(self):

            mask_2d = np.full(fill_value=False, shape=(5, 5))
            mask_2d[2, 2] = True

            mask = aa.PixelMask(array_2d=mask_2d, sub_size=1)

            mask_resized = mask.resized_mask_from_new_shape(
                new_shape=(3, 3), new_centre_pixels=(4, 4)
            )

            mask_resized_manual = np.full(fill_value=False, shape=(3, 3))

            assert type(mask_resized) == aa.PixelMask
            assert (mask_resized == mask_resized_manual).all()


class TestScaledMask:
    
    class TestConstructorsViaShapes:

        def test__mask_all_unmasked__5x5__input__all_are_false(self):
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
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

            assert mask.origin == (0.0, 0.0)
            assert mask.mask_centre == (0.0, 0.0)

        def test__mask_all_unmasked_inverted__5x5__input__all_are_true(self):
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
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

            assert mask.origin == (0.0, 0.0)

        def test__mask_circular__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
                shape=(5, 4), pixel_scales=(2.7, 2.7), radius_arcsec=3.5, centre=(0.0, 0.0)
            )

            mask = aa.ScaledMask.circular(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                sub_size=1,
                radius_arcsec=3.5,
                centre=(0.0, 0.0),
            )

            assert (mask == mask_via_util).all()
            assert mask.origin == (0.0, 0.0)
            assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

        def test__mask_circular__inverted__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
                shape=(5, 4), pixel_scales=(2.7, 2.7), radius_arcsec=3.5, centre=(0.0, 0.0)
            )

            mask = aa.ScaledMask.circular(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                sub_size=1,
                radius_arcsec=3.5,
                centre=(0.0, 0.0),
                invert=True,
            )

            assert (mask == np.invert(mask_via_util)).all()
            assert mask.origin == (0.0, 0.0)
            assert mask.mask_centre == (0.0, 0.0)

        def test__mask_annulus__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=3.5,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledMask.circular_annular(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                sub_size=1,
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=3.5,
                centre=(0.0, 0.0),
            )

            assert (mask == mask_via_util).all()
            assert mask.origin == (0.0, 0.0)
            assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

        def test__mask_annulus_inverted__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=3.5,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledMask.circular_annular(
                shape=(5, 4),
                pixel_scales=(2.7, 2.7),
                sub_size=1,
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=3.5,
                centre=(0.0, 0.0),
                invert=True,
            )

            assert (mask == np.invert(mask_via_util)).all()
            assert mask.origin == (0.0, 0.0)
            assert mask.mask_centre == (0.0, 0.0)

        def test__mask_anti_annulus__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
                shape=(9, 9),
                pixel_scales=(1.2, 1.2),
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=2.2,
                outer_radius_2_arcsec=3.0,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledMask.circular_anti_annular(
                shape=(9, 9),
                pixel_scales=(1.2, 1.2),
                sub_size=1,
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=2.2,
                outer_radius_2_arcsec=3.0,
                centre=(0.0, 0.0),
            )

            assert (mask == mask_via_util).all()
            assert mask.origin == (0.0, 0.0)
            assert mask.mask_centre == (0.0, 0.0)

        def test__mask_anti_annulus_inverted__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
                shape=(9, 9),
                pixel_scales=(1.2, 1.2),
                inner_radius_arcsec=0.8,
                outer_radius_arcsec=2.2,
                outer_radius_2_arcsec=3.0,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledMask.circular_anti_annular(
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
            assert mask.origin == (0.0, 0.0)
            assert mask.mask_centre == (0.0, 0.0)

        def test__mask_elliptical__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
                shape=(8, 5),
                pixel_scales=(2.7, 2.7),
                major_axis_radius_arcsec=5.7,
                axis_ratio=0.4,
                phi=40.0,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledMask.elliptical(
                shape=(8, 5),
                pixel_scales=(2.7, 2.7),
                sub_size=1,
                major_axis_radius_arcsec=5.7,
                axis_ratio=0.4,
                phi=40.0,
                centre=(0.0, 0.0),
            )

            assert (mask == mask_via_util).all()
            assert mask.origin == (0.0, 0.0)
            assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

        def test__mask_elliptical_inverted__compare_to_array_util(self):
            mask_via_util = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
                shape=(8, 5),
                pixel_scales=(2.7, 2.7),
                major_axis_radius_arcsec=5.7,
                axis_ratio=0.4,
                phi=40.0,
                centre=(0.0, 0.0),
            )

            mask = aa.ScaledMask.elliptical(
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
            assert mask.origin == (0.0, 0.0)
            assert mask.mask_centre == (0.0, 0.0)

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

            mask = aa.ScaledMask.elliptical_annular(
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
            assert mask.origin == (0.0, 0.0)
            assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

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

            mask = aa.ScaledMask.elliptical_annular(
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
            assert mask.origin == (0.0, 0.0)
            assert mask.mask_centre == (0.0, 0.0)

    class TestParse:

        def test__load_and_output_mask_to_fits(self):

            mask = aa.ScaledMask.from_fits(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0, sub_size=1, pixel_scales=(1.0, 1.0),
            )

            output_data_dir = "{}/../../test_files/array/output_test/".format(
                os.path.dirname(os.path.realpath(__file__))
            )

            if os.path.exists(output_data_dir):
                shutil.rmtree(output_data_dir)

            os.makedirs(output_data_dir)

            mask.output_mask_to_fits(file_path=output_data_dir + "mask.fits")

            mask = aa.ScaledMask.from_fits(
                file_path=output_data_dir + "mask.fits", hdu=0, sub_size=1, pixel_scales=(1.0, 1.0), origin=(2.0, 2.0))

            assert (mask == np.ones((3, 3))).all()
            assert mask.pixel_scales == (1.0, 1.0)
            assert mask.origin == (2.0, 2.0)

    class TestGeometry:
        
        def test__zero_or_negative_pixel_scale__raises_exception(self):

            with pytest.raises(exc.MaskException):
                aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                    shape=(2, 2), pixel_scales=(0.0, 0.0), sub_size=1
                )

            with pytest.raises(exc.MaskException):
                aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                    shape=(2, 2), pixel_scales=(-0.5, 0.0), sub_size=1
                )

        def test__central_pixel__depends_on_shape_pixel_scale_and_origin(self):
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(0.1, 0.1), sub_size=1
            )
            assert mask.central_pixel_coordinates == (1, 1)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 4), pixel_scales=(0.1, 0.1), sub_size=1
            )
            assert mask.central_pixel_coordinates == (1.5, 1.5)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(5, 3), pixel_scales=(0.1, 0.1), sub_size=1, origin=(1.0, 2.0)
            )
            assert mask.central_pixel_coordinates == (2.0, 1.0)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(2.0, 1.0), sub_size=1
            )
            assert mask.central_pixel_coordinates == (1, 1)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 4), pixel_scales=(2.0, 1.0), sub_size=1
            )
            assert mask.central_pixel_coordinates == (1.5, 1.5)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(5, 3), pixel_scales=(2.0, 1.0), sub_size=1, origin=(1.0, 2.0)
            )
            assert mask.central_pixel_coordinates == (2, 1)

        def test__centring__adapts_to_max_and_min_of_mask(self):
            mask = np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )

            mask = aa.ScaledMask(mask, pixel_scales=(1.0, 1.0), sub_size=1)

            assert mask.mask_centre == (0.0, 0.0)

            mask = np.array(
                [
                    [True, True, True, True],
                    [True, False, False, False],
                    [True, True, True, True],
                ]
            )

            mask = aa.ScaledMask(mask, pixel_scales=(1.0, 1.0), sub_size=1)

            assert mask.mask_centre == (0.0, 0.5)

            mask = np.array(
                [
                    [True, True, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )

            mask = aa.ScaledMask(mask, pixel_scales=(1.0, 1.0), sub_size=1)

            assert mask.mask_centre == (0.5, 0.0)

            mask = np.array(
                [
                    [True, True, True, True],
                    [False, False, False, True],
                    [True, True, True, True],
                ]
            )

            mask = aa.ScaledMask(mask, pixel_scales=(1.0, 1.0), sub_size=1)

            assert mask.mask_centre == (0.0, -0.5)

            mask = np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, True, True],
                ]
            )

            mask = aa.ScaledMask(mask, pixel_scales=(1.0, 1.0), sub_size=1)

            assert mask.mask_centre == (-0.5, 0.0)

            mask = np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [False, True, True, True],
                ]
            )

            mask = aa.ScaledMask(mask, pixel_scales=(1.0, 1.0), sub_size=1)

            assert mask.mask_centre == (-0.5, -0.5)

        def test__pixel_grid__y_and_x_ticks(self):

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(0.5, 0.5), sub_size=1
            )
            assert mask.yticks == pytest.approx(
                np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(6, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask.yticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 1), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(0.5, 0.5), sub_size=1
            )
            assert mask.xticks == pytest.approx(
                np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 6), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask.xticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(1, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(1.0, 5.0), sub_size=1
            )
            assert mask.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(0.5, 5.0), sub_size=1
            )
            assert mask.yticks == pytest.approx(
                np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(6, 3), pixel_scales=(1.0, 5.0), sub_size=1
            )
            assert mask.yticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 6), pixel_scales=(1.0, 5.0), sub_size=1
            )
            assert mask.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(5.0, 1.0), sub_size=1
            )
            assert mask.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(5.0, 0.5), sub_size=1
            )
            assert mask.xticks == pytest.approx(
                np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 6), pixel_scales=(5.0, 1.0), sub_size=1
            )
            assert mask.xticks == pytest.approx(np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(6, 3), pixel_scales=(5.0, 1.0), sub_size=1
            )
            assert mask.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

    class TestGrids:

        def test__unmasked_grid_2d__compare_to_array_util(self):
            grid_2d_util = aa.grid_util.grid_2d_from_shape_pixel_scales_sub_size_and_origin(
                shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1
            )

            assert mask.unmasked_grid.in_2d == pytest.approx(grid_2d_util, 1e-4)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )

            assert (
                    mask.unmasked_grid.in_2d
                    == np.array(
                [
                    [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]],
                    [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]],
                    [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]],
                ]
            )
            ).all()

            grid_2d_util = aa.grid_util.grid_2d_from_shape_pixel_scales_sub_size_and_origin(
                shape=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 7), sub_size=1, pixel_scales=(0.8, 0.56)
            )

            assert mask.unmasked_grid.in_2d == pytest.approx(grid_2d_util, 1e-4)

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), sub_size=1, pixel_scales=(1.0, 2.0)
            )

            assert (
                    mask.unmasked_grid.in_2d
                    == np.array(
                [
                    [[1.0, -2.0], [1.0, 0.0], [1.0, 2.0]],
                    [[0.0, -2.0], [0.0, 0.0], [0.0, 2.0]],
                    [[-1.0, -2.0], [-1.0, 0.0], [-1.0, 2.0]],
                ]
            )
            ).all()

        def test__unmasked_grid_1d__compare_to_array_util(self):
            grid_1d_util = aa.grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
                shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1
            )

            assert mask.unmasked_grid.in_1d == pytest.approx(grid_1d_util, 1e-4)

            grid_1d_util = aa.grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
                shape=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 7), sub_size=1, pixel_scales=(0.8, 0.56)
            )

            assert mask.unmasked_grid.in_1d == pytest.approx(grid_1d_util, 1e-4)

        def test__grid_with_nonzero_origins__compure_to_array_util(self):
            grid_2d_util = aa.grid_util.grid_2d_from_shape_pixel_scales_sub_size_and_origin(
                shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1, origin=(1.0, 3.0)
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1, origin=(1.0, 3.0)
            )

            assert mask.unmasked_grid.in_2d == pytest.approx(grid_2d_util, 1e-4)

            grid_1d_util = aa.grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
                shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1, origin=(-1.0, -4.0)
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1, origin=(-1.0, -4.0)
            )

            assert mask.unmasked_grid.in_1d == pytest.approx(grid_1d_util, 1e-4)

            grid_2d_util = aa.grid_util.grid_2d_from_shape_pixel_scales_sub_size_and_origin(
                shape=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1, origin=(1.0, 2.0)
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 7), sub_size=1, pixel_scales=(0.8, 0.56), origin=(1.0, 2.0)
            )

            assert mask.unmasked_grid.in_2d == pytest.approx(grid_2d_util, 1e-4)

            grid_1d_util = aa.grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
                shape=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1, origin=(-1.0, -4.0)
            )

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1, origin=(-1.0, -4.0)
            )

            assert mask.unmasked_grid.in_1d == pytest.approx(grid_1d_util, 1e-4)

        def test__masked_grids_1d(self):
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )

            assert (
                    mask.masked_grid.in_1d
                    == np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            )
            ).all()

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
            mask[1, 1] = True

            assert (
                    mask.masked_grid.in_1d
                    == np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            )
            ).all()

            mask = aa.ScaledMask(
                array_2d=np.array([[False, True], [True, False], [True, False]]),
                sub_size=1,
                pixel_scales=(1.0, 1.0),
                origin=(3.0, -2.0),
            )

            assert (
                    mask.masked_grid.in_1d == np.array([[4.0, -2.5], [3.0, -1.5], [2.0, -1.5]])
            ).all()

            mask = aa.ScaledMask.circular(
                shape=(4, 7),
                radius_arcsec=4.0,
                pixel_scales=(2.0, 2.0),
                sub_size=1,
                centre=(1.0, 5.0),
            )

            masked_grid_1d_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
                mask=mask, pixel_scales=(2.0, 2.0), sub_size=1
            )

            assert (mask.masked_grid.in_1d == masked_grid_1d_util).all()

        def test__masked_sub_grid(self):
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )

            assert (
                    mask.masked_sub_grid
                    == np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            )
            ).all()

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(2, 2), pixel_scales=(1.0, 1.0), sub_size=2
            )

            assert (
                    mask.masked_sub_grid
                    == np.array(
                [
                    [0.75, -0.75],
                    [0.75, -0.25],
                    [0.25, -0.75],
                    [0.25, -0.25],
                    [0.75, 0.25],
                    [0.75, 0.75],
                    [0.25, 0.25],
                    [0.25, 0.75],
                    [-0.25, -0.75],
                    [-0.25, -0.25],
                    [-0.75, -0.75],
                    [-0.75, -0.25],
                    [-0.25, 0.25],
                    [-0.25, 0.75],
                    [-0.75, 0.25],
                    [-0.75, 0.75],
                ]
            )
            ).all()

            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
            mask[1, 1] = True

            assert (
                    mask.masked_sub_grid
                    == np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            )
            ).all()

            mask = aa.ScaledMask(
                array_2d=np.array([[False, True], [True, False], [True, False]]),
                pixel_scales=(1.0, 1.0),
                sub_size=5,
                origin=(3.0, -2.0),
            )

            masked_grid_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
                mask=mask, pixel_scales=(1.0, 1.0), sub_size=5, origin=(3.0, -2.0)
            )

            assert (mask.masked_sub_grid == masked_grid_util).all()

        def test__edge_grid(self):
            mask = np.array(
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

            mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=1)

            assert mask.edge_grid.in_1d[0:11] == pytest.approx(
                np.array(
                    [
                        [3.0, -3.0],
                        [3.0, -2.0],
                        [3.0, -1.0],
                        [3.0, -0.0],
                        [3.0, 1.0],
                        [3.0, 2.0],
                        [3.0, 3.0],
                        [2.0, -3.0],
                        [2.0, 3.0],
                        [1.0, -3.0],
                        [1.0, -1.0],
                    ]
                ),
                1e-4,
            )

        def test__border_grid(self):
            mask = np.array(
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

            mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=1)

            assert mask.border_grid.in_1d[0:11] == pytest.approx(
                np.array(
                    [
                        [3.0, -3.0],
                        [3.0, -2.0],
                        [3.0, -1.0],
                        [3.0, -0.0],
                        [3.0, 1.0],
                        [3.0, 2.0],
                        [3.0, 3.0],
                        [2.0, -3.0],
                        [2.0, 3.0],
                        [1.0, -3.0],
                        [1.0, 3.0],
                    ]
                ),
                1e-4,
            )

        def test__sub_border_1d_grid__compare_numerical_values(self):
            mask = np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, False, False, True, True, True, True],
                    [True, True, True, True, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )

            mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

            assert (
                    mask.sub_border_grid_1d
                    == np.array([[1.25, -2.25], [1.25, -1.25], [-0.25, 1.25]])
            ).all()

            mask = np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )

            mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

            assert (
                    mask.sub_border_grid_1d
                    == np.array(
                [
                    [1.25, -1.25],
                    [1.25, 0.25],
                    [1.25, 1.25],
                    [-0.25, -1.25],
                    [-0.25, 1.25],
                    [-1.25, -1.25],
                    [-1.25, 0.25],
                    [-1.25, 1.25],
                ]
            )
            ).all()

    class TestResizing:
    
        def test__pad__compare_to_manual_mask(self):
    
            mask_2d = np.full(fill_value=False, shape=(5, 5))
            mask_2d[2, 2] = True
    
            mask = aa.ScaledMask(array_2d=mask_2d, pixel_scales=(1.0, 1.0), sub_size=1)
    
            mask_resized = mask.resized_mask_from_new_shape(
                new_shape=(7, 7), new_centre_pixels=(1, 1)
            )
    
            mask_resized_manual = np.full(fill_value=False, shape=(7, 7))
            mask_resized_manual[4, 4] = True
    
            assert type(mask_resized) == aa.ScaledMask
            assert (mask_resized == mask_resized_manual).all()
    
        def test__trim__compare_to_manual_mask(self):
    
            mask_2d = np.full(fill_value=False, shape=(5, 5))
            mask_2d[2, 2] = True
    
            mask = aa.ScaledMask(array_2d=mask_2d, pixel_scales=(1.0, 1.0), sub_size=1)
    
            mask_resized = mask.resized_mask_from_new_shape(
                new_shape=(3, 3), new_centre_pixels=(4, 4)
            )
    
            mask_resized_manual = np.full(fill_value=False, shape=(3, 3))
    
            assert type(mask_resized) == aa.ScaledMask
            assert (mask_resized == mask_resized_manual).all()
    
        def test__new_centre_is_in_arcsec(self):
    
            mask_2d = np.full(fill_value=False, shape=(5, 5))
            mask_2d[2, 2] = True
    
            mask = aa.ScaledMask(array_2d=mask_2d, pixel_scales=(1.0, 1.0), sub_size=1)
    
            mask_resized = mask.resized_mask_from_new_shape(
                new_shape=(3, 3), new_centre_arcsec=(6.0, 6.0)
            )
            mask_resized_util = aa.array_util.resized_array_2d_from_array_2d_and_resized_shape(
                array_2d=mask_2d, resized_shape=(3, 3), origin=(0, 4)
            )
            assert (mask_resized == mask_resized_util).all()
    
            mask_resized = mask.resized_mask_from_new_shape(
                new_shape=(3, 3), new_centre_arcsec=(7.49, 4.51)
            )
            mask_resized_util = aa.array_util.resized_array_2d_from_array_2d_and_resized_shape(
                array_2d=mask_2d, resized_shape=(3, 3), origin=(0, 4)
            )
            assert (mask_resized == mask_resized_util).all()
    
            mask_resized = mask.resized_mask_from_new_shape(
                new_shape=(3, 3), new_centre_arcsec=(7.49, 7.49)
            )
            mask_resized_util = aa.array_util.resized_array_2d_from_array_2d_and_resized_shape(
                array_2d=mask_2d, resized_shape=(3, 3), origin=(0, 4)
            )
            assert (mask_resized == mask_resized_util).all()
    
            mask_resized = mask.resized_mask_from_new_shape(
                new_shape=(3, 3), new_centre_arcsec=(4.51, 4.51)
            )
            mask_resized_util = aa.array_util.resized_array_2d_from_array_2d_and_resized_shape(
                array_2d=mask_2d, resized_shape=(3, 3), origin=(0, 4)
            )
            assert (mask_resized == mask_resized_util).all()
    
            mask_resized = mask.resized_mask_from_new_shape(
                new_shape=(3, 3), new_centre_arcsec=(4.51, 7.49)
            )
            mask_resized_util = aa.array_util.resized_array_2d_from_array_2d_and_resized_shape(
                array_2d=mask_2d, resized_shape=(3, 3), origin=(0, 4)
            )
            assert (mask_resized == mask_resized_util).all()

    class TestBinnedMaskFromMask:

        def test__compare_to_mask_via_util(self):
            mask = np.full(shape=(14, 19), fill_value=True)
            mask[1, 5] = False
            mask[6, 5] = False
            mask[4, 9] = False
            mask[11, 10] = False

            binned_up_mask_via_util = aa.binning_util.binned_up_mask_from_mask_2d_and_bin_up_factor(
                mask_2d=mask, bin_up_factor=2
            )

            mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=1)
            mask = mask.binned_up_mask_from_bin_up_factor(bin_up_factor=2)
            assert (mask == binned_up_mask_via_util).all()
            assert mask.pixel_scale == 2.0

            binned_up_mask_via_util = aa.binning_util.binned_up_mask_from_mask_2d_and_bin_up_factor(
                mask_2d=mask, bin_up_factor=3
            )

            mask = aa.ScaledMask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=1)
            mask = mask.binned_up_mask_from_bin_up_factor(bin_up_factor=3)
            assert (mask == binned_up_mask_via_util).all()
            assert mask.pixel_scale == 6.0

    class TestZoomCentreAndOffet:
        def test__odd_sized_false_mask__centre_is_0_0__pixels_from_centre_are_0_0(self):
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask._zoom_centre == (1.0, 1.0)
            assert mask._zoom_offset_pixels == (0, 0)
    
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(5, 5), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask._zoom_centre == (2.0, 2.0)
            assert mask._zoom_offset_pixels == (0, 0)
    
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(3, 5), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask._zoom_centre == (1.0, 2.0)
            assert mask._zoom_offset_pixels == (0, 0)
    
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(5, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask._zoom_centre == (2.0, 1.0)
            assert mask._zoom_offset_pixels == (0, 0)
    
        def test__even_sized_false_mask__centre_is_0_0__pixels_from_centre_are_0_0(self):
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 4), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask._zoom_centre == (1.5, 1.5)
            assert mask._zoom_offset_pixels == (0, 0)
    
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(6, 6), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask._zoom_centre == (2.5, 2.5)
            assert mask._zoom_offset_pixels == (0, 0)
    
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(4, 6), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask._zoom_centre == (1.5, 2.5)
            assert mask._zoom_offset_pixels == (0, 0)
    
            mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
                shape=(6, 4), pixel_scales=(1.0, 1.0), sub_size=1
            )
            assert mask._zoom_centre == (2.5, 1.5)
            assert mask._zoom_offset_pixels == (0, 0)
    
        def test__mask_is_single_false__extraction_centre_is_central_pixel(self):
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[False, True, True], [True, True, True], [True, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (0, 0)
            assert mask._zoom_offset_pixels == (-1, -1)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[True, True, False], [True, True, True], [True, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (0, 2)
            assert mask._zoom_offset_pixels == (-1, 1)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[True, True, True], [True, True, True], [False, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (2, 0)
            assert mask._zoom_offset_pixels == (1, -1)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[True, True, True], [True, True, True], [True, True, False]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (2, 2)
            assert mask._zoom_offset_pixels == (1, 1)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[True, False, True], [True, True, True], [True, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (0, 1)
            assert mask._zoom_offset_pixels == (-1, 0)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[True, True, True], [False, True, True], [True, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (1, 0)
            assert mask._zoom_offset_pixels == (0, -1)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[True, True, True], [True, True, False], [True, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (1, 2)
            assert mask._zoom_offset_pixels == (0, 1)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[True, True, True], [True, True, True], [True, False, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (2, 1)
            assert mask._zoom_offset_pixels == (1, 0)
    
        def test__mask_is_x2_false__extraction_centre_is_central_pixel(self):
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[False, True, True], [True, True, True], [True, True, False]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (1, 1)
            assert mask._zoom_offset_pixels == (0, 0)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[False, True, True], [True, True, True], [False, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (1, 0)
            assert mask._zoom_offset_pixels == (0, -1)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[False, True, False], [True, True, True], [True, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (0, 1)
            assert mask._zoom_offset_pixels == (-1, 0)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [[False, False, True], [True, True, True], [True, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
            assert mask._zoom_centre == (0, 0.5)
            assert mask._zoom_offset_pixels == (-1, -0.5)
    
        def test__rectangular_mask(self):
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [
                        [False, True, True, True],
                        [True, True, True, True],
                        [True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
    
            assert mask._zoom_centre == (0, 0)
            assert mask._zoom_offset_pixels == (-1.0, -1.5)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [
                        [True, True, True, True],
                        [True, True, True, True],
                        [True, True, True, False],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
    
            assert mask._zoom_centre == (2, 3)
            assert mask._zoom_offset_pixels == (1.0, 1.5)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [
                        [True, True, True, True, True],
                        [True, True, True, True, True],
                        [True, True, True, True, False],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
    
            assert mask._zoom_centre == (2, 4)
            assert mask._zoom_offset_pixels == (1, 2)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, False],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
    
            assert mask._zoom_centre == (2, 6)
            assert mask._zoom_offset_pixels == (1, 3)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, False],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
    
            assert mask._zoom_centre == (4, 2)
            assert mask._zoom_offset_pixels == (2, 1)
    
            mask = aa.ScaledMask(
                array_2d=np.array(
                    [
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, False],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )
    
            assert mask._zoom_centre == (6, 2)
            assert mask._zoom_offset_pixels == (3, 1)