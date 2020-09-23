import os

import numpy as np
import pytest

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


class TestUnmasked:
    def test__mask_all_unmasked__5x5__input__all_are_false(self):

        mask = aa.Mask.unmasked(shape_2d=(5, 5), invert=False)

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

        mask = aa.Mask.unmasked(
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

        mask = aa.Mask.unmasked(
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


class TestCircular:
    def test__mask_circular__compare_to_array_util(self):

        mask_via_util = aa.util.mask.mask_circular_from(
            shape_2d=(5, 4), pixel_scales=(2.7, 2.7), radius=3.5, centre=(0.0, 0.0)
        )

        mask = aa.Mask.circular(
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

        mask = aa.Mask.circular(
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


class TestAnnular:
    def test__mask_annulus__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_circular_annular_from(
            shape_2d=(5, 4),
            pixel_scales=(2.7, 2.7),
            inner_radius=0.8,
            outer_radius=3.5,
            centre=(0.0, 0.0),
        )

        mask = aa.Mask.circular_annular(
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

        mask = aa.Mask.circular_annular(
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


class TestAntiAnnular:
    def test__mask_anti_annulus__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_circular_anti_annular_from(
            shape_2d=(9, 9),
            pixel_scales=(1.2, 1.2),
            inner_radius=0.8,
            outer_radius=2.2,
            outer_radius_2_scaled=3.0,
            centre=(0.0, 0.0),
        )

        mask = aa.Mask.circular_anti_annular(
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

        mask = aa.Mask.circular_anti_annular(
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


class TestElliptical:
    def test__mask_elliptical__compare_to_array_util(self):
        mask_via_util = aa.util.mask.mask_elliptical_from(
            shape_2d=(8, 5),
            pixel_scales=(2.7, 2.7),
            major_axis_radius=5.7,
            axis_ratio=0.4,
            phi=40.0,
            centre=(0.0, 0.0),
        )

        mask = aa.Mask.elliptical(
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

        mask = aa.Mask.elliptical(
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


class TestEllipiticalAnnular:
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

        mask = aa.Mask.elliptical_annular(
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

        mask = aa.Mask.elliptical_annular(
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


class TestFromPixelCoordinates:
    def test__mask_with_or_without_buffer__false_at_buffed_coordinates(self):

        mask = aa.Mask.from_pixel_coordinates(
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

        mask = aa.Mask.from_pixel_coordinates(
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

        mask = aa.Mask.from_pixel_coordinates(
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
