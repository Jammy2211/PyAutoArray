import autoarray as aa
from autoarray import exc
import os

import numpy as np
import pytest


test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestTotalPixels:
    def test__total_image_pixels_from_mask(self):
        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        assert aa.mask_util.total_pixels_from_mask(mask=mask) == 5

    def test__total_sub_pixels_from_mask(self):
        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        assert (
            aa.mask_util.total_sub_pixels_from_mask_and_sub_size(mask, sub_size=2) == 20
        )

    def test__total_edge_pixels_from_mask(self):

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        assert aa.mask_util.total_edge_pixels_from_mask(mask=mask) == 8

    class TestTotalSparsePixels:
        def test__mask_full_false__full_pixelization_grid_pixels_in_mask(self):

            ma = aa.Mask(
                array_2d=np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            full_pix_grid_pixel_centres = np.array([[0, 0], [0, 1], [0, 2], [1, 0]])

            total_masked_pixels = aa.mask_util.total_sparse_pixels_from_mask(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 4

            full_pix_grid_pixel_centres = np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 1]]
            )

            total_masked_pixels = aa.mask_util.total_sparse_pixels_from_mask(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 6

        def test__mask_is_cross__only_pixelization_grid_pixels_in_mask_are_counted(
            self
        ):

            ma = aa.Mask(
                array_2d=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            full_pix_grid_pixel_centres = np.array([[0, 0], [0, 1], [0, 2], [1, 0]])

            total_masked_pixels = aa.mask_util.total_sparse_pixels_from_mask(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 2

            full_pix_grid_pixel_centres = np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 1]]
            )

            total_masked_pixels = aa.mask_util.total_sparse_pixels_from_mask(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 4

        def test__same_as_above_but_3x4_mask(self):

            ma = aa.Mask(
                array_2d=np.array(
                    [
                        [True, True, False, True],
                        [False, False, False, False],
                        [True, True, False, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            full_pix_grid_pixel_centres = np.array([[0, 0], [0, 1], [0, 2], [1, 0]])

            total_masked_pixels = aa.mask_util.total_sparse_pixels_from_mask(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 2

            full_pix_grid_pixel_centres = np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3], [2, 2]]
            )

            total_masked_pixels = aa.mask_util.total_sparse_pixels_from_mask(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 6

        def test__same_as_above_but_4x3_mask(self):

            ma = aa.Mask(
                array_2d=np.array(
                    [
                        [True, False, True],
                        [True, False, True],
                        [False, False, False],
                        [True, False, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            full_pix_grid_pixel_centres = np.array([[0, 0], [0, 1], [0, 2], [1, 1]])

            total_masked_pixels = aa.mask_util.total_sparse_pixels_from_mask(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 2

            full_pix_grid_pixel_centres = np.array(
                [[0, 0], [0, 1], [0, 2], [1, 1], [2, 0], [2, 1], [2, 2], [3, 1]]
            )

            total_masked_pixels = aa.mask_util.total_sparse_pixels_from_mask(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 6


class TestMaskCircular(object):
    def test__3x3_mask_input_radius_small__medium__big__masks(self):
        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(3, 3), pixel_scales=(1.0, 1.0), radius_arcsec=0.5
        )

        assert (
            mask
            == np.array([[True, True, True], [True, False, True], [True, True, True]])
        ).all()

        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(3, 3), pixel_scales=(1.0, 1.0), radius_arcsec=1.3
        )

        assert (
            mask
            == np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            )
        ).all()

        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(3, 3), pixel_scales=(1.0, 1.0), radius_arcsec=3.0
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(3, 3), pixel_scales=(0.5, 1.0), radius_arcsec=0.5
        )

        assert (
            mask
            == np.array([[True, False, True], [True, False, True], [True, False, True]])
        ).all()

    def test__4x3_mask_input_radius_small__medium__big__masks(self):

        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(4, 3), pixel_scales=(1.0, 1.0), radius_arcsec=0.5
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True],
                    [True, False, True],
                    [True, False, True],
                    [True, True, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(4, 3), pixel_scales=(1.0, 1.0), radius_arcsec=1.5001
        )

        assert (
            mask
            == np.array(
                [
                    [True, False, True],
                    [False, False, False],
                    [False, False, False],
                    [True, False, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(4, 3), pixel_scales=(1.0, 1.0), radius_arcsec=3.0
        )

        assert (
            mask
            == np.array(
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ]
            )
        ).all()

    def test__4x4_mask_input_radius_small__medium__big__masks(self):
        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(4, 4), pixel_scales=(1.0, 1.0), radius_arcsec=0.72
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(4, 4), pixel_scales=(1.0, 1.0), radius_arcsec=1.7
        )

        assert (
            mask
            == np.array(
                [
                    [True, False, False, True],
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, False, False, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(4, 4), pixel_scales=(1.0, 1.0), radius_arcsec=3.0
        )

        assert (
            mask
            == np.array(
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            )
        ).all()

    def test__origin_shifts__downwards__right__diagonal(self):

        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(3, 3), pixel_scales=(3.0, 3.0), radius_arcsec=0.5, centre=(-3, 0)
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array([[True, True, True], [True, True, True], [True, False, True]])
        ).all()

        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(3, 3), pixel_scales=(3.0, 3.0), radius_arcsec=0.5, centre=(0.0, 3.0)
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array([[True, True, True], [True, True, False], [True, True, True]])
        ).all()

        mask = aa.mask_util.mask_circular_from_shape_pixel_scales_and_radius(
            shape=(3, 3), pixel_scales=(3.0, 3.0), radius_arcsec=0.5, centre=(3, 3)
        )

        assert (
            mask
            == np.array([[True, True, False], [True, True, True], [True, True, True]])
        ).all()


class TestMaskAnnular(object):
    def test__mask_inner_radius_zero_outer_radius_small_medium_and_large__mask(self):
        mask = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_radius_arcsec=0.0,
            outer_radius_arcsec=0.5,
        )

        assert (
            mask
            == np.array([[True, True, True], [True, False, True], [True, True, True]])
        ).all()

        mask = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=(4, 4),
            pixel_scales=(1.0, 1.0),
            inner_radius_arcsec=0.81,
            outer_radius_arcsec=2.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, False, False, True],
                    [False, True, True, False],
                    [False, True, True, False],
                    [True, False, False, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_radius_arcsec=0.5,
            outer_radius_arcsec=3.0,
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

        mask = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=(4, 4),
            pixel_scales=(0.5, 1.0),
            inner_radius_arcsec=1.1,
            outer_radius_arcsec=2.0,
        )

        assert (
            mask
            == np.array(
                [
                    [False, True, True, False],
                    [False, True, True, False],
                    [False, True, True, False],
                    [False, True, True, False],
                ]
            )
        ).all()

    def test__4x3_mask_inner_radius_small_outer_radius_medium__mask(self):
        mask = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=(4, 3),
            pixel_scales=(1.0, 1.0),
            inner_radius_arcsec=0.51,
            outer_radius_arcsec=1.51,
        )

        assert (
            mask
            == np.array(
                [
                    [True, False, True],
                    [False, True, False],
                    [False, True, False],
                    [True, False, True],
                ]
            )
        ).all()

    def test__4x3_mask_inner_radius_medium_outer_radius_large__mask(self):
        mask = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=(4, 3),
            pixel_scales=(1.0, 1.0),
            inner_radius_arcsec=1.51,
            outer_radius_arcsec=3.0,
        )

        assert (
            mask
            == np.array(
                [
                    [False, True, False],
                    [True, True, True],
                    [True, True, True],
                    [False, True, False],
                ]
            )
        ).all()

    def test__4x4_mask_inner_radius_medium_outer_radius_large__mask(self):
        mask = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=(4, 4),
            pixel_scales=(1.0, 1.0),
            inner_radius_arcsec=1.71,
            outer_radius_arcsec=3.0,
        )

        assert (
            mask
            == np.array(
                [
                    [False, True, True, False],
                    [True, True, True, True],
                    [True, True, True, True],
                    [False, True, True, False],
                ]
            )
        ).all()

    def test__origin_shift__simple_shift_upwards__right_diagonal(self):

        mask = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=(3, 3),
            pixel_scales=(3.0, 3.0),
            inner_radius_arcsec=0.5,
            outer_radius_arcsec=9.0,
            centre=(3.0, 0.0),
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array(
                [[False, True, False], [False, False, False], [False, False, False]]
            )
        ).all()

        mask = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=(3, 3),
            pixel_scales=(3.0, 3.0),
            inner_radius_arcsec=0.5,
            outer_radius_arcsec=9.0,
            centre=(0.0, 3.0),
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array(
                [[False, False, False], [False, False, True], [False, False, False]]
            )
        ).all()

        mask = aa.mask_util.mask_circular_annular_from_shape_pixel_scales_and_radii(
            shape=(3, 3),
            pixel_scales=(3.0, 3.0),
            inner_radius_arcsec=0.5,
            outer_radius_arcsec=9.0,
            centre=(-3.0, 3.0),
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array(
                [[False, False, False], [False, False, False], [False, False, True]]
            )
        ).all()


class TestMaskAntiAnnular(object):
    def test__5x5_mask_inner_radius_includes_central_pixel__outer_extended_beyond_radius(
        self
    ):

        mask = aa.mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
            shape=(5, 5),
            pixel_scales=(1.0, 1.0),
            inner_radius_arcsec=0.5,
            outer_radius_arcsec=10.0,
            outer_radius_2_arcsec=20.0,
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

    def test__5x5_mask_inner_radius_includes_3x3_central_pixels__outer_extended_beyond_radius(
        self
    ):

        mask = aa.mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
            shape=(5, 5),
            pixel_scales=(1.0, 1.0),
            inner_radius_arcsec=1.5,
            outer_radius_arcsec=10.0,
            outer_radius_2_arcsec=20.0,
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

        mask = aa.mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
            shape=(5, 5),
            pixel_scales=(0.1, 1.0),
            inner_radius_arcsec=1.5,
            outer_radius_arcsec=10.0,
            outer_radius_2_arcsec=20.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                ]
            )
        ).all()

    def test__5x5_mask_inner_radius_includes_central_pixel__outer_radius_includes_outer_pixels(
        self
    ):

        mask = aa.mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
            shape=(5, 5),
            pixel_scales=(1.0, 1.0),
            inner_radius_arcsec=0.5,
            outer_radius_arcsec=1.5,
            outer_radius_2_arcsec=20.0,
        )

        assert (
            mask
            == np.array(
                [
                    [False, False, False, False, False],
                    [False, True, True, True, False],
                    [False, True, False, True, False],
                    [False, True, True, True, False],
                    [False, False, False, False, False],
                ]
            )
        ).all()

    def test__7x7_second_outer_radius_mask_works_too(self):

        mask = aa.mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
            shape=(7, 7),
            pixel_scales=(1.0, 1.0),
            inner_radius_arcsec=0.5,
            outer_radius_arcsec=1.5,
            outer_radius_2_arcsec=2.9,
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, False, False, False, False, False, True],
                    [True, False, True, True, True, False, True],
                    [True, False, True, False, True, False, True],
                    [True, False, True, True, True, False, True],
                    [True, False, False, False, False, False, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__origin_shift__diagonal_shift(self):

        mask = aa.mask_util.mask_circular_anti_annular_from_shape_pixel_scales_and_radii(
            shape=(7, 7),
            pixel_scales=(3.0, 3.0),
            inner_radius_arcsec=1.5,
            outer_radius_arcsec=4.5,
            outer_radius_2_arcsec=8.7,
            centre=(-3.0, 3.0),
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, False, False],
                    [True, True, False, True, True, True, False],
                    [True, True, False, True, False, True, False],
                    [True, True, False, True, True, True, False],
                    [True, True, False, False, False, False, False],
                ]
            )
        ).all()


class TestMaskElliptical(object):
    def test__input_circular_params__small_medium_and_large_masks(self):

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=0.5,
            axis_ratio=1.0,
            phi=0.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [True, False, True], [True, True, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.3,
            axis_ratio=1.0,
            phi=0.0,
        )

        assert (
            mask
            == np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            )
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=3.0,
            axis_ratio=1.0,
            phi=0.0,
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

    def test__input_ellipticl_params__reduce_axis_ratio_makes_side_mask_values_false(
        self
    ):

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.3,
            axis_ratio=0.1,
            phi=0.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [False, False, False], [True, True, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.3,
            axis_ratio=0.1,
            phi=180.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [False, False, False], [True, True, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.3,
            axis_ratio=0.1,
            phi=360.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [False, False, False], [True, True, True]])
        ).all()

    def test__same_as_above_but_90_degree_rotations(self):

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.3,
            axis_ratio=0.1,
            phi=90.0,
        )

        assert (
            mask
            == np.array([[True, False, True], [True, False, True], [True, False, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.3,
            axis_ratio=0.1,
            phi=270.0,
        )

        assert (
            mask
            == np.array([[True, False, True], [True, False, True], [True, False, True]])
        ).all()

    def test__same_as_above_but_diagonal_rotations(self):

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.5,
            axis_ratio=0.1,
            phi=45.0,
        )

        assert (
            mask
            == np.array([[True, True, False], [True, False, True], [False, True, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.5,
            axis_ratio=0.1,
            phi=135.0,
        )

        assert (
            mask
            == np.array([[False, True, True], [True, False, True], [True, True, False]])
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.5,
            axis_ratio=0.1,
            phi=225.0,
        )

        assert (
            mask
            == np.array([[True, True, False], [True, False, True], [False, True, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.5,
            axis_ratio=0.1,
            phi=315.0,
        )

        assert (
            mask
            == np.array([[False, True, True], [True, False, True], [True, True, False]])
        ).all()

    def test__4x3__ellipse_is_formed(self):

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(4, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.5,
            axis_ratio=0.9,
            phi=90.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, False, True],
                    [False, False, False],
                    [False, False, False],
                    [True, False, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(4, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.5,
            axis_ratio=0.1,
            phi=270.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, False, True],
                    [True, False, True],
                    [True, False, True],
                    [True, False, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(4, 3),
            pixel_scales=(1.0, 0.1),
            major_axis_radius_arcsec=1.5,
            axis_ratio=0.1,
            phi=270.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, False, True],
                    [False, False, False],
                    [False, False, False],
                    [True, False, True],
                ]
            )
        ).all()

    def test__3x4__ellipse_is_formed(self):

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 4),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.5,
            axis_ratio=0.9,
            phi=0.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, False, False, True],
                    [False, False, False, False],
                    [True, False, False, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 4),
            pixel_scales=(1.0, 1.0),
            major_axis_radius_arcsec=1.5,
            axis_ratio=0.1,
            phi=180.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True],
                    [False, False, False, False],
                    [True, True, True, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 4),
            pixel_scales=(0.1, 1.0),
            major_axis_radius_arcsec=1.5,
            axis_ratio=0.1,
            phi=180.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, False, False, True],
                    [False, False, False, False],
                    [True, False, False, True],
                ]
            )
        ).all()

    def test__3x3_mask__shifts_dowwards__right__diagonal(self):

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(3.0, 3.0),
            major_axis_radius_arcsec=4.8,
            axis_ratio=0.1,
            phi=45.0,
            centre=(-3.0, 0.0),
        )

        assert (
            mask
            == np.array([[True, True, True], [True, True, False], [True, False, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(3.0, 3.0),
            major_axis_radius_arcsec=4.8,
            axis_ratio=0.1,
            phi=45.0,
            centre=(0.0, 3.0),
        )

        assert (
            mask
            == np.array([[True, True, True], [True, True, False], [True, False, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(3.0, 3.0),
            major_axis_radius_arcsec=4.8,
            axis_ratio=0.1,
            phi=45.0,
            centre=(-3.0, 3.0),
        )

        assert (
            mask
            == np.array([[True, True, True], [True, True, True], [True, True, False]])
        ).all()


class TestMaskEllipticalAnnular(object):
    def test__mask_inner_radius_zero_outer_radius_small_medium_and_large__mask__all_circular_parameters(
        self
    ):

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=0.0,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=0.5,
            outer_axis_ratio=1.0,
            outer_phi=0.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [True, False, True], [True, True, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(4, 4),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=0.81,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=1.0,
            outer_phi=0.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, False, False, True],
                    [False, True, True, False],
                    [False, True, True, False],
                    [True, False, False, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=0.5,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=3.0,
            outer_axis_ratio=1.0,
            outer_phi=0.0,
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

    def test__elliptical_parameters_and_rotations_work_correctly(self):

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=0.0,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=0.1,
            outer_phi=0.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [False, False, False], [True, True, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=0.0,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=0.1,
            outer_phi=90.0,
        )

        assert (
            mask
            == np.array([[True, False, True], [True, False, True], [True, False, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=0.0,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=0.1,
            outer_phi=45.0,
        )

        assert (
            mask
            == np.array([[True, True, False], [True, False, True], [False, True, True]])
        ).all()

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(3, 3),
            pixel_scales=(0.1, 1.0),
            inner_major_axis_radius_arcsec=0.0,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=0.1,
            outer_phi=45.0,
        )

        assert (
            mask
            == np.array([[True, False, True], [True, False, True], [True, False, True]])
        ).all()

    def test__large_mask_array__can_see_elliptical_annuli_form(self):

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=0.1,
            outer_phi=90.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                    [True, True, True, True, True],
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                    [True, True, True, True, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=0.5,
            outer_phi=90.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                    [True, False, True, False, True],
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                    [True, True, True, True, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=2.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=0.5,
            outer_phi=90.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                    [True, True, True, True, True],
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                    [True, True, True, True, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=0.8,
            outer_phi=90.0,
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, True, True],
                    [True, False, False, False, True],
                    [True, False, True, False, True],
                    [True, False, False, False, True],
                    [True, True, False, True, True],
                    [True, True, True, True, True],
                ]
            )
        ).all()

    def test__shifts(self):

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=0.1,
            outer_phi=90.0,
            centre=(-1.0, 0.0),
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                    [True, True, True, True, True],
                    [True, True, False, True, True],
                    [True, True, False, True, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=0.1,
            outer_phi=90.0,
            centre=(0.0, 1.0),
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, True, False, True],
                    [True, True, True, False, True],
                    [True, True, True, True, True],
                    [True, True, True, False, True],
                    [True, True, True, False, True],
                    [True, True, True, True, True],
                ]
            )
        ).all()

        mask = aa.mask_util.mask_elliptical_annular_from_shape_pixel_scales_and_radius(
            shape=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius_arcsec=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius_arcsec=2.0,
            outer_axis_ratio=0.1,
            outer_phi=90.0,
            centre=(-1.0, 1.0),
        )

        assert (
            mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, False, True],
                    [True, True, True, False, True],
                    [True, True, True, True, True],
                    [True, True, True, False, True],
                    [True, True, True, False, True],
                ]
            )
        ).all()


class TestMaskBlurring(object):
    def test__size__3x3_small_mask(self):
        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(3, 3)
        )

        assert (
            blurring_mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

    def test__size__3x3__large_mask(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(3, 3)
        )

        assert (
            blurring_mask
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, True, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__size__5x5__large_mask(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(5, 5)
        )

        assert (
            blurring_mask
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, False, False, False, False, False, True],
                    [True, False, False, False, False, False, True],
                    [True, False, False, True, False, False, True],
                    [True, False, False, False, False, False, True],
                    [True, False, False, False, False, False, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__size__5x3__large_mask(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(5, 3)
        )

        assert (
            blurring_mask
            == np.rot90(
                np.array(
                    [
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, False, False, False, False, False, True],
                        [True, False, False, True, False, False, True],
                        [True, False, False, False, False, False, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                    ]
                )
            )
        ).all()

    def test__size__3x5__large_mask(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(3, 5)
        )

        assert (
            blurring_mask
            == np.rot90(
                np.array(
                    [
                        [True, True, True, True, True, True, True],
                        [True, True, False, False, False, True, True],
                        [True, True, False, False, False, True, True],
                        [True, True, False, True, False, True, True],
                        [True, True, False, False, False, True, True],
                        [True, True, False, False, False, True, True],
                        [True, True, True, True, True, True, True],
                    ]
                )
            )
        ).all()

    def test__size__3x3__multiple_points(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, False, True, True, True, False, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, False, True, True, True, False, True],
                [True, True, True, True, True, True, True],
            ]
        )

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(3, 3)
        )

        assert (
            blurring_mask
            == np.array(
                [
                    [False, False, False, True, False, False, False],
                    [False, True, False, True, False, True, False],
                    [False, False, False, True, False, False, False],
                    [True, True, True, True, True, True, True],
                    [False, False, False, True, False, False, False],
                    [False, True, False, True, False, True, False],
                    [False, False, False, True, False, False, False],
                ]
            )
        ).all()

    def test__size__5x5__multiple_points(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(5, 5)
        )

        assert (
            blurring_mask
            == np.array(
                [
                    [False, False, False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False, False, False],
                    [False, False, True, False, False, False, True, False, False],
                    [False, False, False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False, False, False],
                    [False, False, True, False, False, False, True, False, False],
                    [False, False, False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False, False, False],
                ]
            )
        ).all()

    def test__size__5x3__multiple_points(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(5, 3)
        )

        assert (
            blurring_mask
            == np.rot90(
                np.array(
                    [
                        [True, True, True, True, True, True, True, True, True],
                        [False, False, False, False, False, False, False, False, False],
                        [False, False, True, False, False, False, True, False, False],
                        [False, False, False, False, False, False, False, False, False],
                        [True, True, True, True, True, True, True, True, True],
                        [False, False, False, False, False, False, False, False, False],
                        [False, False, True, False, False, False, True, False, False],
                        [False, False, False, False, False, False, False, False, False],
                        [True, True, True, True, True, True, True, True, True],
                    ]
                )
            )
        ).all()

    def test__size__3x5__multiple_points(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(3, 5)
        )

        assert (
            blurring_mask
            == np.rot90(
                np.array(
                    [
                        [True, False, False, False, True, False, False, False, True],
                        [True, False, False, False, True, False, False, False, True],
                        [True, False, True, False, True, False, True, False, True],
                        [True, False, False, False, True, False, False, False, True],
                        [True, False, False, False, True, False, False, False, True],
                        [True, False, False, False, True, False, False, False, True],
                        [True, False, True, False, True, False, True, False, True],
                        [True, False, False, False, True, False, False, False, True],
                        [True, False, False, False, True, False, False, False, True],
                    ]
                )
            )
        ).all()

    def test__size__3x3__even_sized_image(self):
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
            ]
        )

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(3, 3)
        )

        assert (
            blurring_mask
            == np.array(
                [
                    [False, False, False, True, False, False, False, True],
                    [False, True, False, True, False, True, False, True],
                    [False, False, False, True, False, False, False, True],
                    [True, True, True, True, True, True, True, True],
                    [False, False, False, True, False, False, False, True],
                    [False, True, False, True, False, True, False, True],
                    [False, False, False, True, False, False, False, True],
                    [True, True, True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__size__5x5__even_sized_image(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        )

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(5, 5)
        )

        assert (
            blurring_mask
            == np.array(
                [
                    [True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True],
                    [True, True, True, False, False, False, False, False],
                    [True, True, True, False, False, False, False, False],
                    [True, True, True, False, False, True, False, False],
                    [True, True, True, False, False, False, False, False],
                    [True, True, True, False, False, False, False, False],
                ]
            )
        ).all()

    def test__size__3x3__rectangular_8x9_image(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, False, True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(3, 3)
        )

        assert (
            blurring_mask
            == np.array(
                [
                    [False, False, False, True, False, False, False, True, True],
                    [False, True, False, True, False, True, False, True, True],
                    [False, False, False, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True, True, True],
                    [False, False, False, True, False, False, False, True, True],
                    [False, True, False, True, False, True, False, True, True],
                    [False, False, False, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__size__3x3__rectangular_9x8_image(self):
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

        blurring_mask = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask, kernel_shape=(3, 3)
        )

        assert (
            blurring_mask
            == np.array(
                [
                    [False, False, False, True, False, False, False, True],
                    [False, True, False, True, False, True, False, True],
                    [False, False, False, True, False, False, False, True],
                    [True, True, True, True, True, True, True, True],
                    [False, False, False, True, False, False, False, True],
                    [False, True, False, True, False, True, False, True],
                    [False, False, False, True, False, False, False, True],
                    [True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__size__5x5__multiple_points__mask_extends_beyond_edge_so_raises_mask_exception(
        self
    ):
        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, False, True, True, True, False, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, False, True, True, True, False, True],
                [True, True, True, True, True, True, True],
            ]
        )

        with pytest.raises(exc.MaskException):
            aa.mask_util.blurring_mask_from_mask_and_kernel_shape(mask, kernel_shape=(5, 5))


class TestMaskFromShapeAndMask2dIndexForMask1dIndex(object):
    def test__2d_array_is_2x2__is_not_masked__maps_correctly(self):

        one_to_two = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        shape = (2, 2)

        mask_2d = aa.mask_util.mask_from_shape_and_mask_2d_index_for_mask_1d_index(
            shape=shape, mask_2d_index_for_mask_1d_index=one_to_two
        )

        assert (mask_2d == np.array([[False, False], [False, False]])).all()

    def test__2d_mask_is_2x2__is_masked__maps_correctly(self):

        one_to_two = np.array([[0, 0], [0, 1], [1, 0]])
        shape = (2, 2)

        mask_2d = aa.mask_util.mask_from_shape_and_mask_2d_index_for_mask_1d_index(
            shape=shape, mask_2d_index_for_mask_1d_index=one_to_two
        )

        assert (mask_2d == np.array([[False, False], [False, True]])).all()

    def test__different_shape_and_masks(self):

        one_to_two = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [2, 3]])
        shape = (3, 4)

        mask_2d = aa.mask_util.mask_from_shape_and_mask_2d_index_for_mask_1d_index(
            shape=shape, mask_2d_index_for_mask_1d_index=one_to_two
        )

        assert (
            mask_2d
            == np.array(
                [
                    [False, False, True, True],
                    [False, True, True, True],
                    [False, False, True, False],
                ]
            )
        ).all()


class TestEdgePixels(object):
    def test__7x7_mask_one_central_pixel__is_entire_edge(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        edge_pixels = aa.mask_util.edge_1d_indexes_from_mask(mask=mask)

        assert (edge_pixels == np.array([0])).all()

    def test__7x7_mask_nine_central_pixels__is_edge(self):
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

        edge_pixels = aa.mask_util.edge_1d_indexes_from_mask(mask=mask)

        assert (edge_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

    def test__7x7_mask_rectangle_of_fifteen_central_pixels__is_edge(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        edge_pixels = aa.mask_util.edge_1d_indexes_from_mask(mask=mask)

        assert (edge_pixels == np.array([0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14])).all()

    def test__8x7_mask_add_edge_pixels__also_in_edge(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, False, False, False, False, False, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        edge_pixels = aa.mask_util.edge_1d_indexes_from_mask(mask=mask)

        assert (
            edge_pixels
            == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17])
        ).all()

    def test__8x7_mask_big_square(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, True, True, True, True, True, True],
            ]
        )

        edge_pixels = aa.mask_util.edge_1d_indexes_from_mask(mask=mask)

        assert (
            edge_pixels
            == np.array(
                [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 26, 27, 28, 29]
            )
        ).all()

    def test__7x8_mask_add_edge_pixels__also_in_edge(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True, True],
                [True, True, False, False, False, True, True, True],
                [True, True, False, False, False, True, True, True],
                [True, False, False, False, False, False, True, True],
                [True, True, False, False, False, True, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        )

        edge_pixels = aa.mask_util.edge_1d_indexes_from_mask(mask=mask)

        assert (
            edge_pixels == np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14])
        ).all()

    def test__7x8_mask_big_square(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, True, True],
                [True, False, False, False, False, False, True, True],
                [True, False, False, False, False, False, True, True],
                [True, False, False, False, False, False, True, True],
                [True, False, False, False, False, False, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        )

        edge_pixels = aa.mask_util.edge_1d_indexes_from_mask(mask=mask)

        assert (
            edge_pixels
            == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])
        ).all()


class TestBorderPixels(object):
    def test__7x7_mask_with_small_numbers_of_pixels__border_is_pixel_indexes(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, False, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        border_pixels = aa.mask_util.border_1d_indexes_from_mask(mask=mask)

        assert (border_pixels == np.array([0])).all()

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

        border_pixels = aa.mask_util.border_1d_indexes_from_mask(mask=mask)

        assert (border_pixels == np.array([0, 1, 2])).all()

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

        border_pixels = aa.mask_util.border_1d_indexes_from_mask(mask=mask)

        assert (border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

    def test__9x9_annulus_mask__inner_pixels_excluded(self):

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

        border_pixels = aa.mask_util.border_1d_indexes_from_mask(mask=mask)

        assert (
            border_pixels
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    13,
                    14,
                    17,
                    18,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                ]
            )
        ).all()

    def test__same_as_above_but_10x9_annulus_mask__true_values_at_top_or_bottom(self):

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
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        border_pixels = aa.mask_util.border_1d_indexes_from_mask(mask=mask)

        assert (
            border_pixels
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    13,
                    14,
                    17,
                    18,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                ]
            )
        ).all()

    def test__same_as_above_but_7x8_annulus_mask__true_values_at_right_or_left(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True, True],
                [True, False, True, True, True, True, True, False, True, True],
                [True, False, True, False, False, False, True, False, True, True],
                [True, False, True, False, True, False, True, False, True, True],
                [True, False, True, False, False, False, True, False, True, True],
                [True, False, True, True, True, True, True, False, True, True],
                [True, False, False, False, False, False, False, False, True, True],
                [True, True, True, True, True, True, True, True, True, True],
            ]
        )

        border_pixels = aa.mask_util.border_1d_indexes_from_mask(mask=mask)

        assert (
            border_pixels
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    13,
                    14,
                    17,
                    18,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                ]
            )
        ).all()


class TestSubBorderPixels(object):
    def test__7x7_mask_with_small_numbers_of_pixels__sub_size_1__border_is_pixel_indexes(
        self
    ):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, False, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=1
        )

        assert (sub_border_pixels == np.array([0])).all()

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

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=1
        )

        assert (sub_border_pixels == np.array([0, 1, 2])).all()

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

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=1
        )

        assert (sub_border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

    def test__7x7_mask_with_small_numbers_of_pixels__sub_size_2__border_is_central_sub_pixel_indexes(
        self
    ):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, False, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert (sub_border_pixels == np.array([3])).all()

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

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert (sub_border_pixels == np.array([0, 4, 11])).all()

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

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert (sub_border_pixels == np.array([0, 5, 9, 14, 23, 26, 31, 35])).all()

    def test__7x7_mask_with_small_numbers_of_pixels__sub_size_3__border_is_central_sub_pixel_indexes(
        self
    ):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, False, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=3
        )

        assert (sub_border_pixels == np.array([6])).all()

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

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=3
        )

        assert (sub_border_pixels == np.array([0, 9, 26])).all()

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

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=3
        )

        assert (sub_border_pixels == np.array([0, 11, 20, 33, 53, 60, 71, 80])).all()

    def test__9x9_annulus_mask__inner_pixels_excluded(self):

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

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert (
            sub_border_pixels
            == np.array(
                [
                    0,
                    4,
                    8,
                    13,
                    17,
                    21,
                    25,
                    28,
                    33,
                    36,
                    53,
                    58,
                    71,
                    74,
                    91,
                    94,
                    99,
                    102,
                    106,
                    110,
                    115,
                    119,
                    123,
                    127,
                ]
            )
        ).all()

    def test__same_as_above_but_10x9_annulus_mask__true_values_at_top_or_bottom(self):

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
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert (
            sub_border_pixels
            == np.array(
                [
                    0,
                    4,
                    8,
                    13,
                    17,
                    21,
                    25,
                    28,
                    33,
                    36,
                    53,
                    58,
                    71,
                    74,
                    91,
                    94,
                    99,
                    102,
                    106,
                    110,
                    115,
                    119,
                    123,
                    127,
                ]
            )
        ).all()

    def test__same_as_above_but_7x8_annulus_mask__true_values_at_right_or_left(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True, True],
                [True, False, True, True, True, True, True, False, True, True],
                [True, False, True, False, False, False, True, False, True, True],
                [True, False, True, False, True, False, True, False, True, True],
                [True, False, True, False, False, False, True, False, True, True],
                [True, False, True, True, True, True, True, False, True, True],
                [True, False, False, False, False, False, False, False, True, True],
                [True, True, True, True, True, True, True, True, True, True],
            ]
        )

        sub_border_pixels = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert (
            sub_border_pixels
            == np.array(
                [
                    0,
                    4,
                    8,
                    13,
                    17,
                    21,
                    25,
                    28,
                    33,
                    36,
                    53,
                    58,
                    71,
                    74,
                    91,
                    94,
                    99,
                    102,
                    106,
                    110,
                    115,
                    119,
                    123,
                    127,
                ]
            )
        ).all()
