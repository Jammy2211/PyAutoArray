from autoarray import exc
from autoarray.mask import mask as msk
from autoarray import util

import numpy as np
import pytest


class TestTotalPixels:
    def test__total_image_pixels_from_mask(self):
        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        assert util.mask.total_pixels_from(mask=mask) == 5

    def test__total_sub_pixels_from_mask(self):
        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        assert util.mask.total_sub_pixels_from(mask, sub_size=2) == 20

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

        assert util.mask.total_edge_pixels_from(mask=mask) == 8

    class TestTotalSparsePixels:
        def test__mask_full_false__full_pixelization_grid_pixels_in_mask(self):

            ma = msk.Mask(
                mask=np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                sub_size=1,
            )

            full_pix_grid_pixel_centres = np.array([[0, 0], [0, 1], [0, 2], [1, 0]])

            total_masked_pixels = util.mask.total_sparse_pixels_from(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 4

            full_pix_grid_pixel_centres = np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 1]]
            )

            total_masked_pixels = util.mask.total_sparse_pixels_from(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 6

        def test__mask_is_cross__only_pixelization_grid_pixels_in_mask_are_counted(
            self
        ):

            ma = msk.Mask(
                mask=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                sub_size=1,
            )

            full_pix_grid_pixel_centres = np.array([[0, 0], [0, 1], [0, 2], [1, 0]])

            total_masked_pixels = util.mask.total_sparse_pixels_from(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 2

            full_pix_grid_pixel_centres = np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 1]]
            )

            total_masked_pixels = util.mask.total_sparse_pixels_from(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 4

        def test__same_as_above_but_3x4_mask(self):

            ma = msk.Mask(
                mask=np.array(
                    [
                        [True, True, False, True],
                        [False, False, False, False],
                        [True, True, False, True],
                    ]
                ),
                sub_size=1,
            )

            full_pix_grid_pixel_centres = np.array([[0, 0], [0, 1], [0, 2], [1, 0]])

            total_masked_pixels = util.mask.total_sparse_pixels_from(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 2

            full_pix_grid_pixel_centres = np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3], [2, 2]]
            )

            total_masked_pixels = util.mask.total_sparse_pixels_from(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 6

        def test__same_as_above_but_4x3_mask(self):

            ma = msk.Mask(
                mask=np.array(
                    [
                        [True, False, True],
                        [True, False, True],
                        [False, False, False],
                        [True, False, True],
                    ]
                ),
                sub_size=1,
            )

            full_pix_grid_pixel_centres = np.array([[0, 0], [0, 1], [0, 2], [1, 1]])

            total_masked_pixels = util.mask.total_sparse_pixels_from(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 2

            full_pix_grid_pixel_centres = np.array(
                [[0, 0], [0, 1], [0, 2], [1, 1], [2, 0], [2, 1], [2, 2], [3, 1]]
            )

            total_masked_pixels = util.mask.total_sparse_pixels_from(
                mask=ma, unmasked_sparse_grid_pixel_centres=full_pix_grid_pixel_centres
            )

            assert total_masked_pixels == 6


class TestMaskCircular:
    def test__3x3_mask_input_radius_small__medium__big__masks(self):
        mask = util.mask.mask_circular_from(
            shape_2d=(3, 3), pixel_scales=(1.0, 1.0), radius=0.5
        )

        assert (
            mask
            == np.array([[True, True, True], [True, False, True], [True, True, True]])
        ).all()

        mask = util.mask.mask_circular_from(
            shape_2d=(3, 3), pixel_scales=(1.0, 1.0), radius=1.3
        )

        assert (
            mask
            == np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            )
        ).all()

        mask = util.mask.mask_circular_from(
            shape_2d=(3, 3), pixel_scales=(1.0, 1.0), radius=3.0
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        mask = util.mask.mask_circular_from(
            shape_2d=(3, 3), pixel_scales=(0.5, 1.0), radius=0.5
        )

        assert (
            mask
            == np.array([[True, False, True], [True, False, True], [True, False, True]])
        ).all()

    def test__4x3_mask_input_radius_small__medium__big__masks(self):

        mask = util.mask.mask_circular_from(
            shape_2d=(4, 3), pixel_scales=(1.0, 1.0), radius=0.5
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

        mask = util.mask.mask_circular_from(
            shape_2d=(4, 3), pixel_scales=(1.0, 1.0), radius=1.5001
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

        mask = util.mask.mask_circular_from(
            shape_2d=(4, 3), pixel_scales=(1.0, 1.0), radius=3.0
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
        mask = util.mask.mask_circular_from(
            shape_2d=(4, 4), pixel_scales=(1.0, 1.0), radius=0.72
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

        mask = util.mask.mask_circular_from(
            shape_2d=(4, 4), pixel_scales=(1.0, 1.0), radius=1.7
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

        mask = util.mask.mask_circular_from(
            shape_2d=(4, 4), pixel_scales=(1.0, 1.0), radius=3.0
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

        mask = util.mask.mask_circular_from(
            shape_2d=(3, 3), pixel_scales=(3.0, 3.0), radius=0.5, centre=(-3, 0)
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array([[True, True, True], [True, True, True], [True, False, True]])
        ).all()

        mask = util.mask.mask_circular_from(
            shape_2d=(3, 3), pixel_scales=(3.0, 3.0), radius=0.5, centre=(0.0, 3.0)
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array([[True, True, True], [True, True, False], [True, True, True]])
        ).all()

        mask = util.mask.mask_circular_from(
            shape_2d=(3, 3), pixel_scales=(3.0, 3.0), radius=0.5, centre=(3, 3)
        )

        assert (
            mask
            == np.array([[True, True, False], [True, True, True], [True, True, True]])
        ).all()


class TestMaskAnnular:
    def test__mask_inner_radius_zero_outer_radius_small_medium_and_large__mask(self):
        mask = util.mask.mask_circular_annular_from(
            shape_2d=(3, 3), pixel_scales=(1.0, 1.0), inner_radius=0.0, outer_radius=0.5
        )

        assert (
            mask
            == np.array([[True, True, True], [True, False, True], [True, True, True]])
        ).all()

        mask = util.mask.mask_circular_annular_from(
            shape_2d=(4, 4),
            pixel_scales=(1.0, 1.0),
            inner_radius=0.81,
            outer_radius=2.0,
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

        mask = util.mask.mask_circular_annular_from(
            shape_2d=(3, 3), pixel_scales=(1.0, 1.0), inner_radius=0.5, outer_radius=3.0
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

        mask = util.mask.mask_circular_annular_from(
            shape_2d=(4, 4), pixel_scales=(0.5, 1.0), inner_radius=1.1, outer_radius=2.0
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
        mask = util.mask.mask_circular_annular_from(
            shape_2d=(4, 3),
            pixel_scales=(1.0, 1.0),
            inner_radius=0.51,
            outer_radius=1.51,
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
        mask = util.mask.mask_circular_annular_from(
            shape_2d=(4, 3),
            pixel_scales=(1.0, 1.0),
            inner_radius=1.51,
            outer_radius=3.0,
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
        mask = util.mask.mask_circular_annular_from(
            shape_2d=(4, 4),
            pixel_scales=(1.0, 1.0),
            inner_radius=1.71,
            outer_radius=3.0,
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

        mask = util.mask.mask_circular_annular_from(
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            inner_radius=0.5,
            outer_radius=9.0,
            centre=(3.0, 0.0),
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array(
                [[False, True, False], [False, False, False], [False, False, False]]
            )
        ).all()

        mask = util.mask.mask_circular_annular_from(
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            inner_radius=0.5,
            outer_radius=9.0,
            centre=(0.0, 3.0),
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array(
                [[False, False, False], [False, False, True], [False, False, False]]
            )
        ).all()

        mask = util.mask.mask_circular_annular_from(
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            inner_radius=0.5,
            outer_radius=9.0,
            centre=(-3.0, 3.0),
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array(
                [[False, False, False], [False, False, False], [False, False, True]]
            )
        ).all()


class TestMaskAntiAnnular:
    def test__5x5_mask_inner_radius_includes_central_pixel__outer_extended_beyond_radius(
        self
    ):

        mask = util.mask.mask_circular_anti_annular_from(
            shape_2d=(5, 5),
            pixel_scales=(1.0, 1.0),
            inner_radius=0.5,
            outer_radius=10.0,
            outer_radius_2_scaled=20.0,
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

        mask = util.mask.mask_circular_anti_annular_from(
            shape_2d=(5, 5),
            pixel_scales=(1.0, 1.0),
            inner_radius=1.5,
            outer_radius=10.0,
            outer_radius_2_scaled=20.0,
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

        mask = util.mask.mask_circular_anti_annular_from(
            shape_2d=(5, 5),
            pixel_scales=(0.1, 1.0),
            inner_radius=1.5,
            outer_radius=10.0,
            outer_radius_2_scaled=20.0,
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

        mask = util.mask.mask_circular_anti_annular_from(
            shape_2d=(5, 5),
            pixel_scales=(1.0, 1.0),
            inner_radius=0.5,
            outer_radius=1.5,
            outer_radius_2_scaled=20.0,
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

        mask = util.mask.mask_circular_anti_annular_from(
            shape_2d=(7, 7),
            pixel_scales=(1.0, 1.0),
            inner_radius=0.5,
            outer_radius=1.5,
            outer_radius_2_scaled=2.9,
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

        mask = util.mask.mask_circular_anti_annular_from(
            shape_2d=(7, 7),
            pixel_scales=(3.0, 3.0),
            inner_radius=1.5,
            outer_radius=4.5,
            outer_radius_2_scaled=8.7,
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


class TestMaskElliptical:
    def test__input_circular_params__small_medium_and_large_masks(self):

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=0.5,
            axis_ratio=1.0,
            phi=0.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [True, False, True], [True, True, True]])
        ).all()

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.3,
            axis_ratio=1.0,
            phi=0.0,
        )

        assert (
            mask
            == np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            )
        ).all()

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=3.0,
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

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.3,
            axis_ratio=0.1,
            phi=0.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [False, False, False], [True, True, True]])
        ).all()

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.3,
            axis_ratio=0.1,
            phi=180.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [False, False, False], [True, True, True]])
        ).all()

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.3,
            axis_ratio=0.1,
            phi=360.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [False, False, False], [True, True, True]])
        ).all()

    def test__same_as_above_but_90_degree_rotations(self):

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.3,
            axis_ratio=0.1,
            phi=90.0,
        )

        assert (
            mask
            == np.array([[True, False, True], [True, False, True], [True, False, True]])
        ).all()

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.3,
            axis_ratio=0.1,
            phi=270.0,
        )

        assert (
            mask
            == np.array([[True, False, True], [True, False, True], [True, False, True]])
        ).all()

    def test__same_as_above_but_diagonal_rotations(self):

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.5,
            axis_ratio=0.1,
            phi=45.0,
        )

        assert (
            mask
            == np.array([[True, True, False], [True, False, True], [False, True, True]])
        ).all()

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.5,
            axis_ratio=0.1,
            phi=135.0,
        )

        assert (
            mask
            == np.array([[False, True, True], [True, False, True], [True, True, False]])
        ).all()

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.5,
            axis_ratio=0.1,
            phi=225.0,
        )

        assert (
            mask
            == np.array([[True, True, False], [True, False, True], [False, True, True]])
        ).all()

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.5,
            axis_ratio=0.1,
            phi=315.0,
        )

        assert (
            mask
            == np.array([[False, True, True], [True, False, True], [True, True, False]])
        ).all()

    def test__4x3__ellipse_is_formed(self):

        mask = util.mask.mask_elliptical_from(
            shape_2d=(4, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.5,
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

        mask = util.mask.mask_elliptical_from(
            shape_2d=(4, 3),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.5,
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

        mask = util.mask.mask_elliptical_from(
            shape_2d=(4, 3),
            pixel_scales=(1.0, 0.1),
            major_axis_radius=1.5,
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

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 4),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.5,
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

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 4),
            pixel_scales=(1.0, 1.0),
            major_axis_radius=1.5,
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

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 4),
            pixel_scales=(0.1, 1.0),
            major_axis_radius=1.5,
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

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            major_axis_radius=4.8,
            axis_ratio=0.1,
            phi=45.0,
            centre=(-3.0, 0.0),
        )

        assert (
            mask
            == np.array([[True, True, True], [True, True, False], [True, False, True]])
        ).all()

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            major_axis_radius=4.8,
            axis_ratio=0.1,
            phi=45.0,
            centre=(0.0, 3.0),
        )

        assert (
            mask
            == np.array([[True, True, True], [True, True, False], [True, False, True]])
        ).all()

        mask = util.mask.mask_elliptical_from(
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            major_axis_radius=4.8,
            axis_ratio=0.1,
            phi=45.0,
            centre=(-3.0, 3.0),
        )

        assert (
            mask
            == np.array([[True, True, True], [True, True, True], [True, True, False]])
        ).all()


class TestMaskEllipticalAnnular:
    def test__mask_inner_radius_zero_outer_radius_small_medium_and_large__mask__all_circular_parameters(
        self
    ):

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=0.0,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius=0.5,
            outer_axis_ratio=1.0,
            outer_phi=0.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [True, False, True], [True, True, True]])
        ).all()

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(4, 4),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=0.81,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
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

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=0.5,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius=3.0,
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

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=0.0,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
            outer_axis_ratio=0.1,
            outer_phi=0.0,
        )

        assert (
            mask
            == np.array([[True, True, True], [False, False, False], [True, True, True]])
        ).all()

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=0.0,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
            outer_axis_ratio=0.1,
            outer_phi=90.0,
        )

        assert (
            mask
            == np.array([[True, False, True], [True, False, True], [True, False, True]])
        ).all()

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(3, 3),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=0.0,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
            outer_axis_ratio=0.1,
            outer_phi=45.0,
        )

        assert (
            mask
            == np.array([[True, True, False], [True, False, True], [False, True, True]])
        ).all()

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(3, 3),
            pixel_scales=(0.1, 1.0),
            inner_major_axis_radius=0.0,
            inner_axis_ratio=1.0,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
            outer_axis_ratio=0.1,
            outer_phi=45.0,
        )

        assert (
            mask
            == np.array([[True, False, True], [True, False, True], [True, False, True]])
        ).all()

    def test__large_mask_array__can_see_elliptical_annuli_form(self):

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
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

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
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

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=2.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
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

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
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

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
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

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
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

        mask = util.mask.mask_elliptical_annular_from(
            shape_2d=(7, 5),
            pixel_scales=(1.0, 1.0),
            inner_major_axis_radius=1.0,
            inner_axis_ratio=0.1,
            inner_phi=0.0,
            outer_major_axis_radius=2.0,
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


class TestMaskFromPixelCoordinates:
    def test__mask_without_buffer__false_at_coordinates(self):

        mask = util.mask.mask_via_pixel_coordinates_from(
            shape_2d=(3, 3), pixel_coordinates=[[0, 0]]
        )

        assert (
            mask
            == np.array([[False, True, True], [True, True, True], [True, True, True]])
        ).all()

        mask = util.mask.mask_via_pixel_coordinates_from(
            shape_2d=(2, 3), pixel_coordinates=[[0, 1], [1, 1], [1, 2]]
        )

        assert (mask == np.array([[True, False, True], [True, False, False]])).all()

    def test__mask_with_buffer__false_at_buffed_coordinates(self):

        mask = util.mask.mask_via_pixel_coordinates_from(
            shape_2d=(5, 5), pixel_coordinates=[[2, 2]], buffer=1
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

        mask = util.mask.mask_via_pixel_coordinates_from(
            shape_2d=(7, 7), pixel_coordinates=[[2, 2], [5, 5]], buffer=1
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


class TestMaskBlurring:
    def test__size__3x3_small_mask(self):
        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(3, 3))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(3, 3))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(5, 5))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(5, 3))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(3, 5))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(3, 3))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(5, 5))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(5, 3))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(3, 5))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(3, 3))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(5, 5))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(3, 3))

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

        blurring_mask = util.mask.blurring_mask_from(mask, kernel_shape_2d=(3, 3))

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
            util.mask.blurring_mask_from(mask, kernel_shape_2d=(5, 5))


class TestMaskFromShapeAndMask2dIndexForMask1dIndex:
    def test__2d_array_is_2x2__is_not_masked__maps_correctly(self):

        one_to_two = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        shape = (2, 2)

        mask = util.mask.mask_via_shape_2d_and_mask_index_for_mask_1d_index_from(
            shape_2d=shape, mask_index_for_mask_1d_index=one_to_two
        )

        assert (mask == np.array([[False, False], [False, False]])).all()

    def test__2d_mask_is_2x2__is_masked__maps_correctly(self):

        one_to_two = np.array([[0, 0], [0, 1], [1, 0]])
        shape = (2, 2)

        mask = util.mask.mask_via_shape_2d_and_mask_index_for_mask_1d_index_from(
            shape_2d=shape, mask_index_for_mask_1d_index=one_to_two
        )

        assert (mask == np.array([[False, False], [False, True]])).all()

    def test__different_shape_and_masks(self):

        one_to_two = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [2, 3]])
        shape = (3, 4)

        mask = util.mask.mask_via_shape_2d_and_mask_index_for_mask_1d_index_from(
            shape_2d=shape, mask_index_for_mask_1d_index=one_to_two
        )

        assert (
            mask
            == np.array(
                [
                    [False, False, True, True],
                    [False, True, True, True],
                    [False, False, True, False],
                ]
            )
        ).all()


class TestEdgePixels:
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

        edge_pixels = util.mask.edge_1d_indexes_from(mask=mask)

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

        edge_pixels = util.mask.edge_1d_indexes_from(mask=mask)

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

        edge_pixels = util.mask.edge_1d_indexes_from(mask=mask)

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

        edge_pixels = util.mask.edge_1d_indexes_from(mask=mask)

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

        edge_pixels = util.mask.edge_1d_indexes_from(mask=mask)

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

        edge_pixels = util.mask.edge_1d_indexes_from(mask=mask)

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

        edge_pixels = util.mask.edge_1d_indexes_from(mask=mask)

        assert (
            edge_pixels
            == np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])
        ).all()


class TestBorderPixels:
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

        border_pixels = util.mask.border_1d_indexes_from(mask=mask)

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

        border_pixels = util.mask.border_1d_indexes_from(mask=mask)

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

        border_pixels = util.mask.border_1d_indexes_from(mask=mask)

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

        border_pixels = util.mask.border_1d_indexes_from(mask=mask)

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

        border_pixels = util.mask.border_1d_indexes_from(mask=mask)

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

        border_pixels = util.mask.border_1d_indexes_from(mask=mask)

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


class TestSubBorderPixels:
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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

        sub_border_pixels = util.mask.sub_border_pixel_1d_indexes_from(
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


class TestMask2DIndexFromSubMask1DIndex:
    def test__3x3_mask_with_1_pixel__2x2_sub_grid__correct_mask_1d_index_for_sub_mask_1d_index(
        self
    ):
        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        mask_1d_index_for_sub_mask_1d_index = util.mask.mask_1d_index_for_sub_mask_1d_index_via_mask_from(
            mask, sub_size=2
        )

        assert (mask_1d_index_for_sub_mask_1d_index == np.array([0, 0, 0, 0])).all()

    def test__3x3_mask_with_row_of_pixels_pixel__2x2_sub_grid__correct_mask_1d_index_for_sub_mask_1d_index(
        self
    ):
        mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

        mask_1d_index_for_sub_mask_1d_index = util.mask.mask_1d_index_for_sub_mask_1d_index_via_mask_from(
            mask, sub_size=2
        )

        assert (
            mask_1d_index_for_sub_mask_1d_index
            == np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        ).all()

    def test__3x3_mask_with_row_of_pixels_pixel__3x3_sub_grid__correct_mask_1d_index_for_sub_mask_1d_index(
        self
    ):
        mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

        mask_1d_index_for_sub_mask_1d_index = util.mask.mask_1d_index_for_sub_mask_1d_index_via_mask_from(
            mask, sub_size=3
        )

        assert (
            mask_1d_index_for_sub_mask_1d_index
            == np.array(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                ]
            )
        ).all()


class TestSubMask1DIndexFromMask1DIndexes:
    def test__3x3_mask_with_1_pixel__2x2_sub_grid__correct_sub_mask_1d_indexes_for_mask_1d_index(
        self
    ):

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        sub_mask_1d_indexes_for_mask_1d_index = util.mask.sub_mask_1d_indexes_for_mask_1d_index_via_mask_from(
            mask, sub_size=2
        )

        assert sub_mask_1d_indexes_for_mask_1d_index == [[0, 1, 2, 3]]

    def test__3x3_mask_with_row_of_pixels_pixel__2x2_sub_grid__correct_sub_mask_1d_indexes_for_mask_1d_index(
        self
    ):
        mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

        sub_mask_1d_indexes_for_mask_1d_index = util.mask.sub_mask_1d_indexes_for_mask_1d_index_via_mask_from(
            mask, sub_size=2
        )

        assert sub_mask_1d_indexes_for_mask_1d_index == [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ]

    def test__3x3_mask_with_row_of_pixels_pixel__3x3_sub_grid__correct_sub_mask_1d_indexes_for_mask_1d_index(
        self
    ):
        mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

        sub_mask_1d_indexes_for_mask_1d_index = util.mask.sub_mask_1d_indexes_for_mask_1d_index_via_mask_from(
            mask, sub_size=3
        )

        assert sub_mask_1d_indexes_for_mask_1d_index == [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23, 24, 25, 26],
        ]


class TestSubMask2dToSubMask1d:
    def test__mask_if_full_of_false__indexes_are_ascending_order(self):

        mask = np.full(fill_value=False, shape=(3, 3))

        sub_mask_1d_index_for_sub_mask_index = util.mask.sub_mask_1d_index_for_sub_mask_index_from_sub_mask_from(
            sub_mask=mask
        )

        assert (
            sub_mask_1d_index_for_sub_mask_index
            == np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        ).all()

        mask = np.full(fill_value=False, shape=(2, 3))

        sub_mask_1d_index_for_sub_mask_index = util.mask.sub_mask_1d_index_for_sub_mask_index_from_sub_mask_from(
            sub_mask=mask
        )

        assert (
            sub_mask_1d_index_for_sub_mask_index == np.array([[0, 1, 2], [3, 4, 5]])
        ).all()

        mask = np.full(fill_value=False, shape=(3, 2))

        sub_mask_1d_index_for_sub_mask_index = util.mask.sub_mask_1d_index_for_sub_mask_index_from_sub_mask_from(
            sub_mask=mask
        )

        assert (
            sub_mask_1d_index_for_sub_mask_index == np.array([[0, 1], [2, 3], [4, 5]])
        ).all()

    def test__mask_has_true_and_falses__minus_ones_in_place_of_trues_and_falses_count_correctly(
        self
    ):

        mask = np.array(
            [[False, True, False], [True, True, False], [False, False, True]]
        )

        sub_mask_1d_index_for_sub_mask_index = util.mask.sub_mask_1d_index_for_sub_mask_index_from_sub_mask_from(
            sub_mask=mask
        )

        assert (
            sub_mask_1d_index_for_sub_mask_index
            == np.array([[0, -1, 1], [-1, -1, 2], [3, 4, -1]])
        ).all()

        mask = np.array(
            [
                [False, True, True, False],
                [True, True, False, False],
                [False, False, True, False],
            ]
        )

        sub_mask_1d_index_for_sub_mask_index = util.mask.sub_mask_1d_index_for_sub_mask_index_from_sub_mask_from(
            sub_mask=mask
        )

        assert (
            sub_mask_1d_index_for_sub_mask_index
            == np.array([[0, -1, -1, 1], [-1, -1, 2, 3], [4, 5, -1, 6]])
        ).all()

        mask = np.array(
            [
                [False, True, False],
                [True, True, False],
                [False, False, True],
                [False, False, True],
            ]
        )

        sub_mask_1d_index_for_sub_mask_index = util.mask.sub_mask_1d_index_for_sub_mask_index_from_sub_mask_from(
            sub_mask=mask
        )

        assert (
            sub_mask_1d_index_for_sub_mask_index
            == np.array([[0, -1, 1], [-1, -1, 2], [3, 4, -1], [5, 6, -1]])
        ).all()


class TestSubMask2DForSubMask1D:
    def test__simple_mappings__sub_size_is_1(self):

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        sub_mask_index_for_sub_mask_1d_index = util.mask.sub_mask_index_for_sub_mask_1d_index_via_mask_from(
            mask=mask, sub_size=1
        )

        assert (sub_mask_index_for_sub_mask_1d_index == np.array([[1, 1]])).all()

        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        sub_mask_index_for_sub_mask_1d_index = util.mask.sub_mask_index_for_sub_mask_1d_index_via_mask_from(
            mask=mask, sub_size=1
        )

        assert (
            sub_mask_index_for_sub_mask_1d_index
            == np.array([[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]])
        ).all()

        mask = np.array(
            [
                [True, False, True, True],
                [False, False, False, True],
                [True, False, True, False],
            ]
        )

        sub_mask_index_for_sub_mask_1d_index = util.mask.sub_mask_index_for_sub_mask_1d_index_via_mask_from(
            mask=mask, sub_size=1
        )

        assert (
            sub_mask_index_for_sub_mask_1d_index
            == np.array([[0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 3]])
        ).all()

        mask = np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, True],
                [True, True, False],
            ]
        )

        sub_mask_index_for_sub_mask_1d_index = util.mask.sub_mask_index_for_sub_mask_1d_index_via_mask_from(
            mask=mask, sub_size=1
        )

        assert (
            sub_mask_index_for_sub_mask_1d_index
            == np.array([[0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [3, 2]])
        ).all()

    def test__simple_grid_mappings__sub_size_2(self):

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        sub_mask_index_for_sub_mask_1d_index = util.mask.sub_mask_index_for_sub_mask_1d_index_via_mask_from(
            mask=mask, sub_size=2
        )

        assert (
            sub_mask_index_for_sub_mask_1d_index
            == np.array([[2, 2], [2, 3], [3, 2], [3, 3]])
        ).all()

        sub_mask_index_for_sub_mask_1d_index = util.mask.sub_mask_index_for_sub_mask_1d_index_via_mask_from(
            mask=mask, sub_size=3
        )

        assert (
            sub_mask_index_for_sub_mask_1d_index
            == np.array(
                [[3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5]]
            )
        ).all()

        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        sub_mask_index_for_sub_mask_1d_index = util.mask.sub_mask_index_for_sub_mask_1d_index_via_mask_from(
            mask=mask, sub_size=2
        )

        assert (
            sub_mask_index_for_sub_mask_1d_index
            == np.array(
                [
                    [0, 2],
                    [0, 3],
                    [1, 2],
                    [1, 3],
                    [2, 0],
                    [2, 1],
                    [3, 0],
                    [3, 1],
                    [2, 2],
                    [2, 3],
                    [3, 2],
                    [3, 3],
                    [2, 4],
                    [2, 5],
                    [3, 4],
                    [3, 5],
                    [4, 2],
                    [4, 3],
                    [5, 2],
                    [5, 3],
                ]
            )
        ).all()

        mask = np.array(
            [
                [True, True, True, True],
                [False, True, True, True],
                [True, False, True, True],
            ]
        )

        sub_mask_index_for_sub_mask_1d_index = util.mask.sub_mask_index_for_sub_mask_1d_index_via_mask_from(
            mask=mask, sub_size=2
        )

        assert (
            sub_mask_index_for_sub_mask_1d_index
            == np.array(
                [[2, 0], [2, 1], [3, 0], [3, 1], [4, 2], [4, 3], [5, 2], [5, 3]]
            )
        ).all()

        mask = np.array(
            [
                [True, True, True],
                [True, False, True],
                [True, True, True],
                [True, True, False],
            ]
        )

        sub_mask_index_for_sub_mask_1d_index = util.mask.sub_mask_index_for_sub_mask_1d_index_via_mask_from(
            mask=mask, sub_size=2
        )

        assert (
            sub_mask_index_for_sub_mask_1d_index
            == np.array(
                [[2, 2], [2, 3], [3, 2], [3, 3], [6, 4], [6, 5], [7, 4], [7, 5]]
            )
        ).all()


class TestRescaledMaskFromMask:
    def test__mask_7x7_central_pixel__rescale_factor_is_1__returns_same_mask(self):
        mask = np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        rescaled_mask = util.mask.rescaled_mask_from(mask=mask, rescale_factor=1.0)

        assert (
            rescaled_mask
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

    def test__mask_7x7_central_pixel__rescale_factor_is_2__returns_10x10_mask_4_central_values(
        self
    ):
        mask = np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        rescaled_mask = util.mask.rescaled_mask_from(mask=mask, rescale_factor=2.0)

        assert (
            rescaled_mask
            == np.array(
                [
                    [True, True, True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True, True, True],
                    [True, True, True, False, False, False, False, True, True, True],
                    [True, True, True, False, False, False, False, True, True, True],
                    [True, True, True, False, False, False, False, True, True, True],
                    [True, True, True, False, False, False, False, True, True, True],
                    [True, True, True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__same_as_above__off_centre_pixels(self):
        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
            ]
        )

        rescaled_mask = util.mask.rescaled_mask_from(mask=mask, rescale_factor=2.0)

        assert (
            rescaled_mask
            == np.array(
                [
                    [True, True, True, True, True, True, True, True, True, True],
                    [True, False, False, False, False, True, True, True, True, True],
                    [True, False, False, False, False, True, True, True, True, True],
                    [True, False, False, False, False, True, True, True, True, True],
                    [True, False, False, False, False, True, True, True, True, True],
                    [True, True, True, True, True, False, False, False, False, True],
                    [True, True, True, True, True, False, False, False, False, True],
                    [True, True, True, True, True, False, False, False, False, True],
                    [True, True, True, True, True, False, False, False, False, True],
                    [True, True, True, True, True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__mask_4x3_two_central_pixels__rescale_near_1__returns_slightly_different_masks(
        self
    ):
        mask = np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        rescaled_mask = util.mask.rescaled_mask_from(mask=mask, rescale_factor=1.2)

        assert (
            rescaled_mask
            == np.array(
                [
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, False, False, True, True],
                    [True, True, False, False, True, True],
                    [True, True, False, False, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                ]
            )
        ).all()

        rescaled_mask = util.mask.rescaled_mask_from(mask=mask, rescale_factor=0.8)

        assert (
            rescaled_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

    def test__mask_3x4_two_central_pixels__rescale_near_1__returns_slightly_different_masks(
        self
    ):
        mask = np.array(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, True, True],
                [True, True, False, False, True, True],
                [True, True, True, True, True, True],
                [True, True, True, True, True, True],
            ]
        )

        rescaled_mask = util.mask.rescaled_mask_from(mask=mask, rescale_factor=1.2)

        assert (
            rescaled_mask
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, False, False, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

        rescaled_mask = util.mask.rescaled_mask_from(mask=mask, rescale_factor=0.8)

        assert (
            rescaled_mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, True, True, True, True],
                ]
            )
        ).all()


class TestBuffedMaskFromMask:
    def test__5x5_mask_false_centre_pixel__3x3_falses_in_centre_of_buffed_mask(self):
        mask = np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        buffed_mask = util.mask.buffed_mask_from(mask=mask, buffer=1)

        assert (
            buffed_mask
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

    def test__5x5_mask_false_offset_pixel__3x3_falses_in_centre_of_buffed_mask(self):
        mask = np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
            ]
        )

        buffed_mask = util.mask.buffed_mask_from(mask=mask, buffer=1)

        assert (
            buffed_mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                ]
            )
        ).all()

    def test__mask_4x3__buffed_mask_same_shape(self):
        mask = np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        buffed_mask = util.mask.buffed_mask_from(mask=mask, buffer=1)

        assert (
            buffed_mask
            == np.array(
                [
                    [True, True, True, True, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, True, True, True, True],
                ]
            )
        ).all()

    def test__mask_3x4_two_central_pixels__rescale_near_1__returns_slightly_different_masks(
        self
    ):
        mask = np.array(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, True, True],
                [True, True, False, False, True, True],
                [True, True, True, True, True, True],
                [True, True, True, True, True, True],
            ]
        )

        buffed_mask = util.mask.buffed_mask_from(mask=mask, buffer=1)

        assert (
            buffed_mask
            == np.array(
                [
                    [True, True, True, True, True, True],
                    [True, False, False, False, False, True],
                    [True, False, False, False, False, True],
                    [True, False, False, False, False, True],
                    [True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__buffer_is_above_2__mask_includes_buffing(self):

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

        buffed_mask = util.mask.buffed_mask_from(mask=mask, buffer=2)

        assert (
            buffed_mask
            == np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, False, False, False, False, False, True],
                    [True, False, False, False, False, False, True],
                    [True, False, False, False, False, False, True],
                    [True, False, False, False, False, False, True],
                    [True, False, False, False, False, False, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, False, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        buffed_mask = util.mask.buffed_mask_from(mask=mask, buffer=2)

        assert (
            buffed_mask
            == np.array(
                [
                    [True, True, True, False, False, False, False],
                    [True, True, True, False, False, False, False],
                    [True, True, True, False, False, False, False],
                    [True, True, True, False, False, False, False],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            )
        ).all()


class TestMaskNeighbors:
    def test__gives_right_neighbor_then_down_if_not_available(self):

        mask = np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )

        mask_neighbors = util.mask.mask_neighbors_from(mask=mask)

        assert (mask_neighbors == np.array([1, 3, 3, 2])).all()

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask_neighbors = util.mask.mask_neighbors_from(mask=mask)

        assert (mask_neighbors == np.array([1, 2, 5, 4, 5, 4])).all()

        mask = np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )

        mask_neighbors = util.mask.mask_neighbors_from(mask=mask)

        assert (mask_neighbors == np.array([1, 3, 3, 5, 5, 4])).all()

    def test__mask_has_false_entries_on_edge__does_not_raise_error(self):

        mask = np.array([[False, False], [False, False]])

        mask_neighbors = util.mask.mask_neighbors_from(mask=mask)

        assert (mask_neighbors == np.array([1, 3, 3, 2])).all()

        mask = np.array([[False, False, False], [False, False, False]])

        mask_neighbors = util.mask.mask_neighbors_from(mask=mask)

        assert (mask_neighbors == np.array([1, 2, 5, 4, 5, 4])).all()

        mask = np.array([[False, False], [False, False], [False, False]])

        mask_neighbors = util.mask.mask_neighbors_from(mask=mask)

        assert (mask_neighbors == np.array([1, 3, 3, 5, 5, 4])).all()

    def test__pixel_with_no_adjacent_neighbor__gives_minus_1(self):

        mask = np.array(
            [
                [True, True, True, True],
                [True, False, True, False],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )

        mask_neighbors = util.mask.mask_neighbors_from(mask=mask)

        assert (mask_neighbors == np.array([2, -1, 3, 2])).all()
