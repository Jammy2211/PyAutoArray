import os

import numpy as np
import pytest

import autoarray as aa

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)

class TestMapping:

    def test__array_from_array_2d__compare_to_util(self):
        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        mask = np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, True],
                [True, True, True],
            ]
        )

        masked_array_2d = array_2d * np.invert(mask)

        array_1d_util = aa.array_util.sub_array_1d_from_sub_array_2d_mask_and_sub_size(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        mapping = aa.Mapping(mask_2d=mask)

        scaled_array = mapping.array_from_array_2d(array_2d=array_2d)

        assert (scaled_array == array_1d_util).all()
        assert (scaled_array.in_1d == array_1d_util).all()
        assert (scaled_array.in_2d == masked_array_2d).all()

    def test__trimmed_array_2d_from_padded_array_1d_and_image_shape(self):
        mapping = aa.Mapping(
            mask_2d=np.full((4, 4), False),
        )

        array_1d = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
            ]
        )

        array_2d = mapping.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(2, 2)
        )

        assert (array_2d == np.array([[6.0, 7.0], [1.0, 2.0]])).all()

        mapping = aa.Mapping(
            mask_2d=np.full((5, 3), False)
        )

        array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

        array_2d = mapping.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(3, 1)
        )

        assert (array_2d == np.array([[5.0], [8.0], [2.0]])).all()

        mapping = aa.Mapping(
            mask_2d=np.full((3, 5), False)
        )

        array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

        array_2d = mapping.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(1, 3)
        )

        assert (array_2d == np.array([[7.0, 8.0, 9.0]])).all()

    class TestResizedMask:

        def test__resized_mask__pad__compare_to_manual_mask(self):
            mask_2d = np.full(fill_value=False, shape=(5, 5))
            mask_2d[2, 2] = True
    
            mapping = aa.Mapping(mask_2d=mask_2d)
    
            mask_resized = mapping.resized_mask_from_new_shape(
                new_shape=(7, 7),
            )
    
            mask_resized_manual = np.full(fill_value=False, shape=(7, 7))
            mask_resized_manual[3, 3] = True
    
            assert (mask_resized == mask_resized_manual).all()
    
        def test__resized_mask__trim__compare_to_manual_mask(self):
            mask_2d = np.full(fill_value=False, shape=(5, 5))
            mask_2d[2, 2] = True
    
            mask = aa.Mapping(mask_2d=mask_2d)
    
            mask_resized = mask.resized_mask_from_new_shape(
                new_shape=(3, 3),
            )
    
            mask_resized_manual = np.full(fill_value=False, shape=(3, 3))
            mask_resized_manual[1, 1] = True
    
            assert (mask_resized == mask_resized_manual).all()

    class TestBinnedMask:

        def test__compare_to_mask_via_util(self):
            mask = np.full(shape=(14, 19), fill_value=True)
            mask[1, 5] = False
            mask[6, 5] = False
            mask[4, 9] = False
            mask[11, 10] = False

            binned_up_mask_via_util = aa.binning_util.binned_up_mask_from_mask_2d_and_bin_up_factor(
                mask_2d=mask, bin_up_factor=2
            )

            mapping = aa.Mapping(mask_2d=mask)
            mask = mapping.binned_mask_from_bin_up_factor(bin_up_factor=2)
            assert (mask == binned_up_mask_via_util).all()

            binned_up_mask_via_util = aa.binning_util.binned_up_mask_from_mask_2d_and_bin_up_factor(
                mask_2d=mask, bin_up_factor=3
            )

            mapping = aa.Mapping(mask_2d=mask)
            mask = mapping.binned_mask_from_bin_up_factor(bin_up_factor=3)
            assert (mask == binned_up_mask_via_util).all()


class TestScaledMapping:
    
    def test__array_from_array_1d__compare_to_util(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )

        array_1d = np.array([1.0, 6.0, 4.0, 5.0, 2.0])

        array_2d_util = aa.array_util.sub_array_2d_from_sub_array_1d_mask_and_sub_size(
            sub_array_1d=array_1d, mask=mask, sub_size=1
        )

        masked_array_2d = array_2d_util * np.invert(mask)

        mapping = aa.ScaledMapping(mask_2d=mask, pixel_scales=(3.0, 3.0))

        scaled_array = mapping.array_from_array_1d(array_1d=array_1d)

        assert (scaled_array == array_1d).all()
        assert (scaled_array.in_1d == array_1d).all()
        assert (scaled_array.in_2d == masked_array_2d).all()
        assert scaled_array.mapping.pixel_scale == 3.0
        assert scaled_array.mapping.origin == (0.0, 0.0)
        assert (scaled_array.geometry.xticks == np.array([-6.0, -2.0, 2.0, 6.0])).all()
        assert (scaled_array.geometry.yticks == np.array([-4.5, -1.5, 1.5, 4.5])).all()
        assert scaled_array.geometry.shape_arcsec == (9.0, 12.0)

    def test__grid_from_grid_2d__compare_to_util(self):
        grid_2d = np.array(
            [
                [[1, 1], [2, 2], [3, 3], [4, 4]],
                [[5, 5], [6, 6], [7, 7], [8, 8]],
                [[9, 9], [10, 10], [11, 11], [12, 12]],
            ]
        )

        mask = np.array(
            [
                [True, False, True, True],
                [False, False, False, True],
                [True, False, True, False],
            ]
        )

        masked_grid_2d = grid_2d * np.invert(mask[:, :, None])

        grid_1d_util = aa.grid_util.sub_grid_1d_from_sub_grid_2d_mask_and_sub_size(
            sub_grid_2d=masked_grid_2d, mask=mask, sub_size=1
        )

        mapping = aa.ScaledMapping(mask_2d=mask, pixel_scales=(1.0, 1.0))

        grid = mapping.grid_from_grid_2d(grid_2d=masked_grid_2d)

        assert (grid == grid_1d_util).all()
        assert (grid.in_1d == grid).all()
        assert (grid.in_2d == masked_grid_2d).all()

    def test__grid_from_grid_1d__compare_to_util(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )

        grid_1d = np.array([[1.0, 1.0], [6.0, 6.0], [4.0, 4.0], [5.0, 5.0], [2.0, 2.0]])

        grid_2d_util = aa.grid_util.sub_grid_2d_from_sub_grid_1d_mask_and_sub_size(
            sub_grid_1d=grid_1d, mask=mask, sub_size=1
        )

        masked_grid_2d = grid_2d_util * np.invert(mask[:, :, None])

        mapping = aa.ScaledMapping(mask_2d=mask, pixel_scales=(1.0, 1.0))

        grid = mapping.grid_from_grid_1d(grid_1d=grid_1d)

        assert (grid == grid_1d).all()
        assert (grid.in_1d == grid_1d).all()
        assert (grid.in_2d == masked_grid_2d).all()

    class TestBinnedMask:

        def test__compare_to_mask_via_util(self):
            mask = np.full(shape=(14, 19), fill_value=True)
            mask[1, 5] = False
            mask[6, 5] = False
            mask[4, 9] = False
            mask[11, 10] = False

            binned_up_mask_via_util = aa.binning_util.binned_up_mask_from_mask_2d_and_bin_up_factor(
                mask_2d=mask, bin_up_factor=2
            )

            mapping = aa.ScaledMapping(mask_2d=mask, pixel_scales=(1.0, 1.0))
            mask = mapping.binned_mask_from_bin_up_factor(bin_up_factor=2)
            assert (mask == binned_up_mask_via_util).all()
            assert mask.geometry.pixel_scale == 2.0

            binned_up_mask_via_util = aa.binning_util.binned_up_mask_from_mask_2d_and_bin_up_factor(
                mask_2d=mask, bin_up_factor=3
            )

            mapping = aa.ScaledMapping(mask_2d=mask, pixel_scales=(2.0, 2.0))
            mask = mapping.binned_mask_from_bin_up_factor(bin_up_factor=3)
            assert (mask == binned_up_mask_via_util).all()
            assert mask.geometry.pixel_scale == 6.0

class TestSubMapping:
    
    def test__sub_array_2d_from_sub_array_1d__use_2x3_mask(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mapping = aa.ScaledSubMapping(mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)

        sub_array_1d = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                3.0,
                4.0,
                4.0,
                4.0,
                4.0,
            ]
        )

        sub_array_2d = mapping.sub_array_2d_from_sub_array_1d(sub_array_1d=sub_array_1d)

        assert (
                sub_array_2d
                == np.array(
            [
                [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
            ]
        )
        ).all()

    def test__sub_array_2d_binned_from_sub_array_1d(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mapping = aa.ScaledSubMapping(mask_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        sub_array_1d = np.array(
            [
                1.0,
                10.0,
                2.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                3.0,
                4.0,
                0.0,
                0.0,
                4.0,
            ]
        )

        sub_array_2d = mapping.array_2d_binned_from_sub_array_1d(sub_array_1d=sub_array_1d)

        assert (sub_array_2d == np.array([[3.5, 2.0, 0.0], [3.0, 0.0, 2.0]])).all()

    def test__sub_grid_2d_from_sub_grid_1d(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mapping = aa.ScaledSubMapping(mask_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        sub_grid_1d = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [4.0, 4.0],
                [4.0, 4.0],
                [4.0, 4.0],
            ]
        )

        sub_grid_2d = mapping.sub_grid_2d_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

        assert (
                sub_grid_2d
                == np.array(
            [
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [3.0, 3.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [4.0, 4.0],
                    [4.0, 4.0],
                ],
                [
                    [3.0, 3.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [4.0, 4.0],
                    [4.0, 4.0],
                ],
            ]
        )
        ).all()

    def test__sub_grid_2d_binned_from_sub_grid_1d(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mapping = aa.ScaledSubMapping(mask_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        sub_grid_1d = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [4.0, 4.0],
                [4.0, 4.0],
                [4.0, 4.0],
            ]
        )

        sub_grid_2d_binned = mapping.grid_2d_binned_from_sub_grid_1d(
            sub_grid_1d=sub_grid_1d
        )

        assert (
                sub_grid_2d_binned
                == np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                [[3.0, 3.0], [0.0, 0.0], [4.0, 4.0]],
            ]
        )
        ).all()

    def test__array_from_sub_array_1d(self):

        mask = np.array([[False, True], [False, False]])
        mapping = aa.ScaledSubMapping(mask_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        sub_array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        )

        scaled_array = mapping.array_from_sub_array_1d(sub_array_1d=sub_array_1d)

        assert (scaled_array.in_1d == sub_array_1d).all()

        assert (
                scaled_array.in_2d
                == np.array(
            [
                [1.0, 2.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 0.0],
                [9.0, 10.0, 13.0, 14.0],
                [11.0, 12.0, 15.0, 16.0],
            ]
        )
        ).all()

    def test__array_from_sub_array_2d(self):
        sub_array_2d = np.array(
            [
                [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
            ]
        )

        mask = np.array([[False, False, True], [False, True, False]])
        mapping = aa.ScaledSubMapping(mask_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        scaled_array = mapping.array_from_sub_array_2d(sub_array_2d=sub_array_2d)

        assert (
                scaled_array.in_1d
                == np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                3.0,
                4.0,
                4.0,
                4.0,
                4.0,
            ]
        )
        ).all()

        assert (scaled_array.in_2d == sub_array_2d).all()

    def test__array_binned_from_sub_array_1d_by_binning_up(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mapping = aa.ScaledSubMapping(mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)

        sub_array_1d = np.array(
            [
                1.0,
                10.0,
                2.0,
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
                3.0,
                3.0,
                3.0,
                3.0,
                4.0,
                0.0,
                0.0,
                4.0,
            ]
        )

        scaled_array = mapping.array_binned_from_sub_array_1d(
            sub_array_1d=sub_array_1d
        )

        assert (scaled_array.in_1d == np.array([3.5, 2.0, 3.0, 2.0])).all()
        assert (
                scaled_array.in_2d == np.array([[3.5, 2.0, 0.0], [3.0, 0.0, 2.0]])
        ).all()
        assert scaled_array.mapping.pixel_scales == (3.0, 3.0)
        assert scaled_array.mapping.origin == (0.0, 0.0)

    def test__grid_from_sub_grid_1d(self):
        mask = np.array([[False, True], [False, False]])
        mapping = aa.ScaledSubMapping(mask_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        sub_grid_1d = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
                [6.0, 6.0],
                [7.0, 7.0],
                [8.0, 8.0],
                [9.0, 9.0],
                [10.0, 10.0],
                [11.0, 11.0],
                [12.0, 12.0],
            ]
        )

        grid = mapping.grid_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

        assert (grid.in_1d == sub_grid_1d).all()

        assert (
                grid.in_2d
                == np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                [[3.0, 3.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
                [[5.0, 5.0], [6.0, 6.0], [9.0, 9.0], [10.0, 10.0]],
                [[7.0, 7.0], [8.0, 8.0], [11.0, 11.0], [12.0, 12.0]],
            ]
        )
        ).all()

    def test__grid_from_sub_grid_2d(self):
        sub_grid_2d = np.array(
            [
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [3.0, 3.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [4.0, 4.0],
                    [4.0, 4.0],
                ],
                [
                    [3.0, 3.0],
                    [3.0, 3.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [4.0, 4.0],
                    [4.0, 4.0],
                ],
            ]
        )

        mask = np.array([[False, False, True], [False, True, False]])
        mapping = aa.ScaledSubMapping(mask_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        grid = mapping.grid_from_sub_grid_2d(sub_grid_2d=sub_grid_2d)

        assert (
                grid.in_1d
                == np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [4.0, 4.0],
                [4.0, 4.0],
                [4.0, 4.0],
            ]
        )
        ).all()

        assert (grid.in_2d == sub_grid_2d).all()

    def test__grid_binned_from_sub_grid_1d(self):
        mask = np.array([[False, True], [False, False]])
        mapping = aa.ScaledSubMapping(mask_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        grid_1d = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 6.0],
                [9.0, 9.0],
                [10.0, 10.0],
                [11.0, 11.0],
                [12.0, 12.0],
                [13.0, 13.0],
                [14.0, 14.0],
                [15.0, 15.0],
                [16.0, 16.0],
            ]
        )

        grid = mapping.grid_binned_from_sub_grid_1d(sub_grid_1d=grid_1d)

        assert (grid.in_1d == np.array([[2.5, 3.0], [10.5, 10.5], [14.5, 14.5]])).all()

        assert (
                grid.in_2d
                == np.array([[[2.5, 3.0], [0.0, 0.0]], [[10.5, 10.5], [14.5, 14.5]]])
        ).all()
        