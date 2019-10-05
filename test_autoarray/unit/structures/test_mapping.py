import os

import numpy as np
import pytest

import autoarray as aa

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)

class TestMapping:
    
    def test__mask_2d_index_for_mask_1d_index__compare_to_array_util(self):
        mask = np.array([[True, True, True], [True, False, False], [True, True, False]])

        mask = aa.PixelMask(array_2d=mask, sub_size=1)

        mask_2d_index_for_mask_1d_index = aa.mask_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=1
        )

        assert mask._mask_2d_index_for_mask_1d_index == pytest.approx(
            mask_2d_index_for_mask_1d_index, 1e-4
        )

    def test__mask_1d_index_for_sub_mask_1d_index__compare_to_util(self):
        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, False]]
        )

        mask_1d_index_for_sub_mask_1d_index_util = aa.mask_util.mask_1d_index_for_sub_mask_1d_index_from_mask(
            mask=mask, sub_size=2
        )
        mask = aa.PixelMask(array_2d=mask, sub_size=2)

        assert (
                mask._mask_1d_index_for_sub_mask_1d_index
                == mask_1d_index_for_sub_mask_1d_index_util
        ).all()

    def test__sub_mask_2d_index_for_sub_mask_1d_index__compare_to_array_util(self):
        mask = np.array([[True, True, True], [True, False, False], [True, True, False]])

        mask = aa.PixelMask(array_2d=mask, sub_size=2)

        sub_mask_2d_index_for_sub_mask_1d_index = aa.mask_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert mask._sub_mask_2d_index_for_sub_mask_1d_index == pytest.approx(
            sub_mask_2d_index_for_sub_mask_1d_index, 1e-4
        )

    def test__sub_array_2d_from_sub_array_1d__use_2x3_mask(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mask = aa.PixelMask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)

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

        sub_array_2d = mask.mapping.sub_array_2d_from_sub_array_1d(sub_array_1d=sub_array_1d)

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
        mask = aa.PixelMask(array_2d=mask, sub_size=2)

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

        sub_array_2d = mask.mapping.array_2d_binned_from_sub_array_1d(sub_array_1d=sub_array_1d)

        assert (sub_array_2d == np.array([[3.5, 2.0, 0.0], [3.0, 0.0, 2.0]])).all()

    def test__sub_grid_2d_from_sub_grid_1d(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mask = aa.PixelMask(array_2d=mask, sub_size=2)

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

        sub_grid_2d = mask.mapping.sub_grid_2d_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

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
        mask = aa.PixelMask(array_2d=mask, sub_size=2)

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

        sub_grid_2d_binned = mask.mapping.grid_2d_binned_from_sub_grid_1d(
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

    def test__trimmed_array_2d_from_padded_array_1d_and_image_shape(self):
        mask = aa.PixelMask(
            array_2d=np.full((4, 4), False), sub_size=1
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

        array_2d = mask.mapping.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(2, 2)
        )

        assert (array_2d == np.array([[6.0, 7.0], [1.0, 2.0]])).all()

        mask = aa.PixelMask(
            array_2d=np.full((5, 3), False), sub_size=1
        )

        array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

        array_2d = mask.mapping.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(3, 1)
        )

        assert (array_2d == np.array([[5.0], [8.0], [2.0]])).all()

        mask = aa.PixelMask(
            array_2d=np.full((3, 5), False), sub_size=1
        )

        array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )

        array_2d = mask.mapping.trimmed_array_2d_from_padded_array_1d_and_image_shape(
            padded_array_1d=array_1d, image_shape=(1, 3)
        )

        assert (array_2d == np.array([[7.0, 8.0, 9.0]])).all()

class TestScaledMappingGrids:
    
    def test__scaled_array_from_array_1d__compare_to_util(self):
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

        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=1)

        scaled_array = mask.mapping.scaled_array_from_array_1d(array_1d=array_1d)

        assert (scaled_array == array_1d).all()
        assert (scaled_array.in_1d == array_1d).all()
        assert (scaled_array.in_2d == masked_array_2d).all()
        assert (scaled_array.mask.xticks == np.array([-6.0, -2.0, 2.0, 6.0])).all()
        assert (scaled_array.mask.yticks == np.array([-4.5, -1.5, 1.5, 4.5])).all()
        assert scaled_array.mask.shape_arcsec == (9.0, 12.0)
        assert scaled_array.mask.pixel_scale == 3.0
        assert scaled_array.mask.origin == (0.0, 0.0)

    def test__scaled_array_from_array_2d__compare_to_util(self):
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

        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=1)

        scaled_array = mask.mapping.scaled_array_from_array_2d(array_2d=array_2d)

        assert (scaled_array == array_1d_util).all()
        assert (scaled_array.in_1d == array_1d_util).all()
        assert (scaled_array.in_2d == masked_array_2d).all()

    def test__scaled_array_from_sub_array_1d(self):
        mask = np.array([[False, True], [False, False]])
        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        sub_array_1d = np.array(
            [1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        )

        scaled_array = mask.mapping.scaled_array_from_sub_array_1d(sub_array_1d=sub_array_1d)

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

    def test__scaled_array_from_sub_array_2d(self):
        sub_array_2d = np.array(
            [
                [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, 2.0, 0.0, 0.0],
                [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
                [3.0, 3.0, 0.0, 0.0, 4.0, 4.0],
            ]
        )

        mask = np.array([[False, False, True], [False, True, False]])
        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        scaled_array = mask.mapping.scaled_array_from_sub_array_2d(sub_array_2d=sub_array_2d)

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

    def test__scaled_array_binned_from_sub_array_1d_by_binning_up(self):
        mask = np.array([[False, False, True], [False, True, False]])
        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2)

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

        scaled_array = mask.mapping.scaled_array_binned_from_sub_array_1d(
            sub_array_1d=sub_array_1d
        )

        assert (scaled_array.in_1d == np.array([3.5, 2.0, 3.0, 2.0])).all()
        assert (
                scaled_array.in_2d == np.array([[3.5, 2.0, 0.0], [3.0, 0.0, 2.0]])
        ).all()
        assert scaled_array.mask.pixel_scales == (3.0, 3.0)
        assert scaled_array.mask.origin == (0.0, 0.0)
    
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

        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = mask.mapping.grid_from_grid_2d(grid_2d=masked_grid_2d)

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

        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = mask.mapping.grid_from_grid_1d(grid_1d=grid_1d)

        assert (grid == grid_1d).all()
        assert (grid.in_1d == grid_1d).all()
        assert (grid.in_2d == masked_grid_2d).all()

    def test__grid_from_sub_grid_1d(self):
        mask = np.array([[False, True], [False, False]])
        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

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

        grid = mask.mapping.grid_from_sub_grid_1d(sub_grid_1d=sub_grid_1d)

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
        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        grid = mask.mapping.grid_from_sub_grid_2d(sub_grid_2d=sub_grid_2d)

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
        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(1.0, 1.0), sub_size=2)

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

        grid = mask.mapping.grid_binned_from_sub_grid_1d(sub_grid_1d=grid_1d)

        assert (grid.in_1d == np.array([[2.5, 3.0], [10.5, 10.5], [14.5, 14.5]])).all()

        assert (
                grid.in_2d
                == np.array([[[2.5, 3.0], [0.0, 0.0]], [[10.5, 10.5], [14.5, 14.5]]])
        ).all()
        

class TestGridConversions:
    
    def test__pixel_coordinates_from_arcsec_coordinates__arcsec_are_pixel_centres(self):
        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1
        )

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(1.0, -1.0)
        ) == (0, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(1.0, 1.0)
        ) == (0, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-1.0, -1.0)
        ) == (1, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-1.0, 1.0)
        ) == (1, 1)

        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(3, 3), pixel_scales=(3.0, 3.0), sub_size=1
        )

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(3.0, -3.0)
        ) == (0, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(3.0, 0.0)
        ) == (0, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(3.0, 3.0)
        ) == (0, 2)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.0, -3.0)
        ) == (1, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.0, 0.0)
        ) == (1, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.0, 3.0)
        ) == (1, 2)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-3.0, -3.0)
        ) == (2, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-3.0, 0.0)
        ) == (2, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-3.0, 3.0)
        ) == (2, 2)

    def test__pixel_coordinates_from_arcsec_coordinates__arcsec_are_pixel_corners(self):
        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1
        )

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(1.99, -1.99)
        ) == (0, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(1.99, -0.01)
        ) == (0, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.01, -1.99)
        ) == (0, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.01, -0.01)
        ) == (0, 0)

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(2.01, 0.01)
        ) == (0, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(2.01, 1.99)
        ) == (0, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.01, 0.01)
        ) == (0, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.01, 1.99)
        ) == (0, 1)

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-0.01, -1.99)
        ) == (1, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-0.01, -0.01)
        ) == (1, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-1.99, -1.99)
        ) == (1, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-1.99, -0.01)
        ) == (1, 0)

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-0.01, 0.01)
        ) == (1, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-0.01, 1.99)
        ) == (1, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-1.99, 0.01)
        ) == (1, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-1.99, 1.99)
        ) == (1, 1)

    def test__pixel_coordinates_from_arcsec_coordinates___arcsec_are_pixel_centres__nonzero_centre(
        self
    ):
        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 2.0), origin=(1.0, 1.0), sub_size=1
        )

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(2.0, 0.0)
        ) == (0, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(2.0, 2.0)
        ) == (0, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.0, 0.0)
        ) == (1, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.0, 2.0)
        ) == (1, 1)

        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(3, 3), pixel_scales=(3.0, 3.0), sub_size=1, origin=(3.0, 3.0)
        )

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(6.0, 0.0)
        ) == (0, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(6.0, 3.0)
        ) == (0, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(6.0, 6.0)
        ) == (0, 2)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(3.0, 0.0)
        ) == (1, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(3.0, 3.0)
        ) == (1, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(3.0, 6.0)
        ) == (1, 2)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.0, 0.0)
        ) == (2, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.0, 3.0)
        ) == (2, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.0, 6.0)
        ) == (2, 2)

    def test__pixel_coordinates_from_arcsec_coordinates__arcsec_are_pixel_corners__nonzero_centre(
        self
    ):
        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1, origin=(1.0, 1.0)
        )

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(2.99, -0.99)
        ) == (0, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(2.99, 0.99)
        ) == (0, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(1.01, -0.99)
        ) == (0, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(1.01, 0.99)
        ) == (0, 0)

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(3.01, 1.01)
        ) == (0, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(3.01, 2.99)
        ) == (0, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(1.01, 1.01)
        ) == (0, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(1.01, 2.99)
        ) == (0, 1)

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.99, -0.99)
        ) == (1, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.99, 0.99)
        ) == (1, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-0.99, -0.99)
        ) == (1, 0)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-0.99, 0.99)
        ) == (1, 0)

        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.99, 1.01)
        ) == (1, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(0.99, 2.99)
        ) == (1, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-0.99, 1.01)
        ) == (1, 1)
        assert mask.pixel_coordinates_from_arcsec_coordinates(
            arcsec_coordinates=(-0.99, 2.99)
        ) == (1, 1)

    def test__grid_pixels_from_grid_arcsec(self):

        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 4.0), sub_size=1
        )

        grid_arcsec_1d = aa.Grid(
            sub_grid_1d=np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]]),
            mask=mask,
        )

        grid_pixels_util = aa.grid_util.grid_pixels_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec_1d, shape=(2, 2), pixel_scales=(2.0, 4.0)
        )
        grid_pixels = mask.mapping.grid_pixels_from_grid_arcsec(grid_arcsec_1d=grid_arcsec_1d)

        assert (grid_pixels == grid_pixels_util).all()
        assert (grid_pixels.in_1d == grid_pixels_util).all()

    def test__grid_pixel_centres_1d_from_grid_arcsec_1d__same_as_grid_util(self):

        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid_arcsec_1d = aa.Grid(
            sub_grid_1d=np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5]]),
            mask=mask,
        )

        grid_pixels_util = aa.grid_util.grid_pixel_centres_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec_1d, shape=(2, 2), pixel_scales=(2.0, 2.0)
        )

        grid_pixels = mask.mapping.grid_pixel_centres_from_grid_arcsec_1d(
            grid_arcsec_1d=grid_arcsec_1d
        )

        assert (grid_pixels == grid_pixels_util).all()

        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(7.0, 2.0), sub_size=1
        )

        grid_arcsec_1d = aa.Grid(
            sub_grid_1d=np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]]),
            mask=mask,
        )

        grid_pixels_util = aa.grid_util.grid_pixel_centres_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec_1d, shape=(2, 2), pixel_scales=(7.0, 2.0)
        )

        grid_pixels = mask.mapping.grid_pixel_centres_from_grid_arcsec_1d(
            grid_arcsec_1d=grid_arcsec_1d
        )

        assert (grid_pixels == grid_pixels_util).all()

    def test__grid_pixel_indexes_1d_from_grid_arcsec_1d__same_as_grid_util(self):
        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid_arcsec = aa.Grid(
            sub_grid_1d=np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]]),
            mask=mask,
        )

        grid_pixel_indexes_util = aa.grid_util.grid_pixel_indexes_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 2.0)
        )

        grid_pixel_indexes = mask.mapping.grid_pixel_indexes_from_grid_arcsec_1d(
            grid_arcsec_1d=grid_arcsec
        )

        assert (grid_pixel_indexes == grid_pixel_indexes_util).all()

        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 4.0), sub_size=1
        )

        grid_arcsec = aa.Grid(
            sub_grid_1d=np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]]),
            mask=mask,
        )

        grid_pixels_util = aa.grid_util.grid_pixel_indexes_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec, shape=(2, 2), pixel_scales=(2.0, 4.0)
        )

        grid_pixels = mask.mapping.grid_pixel_indexes_from_grid_arcsec_1d(
            grid_arcsec_1d=grid_arcsec
        )

        assert (grid_pixels == grid_pixels_util).all()

    def test__grid_arcsec_1d_from_grid_pixels_1d__same_as_grid_util(self):

        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid_pixels = aa.Grid(
            sub_grid_1d=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), mask=mask
        )

        grid_pixels_util = aa.grid_util.grid_arcsec_1d_from_grid_pixels_1d_shape_and_pixel_scales(
            grid_pixels_1d=grid_pixels, shape=(2, 2), pixel_scales=(2.0, 2.0)
        )

        grid_pixels = mask.mapping.grid_arcsec_from_grid_pixels_1d(grid_pixels_1d=grid_pixels)

        assert (grid_pixels == grid_pixels_util).all()

        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid_pixels = aa.Grid(
            sub_grid_1d=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), mask=mask
        )

        grid_pixels_util = aa.grid_util.grid_arcsec_1d_from_grid_pixels_1d_shape_and_pixel_scales(
            grid_pixels_1d=grid_pixels, shape=(2, 2), pixel_scales=(2.0, 2.0)
        )
        grid_pixels = mask.mapping.grid_arcsec_from_grid_pixels_1d(grid_pixels_1d=grid_pixels)

        assert (grid_pixels == grid_pixels_util).all()

    def test__pixel_grid__grids_with_nonzero_centres__same_as_grid_util(self):

        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 2.0), sub_size=1, origin=(1.0, 2.0)
        )

        grid_arcsec = aa.Grid(
            sub_grid_1d=np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]]),
            mask=mask,
        )

        grid_pixels_util = aa.grid_util.grid_pixels_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.mapping.grid_pixels_from_grid_arcsec(grid_arcsec_1d=grid_arcsec)
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = aa.grid_util.grid_pixel_indexes_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.mapping.grid_pixel_indexes_from_grid_arcsec_1d(
            grid_arcsec_1d=grid_arcsec
        )
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels_util = aa.grid_util.grid_pixel_centres_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.mapping.grid_pixel_centres_from_grid_arcsec_1d(
            grid_arcsec_1d=grid_arcsec
        )
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels = aa.Grid(
            sub_grid_1d=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), mask=mask
        )

        grid_arcsec_util = aa.grid_util.grid_arcsec_1d_from_grid_pixels_1d_shape_and_pixel_scales(
            grid_pixels_1d=grid_pixels,
            shape=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )

        grid_arcsec = mask.mapping.grid_arcsec_from_grid_pixels_1d(grid_pixels_1d=grid_pixels)

        assert (grid_arcsec == grid_arcsec_util).all()

        grid_arcsec = aa.Grid(
            sub_grid_1d=np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]]),
            mask=mask,
        )

        mask = aa.ScaledMask.unmasked_from_shape_pixel_scales_and_sub_size(
            shape=(2, 2), pixel_scales=(2.0, 1.0), sub_size=1, origin=(1.0, 2.0)
        )

        grid_pixels_util = aa.grid_util.grid_pixels_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.mapping.grid_pixels_from_grid_arcsec(grid_arcsec_1d=grid_arcsec)
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = aa.grid_util.grid_pixel_indexes_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.mapping.grid_pixel_indexes_from_grid_arcsec_1d(
            grid_arcsec_1d=grid_arcsec
        )
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = aa.grid_util.grid_pixel_centres_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
            grid_arcsec_1d=grid_arcsec,
            shape=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.mapping.grid_pixel_centres_from_grid_arcsec_1d(
            grid_arcsec_1d=grid_arcsec
        )
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels = aa.Grid(
            sub_grid_1d=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), mask=mask
        )

        grid_arcsec_util = aa.grid_util.grid_arcsec_1d_from_grid_pixels_1d_shape_and_pixel_scales(
            grid_pixels_1d=grid_pixels,
            shape=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )

        grid_arcsec = mask.mapping.grid_arcsec_from_grid_pixels_1d(grid_pixels_1d=grid_pixels)

        assert (grid_arcsec == grid_arcsec_util).all()