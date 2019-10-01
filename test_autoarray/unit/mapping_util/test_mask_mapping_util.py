import autoarray as aa
import os

import numpy as np


test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestMask2DIndexFromSubMask1DIndex(object):
    def test__3x3_mask_with_1_pixel__2x2_sub_grid__correct_mask_1d_index_for_sub_mask_1d_index(
        self
    ):
        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        mask_1d_index_for_sub_mask_1d_index = aa.mask_mapping_util.mask_1d_index_for_sub_mask_1d_index_from_mask(
            mask, sub_size=2
        )

        assert (mask_1d_index_for_sub_mask_1d_index == np.array([0, 0, 0, 0])).all()

    def test__3x3_mask_with_row_of_pixels_pixel__2x2_sub_grid__correct_mask_1d_index_for_sub_mask_1d_index(
        self
    ):
        mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

        mask_1d_index_for_sub_mask_1d_index = aa.mask_mapping_util.mask_1d_index_for_sub_mask_1d_index_from_mask(
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

        mask_1d_index_for_sub_mask_1d_index = aa.mask_mapping_util.mask_1d_index_for_sub_mask_1d_index_from_mask(
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


class TestSubMask1DIndexFromMask1DIndexes(object):
    def test__3x3_mask_with_1_pixel__2x2_sub_grid__correct_sub_mask_1d_indexes_for_mask_1d_index(
        self
    ):

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        sub_mask_1d_indexes_for_mask_1d_index = aa.mask_mapping_util.sub_mask_1d_indexes_for_mask_1d_index_from_mask(
            mask, sub_size=2
        )

        assert sub_mask_1d_indexes_for_mask_1d_index == [[0, 1, 2, 3]]

    def test__3x3_mask_with_row_of_pixels_pixel__2x2_sub_grid__correct_sub_mask_1d_indexes_for_mask_1d_index(
        self
    ):
        mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

        sub_mask_1d_indexes_for_mask_1d_index = aa.mask_mapping_util.sub_mask_1d_indexes_for_mask_1d_index_from_mask(
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

        sub_mask_1d_indexes_for_mask_1d_index = aa.mask_mapping_util.sub_mask_1d_indexes_for_mask_1d_index_from_mask(
            mask, sub_size=3
        )

        assert sub_mask_1d_indexes_for_mask_1d_index == [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23, 24, 25, 26],
        ]


class TestSubMask2dToSubMask1d(object):
    def test__mask_if_full_of_false__indexes_are_ascending_order(self):

        mask_2d = np.full(fill_value=False, shape=(3, 3))

        sub_mask_1d_index_for_sub_mask_2d_index = aa.mask_mapping_util.sub_mask_1d_index_for_sub_mask_2d_index_from_sub_mask(
            sub_mask=mask_2d
        )

        assert (sub_mask_1d_index_for_sub_mask_2d_index == np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])).all()

        mask_2d = np.full(fill_value=False, shape=(2, 3))

        sub_mask_1d_index_for_sub_mask_2d_index = aa.mask_mapping_util.sub_mask_1d_index_for_sub_mask_2d_index_from_sub_mask(
            sub_mask=mask_2d
        )

        assert (sub_mask_1d_index_for_sub_mask_2d_index == np.array([[0, 1, 2], [3, 4, 5]])).all()

        mask_2d = np.full(fill_value=False, shape=(3, 2))

        sub_mask_1d_index_for_sub_mask_2d_index = aa.mask_mapping_util.sub_mask_1d_index_for_sub_mask_2d_index_from_sub_mask(
            sub_mask=mask_2d
        )

        assert (sub_mask_1d_index_for_sub_mask_2d_index == np.array([[0, 1], [2, 3], [4, 5]])).all()

    def test__mask_has_true_and_falses__minus_ones_in_place_of_trues_and_falses_count_correctly(
        self
    ):

        mask_2d = np.array(
            [[False, True, False], [True, True, False], [False, False, True]]
        )

        sub_mask_1d_index_for_sub_mask_2d_index = aa.mask_mapping_util.sub_mask_1d_index_for_sub_mask_2d_index_from_sub_mask(
            sub_mask=mask_2d
        )

        assert (sub_mask_1d_index_for_sub_mask_2d_index == np.array([[0, -1, 1], [-1, -1, 2], [3, 4, -1]])).all()

        mask_2d = np.array(
            [
                [False, True, True, False],
                [True, True, False, False],
                [False, False, True, False],
            ]
        )

        sub_mask_1d_index_for_sub_mask_2d_index = aa.mask_mapping_util.sub_mask_1d_index_for_sub_mask_2d_index_from_sub_mask(
            sub_mask=mask_2d
        )

        assert (
            sub_mask_1d_index_for_sub_mask_2d_index == np.array([[0, -1, -1, 1], [-1, -1, 2, 3], [4, 5, -1, 6]])
        ).all()

        mask_2d = np.array(
            [
                [False, True, False],
                [True, True, False],
                [False, False, True],
                [False, False, True],
            ]
        )

        sub_mask_1d_index_for_sub_mask_2d_index = aa.mask_mapping_util.sub_mask_1d_index_for_sub_mask_2d_index_from_sub_mask(
            sub_mask=mask_2d
        )

        assert (
            sub_mask_1d_index_for_sub_mask_2d_index
            == np.array([[0, -1, 1], [-1, -1, 2], [3, 4, -1], [5, 6, -1]])
        ).all()


class TestSubOneToTwo(object):
    def test__simple_mappings__sub_size_is_1(self):

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        sub_mask_2d_index_for_sub_mask_1d_index = aa.mask_mapping_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=1
        )

        assert (sub_mask_2d_index_for_sub_mask_1d_index == np.array([[1, 1]])).all()

        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        sub_mask_2d_index_for_sub_mask_1d_index = aa.mask_mapping_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=1
        )

        assert (
            sub_mask_2d_index_for_sub_mask_1d_index == np.array([[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]])
        ).all()

        mask = np.array(
            [
                [True, False, True, True],
                [False, False, False, True],
                [True, False, True, False],
            ]
        )

        sub_mask_2d_index_for_sub_mask_1d_index = aa.mask_mapping_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=1
        )

        assert (
            sub_mask_2d_index_for_sub_mask_1d_index == np.array([[0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 3]])
        ).all()

        mask = np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, True],
                [True, True, False],
            ]
        )

        sub_mask_2d_index_for_sub_mask_1d_index = aa.mask_mapping_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=1
        )

        assert (
            sub_mask_2d_index_for_sub_mask_1d_index == np.array([[0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [3, 2]])
        ).all()

    def test__simple_grid_mappings__sub_size_2(self):

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        sub_mask_2d_index_for_sub_mask_1d_index = aa.mask_mapping_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert (sub_mask_2d_index_for_sub_mask_1d_index == np.array([[2, 2], [2, 3], [3, 2], [3, 3]])).all()

        sub_mask_2d_index_for_sub_mask_1d_index = aa.mask_mapping_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=3
        )

        assert (
            sub_mask_2d_index_for_sub_mask_1d_index
            == np.array(
                [[3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5]]
            )
        ).all()

        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        sub_mask_2d_index_for_sub_mask_1d_index = aa.mask_mapping_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert (
            sub_mask_2d_index_for_sub_mask_1d_index
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

        sub_mask_2d_index_for_sub_mask_1d_index = aa.mask_mapping_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert (
            sub_mask_2d_index_for_sub_mask_1d_index
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

        sub_mask_2d_index_for_sub_mask_1d_index = aa.mask_mapping_util.sub_mask_2d_index_for_sub_mask_1d_index_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        assert (
            sub_mask_2d_index_for_sub_mask_1d_index
            == np.array(
                [[2, 2], [2, 3], [3, 2], [3, 3], [6, 4], [6, 5], [7, 4], [7, 5]]
            )
        ).all()
