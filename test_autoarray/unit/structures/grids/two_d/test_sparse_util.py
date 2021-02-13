import autoarray as aa
import os

import numpy as np


class TestUnmaskedSparseForSparse:
    def test__mask_full_false__image_mask_and_pixel_centres_fully_overlap__each_sparse_maps_to_unmaked_sparse(
        self,
    ):

        ma = aa.Mask2D.manual(
            mask=np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        total_masked_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
            mask_2d=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        unmasked_sparse_for_sparse = aa.util.sparse.unmasked_sparse_for_sparse_from(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (
            unmasked_sparse_for_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ).all()

    def test__same_as_above__but_remove_some_centre_pixels_and_change_order__order_does_not_change_mapping(
        self,
    ):

        ma = aa.Mask2D.manual(
            mask=np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [2, 2], [1, 1], [0, 2], [2, 0], [0, 2]]
        )

        total_masked_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
            mask_2d=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        unmasked_sparse_for_sparse = aa.util.sparse.unmasked_sparse_for_sparse_from(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (unmasked_sparse_for_sparse == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    def test__mask_is_cross__some_pix_pixels_are_masked__omitted_from_mapping(self):

        ma = aa.Mask2D.manual(
            mask=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        total_masked_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
            mask_2d=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        unmasked_sparse_for_sparse = aa.util.sparse.unmasked_sparse_for_sparse_from(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (unmasked_sparse_for_sparse == np.array([1, 3, 4, 5, 7])).all()

    def test__same_as_above__different_mask_and_centres(self):

        ma = aa.Mask2D.manual(
            mask=np.array(
                [[False, False, True], [False, False, False], [True, False, False]]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1]]
        )

        total_masked_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
            mask_2d=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        unmasked_sparse_for_sparse = aa.util.sparse.unmasked_sparse_for_sparse_from(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (unmasked_sparse_for_sparse == np.array([0, 1, 5])).all()

    def test__same_as_above__but_3x4_mask(self):

        ma = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, False, True],
                    [False, False, False, False],
                    [True, True, False, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 3], [2, 2]]
        )

        total_masked_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
            mask_2d=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        unmasked_sparse_for_sparse = aa.util.sparse.unmasked_sparse_for_sparse_from(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (unmasked_sparse_for_sparse == np.array([2, 3, 4, 5, 7])).all()

    def test__same_as_above__but_4x3_mask(self):

        ma = aa.Mask2D.manual(
            mask=np.array(
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

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 2], [3, 1]]
        )

        total_masked_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
            mask_2d=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        unmasked_sparse_for_sparse = aa.util.sparse.unmasked_sparse_for_sparse_from(
            total_sparse_pixels=total_masked_pixels,
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        )

        assert (unmasked_sparse_for_sparse == np.array([1, 5, 6, 7])).all()


class TestSparseForUnmaskedSparse:
    def test__mask_full_false__image_mask_and_pixel_centres_fully_overlap__each_pix_maps_to_unmaked_pix(
        self,
    ):

        ma = aa.Mask2D.manual(
            mask=np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        sparse_for_unmasked_sparse = aa.util.sparse.sparse_for_unmasked_sparse_from(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=9,
        )

        assert (
            sparse_for_unmasked_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ).all()

    def test__same_as_above__but_remove_some_centre_pixels_and_change_order__order_does_not_change_mapping(
        self,
    ):

        ma = aa.Mask2D.manual(
            mask=np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [2, 2], [1, 1], [0, 2], [2, 0], [0, 2]]
        )

        sparse_for_unmasked_sparse = aa.util.sparse.sparse_for_unmasked_sparse_from(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=9,
        )

        assert (sparse_for_unmasked_sparse == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    def test__mask_is_cross__some_pix_pixels_are_masked__omitted_from_mapping(self):

        ma = aa.Mask2D.manual(
            mask=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        sparse_for_unmasked_sparse = aa.util.sparse.sparse_for_unmasked_sparse_from(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=5,
        )

        assert (
            sparse_for_unmasked_sparse == np.array([0, 0, 1, 1, 2, 3, 4, 4, 4])
        ).all()

    def test__same_as_above__different_mask_and_centres(self):

        ma = aa.Mask2D.manual(
            mask=np.array(
                [[False, False, True], [False, False, False], [True, False, False]]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1]]
        )

        sparse_for_unmasked_sparse = aa.util.sparse.sparse_for_unmasked_sparse_from(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=4,
        )

        assert (sparse_for_unmasked_sparse == np.array([0, 1, 2, 2, 2, 2])).all()

    def test__same_as_above__but_3x4_mask(self):

        ma = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, False, True],
                    [False, False, False, False],
                    [True, True, False, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 3], [0, 2]]
        )

        sparse_for_unmasked_sparse = aa.util.sparse.sparse_for_unmasked_sparse_from(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=5,
        )

        assert (sparse_for_unmasked_sparse == np.array([0, 0, 0, 1, 2, 3, 4, 4])).all()

    def test__same_as_above__but_4x3_mask(self):

        ma = aa.Mask2D.manual(
            mask=np.array(
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

        unmasked_sparse_grid_pixel_centres = np.array(
            [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 2], [3, 1]]
        )

        sparse_for_unmasked_sparse = aa.util.sparse.sparse_for_unmasked_sparse_from(
            mask=ma,
            unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            total_sparse_pixels=6,
        )

        assert (sparse_for_unmasked_sparse == np.array([0, 0, 1, 1, 1, 1, 2, 3])).all()


class TestRegularToSparse:
    def test__simple_cases_for_regular_to_unmasked_sparse_and__sparse_for_unmasked_sparse(
        self,
    ):

        regular_to_unmasked_sparse = np.array([0, 1, 2, 3, 4])
        sparse_for_unmasked_sparse = np.array([0, 1, 2, 3, 4])
        sparse_index_for_slim_index = aa.util.sparse.sparse_slim_index_for_mask_slim_index_from(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            sparse_for_unmasked_sparse=sparse_for_unmasked_sparse,
        )

        assert (sparse_index_for_slim_index == np.array([0, 1, 2, 3, 4])).all()

        regular_to_unmasked_sparse = np.array([0, 1, 2, 3, 4])
        sparse_for_unmasked_sparse = np.array([0, 1, 5, 7, 18])
        sparse_index_for_slim_index = aa.util.sparse.sparse_slim_index_for_mask_slim_index_from(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            sparse_for_unmasked_sparse=sparse_for_unmasked_sparse,
        )

        assert (sparse_index_for_slim_index == np.array([0, 1, 5, 7, 18])).all()

        regular_to_unmasked_sparse = np.array([1, 1, 1, 1, 2])
        sparse_for_unmasked_sparse = np.array([0, 10, 15, 3, 4])
        sparse_index_for_slim_index = aa.util.sparse.sparse_slim_index_for_mask_slim_index_from(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            sparse_for_unmasked_sparse=sparse_for_unmasked_sparse,
        )

        assert (sparse_index_for_slim_index == np.array([10, 10, 10, 10, 15])).all()

        regular_to_unmasked_sparse = np.array([5, 6, 7, 8, 9])
        sparse_for_unmasked_sparse = np.array([0, 1, 2, 3, 4, 19, 18, 17, 16, 15])
        sparse_index_for_slim_index = aa.util.sparse.sparse_slim_index_for_mask_slim_index_from(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            sparse_for_unmasked_sparse=sparse_for_unmasked_sparse,
        )

        assert (sparse_index_for_slim_index == np.array([19, 18, 17, 16, 15])).all()


class TestSparseGridFromUnmaskedSparseGrid:
    def test__simple_unmasked_sparse_grid__full_grid_pix_grid_same_size__straightforward_mappings(
        self,
    ):

        unmasked_sparse_grid = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        )
        unmasked_sparse_for_sparse = np.array([0, 1, 2, 3])
        pix_grid = aa.util.sparse.sparse_grid_via_unmasked_from(
            unmasked_sparse_grid=unmasked_sparse_grid,
            unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
        )

        assert (
            pix_grid == np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        ).all()

        unmasked_sparse_grid = np.array(
            [[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]]
        )
        unmasked_sparse_for_sparse = np.array([0, 1, 2, 3])
        pix_grid = aa.util.sparse.sparse_grid_via_unmasked_from(
            unmasked_sparse_grid=unmasked_sparse_grid,
            unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
        )

        assert (
            pix_grid == np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
        ).all()

        unmasked_sparse_grid = np.array(
            [[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]]
        )
        unmasked_sparse_for_sparse = np.array([1, 0, 3, 2])
        pix_grid = aa.util.sparse.sparse_grid_via_unmasked_from(
            unmasked_sparse_grid=unmasked_sparse_grid,
            unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
        )

        assert (
            pix_grid == np.array([[4.0, 5.0], [0.0, 0.0], [8.0, 7.0], [2.0, 2.0]])
        ).all()

    def test__simple_unmasked_sparse_grid__full_grid_pix_bigger_than_pix__straightforward_mappings(
        self,
    ):

        unmasked_sparse_grid = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        )
        unmasked_sparse_for_sparse = np.array([1, 2])
        pix_grid = aa.util.sparse.sparse_grid_via_unmasked_from(
            unmasked_sparse_grid=unmasked_sparse_grid,
            unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
        )

        assert (pix_grid == np.array([[1.0, 1.0], [2.0, 2.0]])).all()

        unmasked_sparse_grid = np.array(
            [[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]]
        )
        unmasked_sparse_for_sparse = np.array([2, 2, 3])
        pix_grid = aa.util.sparse.sparse_grid_via_unmasked_from(
            unmasked_sparse_grid=unmasked_sparse_grid,
            unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
        )

        assert (pix_grid == np.array([[2.0, 2.0], [2.0, 2.0], [8.0, 7.0]])).all()

        unmasked_sparse_grid = np.array(
            [
                [0.0, 0.0],
                [4.0, 5.0],
                [2.0, 2.0],
                [8.0, 7.0],
                [11.0, 11.0],
                [-20.0, -15.0],
            ]
        )
        unmasked_sparse_for_sparse = np.array([1, 0, 5, 2])
        pix_grid = aa.util.sparse.sparse_grid_via_unmasked_from(
            unmasked_sparse_grid=unmasked_sparse_grid,
            unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
        )

        assert (
            pix_grid == np.array([[4.0, 5.0], [0.0, 0.0], [-20.0, -15.0], [2.0, 2.0]])
        ).all()
