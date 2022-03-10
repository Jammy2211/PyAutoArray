from os import path
import numpy as np
import pytest

from autoarray.structures.two_d.grids.mock.mock_grid_decorators import (
    ndarray_1d_from,
    ndarray_2d_from,
)

import autoarray as aa

test_coordinates_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files"
)


class TestObj:
    def test__grid_via_deflection_grid_from(self):

        grid = aa.Grid2DIterate.uniform(
            shape_native=(2, 2),
            pixel_scales=2.0,
            fractional_accuracy=0.1,
            sub_steps=[2, 3],
        )

        grid_deflected = grid.grid_via_deflection_grid_from(deflection_grid=grid)

        assert type(grid_deflected) == aa.Grid2DIterate
        assert (
            grid_deflected == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        ).all()
        assert (
            grid_deflected.native
            == np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        ).all()
        assert (
            grid_deflected.slim
            == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        ).all()
        assert (grid_deflected.mask == grid.mask).all()
        assert grid_deflected.pixel_scales == (2.0, 2.0)
        assert grid_deflected.origin == (0.0, 0.0)
        assert grid_deflected.fractional_accuracy == 0.1
        assert grid_deflected.sub_steps == [2, 3]

    def test__blurring_grid_from_mask__compare_to_array_util(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, False, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_mask_util = aa.util.mask_2d.blurring_mask_2d_from(
            mask_2d=mask, kernel_shape_native=(3, 5)
        )

        blurring_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid = aa.Grid2DIterate.from_mask(mask=mask)

        blurring_grid = grid.blurring_grid_via_kernel_shape_from(
            kernel_shape_native=(3, 5)
        )

        assert isinstance(blurring_grid, aa.Grid2DIterate)
        assert len(blurring_grid.shape) == 2
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)

    def test__blurring_grid_via_kernel_shape_from__compare_to_array_util(self):
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

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0))

        blurring_mask_util = aa.util.mask_2d.blurring_mask_2d_from(
            mask_2d=mask, kernel_shape_native=(3, 5)
        )

        blurring_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0))

        blurring_grid = aa.Grid2DIterate.blurring_grid_from(
            mask=mask, kernel_shape_native=(3, 5)
        )

        assert isinstance(blurring_grid, aa.Grid2DIterate)
        assert len(blurring_grid.shape) == 2
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)

    def test__padded_grid_from__matches_grid_2d_after_padding(self):

        grid = aa.Grid2DIterate.uniform(
            shape_native=(4, 4),
            pixel_scales=3.0,
            fractional_accuracy=0.1,
            sub_steps=[2, 3],
        )

        padded_grid = grid.padded_grid_from(kernel_shape_native=(3, 3))

        padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=np.full((6, 6), False), pixel_scales=(3.0, 3.0), sub_size=1
        )

        assert isinstance(padded_grid, aa.Grid2DIterate)
        assert padded_grid.shape == (36, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 6))).all()
        assert padded_grid.fractional_accuracy == 0.1
        assert padded_grid.sub_steps == [2, 3]
        assert (padded_grid == padded_grid_util).all()

        mask = aa.Mask2D.manual(
            mask=np.full((2, 5), False), pixel_scales=(8.0, 8.0), sub_size=1
        )

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.1, sub_steps=[2, 3]
        )

        padded_grid = grid.padded_grid_from(kernel_shape_native=(5, 5))

        padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=1
        )

        assert padded_grid.shape == (54, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 9))).all()
        assert padded_grid.fractional_accuracy == 0.1
        assert padded_grid.sub_steps == [2, 3]
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)


class TestIteratedArray:
    def test__threshold_mask_from(self):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = aa.Grid2DIterate.from_mask(mask=mask, fractional_accuracy=0.9999)

        arr = aa.Array2D.manual_mask(
            array=[
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask,
        )

        threshold_mask = grid.threshold_mask_via_arrays_from(
            array_lower_sub_2d=arr.binned.native, array_higher_sub_2d=arr.binned.native
        )

        assert (
            threshold_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            )
        ).all()

        result_array_lower_sub = aa.Array2D.manual_mask(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask,
        )

        result_array_higher_sub = aa.Array2D.manual_mask(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask,
        )

        threshold_mask = grid.threshold_mask_via_arrays_from(
            array_lower_sub_2d=result_array_lower_sub.binned.native,
            array_higher_sub_2d=result_array_higher_sub.binned.native,
        )

        assert (
            threshold_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

        grid = aa.Grid2DIterate.from_mask(mask=mask, fractional_accuracy=0.5)

        result_array_lower_sub = aa.Array2D.manual_mask(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.9, 0.001, 0.0],
                [0.0, 0.999, 1.9, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask,
        )

        result_array_higher_sub = aa.Array2D.manual_mask(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask,
        )

        threshold_mask = grid.threshold_mask_via_arrays_from(
            array_lower_sub_2d=result_array_lower_sub.binned.native,
            array_higher_sub_2d=result_array_higher_sub.binned.native,
        )

        assert (
            threshold_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, True, False, True],
                    [True, False, True, True],
                    [True, True, True, True],
                ]
            )
        ).all()

    def test__threshold_mask_from__uses_higher_sub_grids_mask(self):

        mask_lower_sub = aa.Mask2D.manual(
            mask=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        mask_higher_sub = aa.Mask2D.manual(
            mask=[
                [True, True, True, True],
                [True, False, True, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = aa.Grid2DIterate.from_mask(mask=mask_lower_sub, fractional_accuracy=0.5)

        array_lower_sub = aa.Array2D.manual_mask(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask_lower_sub,
        )

        array_higher_sub = aa.Array2D.manual_mask(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 5.0, 0.0],
                [0.0, 5.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask_higher_sub,
        )

        threshold_mask = grid.threshold_mask_via_arrays_from(
            array_lower_sub_2d=array_lower_sub.binned.native,
            array_higher_sub_2d=array_higher_sub.binned.native,
        )

        assert (
            threshold_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, True, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

    def test__iterated_array_from__extreme_fractional_accuracies_uses_last_or_first_sub(
        self,
    ):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
        )

        mask_sub_1 = mask.mask_new_sub_size_from(mask=mask, sub_size=1)
        grid_sub_1 = aa.Grid2D.from_mask(mask=mask_sub_1)
        values_sub_1 = ndarray_1d_from(grid=grid_sub_1, profile=None)
        values_sub_1 = grid_sub_1.structure_2d_from(result=values_sub_1)

        values = grid.iterated_array_from(
            func=ndarray_1d_from,
            cls=None,
            array_lower_sub_2d=values_sub_1.binned.native,
        )

        mask_sub_3 = mask.mask_new_sub_size_from(mask=mask, sub_size=3)
        grid_sub_3 = aa.Grid2D.from_mask(mask=mask_sub_3)
        values_sub_3 = ndarray_1d_from(grid=grid_sub_3, profile=None)
        values_sub_3 = grid_sub_3.structure_2d_from(result=values_sub_3)

        assert (values == values_sub_3.binned).all()

        # This test ensures that if the fractional accuracy is met on the last sub_size jump (e.g. 2 doesnt meet it,
        # but 3 does) that the sub_size of 3 is used. There was a bug where the mask was not updated correctly and the
        # iterated array double counted the values.

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.9, sub_steps=[2, 3]
        )

        values = grid.iterated_array_from(
            func=ndarray_1d_from,
            cls=None,
            array_lower_sub_2d=values_sub_1.binned.native,
        )

        assert (values == values_sub_3.binned).all()

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
        )

        values = grid.iterated_array_from(
            func=ndarray_1d_from,
            cls=None,
            array_lower_sub_2d=values_sub_1.binned.native,
        )

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid2D.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_1d_from(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_2d_from(result=values_sub_2)

        assert (values == values_sub_2.binned).all()

    def test__iterated_array_from__check_values_computed_to_fractional_accuracy(self,):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.5, sub_steps=[2, 4]
        )

        mask_sub_1 = mask.mask_new_sub_size_from(mask=mask, sub_size=1)
        grid_sub_1 = aa.Grid2D.from_mask(mask=mask_sub_1)
        values_sub_1 = ndarray_1d_from(grid=grid_sub_1, profile=None)
        values_sub_1 = grid_sub_1.structure_2d_from(result=values_sub_1)

        values = grid.iterated_array_from(
            func=ndarray_1d_from,
            cls=None,
            array_lower_sub_2d=values_sub_1.binned.native,
        )

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid2D.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_1d_from(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_2d_from(result=values_sub_2)

        mask_sub_4 = mask.mask_new_sub_size_from(mask=mask, sub_size=4)
        grid_sub_4 = aa.Grid2D.from_mask(mask=mask_sub_4)
        values_sub_4 = ndarray_1d_from(grid=grid_sub_4, profile=None)
        values_sub_4 = grid_sub_4.structure_2d_from(result=values_sub_4)

        assert values.native[1, 1] == values_sub_2.binned.native[1, 1]
        assert values.native[2, 2] != values_sub_2.binned.native[2, 2]

        assert values.native[1, 1] != values_sub_4.binned.native[1, 1]
        assert values.native[2, 2] == values_sub_4.binned.native[2, 2]

    def test__func_returns_all_zeros__iteration_terminated(self):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
        )

        arr = aa.Array2D.manual_mask(array=np.zeros(9), mask=mask)

        values = grid.iterated_array_from(
            func=ndarray_1d_from, cls=None, array_lower_sub_2d=arr
        )

        assert (values == np.zeros((9,))).all()


class TestIteratedGrid:
    def test__threshold_mask_from(self):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        iterate = aa.Grid2DIterate.from_mask(mask=mask, fractional_accuracy=0.9999)

        grid = aa.Grid2D.manual_mask(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask,
        )

        threshold_mask = iterate.threshold_mask_via_grids_from(
            grid_lower_sub_2d=grid.binned.native, grid_higher_sub_2d=grid.binned.native
        )

        assert (
            threshold_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            )
        ).all()

        grid_lower_sub = aa.Grid2D.manual_mask(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask,
        )

        grid_higher_sub = aa.Grid2D.manual_mask(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask,
        )

        threshold_mask = iterate.threshold_mask_via_grids_from(
            grid_lower_sub_2d=grid_lower_sub.binned.native,
            grid_higher_sub_2d=grid_higher_sub.binned.native,
        )

        assert (
            threshold_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

        iterate = aa.Grid2DIterate.from_mask(mask=mask, fractional_accuracy=0.5)

        grid_lower_sub = aa.Grid2D.manual_mask(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.9, 1.9], [0.001, 0.001], [0.0, 0.0]],
                [[0.0, 0.0], [0.999, 0.999], [1.9, 0.001], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask,
        )

        grid_higher_sub = aa.Grid2D.manual_mask(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask,
        )

        threshold_mask = iterate.threshold_mask_via_grids_from(
            grid_lower_sub_2d=grid_lower_sub.binned.native,
            grid_higher_sub_2d=grid_higher_sub.binned.native,
        )

        assert (
            threshold_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, True, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

    def test__threshold_mask_from__uses_higher_sub_grids_mask(self):

        mask_lower_sub = aa.Mask2D.manual(
            mask=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        mask_higher_sub = aa.Mask2D.manual(
            mask=[
                [True, True, True, True],
                [True, False, True, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        iterate = aa.Grid2DIterate.from_mask(
            mask=mask_lower_sub, fractional_accuracy=0.5
        )

        grid_lower_sub = aa.Grid2D.manual_mask(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask_lower_sub,
        )

        grid_higher_sub = aa.Grid2D.manual_mask(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.1, 2.0], [0.1, 0.1], [0.0, 0.0]],
                [[0.0, 0.0], [0.1, 0.1], [0.1, 0.1], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask_higher_sub,
        )

        threshold_mask = iterate.threshold_mask_via_grids_from(
            grid_lower_sub_2d=grid_lower_sub.binned.native,
            grid_higher_sub_2d=grid_higher_sub.binned.native,
        )

        assert (
            threshold_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, True, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

    def test__iterated_grid_from__extreme_fractional_accuracies_uses_last_or_first_sub(
        self,
    ):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
        )

        mask_sub_1 = mask.mask_new_sub_size_from(mask=mask, sub_size=1)
        grid_sub_1 = aa.Grid2D.from_mask(mask=mask_sub_1)
        values_sub_1 = ndarray_2d_from(grid=grid_sub_1, profile=None)
        values_sub_1 = grid_sub_1.structure_2d_from(result=values_sub_1)

        values = grid.iterated_grid_from(
            func=ndarray_2d_from, cls=None, grid_lower_sub_2d=values_sub_1.binned.native
        )

        mask_sub_3 = mask.mask_new_sub_size_from(mask=mask, sub_size=3)
        grid_sub_3 = aa.Grid2D.from_mask(mask=mask_sub_3)
        values_sub_3 = ndarray_2d_from(grid=grid_sub_3, profile=None)
        values_sub_3 = grid_sub_3.structure_2d_from(result=values_sub_3)

        assert (values == values_sub_3.binned).all()

        # This test ensures that if the fractional accuracy is met on the last sub_size jump (e.g. 2 doesnt meet it,
        # but 3 does) that the sub_size of 3 is used. There was a bug where the mask was not updated correctly and the
        # iterated grid double counted the values.

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.99, sub_steps=[2, 3]
        )

        values = grid.iterated_grid_from(
            func=ndarray_2d_from, cls=None, grid_lower_sub_2d=values_sub_1.binned.native
        )

        assert (values == values_sub_3.binned).all()

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
        )

        values = grid.iterated_grid_from(
            func=ndarray_2d_from, cls=None, grid_lower_sub_2d=values_sub_1.binned.native
        )

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid2D.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_2d_from(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_2d_from(result=values_sub_2)

        assert (values == values_sub_2.binned).all()

    def test__iterated_grid_from__check_values_computed_to_fractional_accuracy(self,):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.5, sub_steps=[2, 4]
        )

        mask_sub_1 = mask.mask_new_sub_size_from(mask=mask, sub_size=1)
        grid_sub_1 = aa.Grid2D.from_mask(mask=mask_sub_1)
        values_sub_1 = ndarray_2d_from(grid=grid_sub_1, profile=None)
        values_sub_1 = grid_sub_1.structure_2d_from(result=values_sub_1)

        values = grid.iterated_grid_from(
            func=ndarray_2d_from, cls=None, grid_lower_sub_2d=values_sub_1.binned.native
        )

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid2D.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_2d_from(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_2d_from(result=values_sub_2)

        mask_sub_4 = mask.mask_new_sub_size_from(mask=mask, sub_size=4)
        grid_sub_4 = aa.Grid2D.from_mask(mask=mask_sub_4)
        values_sub_4 = ndarray_2d_from(grid=grid_sub_4, profile=None)
        values_sub_4 = grid_sub_4.structure_2d_from(result=values_sub_4)

        assert values.native[1, 1, 0] == values_sub_2.binned.native[1, 1, 0]
        assert values.native[2, 2, 0] != values_sub_2.binned.native[2, 2, 0]

        assert values.native[1, 1, 0] != values_sub_4.binned.native[1, 1, 0]
        assert values.native[2, 2, 0] == values_sub_4.binned.native[2, 2, 0]

        assert values.native[1, 1, 1] == values_sub_2.binned.native[1, 1, 1]
        assert values.native[2, 2, 1] != values_sub_2.binned.native[2, 2, 1]

        assert values.native[1, 1, 1] != values_sub_4.binned.native[1, 1, 1]
        assert values.native[2, 2, 1] == values_sub_4.binned.native[2, 2, 1]

    def test__func_returns_all_zeros__iteration_terminated(self):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
        )

        grid_lower = aa.Grid2D.manual_mask(grid=np.zeros((9, 2)), mask=mask)

        values = grid.iterated_grid_from(
            func=ndarray_1d_from, cls=None, grid_lower_sub_2d=grid_lower
        )

        assert (values == np.zeros((9, 2))).all()


class TestAPI:
    def test__manual_slim(self):

        grid = aa.Grid2DIterate.manual_slim(
            grid=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            shape_native=(2, 2),
            pixel_scales=1.0,
            fractional_accuracy=0.1,
            sub_steps=[2, 3, 4],
            origin=(0.0, 1.0),
        )

        assert type(grid) == aa.Grid2DIterate
        assert type(grid.slim) == aa.Grid2DIterate
        assert type(grid.native) == aa.Grid2DIterate
        assert (
            grid == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            grid.native
            == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.fractional_accuracy == 0.1
        assert grid.sub_steps == [2, 3, 4]
        assert grid.origin == (0.0, 1.0)

    def test__from_mask(self):

        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        grid_via_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, sub_size=1, pixel_scales=(2.0, 2.0)
        )

        grid = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.1, sub_steps=[2, 3, 4]
        )

        assert type(grid) == aa.Grid2DIterate
        assert type(grid.slim) == aa.Grid2DIterate
        assert type(grid.native) == aa.Grid2DIterate
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.sub_steps == [2, 3, 4]
        assert grid.sub_size == 1

    def test__uniform(self):

        grid = aa.Grid2DIterate.uniform(
            shape_native=(2, 2),
            pixel_scales=2.0,
            fractional_accuracy=0.1,
            sub_steps=[2, 3, 4],
        )

        assert type(grid) == aa.Grid2DIterate
        assert type(grid.slim) == aa.Grid2DIterate
        assert type(grid.native) == aa.Grid2DIterate
        assert (
            grid == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert (
            grid.native
            == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
        ).all()
        assert (
            grid.slim == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.fractional_accuracy == 0.1
        assert grid.sub_steps == [2, 3, 4]
        assert grid.origin == (0.0, 0.0)
