import os
import numpy as np
import pytest

import autoarray as aa
from autoarray.structures import grids
from test_autoarray.mock.mock_grids import ndarray_1d_from_grid, ndarray_2d_from_grid

test_coordinates_dir = "{}/files/coordinates/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestObj:
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

        mask = aa.Mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_mask_util = aa.util.mask.blurring_mask_2d_from(
            mask_2d=mask, kernel_shape_2d=(3, 5)
        )

        blurring_grid_util = aa.util.grid.grid_1d_via_mask_2d_from(
            mask_2d=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid = aa.MaskedGrid.from_mask(mask=mask)

        blurring_grid = grid.blurring_grid_from_kernel_shape(kernel_shape_2d=(3, 5))

        assert isinstance(blurring_grid, grids.GridIterator)
        assert len(blurring_grid.shape) == 2
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)

    def test__blurring_grid_from_kernel_shape__compare_to_array_util(self):
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

        mask = aa.Mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0))

        blurring_mask_util = aa.util.mask.blurring_mask_2d_from(
            mask_2d=mask, kernel_shape_2d=(3, 5)
        )

        blurring_grid_util = aa.util.grid.grid_1d_via_mask_2d_from(
            mask_2d=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        mask = aa.Mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0))

        blurring_grid = grids.GridIterator.blurring_grid_from_mask_and_kernel_shape(
            mask=mask, kernel_shape_2d=(3, 5)
        )

        assert isinstance(blurring_grid, grids.GridIterator)
        assert len(blurring_grid.shape) == 2
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)

        blurring_grid = grids.GridIterator.blurring_grid_from_mask_and_kernel_shape(
            mask=mask, kernel_shape_2d=(3, 5), store_in_1d=False
        )

        assert isinstance(blurring_grid, grids.GridIterator)
        assert len(blurring_grid.shape) == 3
        assert blurring_grid.pixel_scales == (2.0, 2.0)


class TestIteratedArray:
    def test__fractional_mask_from_arrays(self):

        mask = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = aa.GridIterator.from_mask(mask=mask, fractional_accuracy=0.9999)

        arr = aa.MaskedArray.manual_2d(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask,
        )

        fractional_mask = grid.fractional_mask_from_arrays(
            array_lower_sub_2d=arr.in_2d_binned, array_higher_sub_2d=arr.in_2d_binned
        )

        assert (
            fractional_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            )
        ).all()

        result_array_lower_sub = aa.MaskedArray.manual_2d(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask,
        )

        result_array_higher_sub = aa.MaskedArray.manual_2d(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask,
        )

        fractional_mask = grid.fractional_mask_from_arrays(
            array_lower_sub_2d=result_array_lower_sub.in_2d_binned,
            array_higher_sub_2d=result_array_higher_sub.in_2d_binned,
        )

        assert (
            fractional_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

        grid = aa.GridIterator.from_mask(mask=mask, fractional_accuracy=0.5)

        result_array_lower_sub = aa.MaskedArray.manual_2d(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.9, 0.001, 0.0],
                [0.0, 0.999, 1.9, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask,
        )

        result_array_higher_sub = aa.MaskedArray.manual_2d(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask,
        )

        fractional_mask = grid.fractional_mask_from_arrays(
            array_lower_sub_2d=result_array_lower_sub.in_2d_binned,
            array_higher_sub_2d=result_array_higher_sub.in_2d_binned,
        )

        assert (
            fractional_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, True, False, True],
                    [True, False, True, True],
                    [True, True, True, True],
                ]
            )
        ).all()

    def test__fractional_mask_from_arrays__uses_higher_sub_grids_mask(self):

        mask_lower_sub = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        mask_higher_sub = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True],
                [True, False, True, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = aa.GridIterator.from_mask(mask=mask_lower_sub, fractional_accuracy=0.5)

        array_lower_sub = aa.MaskedArray.manual_2d(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask_lower_sub,
        )

        array_higher_sub = aa.MaskedArray.manual_2d(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.1, 0.0],
                [0.0, 0.1, 0.1, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            mask=mask_higher_sub,
        )

        fractional_mask = grid.fractional_mask_from_arrays(
            array_lower_sub_2d=array_lower_sub.in_2d_binned,
            array_higher_sub_2d=array_higher_sub.in_2d_binned,
        )

        assert (
            fractional_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, True, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

    def test__iterated_array_from_func__extreme_fractional_accuracies_uses_last_or_first_sub(
        self
    ):

        mask = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.GridIterator.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
        )

        mask_sub_1 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=1)
        grid_sub_1 = aa.Grid.from_mask(mask=mask_sub_1)
        values_sub_1 = ndarray_1d_from_grid(grid=grid_sub_1, profile=None)
        values_sub_1 = grid_sub_1.structure_from_result(result=values_sub_1)

        values = grid.iterated_array_from_func(
            func=ndarray_1d_from_grid,
            profile=None,
            array_lower_sub_2d=values_sub_1.in_2d_binned,
        )

        mask_sub_3 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=3)
        grid_sub_3 = aa.Grid.from_mask(mask=mask_sub_3)
        values_sub_3 = ndarray_1d_from_grid(grid=grid_sub_3, profile=None)
        values_sub_3 = grid_sub_3.structure_from_result(result=values_sub_3)

        assert (values == values_sub_3.in_1d_binned).all()

        grid = aa.GridIterator.from_mask(
            mask=mask, fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
        )

        values = grid.iterated_array_from_func(
            func=ndarray_1d_from_grid,
            profile=None,
            array_lower_sub_2d=values_sub_1.in_2d_binned,
        )

        mask_sub_2 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_1d_from_grid(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_from_result(result=values_sub_2)

        assert (values == values_sub_2.in_1d_binned).all()

    def test__iterated_array_from_func__check_values_computed_to_fractional_accuracy(
        self
    ):

        mask = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.GridIterator.from_mask(
            mask=mask, fractional_accuracy=0.5, sub_steps=[2, 4]
        )

        mask_sub_1 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=1)
        grid_sub_1 = aa.Grid.from_mask(mask=mask_sub_1)
        values_sub_1 = ndarray_1d_from_grid(grid=grid_sub_1, profile=None)
        values_sub_1 = grid_sub_1.structure_from_result(result=values_sub_1)

        values = grid.iterated_array_from_func(
            func=ndarray_1d_from_grid,
            profile=None,
            array_lower_sub_2d=values_sub_1.in_2d_binned,
        )

        mask_sub_2 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_1d_from_grid(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_from_result(result=values_sub_2)

        mask_sub_4 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
        grid_sub_4 = aa.Grid.from_mask(mask=mask_sub_4)
        values_sub_4 = ndarray_1d_from_grid(grid=grid_sub_4, profile=None)
        values_sub_4 = grid_sub_4.structure_from_result(result=values_sub_4)

        assert values.in_2d[1, 1] == values_sub_2.in_2d_binned[1, 1]
        assert values.in_2d[2, 2] != values_sub_2.in_2d_binned[2, 2]

        assert values.in_2d[1, 1] != values_sub_4.in_2d_binned[1, 1]
        assert values.in_2d[2, 2] == values_sub_4.in_2d_binned[2, 2]

    def test__func_returns_all_zeros__iteration_terminated(self):

        mask = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.GridIterator.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
        )

        arr = aa.MaskedArray(array=np.zeros(9), mask=mask)

        values = grid.iterated_array_from_func(
            func=ndarray_1d_from_grid, profile=None, array_lower_sub_2d=arr
        )

        assert (values == np.zeros((9,))).all()


class TestIteratedGrid:
    def test__fractional_mask_from_grids(self):

        mask = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        iterator = aa.GridIterator.from_mask(mask=mask, fractional_accuracy=0.9999)

        grid = aa.MaskedGrid.manual_2d(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask,
        )

        fractional_mask = iterator.fractional_mask_from_grids(
            grid_lower_sub_2d=grid.in_2d_binned, grid_higher_sub_2d=grid.in_2d_binned
        )

        assert (
            fractional_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            )
        ).all()

        grid_lower_sub = aa.MaskedGrid.manual_2d(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask,
        )

        grid_higher_sub = aa.MaskedGrid.manual_2d(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask,
        )

        fractional_mask = iterator.fractional_mask_from_grids(
            grid_lower_sub_2d=grid_lower_sub.in_2d_binned,
            grid_higher_sub_2d=grid_higher_sub.in_2d_binned,
        )

        assert (
            fractional_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

        iterator = aa.GridIterator.from_mask(mask=mask, fractional_accuracy=0.5)

        grid_lower_sub = aa.MaskedGrid.manual_2d(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.9, 1.9], [0.001, 0.001], [0.0, 0.0]],
                [[0.0, 0.0], [0.999, 0.999], [1.9, 0.001], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask,
        )

        grid_higher_sub = aa.MaskedGrid.manual_2d(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask,
        )

        fractional_mask = iterator.fractional_mask_from_grids(
            grid_lower_sub_2d=grid_lower_sub.in_2d_binned,
            grid_higher_sub_2d=grid_higher_sub.in_2d_binned,
        )

        assert (
            fractional_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, True, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

    def test__fractional_mask_from_grids__uses_higher_sub_grids_mask(self):

        mask_lower_sub = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        mask_higher_sub = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True],
                [True, False, True, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        iterator = aa.GridIterator.from_mask(
            mask=mask_lower_sub, fractional_accuracy=0.5
        )

        grid_lower_sub = aa.MaskedGrid.manual_2d(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask_lower_sub,
        )

        grid_higher_sub = aa.MaskedGrid.manual_2d(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.1, 2.0], [0.1, 0.1], [0.0, 0.0]],
                [[0.0, 0.0], [0.1, 0.1], [0.1, 0.1], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            mask=mask_higher_sub,
        )

        fractional_mask = iterator.fractional_mask_from_grids(
            grid_lower_sub_2d=grid_lower_sub.in_2d_binned,
            grid_higher_sub_2d=grid_higher_sub.in_2d_binned,
        )

        assert (
            fractional_mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, True, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

    def test__iterated_grid_from_func__extreme_fractional_accuracies_uses_last_or_first_sub(
        self
    ):

        mask = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.GridIterator.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
        )

        mask_sub_1 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=1)
        grid_sub_1 = aa.Grid.from_mask(mask=mask_sub_1)
        values_sub_1 = ndarray_2d_from_grid(grid=grid_sub_1, profile=None)
        values_sub_1 = grid_sub_1.structure_from_result(result=values_sub_1)

        values = grid.iterated_grid_from_func(
            func=ndarray_2d_from_grid,
            profile=None,
            grid_lower_sub_2d=values_sub_1.in_2d_binned,
        )

        mask_sub_3 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=3)
        grid_sub_3 = aa.Grid.from_mask(mask=mask_sub_3)
        values_sub_3 = ndarray_2d_from_grid(grid=grid_sub_3, profile=None)
        values_sub_3 = grid_sub_3.structure_from_result(result=values_sub_3)

        assert (values == values_sub_3.in_1d_binned).all()

        grid = aa.GridIterator.from_mask(
            mask=mask, fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
        )

        values = grid.iterated_grid_from_func(
            func=ndarray_2d_from_grid,
            profile=None,
            grid_lower_sub_2d=values_sub_1.in_2d_binned,
        )

        mask_sub_2 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_2d_from_grid(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_from_result(result=values_sub_2)

        assert (values == values_sub_2.in_1d_binned).all()

    def test__iterated_grid_from_func__check_values_computed_to_fractional_accuracy(
        self
    ):

        mask = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.GridIterator.from_mask(
            mask=mask, fractional_accuracy=0.5, sub_steps=[2, 4]
        )

        mask_sub_1 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=1)
        grid_sub_1 = aa.Grid.from_mask(mask=mask_sub_1)
        values_sub_1 = ndarray_2d_from_grid(grid=grid_sub_1, profile=None)
        values_sub_1 = grid_sub_1.structure_from_result(result=values_sub_1)

        values = grid.iterated_grid_from_func(
            func=ndarray_2d_from_grid,
            profile=None,
            grid_lower_sub_2d=values_sub_1.in_2d_binned,
        )

        mask_sub_2 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_2d_from_grid(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_from_result(result=values_sub_2)

        mask_sub_4 = mask.mapping.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
        grid_sub_4 = aa.Grid.from_mask(mask=mask_sub_4)
        values_sub_4 = ndarray_2d_from_grid(grid=grid_sub_4, profile=None)
        values_sub_4 = grid_sub_4.structure_from_result(result=values_sub_4)

        assert values.in_2d[1, 1, 0] == values_sub_2.in_2d_binned[1, 1, 0]
        assert values.in_2d[2, 2, 0] != values_sub_2.in_2d_binned[2, 2, 0]

        assert values.in_2d[1, 1, 0] != values_sub_4.in_2d_binned[1, 1, 0]
        assert values.in_2d[2, 2, 0] == values_sub_4.in_2d_binned[2, 2, 0]

        assert values.in_2d[1, 1, 1] == values_sub_2.in_2d_binned[1, 1, 1]
        assert values.in_2d[2, 2, 1] != values_sub_2.in_2d_binned[2, 2, 1]

        assert values.in_2d[1, 1, 1] != values_sub_4.in_2d_binned[1, 1, 1]
        assert values.in_2d[2, 2, 1] == values_sub_4.in_2d_binned[2, 2, 1]

    def test__func_returns_all_zeros__iteration_terminated(self):

        mask = aa.Mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = aa.GridIterator.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
        )

        grid_lower = aa.MaskedGrid(grid=np.zeros((9, 2)), mask=mask)

        values = grid.iterated_grid_from_func(
            func=ndarray_1d_from_grid, profile=None, grid_lower_sub_2d=grid_lower
        )

        assert (values == np.zeros((9, 2))).all()


class TestAPI:
    def test__manual_1d(self):
        grid = aa.GridIterator.manual_1d(
            grid=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            shape_2d=(2, 2),
            pixel_scales=1.0,
            fractional_accuracy=0.1,
            sub_steps=[2, 3, 4],
            origin=(0.0, 1.0),
        )

        assert type(grid) == grids.GridIterator
        assert (
            grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
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
        mask = aa.Mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        grid_via_util = aa.util.grid.grid_1d_via_mask_2d_from(
            mask_2d=mask, sub_size=1, pixel_scales=(2.0, 2.0)
        )

        grid = aa.GridIterator.from_mask(mask=mask, fractional_accuracy=0.1)

        assert type(grid) == grids.GridIterator
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.interpolator == None
        assert grid.sub_size == 1

    def test__uniform(self):

        grid = aa.GridIterator.uniform(
            shape_2d=(2, 2), pixel_scales=2.0, fractional_accuracy=0.1
        )

        assert type(grid) == grids.GridIterator
        assert (
            grid.in_2d
            == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (0.0, 0.0)
