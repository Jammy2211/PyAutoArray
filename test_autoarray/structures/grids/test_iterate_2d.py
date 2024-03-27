import numpy as np

import autoarray as aa

from autoarray.structures.mock.mock_structure_decorators import (
    ndarray_1d_from,
    ndarray_2d_from,
)


def test__threshold_mask_via_arrays_from():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    arr = aa.Array2D(
        values=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask,
    )

    iterator = aa.Iterator(fractional_accuracy = 0.9999)

    threshold_mask = iterator.threshold_mask_via_arrays_from(
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

    mask_lower_sub = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    mask_higher_sub = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    iterator = aa.Iterator(
        fractional_accuracy=0.5
    )

    array_lower_sub = aa.Array2D(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask_lower_sub,
    )

    array_higher_sub = aa.Array2D(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 5.0, 0.0],
            [0.0, 5.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask_higher_sub,
    )

    threshold_mask = iterator.threshold_mask_via_arrays_from(
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


def test__iterated_array_from__extreme_fractional_accuracies_uses_last_or_first_sub():
    mask = aa.Mask2D(
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
    
    grid = aa.Grid2D.from_mask(mask=mask)
    
    iterator = aa.Iterator(
        fractional_accuracy = 1.0,
        sub_steps = [2, 3]
    )

    sub_1 = mask.mask_new_sub_size_from(mask=mask, sub_size=1)
    grid_sub_1 = aa.Grid2D.from_mask(mask=sub_1)
    values_sub_1 = ndarray_1d_from(grid=grid_sub_1, profile=None)
    values_sub_1 = grid_sub_1.structure_2d_from(result=values_sub_1)

    values = iterator.iterated_array_from(
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

    values = iterator.iterated_array_from(
        func=ndarray_1d_from,
        cls=None,
        array_lower_sub_2d=values_sub_1.binned.native,
    )

    assert (values == values_sub_3.binned).all()

    iterator = aa.Iterator(
        fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
    )

    values = iterator.iterated_array_from(
        func=ndarray_1d_from,
        cls=None,
        array_lower_sub_2d=values_sub_1.binned.native,
    )

    mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
    grid_sub_2 = aa.Grid2D.from_mask(mask=mask_sub_2)
    values_sub_2 = ndarray_1d_from(grid=grid_sub_2, profile=None)
    values_sub_2 = grid_sub_2.structure_2d_from(result=values_sub_2)

    assert (values == values_sub_2.binned).all()


def test__iterated_array_from__check_values_computed_to_fractional_accuracy():
    mask = aa.Mask2D(
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

    grid = aa.Grid2D.from_mask(
        mask=mask, 
    )

    iterator = aa.Iterator(fractional_accuracy=0.5, sub_steps=[2, 4]
    )

    sub_1 = mask.mask_new_sub_size_from(mask=mask, sub_size=1)
    grid_sub_1 = aa.Grid2D.from_mask(mask=sub_1)
    values_sub_1 = ndarray_1d_from(grid=grid_sub_1, profile=None)
    values_sub_1 = grid_sub_1.structure_2d_from(result=values_sub_1)

    values = iterator.iterated_array_from(
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


def test__iterated_array_from__func_returns_all_zeros__iteration_terminated():
    mask = aa.Mask2D(
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
    
    grid = aa.Grid2D.from_mask(
        mask=mask, 
    )

    iterator = aa.Iterator(fractional_accuracy=1.0, sub_steps=[2, 3]
    )

    arr = aa.Array2D(values=np.zeros(9), mask=mask)

    values = iterator.iterated_array_from(
        func=ndarray_1d_from, cls=None, array_lower_sub_2d=arr
    )

    assert (values == np.zeros((9,))).all()


def test__threshold_mask_via_grids_from():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid = aa.Grid2D.from_mask(
        mask=mask, 
    )

    iterator = aa.Iterator(fractional_accuracy=0.9999, 
    )

    grid = aa.Grid2D(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        mask=mask,
    )

    threshold_mask = iterator.threshold_mask_via_grids_from(
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

    grid = aa.Grid2D.from_mask(
        mask=mask,
    )

    iterator = aa.Iterator(fractional_accuracy=0.5,
    )

    grid_lower_sub = aa.Grid2D(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.9, 1.9], [0.001, 0.001], [0.0, 0.0]],
            [[0.0, 0.0], [0.999, 0.999], [1.9, 0.001], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        mask=mask,
    )

    grid_higher_sub = aa.Grid2D(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        mask=mask,
    )

    threshold_mask = iterator.threshold_mask_via_grids_from(
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

    mask_lower_sub = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    mask_higher_sub = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid = aa.Grid2D.from_mask(
        mask=mask_lower_sub,
    )

    iterator = aa.Iterator(fractional_accuracy=0.9999,
    )

    grid_lower_sub = aa.Grid2D(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
            [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        mask=mask_lower_sub,
    )

    grid_higher_sub = aa.Grid2D(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.1, 2.0], [0.1, 0.1], [0.0, 0.0]],
            [[0.0, 0.0], [0.1, 0.1], [0.1, 0.1], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        mask=mask_higher_sub,
    )

    threshold_mask = iterator.threshold_mask_via_grids_from(
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


def test__iterated_grid_from__extreme_fractional_accuracies_uses_last_or_first_sub():
    mask = aa.Mask2D(
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

    grid = aa.Grid2D.from_mask(
        mask=mask,
    )

    iterator = aa.Iterator(fractional_accuracy=1.0, sub_steps=[2, 3]
    )

    sub_1 = mask.mask_new_sub_size_from(mask=mask, sub_size=1)
    grid_sub_1 = aa.Grid2D.from_mask(mask=sub_1)
    values_sub_1 = ndarray_2d_from(grid=grid_sub_1, profile=None)
    values_sub_1 = grid_sub_1.structure_2d_from(result=values_sub_1)

    values = iterator.iterated_grid_from(
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

    iterator = aa.Iterator(fractional_accuracy=0.99, sub_steps=[2, 3]
    )

    values = iterator.iterated_grid_from(
        func=ndarray_2d_from, cls=None, grid_lower_sub_2d=values_sub_1.binned.native
    )

    assert (values == values_sub_3.binned).all()

    iterator = aa.Iterator(fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
    )


    values = iterator.iterated_grid_from(
        func=ndarray_2d_from, cls=None, grid_lower_sub_2d=values_sub_1.binned.native
    )

    mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
    grid_sub_2 = aa.Grid2D.from_mask(mask=mask_sub_2)
    values_sub_2 = ndarray_2d_from(grid=grid_sub_2, profile=None)
    values_sub_2 = grid_sub_2.structure_2d_from(result=values_sub_2)

    assert (values == values_sub_2.binned).all()


def test__iterated_grid_from__check_values_computed_to_fractional_accuracy():
    mask = aa.Mask2D(
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

    grid = aa.Grid2D.from_mask(mask=mask)

    iterator = aa.Iterator(fractional_accuracy=0.5, sub_steps=[2, 4]
    )

    sub_1 = mask.mask_new_sub_size_from(mask=mask, sub_size=1)
    grid_sub_1 = aa.Grid2D.from_mask(mask=sub_1)
    values_sub_1 = ndarray_2d_from(grid=grid_sub_1, profile=None)
    values_sub_1 = grid_sub_1.structure_2d_from(result=values_sub_1)

    values = iterator.iterated_grid_from(
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


def test__iterated_grid_from__func_returns_all_zeros__iteration_terminated():
    mask = aa.Mask2D(
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

    grid = aa.Grid2D.from_mask(mask=mask)

    iterator = aa.Iterator(fractional_accuracy=1.0, sub_steps=[2, 3]
    )

    grid_lower = aa.Grid2D(values=np.zeros((9, 2)), mask=mask)

    values = iterator.iterated_grid_from(
        func=ndarray_1d_from, cls=None, grid_lower_sub_2d=grid_lower
    )

    assert (values == np.zeros((9, 2))).all()
