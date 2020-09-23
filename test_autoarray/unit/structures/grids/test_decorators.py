import os
import numpy as np

import autoarray as aa
from test_autoarray.mock import (
    MockGridLikeIteratorObj,
    MockGridLikeObj,
    ndarray_1d_from_grid,
    ndarray_2d_from_grid,
)

test_coordinates_dir = "{}/files/coordinates/".format(
    os.path.dirname(os.path.realpath(__file__))
)


def test__grid_in__output_values_same_format():
    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=1)

    grid = aa.Grid.from_mask(mask=mask)

    grid_like_object = MockGridLikeObj()

    array_output = grid_like_object.ndarray_1d_from_grid(grid=grid)

    assert isinstance(array_output, aa.Array)
    assert (
        array_output.in_2d
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    grid_output = grid_like_object.ndarray_2d_from_grid(grid=grid)

    assert isinstance(grid_output, aa.Grid)
    assert (
        grid_output.in_2d
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__grid_in__output_is_list__list_of_same_format():
    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=1)

    grid = aa.Grid.from_mask(mask=mask)

    grid_like_object = MockGridLikeObj()

    array_output = grid_like_object.ndarray_1d_list_from_grid(grid=grid)

    assert isinstance(array_output[0], aa.Array)
    assert (
        array_output[0].in_2d
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    assert isinstance(array_output[1], aa.Array)
    assert (
        array_output[1].in_2d
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    grid_output = grid_like_object.ndarray_2d_list_from_grid(grid=grid)

    assert isinstance(grid_output[0], aa.Grid)
    assert (
        grid_output[0].in_2d
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.5, -0.5], [0.5, 0.5], [0.0, 0.0]],
                [[0.0, 0.0], [-0.5, -0.5], [-0.5, 0.5], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()

    assert isinstance(grid_output[1], aa.Grid)
    assert (
        grid_output[1].in_2d
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__grid_coordinates_in__output_values_same_format():

    grid_like_object = MockGridLikeObj()

    coordinates = aa.GridCoordinates(
        coordinates=[[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0)]]
    )

    values_output = grid_like_object.ndarray_1d_from_grid(grid=coordinates)

    assert values_output.in_list == [[1.0, 1.0], [1.0]]

    coordinates_output = grid_like_object.ndarray_2d_from_grid(grid=coordinates)

    assert coordinates_output.in_list == [[(2.0, 4.0), (6.0, 8.0)], [(10.0, 12.0)]]


def test__grid_coordinates_in__output_is_list__list_of_same_format():

    grid_like_object = MockGridLikeObj()

    coordinates = aa.GridCoordinates(
        coordinates=[[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0)]]
    )

    coordinates_output = grid_like_object.ndarray_1d_list_from_grid(grid=coordinates)

    assert coordinates_output[0].in_list == [[1.0, 1.0], [1.0]], [[2.0, 2.0], [2.0]]

    coordinates_output = grid_like_object.ndarray_2d_list_from_grid(grid=coordinates)

    assert coordinates_output[0].in_list == [[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0)]], [
        [(2.0, 4.0), (6.0, 8.0)],
        [(10.0, 12.0)],
    ]


def test__grid_iterate_in__output_values__use_iterated_array_function():

    mask = aa.Mask.manual(
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

    grid = aa.GridIterate.from_mask(
        mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
    )

    grid_like_obj = MockGridLikeIteratorObj()

    values = grid_like_obj.ndarray_1d_from_grid(grid=grid)

    mask_sub_3 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=3)
    grid_sub_3 = aa.Grid.from_mask(mask=mask_sub_3)
    values_sub_3 = ndarray_1d_from_grid(grid=grid_sub_3, profile=None)
    values_sub_3 = grid_sub_3.structure_from_result(result=values_sub_3)

    assert (values == values_sub_3.in_1d_binned).all()

    grid = aa.GridIterate.from_mask(
        mask=mask, fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
    )

    grid_like_obj = MockGridLikeIteratorObj()

    values = grid_like_obj.ndarray_1d_from_grid(grid=grid)

    mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
    grid_sub_2 = aa.Grid.from_mask(mask=mask_sub_2)
    values_sub_2 = ndarray_1d_from_grid(grid=grid_sub_2, profile=None)
    values_sub_2 = grid_sub_2.structure_from_result(result=values_sub_2)

    assert (values == values_sub_2.in_1d_binned).all()

    grid = aa.GridIterate.from_mask(
        mask=mask, fractional_accuracy=0.5, sub_steps=[2, 4]
    )

    iterate_obj = MockGridLikeIteratorObj()

    values = iterate_obj.ndarray_1d_from_grid(grid=grid)

    mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
    grid_sub_2 = aa.Grid.from_mask(mask=mask_sub_2)
    values_sub_2 = ndarray_1d_from_grid(grid=grid_sub_2, profile=None)
    values_sub_2 = grid_sub_2.structure_from_result(result=values_sub_2)

    mask_sub_4 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
    grid_sub_4 = aa.Grid.from_mask(mask=mask_sub_4)
    values_sub_4 = ndarray_1d_from_grid(grid=grid_sub_4, profile=None)
    values_sub_4 = grid_sub_4.structure_from_result(result=values_sub_4)

    assert values.in_2d[1, 1] == values_sub_2.in_2d_binned[1, 1]
    assert values.in_2d[2, 2] != values_sub_2.in_2d_binned[2, 2]

    assert values.in_2d[1, 1] != values_sub_4.in_2d_binned[1, 1]
    assert values.in_2d[2, 2] == values_sub_4.in_2d_binned[2, 2]


def test__grid_iterate_in__output_is_list_of_arrays__use_maximum_sub_size_in_all_pixels():

    mask = aa.Mask.manual(
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

    grid = aa.GridIterate.from_mask(
        mask=mask, fractional_accuracy=0.05, sub_steps=[2, 3]
    )

    grid_like_obj = MockGridLikeIteratorObj()

    values = grid_like_obj.ndarray_1d_list_from_grid(grid=grid)

    mask_sub_3 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=3)
    grid_sub_3 = aa.Grid.from_mask(mask=mask_sub_3)
    values_sub_3 = ndarray_1d_from_grid(grid=grid_sub_3, profile=None)
    values_sub_3 = grid_sub_3.structure_from_result(result=values_sub_3)

    assert (values[0] == values_sub_3.in_1d_binned).all()


def test__grid_iterate_in__output_values__use_iterated_grid_function():

    mask = aa.Mask.manual(
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

    grid = aa.GridIterate.from_mask(
        mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
    )

    grid_like_obj = MockGridLikeIteratorObj()

    values = grid_like_obj.ndarray_2d_from_grid(grid=grid)

    mask_sub_3 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=3)
    grid_sub_3 = aa.Grid.from_mask(mask=mask_sub_3)
    values_sub_3 = ndarray_2d_from_grid(grid=grid_sub_3, profile=None)
    values_sub_3 = grid_sub_3.structure_from_result(result=values_sub_3)

    assert (values == values_sub_3.in_1d_binned).all()

    grid = aa.GridIterate.from_mask(
        mask=mask, fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
    )

    grid_like_obj = MockGridLikeIteratorObj()

    values = grid_like_obj.ndarray_2d_from_grid(grid=grid)

    mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
    grid_sub_2 = aa.Grid.from_mask(mask=mask_sub_2)
    values_sub_2 = ndarray_2d_from_grid(grid=grid_sub_2, profile=None)
    values_sub_2 = grid_sub_2.structure_from_result(result=values_sub_2)

    assert (values == values_sub_2.in_1d_binned).all()

    grid = aa.GridIterate.from_mask(
        mask=mask, fractional_accuracy=0.5, sub_steps=[2, 4]
    )

    iterate_obj = MockGridLikeIteratorObj()

    values = iterate_obj.ndarray_2d_from_grid(grid=grid)

    mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
    grid_sub_2 = aa.Grid.from_mask(mask=mask_sub_2)
    values_sub_2 = ndarray_2d_from_grid(grid=grid_sub_2, profile=None)
    values_sub_2 = grid_sub_2.structure_from_result(result=values_sub_2)

    mask_sub_4 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
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


def test__grid_iterate_in__output_is_list_of_grids__use_maximum_sub_size_in_all_pixels():

    mask = aa.Mask.manual(
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

    grid = aa.GridIterate.from_mask(
        mask=mask, fractional_accuracy=0.05, sub_steps=[2, 3]
    )

    grid_like_obj = MockGridLikeIteratorObj()

    values = grid_like_obj.ndarray_2d_list_from_grid(grid=grid)

    mask_sub_3 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=3)
    grid_sub_3 = aa.Grid.from_mask(mask=mask_sub_3)
    values_sub_3 = ndarray_2d_from_grid(grid=grid_sub_3, profile=None)
    values_sub_3 = grid_sub_3.structure_from_result(result=values_sub_3)

    assert (values[0][0] == values_sub_3.in_1d_binned[0]).all()
    assert (values[0][1] == values_sub_3.in_1d_binned[1]).all()


def test__grid_interpolate_in__output_values__interpolation_used_and_accurate():

    mask = aa.Mask.circular_annular(
        shape_2d=(20, 20),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
        inner_radius=3.0,
        outer_radius=8.0,
    )

    grid_like_obj = MockGridLikeIteratorObj()

    grid = aa.Grid.from_mask(mask=mask)

    true_array = grid_like_obj.ndarray_1d_from_grid(grid=grid)

    grid = aa.GridInterpolate.from_mask(mask=mask, pixel_scales_interp=1.0)

    interpolated_array = grid_like_obj.ndarray_1d_from_grid(grid=grid)

    assert interpolated_array.shape[0] == mask.pixels_in_mask
    assert (true_array == interpolated_array).all()

    grid = aa.GridInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

    interpolated_array = grid_like_obj.ndarray_1d_from_grid(grid=grid)

    assert interpolated_array.shape[0] == mask.pixels_in_mask
    assert true_array[0] != interpolated_array[0]
    assert np.max(true_array - interpolated_array) < 0.001

    grid = aa.Grid.from_mask(mask=mask)

    true_grid = grid_like_obj.ndarray_2d_from_grid(grid=grid)

    grid = aa.GridInterpolate.from_mask(mask=mask, pixel_scales_interp=1.0)

    interpolated_grid = grid_like_obj.ndarray_2d_from_grid(grid=grid)

    assert interpolated_grid.shape[0] == mask.pixels_in_mask
    assert interpolated_grid.shape[1] == 2
    assert (true_grid == interpolated_grid).all()

    grid = aa.GridInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

    interpolated_grid = grid_like_obj.ndarray_2d_from_grid(grid=grid)

    assert interpolated_grid.shape[0] == mask.pixels_in_mask
    assert interpolated_grid.shape[1] == 2
    assert true_grid[0, 0] != interpolated_grid[0, 0]
    assert np.max(true_grid[:, 0] - interpolated_grid[:, 0]) < 0.001
    assert np.max(true_grid[:, 1] - interpolated_grid[:, 1]) < 0.001
