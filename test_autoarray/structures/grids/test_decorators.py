import os
import numpy as np
import pytest

import autoarray as aa
from autoarray.mock.mock_grid import (
    MockGridLikeIteratorObj,
    MockGrid1DLikeObj,
    MockGrid2DLikeObj,
    ndarray_1d_from,
    ndarray_2d_from,
)


class TestGrid1DToStructure:
    def test__grid_1d_in__output_values_projected_format(self):

        grid_2d = aa.Grid1D.manual_native(grid=[1.0, 2.0, 3.0], pixel_scales=1.0)

        grid_like_object = MockGrid1DLikeObj()

        array_output = grid_like_object.ndarray_1d_from(grid=grid_2d)

        assert isinstance(array_output, aa.Array1D)
        assert (array_output.native == np.array([1.0, 1.0, 1.0])).all()
        assert array_output.pixel_scales == (1.0,)

        grid_like_object = MockGrid1DLikeObj(centre=(1.0, 0.0), angle=45.0)

        array_output = grid_like_object.ndarray_1d_from(grid=grid_2d)

        assert isinstance(array_output, aa.Array1D)
        assert (array_output.native == np.array([1.0, 1.0, 1.0])).all()
        assert array_output.pixel_scales == (1.0,)

    def test__grid_2d_in__output_values_projected_format(self):

        grid_2d = aa.Grid2D.uniform(shape_native=(4, 4), pixel_scales=1.0, sub_size=1)

        grid_like_object = MockGrid1DLikeObj()

        array_output = grid_like_object.ndarray_1d_from(grid=grid_2d)

        assert isinstance(array_output, aa.Array1D)
        assert (array_output.native == np.array([1.0])).all()
        assert array_output.pixel_scales == (1.0,)

        grid_like_object = MockGrid1DLikeObj(centre=(1.0, 0.0))

        array_output = grid_like_object.ndarray_1d_from(grid=grid_2d)

        assert isinstance(array_output, aa.Array1D)
        assert (array_output.native == np.array([1.0, 1.0, 1.0])).all()
        assert array_output.pixel_scales == (1.0,)

    def test__grid_2d_irregular_in__output_values_projected_format(self):

        grid_2d = aa.Grid2DIrregular(grid=[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])

        grid_like_object = MockGrid1DLikeObj()

        array_output = grid_like_object.ndarray_1d_from(grid=grid_2d)

        assert isinstance(array_output, aa.ValuesIrregular)
        assert (array_output == np.array([1.0, 1.0, 1.0])).all()


class TestGrid2DToStructure:
    def test__grid_1d_in__output_values_same_format(self):

        mask = aa.Mask1D.manual(
            mask=[True, False, False, True], pixel_scales=(1.0,), sub_size=1
        )

        grid_1d = aa.Grid1D.from_mask(mask=mask)

        grid_like_object = MockGrid2DLikeObj()

        array_output = grid_like_object.ndarray_1d_from(grid=grid_1d)

        assert isinstance(array_output, aa.Array1D)
        assert (array_output.native == np.array([0.0, 1.0, 1.0, 0.0])).all()

        grid_output = grid_like_object.ndarray_2d_from(grid=grid_1d)

        assert isinstance(grid_output, aa.Grid2D)
        assert grid_output.native == pytest.approx(
            np.array([[[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]]]), 1.0e-4
        )

    def test__grid_2d_in__output_values_same_format(self):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        grid_2d = aa.Grid2D.from_mask(mask=mask)

        grid_like_object = MockGrid2DLikeObj()

        array_output = grid_like_object.ndarray_1d_from(grid=grid_2d)

        assert isinstance(array_output, aa.Array2D)
        assert (
            array_output.native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        grid_output = grid_like_object.ndarray_2d_from(grid=grid_2d)

        assert isinstance(grid_output, aa.Grid2D)
        assert (
            grid_output.native
            == np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()

    def test__grid_2d_irregular_in__output_values_same_format(self):

        grid_like_object = MockGrid2DLikeObj()

        grid_2d = aa.Grid2DIrregular(grid=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

        values_output = grid_like_object.ndarray_1d_from(grid=grid_2d)

        assert values_output.in_list == [1.0, 1.0, 1.0]

        grid_output = grid_like_object.ndarray_2d_from(grid=grid_2d)

        assert grid_output.in_list == [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]

    def test__grid_2d_iterate_in__output_values__use_iterated_array_function(self):

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

        grid_2d = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
        )

        grid_like_obj = MockGridLikeIteratorObj()

        values = grid_like_obj.ndarray_1d_from(grid=grid_2d)

        mask_sub_3 = mask.mask_new_sub_size_from(mask=mask, sub_size=3)
        grid_sub_3 = aa.Grid2D.from_mask(mask=mask_sub_3)
        values_sub_3 = ndarray_1d_from(grid=grid_sub_3, profile=None)
        values_sub_3 = grid_sub_3.structure_2d_from(result=values_sub_3)

        assert (values == values_sub_3.binned).all()

        grid_2d = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
        )

        grid_like_obj = MockGridLikeIteratorObj()

        values = grid_like_obj.ndarray_1d_from(grid=grid_2d)

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid2D.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_1d_from(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_2d_from(result=values_sub_2)

        assert (values == values_sub_2.binned).all()

        grid_2d = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.5, sub_steps=[2, 4]
        )

        iterate_obj = MockGridLikeIteratorObj()

        values = iterate_obj.ndarray_1d_from(grid=grid_2d)

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

    def test__grid_2d_iterate_in__output_values__use_iterated_grid_function(self):

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

        grid_2d = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2, 3]
        )

        grid_like_obj = MockGridLikeIteratorObj()

        values = grid_like_obj.ndarray_2d_from(grid=grid_2d)

        mask_sub_3 = mask.mask_new_sub_size_from(mask=mask, sub_size=3)
        grid_sub_3 = aa.Grid2D.from_mask(mask=mask_sub_3)
        values_sub_3 = ndarray_2d_from(grid=grid_sub_3, profile=None)
        values_sub_3 = grid_sub_3.structure_2d_from(result=values_sub_3)

        assert (values == values_sub_3.binned).all()

        grid_2d = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
        )

        grid_like_obj = MockGridLikeIteratorObj()

        values = grid_like_obj.ndarray_2d_from(grid=grid_2d)

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid2D.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_2d_from(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_2d_from(result=values_sub_2)

        assert (values == values_sub_2.binned).all()

        grid_2d = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.5, sub_steps=[2, 4]
        )

        iterate_obj = MockGridLikeIteratorObj()

        values = iterate_obj.ndarray_2d_from(grid=grid_2d)

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


class TestGrid2DToStructreList:
    def test__grid_1d_in__output_is_list__list_of_same_format(self):

        mask = aa.Mask1D.manual(
            mask=[True, False, False, True], pixel_scales=(1.0,), sub_size=1
        )

        grid_1d = aa.Grid1D.from_mask(mask=mask)

        grid_like_object = MockGrid2DLikeObj()

        array_output = grid_like_object.ndarray_1d_list_from(grid=grid_1d)

        assert isinstance(array_output[0], aa.Array1D)
        assert (array_output[0].native == np.array([[0.0, 1.0, 1.0, 0.0]])).all()

        assert isinstance(array_output[1], aa.Array1D)
        assert (array_output[1].native == np.array([[0.0, 2.0, 2.0, 0.0]])).all()

        grid_output = grid_like_object.ndarray_2d_list_from(grid=grid_1d)

        assert isinstance(grid_output[0], aa.Grid2D)
        assert grid_output[0].native == pytest.approx(
            np.array([[[0.0, 0.0], [0.0, -0.5], [0.0, 0.5], [0.0, 0.0]]]), 1.0e-4
        )

        assert isinstance(grid_output[1], aa.Grid2D)
        assert grid_output[1].native == pytest.approx(
            np.array([[[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]]]), 1.0e-4
        )

    def test__grid_2d_in__output_is_list__list_of_same_format(self):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        grid_2d = aa.Grid2D.from_mask(mask=mask)

        grid_like_object = MockGrid2DLikeObj()

        array_output = grid_like_object.ndarray_1d_list_from(grid=grid_2d)

        assert isinstance(array_output[0], aa.Array2D)
        assert (
            array_output[0].native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        assert isinstance(array_output[1], aa.Array2D)
        assert (
            array_output[1].native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 2.0, 0.0],
                    [0.0, 2.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        grid_output = grid_like_object.ndarray_2d_list_from(grid=grid_2d)

        assert isinstance(grid_output[0], aa.Grid2D)
        assert (
            grid_output[0].native
            == np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.5, -0.5], [0.5, 0.5], [0.0, 0.0]],
                    [[0.0, 0.0], [-0.5, -0.5], [-0.5, 0.5], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()

        assert isinstance(grid_output[1], aa.Grid2D)
        assert (
            grid_output[1].native
            == np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()

    def test__grid_2d_irregular_in__output_is_list__list_of_same_format(self):

        grid_like_object = MockGrid2DLikeObj()

        grid_2d = aa.Grid2DIrregular(grid=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

        grid_output = grid_like_object.ndarray_1d_list_from(grid=grid_2d)

        assert grid_output[0].in_list == [1.0, 1.0, 1.0]
        assert grid_output[1].in_list == [2.0, 2.0, 2.0]

        grid_output = grid_like_object.ndarray_2d_list_from(grid=grid_2d)

        assert grid_output[0].in_list == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        assert grid_output[1].in_list == [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]

    def test__grid_2d_iterate_in__output_is_list_of_arrays__use_maximum_sub_size_in_all_pixels(
        self
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

        grid_2d = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.05, sub_steps=[2, 3]
        )

        grid_like_obj = MockGridLikeIteratorObj()

        values = grid_like_obj.ndarray_1d_list_from(grid=grid_2d)

        mask_sub_3 = mask.mask_new_sub_size_from(mask=mask, sub_size=3)
        grid_sub_3 = aa.Grid2D.from_mask(mask=mask_sub_3)
        values_sub_3 = ndarray_1d_from(grid=grid_sub_3, profile=None)
        values_sub_3 = grid_sub_3.structure_2d_from(result=values_sub_3)

        assert (values[0] == values_sub_3.binned).all()

    def test__grid_2d_iterate_in__output_is_list_of_grids__use_maximum_sub_size_in_all_pixels(
        self
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

        grid_2d = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.05, sub_steps=[2, 3]
        )

        grid_like_obj = MockGridLikeIteratorObj()

        values = grid_like_obj.ndarray_2d_list_from(grid=grid_2d)

        mask_sub_3 = mask.mask_new_sub_size_from(mask=mask, sub_size=3)
        grid_sub_3 = aa.Grid2D.from_mask(mask=mask_sub_3)
        values_sub_3 = ndarray_2d_from(grid=grid_sub_3, profile=None)
        values_sub_3 = grid_sub_3.structure_2d_from(result=values_sub_3)

        assert (values[0][0] == values_sub_3.binned[0]).all()
        assert (values[0][1] == values_sub_3.binned[1]).all()


class TestGrid2DToVectorYX:
    def test__grid_2d_in__output_values_same_format(self):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        grid_2d = aa.Grid2D.from_mask(mask=mask)

        grid_like_object = MockGrid2DLikeObj()

        vectors_output = grid_like_object.ndarray_yx_2d_from(grid=grid_2d)

        assert isinstance(vectors_output, aa.VectorYX2D)
        assert (
            vectors_output.native
            == np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()

    def test__grid_2d_irregular_in__output_values_same_format(self):

        grid_like_object = MockGrid2DLikeObj()

        grid_2d = aa.Grid2DIrregular(grid=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

        vectors_output = grid_like_object.ndarray_yx_2d_from(grid=grid_2d)

        assert isinstance(vectors_output, aa.VectorYX2DIrregular)
        assert vectors_output.in_list == [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]

    def test__grid_2d_iterate_in__output_values__use_iterated_grid_function(self):

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

        grid_2d = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
        )

        grid_like_obj = MockGridLikeIteratorObj()

        vectors_output = grid_like_obj.ndarray_yx_2d_from(grid=grid_2d)

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid2D.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_2d_from(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_2d_from(result=values_sub_2)

        assert isinstance(vectors_output, aa.VectorYX2D)
        assert (vectors_output == values_sub_2.binned).all()

        grid_2d = aa.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.5, sub_steps=[2, 4]
        )

        iterate_obj = MockGridLikeIteratorObj()

        vectors_output = iterate_obj.ndarray_yx_2d_from(grid=grid_2d)

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = aa.Grid2D.from_mask(mask=mask_sub_2)
        values_sub_2 = ndarray_2d_from(grid=grid_sub_2, profile=None)
        values_sub_2 = grid_sub_2.structure_2d_from(result=values_sub_2)

        mask_sub_4 = mask.mask_new_sub_size_from(mask=mask, sub_size=4)
        grid_sub_4 = aa.Grid2D.from_mask(mask=mask_sub_4)
        values_sub_4 = ndarray_2d_from(grid=grid_sub_4, profile=None)
        values_sub_4 = grid_sub_4.structure_2d_from(result=values_sub_4)

        assert isinstance(vectors_output, aa.VectorYX2D)

        assert vectors_output.native[1, 1, 0] == values_sub_2.binned.native[1, 1, 0]
        assert vectors_output.native[2, 2, 0] != values_sub_2.binned.native[2, 2, 0]

        assert vectors_output.native[1, 1, 0] != values_sub_4.binned.native[1, 1, 0]
        assert vectors_output.native[2, 2, 0] == values_sub_4.binned.native[2, 2, 0]

        assert vectors_output.native[1, 1, 1] == values_sub_2.binned.native[1, 1, 1]
        assert vectors_output.native[2, 2, 1] != values_sub_2.binned.native[2, 2, 1]

        assert vectors_output.native[1, 1, 1] != values_sub_4.binned.native[1, 1, 1]
        assert vectors_output.native[2, 2, 1] == values_sub_4.binned.native[2, 2, 1]
