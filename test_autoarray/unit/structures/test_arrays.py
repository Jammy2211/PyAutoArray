import os

import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray.structures import arrays
from autoarray import exc

test_data_dir = "{}/files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestArrayAPI:
    class TestManual:
        def test__array__makes_array_without_other_inputs(self):

            arr = aa.Array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]])

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()

            arr = aa.Array.manual_1d(array=[1.0, 2.0, 3.0, 4.0], shape_2d=(2, 2))

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()

            arr = aa.Array.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape_2d=(2, 3), store_in_1d=True
            )

            assert type(arr) == arrays.Array
            assert (arr == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()
            assert (arr.in_2d == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()

            arr = aa.Array.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape_2d=(2, 3), store_in_1d=False
            )

            assert type(arr) == arrays.Array
            assert (arr == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (arr.in_2d == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()

        def test__array__makes_array_with_pixel_scale(self):

            arr = aa.Array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)

            arr = aa.Array.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0],
                shape_2d=(2, 2),
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

            arr = aa.Array.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape_2d=(2, 3),
                pixel_scales=(2.0, 3.0),
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()
            assert arr.pixel_scales == (2.0, 3.0)
            assert arr.geometry.origin == (0.0, 0.0)

        def test__array__makes_with_pixel_scale_and_sub_size(self):

            arr = aa.Array.manual_2d(
                array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, sub_size=1
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 1

            arr = aa.Array.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0],
                shape_2d=(1, 1),
                pixel_scales=1.0,
                sub_size=2,
                origin=(0.0, 1.0),
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)
            assert arr.mask.sub_size == 2

            arr = aa.Array.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                shape_2d=(2, 1),
                pixel_scales=2.0,
                sub_size=2,
            )

            assert type(arr) == arrays.Array
            assert (
                arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert (
                arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            ).all()
            assert (arr.in_2d_binned == np.array([[2.5], [6.5]])).all()
            assert (arr.in_1d_binned == np.array([2.5, 6.5])).all()
            assert arr.pixel_scales == (2.0, 2.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 2

    class TestFull:
        def test__array__makes_array_without_other_inputs(self):

            arr = aa.Array.ones(shape_2d=(2, 2))

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()

            arr = aa.Array.full(fill_value=2.0, shape_2d=(2, 2), store_in_1d=True)

            assert type(arr) == arrays.Array
            assert (arr == np.array([2.0, 2.0, 2.0, 2.0])).all()
            assert (arr.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (arr.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()

            arr = aa.Array.full(fill_value=2.0, shape_2d=(2, 2), store_in_1d=False)

            assert type(arr) == arrays.Array
            assert (arr == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (arr.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (arr.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            arr = aa.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)

            arr = aa.Array.full(
                fill_value=2.0, shape_2d=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0)
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (arr.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            arr = aa.Array.full(
                fill_value=1.0, shape_2d=(1, 4), pixel_scales=1.0, sub_size=1
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 1.0, 1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 1

            arr = aa.Array.full(
                fill_value=2.0,
                shape_2d=(1, 1),
                pixel_scales=1.0,
                sub_size=2,
                origin=(0.0, 1.0),
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (arr.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)
            assert arr.mask.sub_size == 2

    class TestOnesZeros:
        def test__array__makes_array_without_other_inputs(self):

            arr = aa.Array.ones(shape_2d=(2, 2))

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()

            arr = aa.Array.zeros(shape_2d=(2, 2), store_in_1d=True)

            assert type(arr) == arrays.Array
            assert (arr == np.array([0.0, 0.0, 0.0, 0.0])).all()
            assert (arr.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()

            arr = aa.Array.zeros(shape_2d=(2, 2), store_in_1d=False)

            assert type(arr) == arrays.Array
            assert (arr == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            arr = aa.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)

            arr = aa.Array.zeros(shape_2d=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0))

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            arr = aa.Array.ones(shape_2d=(1, 4), pixel_scales=1.0, sub_size=1)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 1.0, 1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 1

            arr = aa.Array.zeros(
                shape_2d=(1, 1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0)
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)
            assert arr.mask.sub_size == 2

    class TestFromFits:
        def test__array__makes_array_without_other_inputs(self):

            arr = aa.Array.from_fits(file_path=test_data_dir + "3x3_ones.fits", hdu=0)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.ones((3, 3))).all()
            assert (arr.in_1d == np.ones(9)).all()

            arr = aa.Array.from_fits(
                file_path=test_data_dir + "4x3_ones.fits", hdu=0, store_in_1d=True
            )

            assert type(arr) == arrays.Array
            assert (arr == np.ones((12,))).all()
            assert (arr.in_2d == np.ones((4, 3))).all()
            assert (arr.in_1d == np.ones((12,))).all()

            arr = aa.Array.from_fits(
                file_path=test_data_dir + "4x3_ones.fits", hdu=0, store_in_1d=False
            )

            assert type(arr) == arrays.Array
            assert (arr == np.ones((4, 3))).all()
            assert (arr.in_2d == np.ones((4, 3))).all()
            assert (arr.in_1d == np.ones((12,))).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            arr = aa.Array.from_fits(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0, pixel_scales=1.0
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.ones((3, 3))).all()
            assert (arr.in_1d == np.ones(9)).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)

            arr = aa.Array.from_fits(
                file_path=test_data_dir + "4x3_ones.fits",
                hdu=0,
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.ones((4, 3))).all()
            assert (arr.in_1d == np.ones((12,))).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            arr = aa.Array.from_fits(
                file_path=test_data_dir + "3x3_ones.fits",
                hdu=0,
                pixel_scales=1.0,
                sub_size=1,
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.ones((3, 3))).all()
            assert (arr.in_1d == np.ones(9)).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 1

            arr = aa.Array.from_fits(
                file_path=test_data_dir + "4x3_ones.fits",
                hdu=0,
                pixel_scales=1.0,
                sub_size=1,
                origin=(0.0, 1.0),
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.ones((4, 3))).all()
            assert (arr.in_1d == np.ones(12)).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)
            assert arr.mask.sub_size == 1

    class TestFromYXValues:
        def test__use_manual_array_values__returns_input_array(self):

            arr = aa.Array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)

            y = arr.mask.geometry.unmasked_grid[:, 0]
            x = arr.mask.geometry.unmasked_grid[:, 1]
            arr_via_yx = aa.Array.manual_yx_and_values(
                y=y, x=x, values=arr, shape_2d=arr.shape_2d, pixel_scales=1.0
            )

            assert (arr == arr_via_yx).all()

            arr = aa.Array.manual_2d(
                array=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], pixel_scales=1.0
            )

            y = arr.mask.geometry.unmasked_grid[:, 0]
            x = arr.mask.geometry.unmasked_grid[:, 1]

            arr_via_yx = aa.Array.manual_yx_and_values(
                y=y, x=x, values=arr, shape_2d=arr.shape_2d, pixel_scales=1.0
            )

            assert (arr == arr_via_yx).all()

            arr = aa.Array.manual_2d(
                array=[[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]], pixel_scales=1.0
            )

            y = arr.mask.geometry.unmasked_grid[:, 0]
            x = arr.mask.geometry.unmasked_grid[:, 1]

            arr_via_yx = aa.Array.manual_yx_and_values(
                y=y, x=x, values=arr, shape_2d=arr.shape_2d, pixel_scales=1.0
            )

            assert (arr == arr_via_yx).all()

        def test__use_input_values_which_swap_values_from_top_left_notation(self):

            arr = aa.Array.manual_yx_and_values(
                y=[0.5, 0.5, -0.5, -0.5],
                x=[-0.5, 0.5, -0.5, 0.5],
                values=[1.0, 2.0, 3.0, 4.0],
                shape_2d=(2, 2),
                pixel_scales=1.0,
            )

            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

            arr = aa.Array.manual_yx_and_values(
                y=[-0.5, 0.5, 0.5, -0.5],
                x=[-0.5, 0.5, -0.5, 0.5],
                values=[1.0, 2.0, 3.0, 4.0],
                shape_2d=(2, 2),
                pixel_scales=1.0,
            )

            assert (arr.in_2d == np.array([[3.0, 2.0], [1.0, 4.0]])).all()

            arr = aa.Array.manual_yx_and_values(
                y=[-0.5, 0.5, 0.5, -0.5],
                x=[0.5, 0.5, -0.5, -0.5],
                values=[1.0, 2.0, 3.0, 4.0],
                shape_2d=(2, 2),
                pixel_scales=1.0,
            )

            assert (arr.in_2d == np.array([[4.0, 2.0], [1.0, 3.0]])).all()

            arr = aa.Array.manual_yx_and_values(
                y=[1.0, 1.0, 0.0, 0.0, -1.0, -1.0],
                x=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
                values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape_2d=(3, 2),
                pixel_scales=1.0,
            )

            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])).all()

            arr = aa.Array.manual_yx_and_values(
                y=[0.0, 1.0, -1.0, 0.0, -1.0, 1.0],
                x=[-0.5, 0.5, 0.5, 0.5, -0.5, -0.5],
                values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape_2d=(3, 2),
                pixel_scales=1.0,
            )

            assert (arr.in_2d == np.array([[3.0, 2.0], [6.0, 4.0], [5.0, 1.0]])).all()


class TestMaskedArrayAPI:
    class TestManual:
        def test__array__makes_array_with_pixel_scale(self):

            mask = aa.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
            arr = aa.MaskedArray.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)

            mask = aa.Mask.manual(
                mask=[[False, False], [True, False]],
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )
            arr = aa.MaskedArray.manual_1d(array=[1.0, 2.0, 4.0], mask=mask)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 4.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

            mask = aa.Mask.manual(
                mask=[[False, False], [True, False]],
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )
            arr = aa.MaskedArray.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 4.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

            mask = aa.Mask.manual(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)
            arr = aa.MaskedArray.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0], mask=mask, store_in_1d=True
            )

            assert type(arr) == arrays.Array
            assert (arr == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert (
                arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
            ).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert (arr.in_2d_binned == np.array([[2.5], [0.0]])).all()
            assert (arr.in_1d_binned == np.array([2.5])).all()
            assert arr.pixel_scales == (2.0, 2.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 2

            arr = aa.MaskedArray.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0], mask=mask, store_in_1d=False
            )

            assert type(arr) == arrays.Array
            assert (
                arr == np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
            ).all()
            assert (
                arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
            ).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert (arr.in_2d_binned == np.array([[2.5], [0.0]])).all()
            assert (arr.in_1d_binned == np.array([2.5])).all()
            assert arr.pixel_scales == (2.0, 2.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 2

        def test__manual_2d__exception_raised_if_input_array_is_2d_and_not_sub_shape_of_mask(
            self
        ):

            with pytest.raises(exc.ArrayException):
                mask = aa.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)
                aa.MaskedArray.manual_2d(array=[[1.0], [3.0]], mask=mask)

            with pytest.raises(exc.ArrayException):
                mask = aa.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=2)
                aa.MaskedArray.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

            with pytest.raises(exc.ArrayException):
                mask = aa.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=2)
                aa.MaskedArray.manual_2d(
                    array=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], mask=mask
                )

        def test__exception_raised_if_input_array_is_1d_and_not_number_of_masked_sub_pixels(
            self
        ):

            with pytest.raises(exc.ArrayException):
                mask = aa.Mask.manual(mask=[[False, False], [True, False]], sub_size=1)
                aa.MaskedArray.manual_1d(array=[1.0, 2.0, 3.0, 4.0], mask=mask)

            with pytest.raises(exc.ArrayException):
                mask = aa.Mask.manual(mask=[[False, False], [True, False]], sub_size=1)
                aa.MaskedArray.manual_1d(array=[1.0, 2.0], mask=mask)

            with pytest.raises(exc.ArrayException):
                mask = aa.Mask.manual(mask=[[False, True], [True, True]], sub_size=2)
                aa.MaskedArray.manual_1d(array=[1.0, 2.0, 4.0], mask=mask)

            with pytest.raises(exc.ArrayException):
                mask = aa.Mask.manual(mask=[[False, True], [True, True]], sub_size=2)
                aa.MaskedArray.manual_1d(array=[1.0, 2.0, 3.0, 4.0, 5.0], mask=mask)

    class TestFull:
        def test__makes_array_using_mask(self):

            mask = aa.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
            arr = aa.MaskedArray.ones(mask=mask)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)

            mask = aa.Mask.manual(
                mask=[[False, False], [True, False]],
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )
            arr = aa.MaskedArray.full(fill_value=2.0, mask=mask)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[2.0, 2.0], [0.0, 2.0]])).all()
            assert (arr.in_1d == np.array([2.0, 2.0, 2.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

            mask = aa.Mask.manual(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)
            arr = aa.MaskedArray.full(fill_value=3.0, mask=mask, store_in_1d=True)

            assert type(arr) == arrays.Array
            assert (arr == np.array([3.0, 3.0, 3.0, 3.0])).all()
            assert (
                arr.in_2d == np.array([[3.0, 3.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0]])
            ).all()
            assert (arr.in_1d == np.array([3.0, 3.0, 3.0, 3.0])).all()
            assert (arr.in_2d_binned == np.array([[3.0], [0.0]])).all()
            assert (arr.in_1d_binned == np.array([3.0])).all()
            assert arr.pixel_scales == (2.0, 2.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 2

            arr = aa.MaskedArray.full(fill_value=3.0, mask=mask, store_in_1d=False)

            assert type(arr) == arrays.Array
            assert (
                arr == np.array([[3.0, 3.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0]])
            ).all()
            assert (
                arr.in_2d == np.array([[3.0, 3.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0]])
            ).all()
            assert (arr.in_1d == np.array([3.0, 3.0, 3.0, 3.0])).all()
            assert (arr.in_2d_binned == np.array([[3.0], [0.0]])).all()
            assert (arr.in_1d_binned == np.array([3.0])).all()
            assert arr.pixel_scales == (2.0, 2.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 2

    class TestOnesZeros:
        def test__makes_array_using_mask(self):

            mask = aa.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
            arr = aa.MaskedArray.ones(mask=mask)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)

            mask = aa.Mask.manual(
                mask=[[False, False], [True, False]],
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )
            arr = aa.MaskedArray.zeros(mask=mask)

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_1d == np.array([0.0, 0.0, 0.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

            mask = aa.Mask.manual(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)
            arr = aa.MaskedArray.ones(mask=mask, store_in_1d=True)

            assert type(arr) == arrays.Array
            assert (arr == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert (
                arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
            ).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert (arr.in_2d_binned == np.array([[1.0], [0.0]])).all()
            assert (arr.in_1d_binned == np.array([1.0])).all()
            assert arr.pixel_scales == (2.0, 2.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 2

            arr = aa.MaskedArray.ones(mask=mask, store_in_1d=False)

            assert type(arr) == arrays.Array
            assert (
                arr == np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
            ).all()
            assert (
                arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
            ).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert (arr.in_2d_binned == np.array([[1.0], [0.0]])).all()
            assert (arr.in_1d_binned == np.array([1.0])).all()
            assert arr.pixel_scales == (2.0, 2.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 2

    class TestFromFits:
        def test__array_from_fits_uses_mask(self):

            mask = aa.Mask.unmasked(shape_2d=(3, 3), pixel_scales=1.0)
            arr = aa.MaskedArray.from_fits(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0, mask=mask
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == np.ones((3, 3))).all()
            assert (arr.in_1d == np.ones(9)).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 1

            mask = aa.Mask.manual(
                [
                    [False, False, False],
                    [False, False, False],
                    [True, False, True],
                    [False, False, False],
                ],
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )
            arr = aa.MaskedArray.from_fits(
                file_path=test_data_dir + "4x3_ones.fits",
                hdu=0,
                mask=mask,
                store_in_1d=True,
            )

            assert type(arr) == arrays.Array
            assert (
                arr == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            ).all()
            assert (
                arr.in_2d
                == np.array(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
                )
            ).all()
            assert (
                arr.in_1d
                == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            ).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)
            assert arr.mask.sub_size == 1

            arr = aa.MaskedArray.from_fits(
                file_path=test_data_dir + "4x3_ones.fits",
                hdu=0,
                mask=mask,
                store_in_1d=False,
            )

            assert type(arr) == arrays.Array
            assert (
                arr
                == np.array(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
                )
            ).all()
            assert (
                arr.in_2d
                == np.array(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
                )
            ).all()
            assert (
                arr.in_1d
                == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            ).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)
            assert arr.mask.sub_size == 1


class TestArray:
    class TestConstructorMethods:
        def test__constructor_class_method_in_2d__store_in_1d(self):

            arr = arrays.Array.manual_2d(
                array=np.ones((3, 3)),
                sub_size=1,
                pixel_scales=(1.0, 1.0),
                store_in_1d=True,
            )

            assert (arr == np.ones((9,))).all()
            assert (arr.in_1d == np.ones((9,))).all()
            assert (arr.in_2d == np.ones((3, 3))).all()
            assert (arr.in_1d_binned == np.ones((9,))).all()
            assert (arr.in_2d_binned == np.ones((3, 3))).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_2d_scaled == pytest.approx((3.0, 3.0))
            assert arr.geometry.scaled_maxima == (1.5, 1.5)
            assert arr.geometry.scaled_minima == (-1.5, -1.5)

            arr = arrays.Array.manual_2d(
                array=np.ones((4, 4)),
                sub_size=2,
                pixel_scales=(0.1, 0.1),
                store_in_1d=True,
            )

            assert (arr == np.ones((16,))).all()
            assert (arr.in_1d == np.ones((16,))).all()
            assert (arr.in_2d == np.ones((4, 4))).all()
            assert (arr.in_1d_binned == np.ones((4,))).all()
            assert (arr.in_2d_binned == np.ones((2, 2))).all()
            assert arr.pixel_scales == (0.1, 0.1)
            assert arr.geometry.central_pixel_coordinates == (0.5, 0.5)
            assert arr.geometry.shape_2d_scaled == pytest.approx((0.2, 0.2))
            assert arr.geometry.scaled_maxima == pytest.approx((0.1, 0.1), 1e-4)
            assert arr.geometry.scaled_minima == pytest.approx((-0.1, -0.1), 1e-4)

            arr = arrays.Array.manual_2d(
                array=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
                pixel_scales=(0.1, 0.1),
                sub_size=2,
                origin=(1.0, 1.0),
                store_in_1d=True,
            )

            assert (arr == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
            assert arr.shape_2d == (2, 1)
            assert arr.sub_shape_2d == (4, 2)
            assert (
                arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            ).all()
            assert (
                arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert arr.in_2d_binned.shape == (2, 1)
            assert (arr.in_1d_binned == np.array([2.5, 6.5])).all()
            assert (arr.in_2d_binned == np.array([[2.5], [6.5]])).all()
            assert arr.pixel_scales == (0.1, 0.1)
            assert arr.geometry.central_pixel_coordinates == (0.5, 0.0)
            assert arr.geometry.shape_2d_scaled == pytest.approx((0.2, 0.1))
            assert arr.geometry.scaled_maxima == pytest.approx((1.1, 1.05), 1e-4)
            assert arr.geometry.scaled_minima == pytest.approx((0.9, 0.95), 1e-4)

            arr = arrays.Array.manual_2d(
                array=np.ones((3, 3)),
                pixel_scales=(2.0, 1.0),
                sub_size=1,
                origin=(-1.0, -2.0),
                store_in_1d=True,
            )

            assert arr == pytest.approx(np.ones((9,)), 1e-4)
            assert arr.in_1d == pytest.approx(np.ones((9,)), 1e-4)
            assert arr.in_2d == pytest.approx(np.ones((3, 3)), 1e-4)
            assert arr.pixel_scales == (2.0, 1.0)
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_2d_scaled == pytest.approx((6.0, 3.0))
            assert arr.geometry.origin == (-1.0, -2.0)
            assert arr.geometry.scaled_maxima == pytest.approx((2.0, -0.5), 1e-4)
            assert arr.geometry.scaled_minima == pytest.approx((-4.0, -3.5), 1e-4)

        def test__constructor_class_method_in_1d__store_in_1d(self):
            arr = arrays.Array.manual_1d(
                array=np.ones((9,)),
                shape_2d=(3, 3),
                pixel_scales=(2.0, 1.0),
                sub_size=1,
                origin=(-1.0, -2.0),
                store_in_1d=True,
            )

            assert arr == pytest.approx(np.ones((9,)), 1e-4)
            assert arr.in_1d == pytest.approx(np.ones((9,)), 1e-4)
            assert arr.in_2d == pytest.approx(np.ones((3, 3)), 1e-4)
            assert arr.pixel_scales == (2.0, 1.0)
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_2d_scaled == pytest.approx((6.0, 3.0))
            assert arr.geometry.origin == (-1.0, -2.0)
            assert arr.geometry.scaled_maxima == pytest.approx((2.0, -0.5), 1e-4)
            assert arr.geometry.scaled_minima == pytest.approx((-4.0, -3.5), 1e-4)

        def test__constructor_class_method_in_2d__store_in_2d(self):

            arr = arrays.Array.manual_2d(
                array=np.ones((3, 3)),
                sub_size=1,
                pixel_scales=(1.0, 1.0),
                store_in_1d=False,
            )

            assert (arr == np.ones((3, 3))).all()
            assert (arr.in_1d == np.ones((9,))).all()
            assert (arr.in_2d == np.ones((3, 3))).all()
            assert (arr.in_1d_binned == np.ones((9,))).all()
            assert (arr.in_2d_binned == np.ones((3, 3))).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_2d_scaled == pytest.approx((3.0, 3.0))
            assert arr.geometry.scaled_maxima == (1.5, 1.5)
            assert arr.geometry.scaled_minima == (-1.5, -1.5)

            arr = arrays.Array.manual_2d(
                array=np.ones((4, 4)),
                sub_size=2,
                pixel_scales=(0.1, 0.1),
                store_in_1d=False,
            )

            assert (arr == np.ones((4, 4))).all()
            assert (arr.in_1d == np.ones((16,))).all()
            assert (arr.in_2d == np.ones((4, 4))).all()
            assert (arr.in_1d_binned == np.ones((4,))).all()
            assert (arr.in_2d_binned == np.ones((2, 2))).all()
            assert arr.pixel_scales == (0.1, 0.1)
            assert arr.geometry.central_pixel_coordinates == (0.5, 0.5)
            assert arr.geometry.shape_2d_scaled == pytest.approx((0.2, 0.2))
            assert arr.geometry.scaled_maxima == pytest.approx((0.1, 0.1), 1e-4)
            assert arr.geometry.scaled_minima == pytest.approx((-0.1, -0.1), 1e-4)

            arr = arrays.Array.manual_2d(
                array=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
                pixel_scales=(0.1, 0.1),
                sub_size=2,
                origin=(1.0, 1.0),
                store_in_1d=False,
            )

            assert (
                arr == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert arr.shape_2d == (2, 1)
            assert arr.sub_shape_2d == (4, 2)
            assert (
                arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            ).all()
            assert (
                arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert arr.in_2d_binned.shape == (2, 1)
            assert (arr.in_1d_binned == np.array([2.5, 6.5])).all()
            assert (arr.in_2d_binned == np.array([[2.5], [6.5]])).all()
            assert arr.pixel_scales == (0.1, 0.1)
            assert arr.geometry.central_pixel_coordinates == (0.5, 0.0)
            assert arr.geometry.shape_2d_scaled == pytest.approx((0.2, 0.1))
            assert arr.geometry.scaled_maxima == pytest.approx((1.1, 1.05), 1e-4)
            assert arr.geometry.scaled_minima == pytest.approx((0.9, 0.95), 1e-4)

            arr = arrays.Array.manual_2d(
                array=np.ones((3, 3)),
                pixel_scales=(2.0, 1.0),
                sub_size=1,
                origin=(-1.0, -2.0),
                store_in_1d=False,
            )

            assert arr == pytest.approx(np.ones((3, 3)), 1e-4)
            assert arr.in_1d == pytest.approx(np.ones((9,)), 1e-4)
            assert arr.in_2d == pytest.approx(np.ones((3, 3)), 1e-4)
            assert arr.pixel_scales == (2.0, 1.0)
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_2d_scaled == pytest.approx((6.0, 3.0))
            assert arr.geometry.origin == (-1.0, -2.0)
            assert arr.geometry.scaled_maxima == pytest.approx((2.0, -0.5), 1e-4)
            assert arr.geometry.scaled_minima == pytest.approx((-4.0, -3.5), 1e-4)

        def test__constructor_class_method_in_1d__store_in_2d(self):
            arr = arrays.Array.manual_1d(
                array=np.ones((9,)),
                shape_2d=(3, 3),
                pixel_scales=(2.0, 1.0),
                sub_size=1,
                origin=(-1.0, -2.0),
                store_in_1d=False,
            )

            assert arr == pytest.approx(np.ones((3, 3)), 1e-4)
            assert arr.in_1d == pytest.approx(np.ones((9,)), 1e-4)
            assert arr.in_2d == pytest.approx(np.ones((3, 3)), 1e-4)
            assert arr.pixel_scales == (2.0, 1.0)
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_2d_scaled == pytest.approx((6.0, 3.0))
            assert arr.geometry.origin == (-1.0, -2.0)
            assert arr.geometry.scaled_maxima == pytest.approx((2.0, -0.5), 1e-4)
            assert arr.geometry.scaled_minima == pytest.approx((-4.0, -3.5), 1e-4)

    class TestNewArrays:
        def test__pad__compare_to_array_util(self):
            array_2d = np.ones((5, 5))
            array_2d[2, 2] = 2.0

            arr = arrays.Array.manual_2d(
                array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
            )

            arr = arr.resized_from_new_shape(new_shape=(7, 7))

            arr_resized_manual = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == arr_resized_manual).all()
            assert arr.mask.pixel_scales == (1.0, 1.0)

        def test__trim__compare_to_array_util(self):
            array_2d = np.ones((5, 5))
            array_2d[2, 2] = 2.0

            arr = arrays.Array.manual_2d(
                array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
            )

            arr = arr.resized_from_new_shape(new_shape=(3, 3))

            arr_resized_manual = np.array(
                [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == arr_resized_manual).all()
            assert arr.mask.pixel_scales == (1.0, 1.0)

        def test__padded_from_kernel_shape__padded_edge_of_zeros_where_extra_psf_blurring_is_performed(
            self
        ):

            array_2d = np.ones((5, 5))
            array_2d[2, 2] = 2.0

            arr = arrays.Array.manual_2d(
                array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
            )

            new_arr = arr.padded_from_kernel_shape(kernel_shape_2d=(3, 3))

            assert type(new_arr) == arrays.Array
            assert new_arr.in_2d[0, 0] == 0.0
            assert new_arr.shape_2d == (7, 7)
            assert new_arr.mask.pixel_scales == (1.0, 1.0)

            new_arr = arr.padded_from_kernel_shape(kernel_shape_2d=(5, 5))

            assert type(new_arr) == arrays.Array
            assert new_arr.in_2d[0, 0] == 0.0
            assert new_arr.shape_2d == (9, 9)
            assert new_arr.mask.pixel_scales == (1.0, 1.0)

            array_2d = np.ones((9, 9))
            array_2d[4, 4] = 2.0

            arr = arrays.Array.manual_2d(
                array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
            )

            new_arr = arr.padded_from_kernel_shape(kernel_shape_2d=(7, 7))

            assert type(new_arr) == arrays.Array
            assert new_arr.in_2d[0, 0] == 0.0
            assert new_arr.shape_2d == (15, 15)
            assert new_arr.mask.pixel_scales == (1.0, 1.0)

        def test__trimmed_from_kernel_shape__trim_edges_where_extra_psf_blurring_is_performed(
            self
        ):
            array_2d = np.ones((5, 5))
            array_2d[2, 2] = 2.0

            arr = arrays.Array.manual_2d(
                array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
            )

            new_arr = arr.trimmed_from_kernel_shape(kernel_shape_2d=(3, 3))

            assert type(new_arr) == arrays.Array
            assert (
                new_arr.in_2d
                == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
            ).all()
            assert new_arr.mask.pixel_scales == (1.0, 1.0)

            new_arr = arr.trimmed_from_kernel_shape(kernel_shape_2d=(5, 5))

            assert type(new_arr) == arrays.Array
            assert (new_arr.in_2d == np.array([[2.0]])).all()
            assert new_arr.mask.pixel_scales == (1.0, 1.0)

            array_2d = np.ones((9, 9))
            array_2d[4, 4] = 2.0

            arr = arrays.Array.manual_2d(
                array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
            )

            new_arr = arr.trimmed_from_kernel_shape(kernel_shape_2d=(7, 7))

            assert type(new_arr) == arrays.Array
            assert (
                new_arr.in_2d
                == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
            ).all()
            assert new_arr.mask.pixel_scales == (1.0, 1.0)

        def test__zoomed__2d_array_zoomed__uses_the_limits_of_the_mask(self):

            array_2d = np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            )

            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, True, True, True],
                        [True, False, False, True],
                        [True, False, False, True],
                        [True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            arr_masked = aa.MaskedArray.manual_2d(array=array_2d, mask=mask)

            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

            assert (arr_zoomed.in_2d == np.array([[6.0, 7.0], [10.0, 11.0]])).all()

            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, True, True, True],
                        [True, False, False, False],
                        [True, False, False, False],
                        [True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            arr_masked = aa.MaskedArray.manual_2d(array=array_2d, mask=mask)
            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
            assert (
                arr_zoomed.in_2d == np.array([[6.0, 7.0, 8.0], [10.0, 11.0, 12.0]])
            ).all()

            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, True, True, True],
                        [True, False, False, True],
                        [True, False, False, True],
                        [True, False, False, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            arr_masked = aa.MaskedArray.manual_2d(array=array_2d, mask=mask)
            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
            assert (
                arr_zoomed.in_2d == np.array([[6.0, 7.0], [10.0, 11.0], [14.0, 15.0]])
            ).all()

            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, True, True, True],
                        [True, False, False, True],
                        [False, False, False, True],
                        [True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            arr_masked = aa.MaskedArray.manual_2d(array=array_2d, mask=mask)
            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

            assert (
                arr_zoomed.in_2d == np.array([[0.0, 6.0, 7.0], [9.0, 10.0, 11.0]])
            ).all()

            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, False, True, True],
                        [True, False, False, True],
                        [True, False, False, True],
                        [True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            arr_masked = aa.MaskedArray.manual_2d(array=array_2d, mask=mask)
            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
            assert (
                arr_zoomed.in_2d == np.array([[2.0, 0.0], [6.0, 7.0], [10.0, 11.0]])
            ).all()

            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, True, True, True],
                        [True, False, False, True],
                        [True, False, False, True],
                        [True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            arr_masked = aa.MaskedArray.manual_2d(array=array_2d, mask=mask)
            arr_zoomed = arr_masked.zoomed_around_mask(buffer=1)

            assert (
                arr_zoomed.in_2d
                == np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 6.0, 7.0, 0.0],
                        [0.0, 10.0, 11.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ).all()

        def test__zoomed_2d_array_zoomed__centre_is_updated_using_original_mask(self):

            array_2d = np.ones(shape=(4, 4))

            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, True, True, True],
                        [True, False, False, True],
                        [True, False, False, True],
                        [True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            arr_masked = aa.MaskedArray.manual_2d(array=array_2d, mask=mask)

            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

            assert arr_zoomed.mask.geometry.origin == (0.0, 0.0)

            array_2d = np.ones(shape=(6, 6))

            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, True, True, True, True, True],
                        [True, True, True, False, False, True],
                        [True, True, True, False, False, True],
                        [True, True, True, True, True, True],
                        [True, True, True, True, True, True],
                        [True, True, True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            arr_masked = aa.MaskedArray.manual_2d(array=array_2d, mask=mask)

            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

            assert arr_zoomed.mask.geometry.origin == (1.0, 1.0)

            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, True, True, True, True, True],
                        [True, True, True, True, True, True],
                        [True, True, True, False, False, True],
                        [True, True, True, False, False, True],
                        [True, True, True, True, True, True],
                        [True, True, True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            arr_masked = aa.MaskedArray.manual_2d(array=array_2d, mask=mask)

            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

            assert arr_zoomed.mask.geometry.origin == (0.0, 1.0)

        def test__zoomed__array_extent__uses_the_limits_of_the_unzoomed_mask(self):

            array_2d = np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            )

            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, True, True, False],
                        [True, False, False, True],
                        [True, False, False, True],
                        [True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 2.0),
                sub_size=1,
            )

            arr_masked = aa.MaskedArray.manual_2d(array=array_2d, mask=mask)

            extent = arr_masked.extent_of_zoomed_array(buffer=1)

            assert extent == pytest.approx(np.array([-4.0, 6.0, -2.0, 3.0]), 1.0e-4)

        def test__binned_up__compare_all_extract_methods_to_array_util(self):
            array_2d = np.array(
                [
                    [1.0, 6.0, 3.0, 7.0, 3.0, 2.0],
                    [2.0, 5.0, 3.0, 7.0, 7.0, 7.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ]
            )

            arr = arrays.Array.manual_2d(
                array=array_2d, sub_size=1, pixel_scales=(0.1, 0.1)
            )

            arr_binned_util = aa.util.binning.bin_array_2d_via_mean(
                array_2d=array_2d, bin_up_factor=4
            )
            arr_binned = arr.binned_from_bin_up_factor(bin_up_factor=4, method="mean")
            assert (arr_binned.in_2d == arr_binned_util).all()
            assert arr_binned.pixel_scales == (0.4, 0.4)

            arr_binned_util = aa.util.binning.bin_array_2d_via_quadrature(
                array_2d=array_2d, bin_up_factor=4
            )
            arr_binned = arr.binned_from_bin_up_factor(
                bin_up_factor=4, method="quadrature"
            )
            assert (arr_binned.in_2d == arr_binned_util).all()
            assert arr_binned.pixel_scales == (0.4, 0.4)

            arr_binned_util = aa.util.binning.bin_array_2d_via_sum(
                array_2d=array_2d, bin_up_factor=4
            )
            arr_binned = arr.binned_from_bin_up_factor(bin_up_factor=4, method="sum")
            assert (arr_binned.in_2d == arr_binned_util).all()
            assert arr_binned.pixel_scales == (0.4, 0.4)

        def test__binned_up__invalid_method__raises_exception(self):
            array_2d = np.array(
                [
                    [1.0, 6.0, 3.0, 7.0, 3.0, 2.0],
                    [2.0, 5.0, 3.0, 7.0, 7.0, 7.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ]
            )

            array_2d = arrays.Array.manual_2d(
                array=array_2d, sub_size=1, pixel_scales=(0.1, 0.1)
            )
            with pytest.raises(exc.ArrayException):
                array_2d.binned_from_bin_up_factor(bin_up_factor=4, method="wrong")

    class TestOutputToFits:
        def test__output_to_fits(self):

            arr = aa.Array.from_fits(file_path=test_data_dir + "3x3_ones.fits", hdu=0)

            output_data_dir = "{}/files/array/output_test/".format(
                os.path.dirname(os.path.realpath(__file__))
            )
            if os.path.exists(output_data_dir):
                shutil.rmtree(output_data_dir)

            os.makedirs(output_data_dir)

            arr.output_to_fits(file_path=output_data_dir + "array.fits")

            array_from_out = aa.Array.from_fits(
                file_path=output_data_dir + "array.fits", hdu=0
            )

            assert (array_from_out.in_2d == np.ones((3, 3))).all()

        def test__output_to_fits__shapes_of_arrays_are_2d(self):

            arr = aa.Array.from_fits(file_path=test_data_dir + "3x3_ones.fits", hdu=0)

            output_data_dir = "{}/files/array/output_test/".format(
                os.path.dirname(os.path.realpath(__file__))
            )
            if os.path.exists(output_data_dir):
                shutil.rmtree(output_data_dir)

            os.makedirs(output_data_dir)

            arr.output_to_fits(file_path=output_data_dir + "array.fits")

            array_from_out = aa.util.array.numpy_array_2d_from_fits(
                file_path=output_data_dir + "array.fits", hdu=0
            )

            assert (array_from_out == np.ones((3, 3))).all()

            mask = aa.Mask.unmasked(shape_2d=(3, 3), pixel_scales=0.1)

            masked_array = aa.MaskedArray(array=arr, mask=mask)

            masked_array.output_to_fits(file_path=output_data_dir + "masked_array.fits")

            masked_array_from_out = aa.util.array.numpy_array_2d_from_fits(
                file_path=output_data_dir + "masked_array.fits", hdu=0
            )

            assert (masked_array_from_out == np.ones((3, 3))).all()


test_values_dir = "{}/files/values/".format(os.path.dirname(os.path.realpath(__file__)))


class TestValues:
    def test__indexes_give_entries_where_list_begin_and_end(self):

        values = aa.Values(values=[[0.0]])

        assert values.lower_indexes == [0]
        assert values.upper_indexes == [1]

        values = aa.Values(values=[[0.0, 0.0]])

        assert values.lower_indexes == [0]
        assert values.upper_indexes == [2]

        values = aa.Values(values=[[0.0, 0.0], [0.0]])

        assert values.lower_indexes == [0, 2]
        assert values.upper_indexes == [2, 3]

        values = aa.Values(values=[[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0], [0.0]])

        assert values.lower_indexes == [0, 2, 5, 7]
        assert values.upper_indexes == [2, 5, 7, 8]

    def test__input_as_list__convert_correctly(self):

        values = aa.Values(values=[1.0, -1.0])

        assert type(values) == arrays.Values
        assert (values == np.array([1.0, -1.0])).all()
        assert values.in_list == [[1.0, -1.0]]

        values = aa.Values(values=[[1.0], [-1.0]])

        assert type(values) == arrays.Values
        assert (values == np.array([1.0, -1.0])).all()
        assert values.in_list == [[1.0], [-1.0]]

    def test__values_from_arr_1d(self):

        values = aa.Values(values=[[1.0, 2.0]])

        values_from_1d = values.values_from_arr_1d(arr_1d=np.array([1.0, 2.0]))

        assert values_from_1d.in_list == [[1.0, 2.0]]

        values = aa.Values(values=[[1.0, 2.0], [3.0]])

        values_from_1d = values.values_from_arr_1d(arr_1d=np.array([1.0, 2.0, 3.0]))

        assert values_from_1d.in_list == [[1.0, 2.0], [3.0]]

    def test__coordinates_from_grid_1d(self):

        values = aa.Values(values=[[1.0, 2.0]])

        coordinate_from_1d = values.coordinates_from_grid_1d(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert coordinate_from_1d.in_list == [[(1.0, 1.0), (2.0, 2.0)]]

        values = aa.Values(values=[[1.0, 2.0], [3.0]])

        coordinate_from_1d = values.coordinates_from_grid_1d(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert coordinate_from_1d.in_list == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]

    def test__load_values__retains_list_structure(self):
        values = aa.Values.from_file(file_path=test_values_dir + "values_test.dat")

        assert values.in_list == [[1.0, 2.0], [3.0, 4.0, 5.0]]

    def test__output_values_to_file(self):

        values = aa.Values([[4.0, 5.0], [6.0, 7.0, 8.0]])

        output_values_dir = "{}/files/values/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_values_dir):
            shutil.rmtree(output_values_dir)

        os.makedirs(output_values_dir)

        values.output_to_file(file_path=output_values_dir + "values_test.dat")

        values = aa.Values.from_file(file_path=output_values_dir + "values_test.dat")

        assert values.in_list == [[4.0, 5.0], [6.0, 7.0, 8.0]]

        with pytest.raises(FileExistsError):
            values.output_to_file(file_path=output_values_dir + "values_test.dat")

        values.output_to_file(
            file_path=output_values_dir + "values_test.dat", overwrite=True
        )
