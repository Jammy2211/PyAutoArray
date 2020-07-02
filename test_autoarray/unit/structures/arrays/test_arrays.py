import os

import numpy as np
import pytest

import autoarray as aa
from autoarray import exc

test_data_dir = "{}/files/array/".format(os.path.dirname(os.path.realpath(__file__)))


class TestArrayAPI:
    class TestManual:
        def test__array__makes_array_without_other_inputs(self):

            arr = aa.Array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]])

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()

            arr = aa.Array.manual_1d(array=[1.0, 2.0, 3.0, 4.0], shape_2d=(2, 2))

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()

            arr = aa.Array.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape_2d=(2, 3), store_in_1d=True
            )

            assert type(arr) == aa.Array
            assert (arr == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()
            assert (arr.in_2d == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()

            arr = aa.Array.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape_2d=(2, 3), store_in_1d=False
            )

            assert type(arr) == aa.Array
            assert (arr == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (arr.in_2d == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()

        def test__array__makes_array_with_pixel_scale(self):

            arr = aa.Array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

            arr = aa.Array.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape_2d=(2, 3),
                pixel_scales=(2.0, 3.0),
            )

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()
            assert arr.pixel_scales == (2.0, 3.0)
            assert arr.geometry.origin == (0.0, 0.0)

        def test__array__makes_with_pixel_scale_and_sub_size(self):

            arr = aa.Array.manual_2d(
                array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, sub_size=1
            )

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()

            arr = aa.Array.full(fill_value=2.0, shape_2d=(2, 2), store_in_1d=True)

            assert type(arr) == aa.Array
            assert (arr == np.array([2.0, 2.0, 2.0, 2.0])).all()
            assert (arr.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (arr.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()

            arr = aa.Array.full(fill_value=2.0, shape_2d=(2, 2), store_in_1d=False)

            assert type(arr) == aa.Array
            assert (arr == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (arr.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (arr.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            arr = aa.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)

            arr = aa.Array.full(
                fill_value=2.0, shape_2d=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0)
            )

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (arr.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            arr = aa.Array.full(
                fill_value=1.0, shape_2d=(1, 4), pixel_scales=1.0, sub_size=1
            )

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (arr.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)
            assert arr.mask.sub_size == 2

    class TestOnesZeros:
        def test__array__makes_array_without_other_inputs(self):

            arr = aa.Array.ones(shape_2d=(2, 2))

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()

            arr = aa.Array.zeros(shape_2d=(2, 2), store_in_1d=True)

            assert type(arr) == aa.Array
            assert (arr == np.array([0.0, 0.0, 0.0, 0.0])).all()
            assert (arr.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()

            arr = aa.Array.zeros(shape_2d=(2, 2), store_in_1d=False)

            assert type(arr) == aa.Array
            assert (arr == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            arr = aa.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)

            arr = aa.Array.zeros(shape_2d=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0))

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            arr = aa.Array.ones(shape_2d=(1, 4), pixel_scales=1.0, sub_size=1)

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[1.0, 1.0, 1.0, 1.0]])).all()
            assert (arr.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 0.0)
            assert arr.mask.sub_size == 1

            arr = aa.Array.zeros(
                shape_2d=(1, 1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0)
            )

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)
            assert arr.mask.sub_size == 2

    class TestFromFits:
        def test__array__makes_array_without_other_inputs(self):

            arr = aa.Array.from_fits(file_path=test_data_dir + "3x3_ones.fits", hdu=0)

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.ones((3, 3))).all()
            assert (arr.in_1d == np.ones(9)).all()

            arr = aa.Array.from_fits(
                file_path=test_data_dir + "4x3_ones.fits", hdu=0, store_in_1d=True
            )

            assert type(arr) == aa.Array
            assert (arr == np.ones((12,))).all()
            assert (arr.in_2d == np.ones((4, 3))).all()
            assert (arr.in_1d == np.ones((12,))).all()

            arr = aa.Array.from_fits(
                file_path=test_data_dir + "4x3_ones.fits", hdu=0, store_in_1d=False
            )

            assert type(arr) == aa.Array
            assert (arr == np.ones((4, 3))).all()
            assert (arr.in_2d == np.ones((4, 3))).all()
            assert (arr.in_1d == np.ones((12,))).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            arr = aa.Array.from_fits(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0, pixel_scales=1.0
            )

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
            assert (arr.in_1d == np.array([1.0, 2.0, 4.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

            mask = aa.Mask.manual(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)
            arr = aa.MaskedArray.manual_1d(
                array=[1.0, 2.0, 3.0, 4.0], mask=mask, store_in_1d=True
            )

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[2.0, 2.0], [0.0, 2.0]])).all()
            assert (arr.in_1d == np.array([2.0, 2.0, 2.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

            mask = aa.Mask.manual(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)
            arr = aa.MaskedArray.full(fill_value=3.0, mask=mask, store_in_1d=True)

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
            assert (arr.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (arr.in_1d == np.array([0.0, 0.0, 0.0])).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.origin == (0.0, 1.0)

            mask = aa.Mask.manual(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)
            arr = aa.MaskedArray.ones(mask=mask, store_in_1d=True)

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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

            assert type(arr) == aa.Array
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
