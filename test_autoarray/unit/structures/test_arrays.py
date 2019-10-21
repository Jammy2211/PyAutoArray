import os

import numpy as np
import pytest

import autoarray as aa
from autoarray.structures import arrays
from autoarray import exc

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestArrayAPI:

    class TestManual:

        def test__array__makes_array_without_other_inputs(self):

            array = aa.array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]])

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()

            array = aa.array.manual_1d(array=[1.0, 2.0, 3.0, 4.0], shape_2d=(2, 2))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()

            array = aa.array.manual_1d(array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape_2d=(2, 3))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()

        def test__array__makes_array_with_pixel_scale(self):

            array = aa.array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)

            array = aa.array.manual_1d(array=[1.0, 2.0, 3.0, 4.0], shape_2d=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

            array = aa.array.manual_1d(array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape_2d=(2, 3), pixel_scales=(2.0, 3.0))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()
            assert array.pixel_scales == (2.0, 3.0)
            assert array.geometry.origin == (0.0, 0.0)

        def test__array__makes_with_pixel_scale_and_sub_size(self):

            array = aa.array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, sub_size=1)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 1

            array = aa.array.manual_1d(array=[1.0, 2.0, 3.0, 4.0], shape_2d=(1, 1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)
            assert array.mask.sub_size == 2

            array = aa.array.manual_1d(array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape_2d=(2, 1), pixel_scales=2.0, sub_size=2)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
            assert (array.in_2d_binned == np.array([[2.5], [6.5]])).all()
            assert (array.in_1d_binned == np.array([2.5, 6.5])).all()
            assert array.pixel_scales == (2.0, 2.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 2

    class TestFull:

        def test__array__makes_array_without_other_inputs(self):

            array = aa.array.full(fill_value=1.0, shape_2d=(2,2))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()

            array = aa.array.full(fill_value=2.0, shape_2d=(2,2))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (array.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            array = aa.array.full(fill_value=1.0, shape_2d=(2,2), pixel_scales=1.0)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)

            array = aa.array.full(fill_value=2.0, shape_2d=(2,2), pixel_scales=1.0, origin=(0.0, 1.0))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (array.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            array = aa.array.full(fill_value=1.0, shape_2d=(1,4), pixel_scales=1.0, sub_size=1)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 1.0, 1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 1

            array = aa.array.full(fill_value=2.0, shape_2d=(1,1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (array.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)
            assert array.mask.sub_size == 2

    class TestOnesZeros:

        def test__array__makes_array_without_other_inputs(self):

            array = aa.array.ones(shape_2d=(2,2))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()

            array = aa.array.zeros(shape_2d=(2,2))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (array.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            array = aa.array.ones(shape_2d=(2,2), pixel_scales=1.0)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)

            array = aa.array.zeros(shape_2d=(2,2), pixel_scales=1.0, origin=(0.0, 1.0))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (array.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            array = aa.array.ones(shape_2d=(1,4), pixel_scales=1.0, sub_size=1)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 1.0, 1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 1

            array = aa.array.zeros(shape_2d=(1,1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (array.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)
            assert array.mask.sub_size == 2

    class TestFromFits:

        def test__array__makes_array_without_other_inputs(self):

            array = aa.array.from_fits(
                file_path=test_data_dir + "3x3_ones.fits",
                hdu=0)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.ones((3,3))).all()
            assert (array.in_1d == np.ones(9,)).all()

            array = aa.array.from_fits(file_path=test_data_dir + "4x3_ones.fits", hdu=0)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.ones((4,3))).all()
            assert (array.in_1d == np.ones((12,))).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            array = aa.array.from_fits(
                file_path=test_data_dir + "3x3_ones.fits",
                hdu=0, pixel_scales=1.0)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.ones((3,3))).all()
            assert (array.in_1d == np.ones(9,)).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)

            array = aa.array.from_fits(file_path=test_data_dir + "4x3_ones.fits", hdu=0, pixel_scales=1.0, origin=(0.0, 1.0))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.ones((4,3))).all()
            assert (array.in_1d == np.ones((12,))).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            array = aa.array.from_fits(file_path=test_data_dir + "3x3_ones.fits",
                hdu=0, pixel_scales=1.0, sub_size=1)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.ones((3,3))).all()
            assert (array.in_1d == np.ones(9,)).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 1

            array = aa.array.from_fits(file_path=test_data_dir + "4x3_ones.fits", hdu=0, pixel_scales=1.0, sub_size=1, origin=(0.0, 1.0))

            assert type(array) == arrays.Array
            assert (array.in_2d == np.ones((4,3))).all()
            assert (array.in_1d == np.ones(12,)).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)
            assert array.mask.sub_size == 1


class TestMaskedArrayAPI:

    class TestManual:

        def test__array__makes_array_with_pixel_scale(self):

            mask = aa.mask.unmasked(shape_2d=(2,2), pixel_scales=1.0)
            array = aa.masked_array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)

            mask = aa.mask.manual(mask_2d=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0))
            array = aa.masked_array.manual_1d(array=[1.0, 2.0, 4.0], mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 4.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

            mask = aa.mask.manual(mask_2d=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0))
            array = aa.masked_array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 4.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

            mask = aa.mask.manual(mask_2d=[[False], [True]], pixel_scales=2.0, sub_size=2)
            array = aa.masked_array.manual_1d(array=[1.0, 2.0, 3.0, 4.0], mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert (array.in_2d_binned == np.array([[2.5], [0.0]])).all()
            assert (array.in_1d_binned == np.array([2.5])).all()
            assert array.pixel_scales == (2.0, 2.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 2

        def test__manual_2d__exception_raised_if_input_array_is_2d_and_not_sub_shape_of_mask(self):

            with pytest.raises(exc.ArrayException):
                mask = aa.mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)
                aa.masked_array.manual_2d(array=[[1.0], [3.0]], mask=mask)

            with pytest.raises(exc.ArrayException):
                mask = aa.mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=2)
                aa.masked_array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

            with pytest.raises(exc.ArrayException):
                mask = aa.mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=2)
                aa.masked_array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], mask=mask)

        def test__exception_raised_if_input_array_is_1d_and_not_number_of_masked_sub_pixels(self):

            with pytest.raises(exc.ArrayException):
                mask = aa.mask.manual(mask_2d=[[False, False], [True, False]], sub_size=1)
                aa.masked_array.manual_1d(array=[1.0, 2.0, 3.0, 4.0], mask=mask)

            with pytest.raises(exc.ArrayException):
                mask = aa.mask.manual(mask_2d=[[False, False], [True, False]], sub_size=1)
                aa.masked_array.manual_1d(array=[1.0, 2.0], mask=mask)

            with pytest.raises(exc.ArrayException):
                mask = aa.mask.manual(mask_2d=[[False, True], [True, True]], sub_size=2)
                aa.masked_array.manual_1d(array=[1.0, 2.0, 4.0], mask=mask)

            with pytest.raises(exc.ArrayException):
                mask = aa.mask.manual(mask_2d=[[False, True], [True, True]], sub_size=2)
                aa.masked_array.manual_1d(array=[1.0, 2.0, 3.0, 4.0, 5.0], mask=mask)

    class TestFull:

        def test__makes_array_using_mask(self):

            mask = aa.mask.unmasked(shape_2d=(2,2), pixel_scales=1.0)
            array = aa.masked_array.full(fill_value=1.0, mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)

            mask = aa.mask.manual(mask_2d=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0))
            array = aa.masked_array.full(fill_value=2.0, mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[2.0, 2.0], [0.0, 2.0]])).all()
            assert (array.in_1d == np.array([2.0, 2.0, 2.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

            mask = aa.mask.manual(mask_2d=[[False], [True]], pixel_scales=2.0, sub_size=2)
            array = aa.masked_array.full(fill_value=3.0, mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[3.0, 3.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0]])).all()
            assert (array.in_1d == np.array([3.0, 3.0, 3.0, 3.0])).all()
            assert (array.in_2d_binned == np.array([[3.0], [0.0]])).all()
            assert (array.in_1d_binned == np.array([3.0])).all()
            assert array.pixel_scales == (2.0, 2.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 2

    class TestOnesZeros:

        def test__makes_array_using_mask(self):

            mask = aa.mask.unmasked(shape_2d=(2,2), pixel_scales=1.0)
            array = aa.masked_array.ones(mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)

            mask = aa.mask.manual(mask_2d=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0))
            array = aa.masked_array.zeros(mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (array.in_1d == np.array([0.0, 0.0, 0.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

            mask = aa.mask.manual(mask_2d=[[False], [True]], pixel_scales=2.0, sub_size=2)
            array = aa.masked_array.ones(mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert (array.in_2d_binned == np.array([[1.0], [0.0]])).all()
            assert (array.in_1d_binned == np.array([1.0])).all()
            assert array.pixel_scales == (2.0, 2.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 2

    class TestFromFits:

        def test__array_from_fits_uses_mask(self):

            mask = aa.mask.unmasked(shape_2d=(3,3), pixel_scales=1.0)
            array = aa.masked_array.from_fits(file_path=test_data_dir + "3x3_ones.fits",
                hdu=0, mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.ones((3,3))).all()
            assert (array.in_1d == np.ones(9,)).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 1

            mask = aa.mask.manual([[False, False, False], [False, False, False], [True, False, True], [False, False, False]], pixel_scales=1.0, origin=(0.0, 1.0))
            array = aa.masked_array.from_fits(file_path=test_data_dir + "4x3_ones.fits", hdu=0, mask=mask)

            assert type(array) == arrays.Array
            assert (array.in_2d == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)
            assert array.mask.sub_size == 1


class TestArray:

    class TestConstructorMethods:

        def test__constructor_class_method_in_2d(self):

            arr = arrays.Array.from_sub_array_2d_pixel_scales_and_sub_size(
                sub_array_2d=np.ones((3, 3)), sub_size=1, pixel_scales=(1.0, 1.0))

            assert (arr.in_1d == np.ones((9,))).all()
            assert (arr.in_2d == np.ones((3, 3))).all()
            assert (arr.in_1d_binned == np.ones((9,))).all()
            assert (arr.in_2d_binned == np.ones((3, 3))).all()
            assert arr.pixel_scales == (1.0, 1.0)
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_2d_arcsec == pytest.approx((3.0, 3.0))
            assert arr.geometry.arc_second_maxima == (1.5, 1.5)
            assert arr.geometry.arc_second_minima == (-1.5, -1.5)

            arr = arrays.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=np.ones((4, 4)), sub_size=2,
                                                                           pixel_scales=(0.1, 0.1))

            assert (arr.in_1d == np.ones((16,))).all()
            assert (arr.in_2d == np.ones((4, 4))).all()
            assert (arr.in_1d_binned == np.ones((4,))).all()
            assert (arr.in_2d_binned == np.ones((2, 2))).all()
            assert arr.pixel_scales == (0.1, 0.1)
            assert arr.geometry.central_pixel_coordinates == (0.5, 0.5)
            assert arr.geometry.shape_2d_arcsec == pytest.approx((0.2, 0.2))
            assert arr.geometry.arc_second_maxima == pytest.approx((0.1, 0.1), 1e-4)
            assert arr.geometry.arc_second_minima == pytest.approx((-0.1, -0.1), 1e-4)

            arr = arrays.Array.from_sub_array_2d_pixel_scales_and_sub_size(
                sub_array_2d=np.array([[1.0, 2.0],
                                       [3.0, 4.0],
                                       [5.0, 6.0],
                                       [7.0, 8.0]]), pixel_scales=(0.1, 0.1), sub_size=2, origin=(1.0, 1.0)
            )

            assert arr.shape_2d == (2, 1)
            assert arr.sub_shape_2d == (4, 2)
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
            assert arr.in_2d_binned.shape == (2, 1)
            assert (arr.in_1d_binned == np.array([2.5, 6.5])).all()
            assert (arr.in_2d_binned == np.array([[2.5], [6.5]])).all()
            assert arr.pixel_scales == (0.1, 0.1)
            assert arr.geometry.central_pixel_coordinates == (0.5, 0.0)
            assert arr.geometry.shape_2d_arcsec == pytest.approx((0.2, 0.1))
            assert arr.geometry.arc_second_maxima == pytest.approx((1.1, 1.05), 1e-4)
            assert arr.geometry.arc_second_minima == pytest.approx((0.9, 0.95), 1e-4)

            arr = arrays.Array.from_sub_array_2d_pixel_scales_and_sub_size(
                sub_array_2d=np.ones((3, 3)), pixel_scales=(2.0, 1.0), sub_size=1, origin=(-1.0, -2.0)
            )

            assert arr.in_1d == pytest.approx(np.ones((9,)), 1e-4)
            assert arr.in_2d == pytest.approx(np.ones((3, 3)), 1e-4)
            assert arr.pixel_scales == (2.0, 1.0)
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_2d_arcsec == pytest.approx((6.0, 3.0))
            assert arr.geometry.origin == (-1.0, -2.0)
            assert arr.geometry.arc_second_maxima == pytest.approx((2.0, -0.5), 1e-4)
            assert arr.geometry.arc_second_minima == pytest.approx((-4.0, -3.5), 1e-4)

        def test__constructor_class_method_in_1d(self):
            arr = arrays.Array.from_sub_array_1d_shape_2d_pixel_scales_and_sub_size(
                sub_array_1d=np.ones((9,)), shape_2d=(3, 3), pixel_scales=(2.0, 1.0), sub_size=1, origin=(-1.0, -2.0)
            )

            assert arr.in_1d == pytest.approx(np.ones((9,)), 1e-4)
            assert arr.in_2d == pytest.approx(np.ones((3, 3)), 1e-4)
            assert arr.pixel_scales == (2.0, 1.0)
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_2d_arcsec == pytest.approx((6.0, 3.0))
            assert arr.geometry.origin == (-1.0, -2.0)
            assert arr.geometry.arc_second_maxima == pytest.approx((2.0, -0.5), 1e-4)
            assert arr.geometry.arc_second_minima == pytest.approx((-4.0, -3.5), 1e-4)

    class TestNewArrays:

        def test__pad__compare_to_array_util(self):
            array_2d = np.ones((5, 5))
            array_2d[2, 2] = 2.0

            arr = arrays.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

            arr = arr.resized_from_new_shape(
                new_shape=(7, 7),
            )

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

            arr = arrays.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

            arr = arr.resized_from_new_shape(
                new_shape=(3, 3),
            )

            arr_resized_manual = np.array(
                [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
            )

            assert type(arr) == arrays.Array
            assert (arr.in_2d == arr_resized_manual).all()
            assert arr.mask.pixel_scales == (1.0, 1.0)

        def test__kernel_trim__trim_edges_where_extra_psf_blurring_is_performed(self):
            array_2d = np.ones((5, 5))
            array_2d[2, 2] = 2.0

            arr = arrays.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

            new_arr = arr.trimmed_from_kernel_shape(
                kernel_shape=(3, 3)
            )

            assert type(new_arr) == arrays.Array
            assert (
                new_arr.in_2d
                == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
            ).all()
            assert new_arr.mask.pixel_scales == (1.0, 1.0)

            new_arr = arr.trimmed_from_kernel_shape(
                kernel_shape=(5, 5)
            )

            assert type(new_arr) == arrays.Array
            assert (new_arr.in_2d == np.array([[2.0]])).all()
            assert new_arr.mask.pixel_scales == (1.0, 1.0)

            array_2d = np.ones((9, 9))
            array_2d[4, 4] = 2.0

            arr = arrays.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

            new_arr = arr.trimmed_from_kernel_shape(
                kernel_shape=(7, 7)
            )

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

            mask = aa.mask.manual(
                mask_2d=np.array(
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

            arr_masked = aa.masked_array.manual_2d(array=array_2d, mask=mask)

            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

            assert (arr_zoomed.in_2d == np.array([[6.0, 7.0], [10.0, 11.0]])).all()

            mask = aa.mask.manual(
                mask_2d=np.array(
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

            arr_masked = aa.masked_array.manual_2d(array=array_2d, mask=mask)
            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
            assert (
                arr_zoomed.in_2d == np.array([[6.0, 7.0, 8.0], [10.0, 11.0, 12.0]])
            ).all()

            mask = aa.mask.manual(
                mask_2d=np.array(
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

            arr_masked = aa.masked_array.manual_2d(array=array_2d, mask=mask)
            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
            assert (
                arr_zoomed.in_2d
                == np.array([[6.0, 7.0], [10.0, 11.0], [14.0, 15.0]])
            ).all()

            mask = aa.mask.manual(
                mask_2d=np.array(
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

            arr_masked = aa.masked_array.manual_2d(array=array_2d, mask=mask)
            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

            assert (
                arr_zoomed.in_2d == np.array([[0.0, 6.0, 7.0], [9.0, 10.0, 11.0]])
            ).all()

            mask = aa.mask.manual(
                mask_2d=np.array(
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

            arr_masked = aa.masked_array.manual_2d(array=array_2d, mask=mask)
            arr_zoomed = arr_masked.zoomed_around_mask(buffer=0
            )
            assert (
                arr_zoomed.in_2d
                == np.array([[2.0, 0.0], [6.0, 7.0], [10.0, 11.0]])
            ).all()

            mask = aa.mask.manual(
                mask_2d=np.array(
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

            arr_masked = aa.masked_array.manual_2d(array=array_2d, mask=mask)
            arr_zoomed = arr_masked.zoomed_around_mask(buffer=1
            )

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

        def test__binned_up__compare_all_extract_methods_to_array_util(self):
            array_2d = np.array(
                [
                    [1.0, 6.0, 3.0, 7.0, 3.0, 2.0],
                    [2.0, 5.0, 3.0, 7.0, 7.0, 7.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ]
            )

            arr = arrays.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(0.1, 0.1))

            arr_binned_util = aa.util.binning.bin_array_2d_via_mean(
                array_2d=array_2d, bin_up_factor=4
            )
            arr_binned = arr.binned_from_bin_up_factor(
                bin_up_factor=4, method="mean"
            )
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
            arr_binned = arr.binned_from_bin_up_factor(
                bin_up_factor=4, method="sum"
            )
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

            array_2d = arrays.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(0.1, 0.1))
            with pytest.raises(exc.ScaledException):
                array_2d.binned_from_bin_up_factor(
                    bin_up_factor=4, method="wrong"
                )