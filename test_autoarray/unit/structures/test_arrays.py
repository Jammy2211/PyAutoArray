import os

import numpy as np
import pytest

import autoarray as aa
from autoarray import exc

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestAPIFactory:

    class TestArray:

        def test__array__makes_array_without_other_inputs(self):

            array = aa.array.manual(array=[[1.0, 2.0], [3.0, 4.0]])

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()

            array = aa.array.manual(array=[1.0, 2.0, 3.0, 4.0], shape_2d=(2,2))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()

            array = aa.array.manual(array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape_2d=(2,3))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            array = aa.array.manual(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)

            array = aa.array.manual(array=[1.0, 2.0, 3.0, 4.0], shape_2d=(2,2), pixel_scales=1.0, origin=(0.0, 1.0))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

            array = aa.array.manual(array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape_2d=(2,3), pixel_scales=(2.0, 3.0))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).all()
            assert array.pixel_scales == (2.0, 3.0)
            assert array.geometry.origin == (0.0, 0.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            array = aa.array.manual(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, sub_size=1)

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 1

            array = aa.array.manual(array=[1.0, 2.0, 3.0, 4.0], shape_2d=(1,1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)
            assert array.mask.sub_size == 2

            array = aa.array.manual(array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape_2d=(2,1), pixel_scales=2.0, sub_size=2)

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
            assert (array.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
            assert array.pixel_scales == (2.0, 2.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 2

        def test__array__input_is_1d_array__no_shape_2d__raises_exception(self):

            with pytest.raises(exc.ArrayException):

                aa.array.manual(array=[1.0, 2.0, 3.0])

            with pytest.raises(exc.ArrayException):

                aa.array.manual(array=[1.0, 2.0, 3.0], pixel_scales=1.0)

            with pytest.raises(exc.ArrayException):

                aa.array.manual(array=[1.0, 2.0, 3.0], sub_size=1)

            with pytest.raises(exc.ArrayException):

                aa.array.manual(array=[1.0, 2.0, 3.0], pixel_scales=1.0, sub_size=1)

    class TestFull:

        def test__array__makes_array_without_other_inputs(self):

            array = aa.array.full(fill_value=1.0, shape_2d=(2,2))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()

            array = aa.array.full(fill_value=2.0, shape_2d=(2,2))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (array.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            array = aa.array.full(fill_value=1.0, shape_2d=(2,2), pixel_scales=1.0)

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)

            array = aa.array.full(fill_value=2.0, shape_2d=(2,2), pixel_scales=1.0, origin=(0.0, 1.0))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (array.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            array = aa.array.full(fill_value=1.0, shape_2d=(1,4), pixel_scales=1.0, sub_size=1)

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 1.0, 1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 1

            array = aa.array.full(fill_value=2.0, shape_2d=(1,1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
            assert (array.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)
            assert array.mask.sub_size == 2

    class TestOnesZeros:

        def test__array__makes_array_without_other_inputs(self):

            array = aa.array.ones(shape_2d=(2,2))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()

            array = aa.array.zeros(shape_2d=(2,2))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (array.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            array = aa.array.ones(shape_2d=(2,2), pixel_scales=1.0)

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)

            array = aa.array.zeros(shape_2d=(2,2), pixel_scales=1.0, origin=(0.0, 1.0))

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
            assert (array.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            array = aa.array.ones(shape_2d=(1,4), pixel_scales=1.0, sub_size=1)

            assert type(array) == aa.Array
            assert (array.in_2d == np.array([[1.0, 1.0, 1.0, 1.0]])).all()
            assert (array.in_1d == np.array([1.0, 1.0, 1.0, 1.0])).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 1

            array = aa.array.zeros(shape_2d=(1,1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0))

            assert type(array) == aa.Array
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

            assert type(array) == aa.Array
            assert (array.in_2d == np.ones((3,3))).all()
            assert (array.in_1d == np.ones(9,)).all()

            array = aa.array.from_fits(file_path=test_data_dir + "4x3_ones.fits", hdu=0)

            assert type(array) == aa.Array
            assert (array.in_2d == np.ones((4,3))).all()
            assert (array.in_1d == np.ones((12,))).all()

        def test__array__makes_scaled_array_with_pixel_scale(self):

            array = aa.array.from_fits(
                file_path=test_data_dir + "3x3_ones.fits",
                hdu=0, pixel_scales=1.0)

            assert type(array) == aa.Array
            assert (array.in_2d == np.ones((3,3))).all()
            assert (array.in_1d == np.ones(9,)).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)

            array = aa.array.from_fits(file_path=test_data_dir + "4x3_ones.fits", hdu=0, pixel_scales=1.0, origin=(0.0, 1.0))

            assert type(array) == aa.Array
            assert (array.in_2d == np.ones((4,3))).all()
            assert (array.in_1d == np.ones((12,))).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)

        def test__array__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

            array = aa.array.from_fits(file_path=test_data_dir + "3x3_ones.fits",
                hdu=0, pixel_scales=1.0, sub_size=1)

            assert type(array) == aa.Array
            assert (array.in_2d == np.ones((3,3))).all()
            assert (array.in_1d == np.ones(9,)).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 0.0)
            assert array.mask.sub_size == 1

            array = aa.array.from_fits(file_path=test_data_dir + "4x3_ones.fits", hdu=0, pixel_scales=1.0, sub_size=1, origin=(0.0, 1.0))

            assert type(array) == aa.Array
            assert (array.in_2d == np.ones((4,3))).all()
            assert (array.in_1d == np.ones(12,)).all()
            assert array.pixel_scales == (1.0, 1.0)
            assert array.geometry.origin == (0.0, 1.0)
            assert array.mask.sub_size == 1


class TestArray:

    class TestConstructorMethods:

        def test__constructor_class_method_in_2d(self):

            arr = aa.Array.from_sub_array_2d_pixel_scales_and_sub_size(
                sub_array_2d=np.ones((3, 3)), sub_size=1, pixel_scales=(1.0, 1.0))

            assert (arr.in_1d == np.ones((9,))).all()
            assert (arr.in_2d == np.ones((3, 3))).all()
            assert (arr.in_1d_binned == np.ones((9,))).all()
            assert (arr.in_2d_binned == np.ones((3, 3))).all()
            assert arr.pixel_scale == 1.0
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_arcsec == pytest.approx((3.0, 3.0))
            assert arr.geometry.arc_second_maxima == (1.5, 1.5)
            assert arr.geometry.arc_second_minima == (-1.5, -1.5)

            arr = aa.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=np.ones((4, 4)), sub_size=2,
                                                                       pixel_scales=(0.1, 0.1))

            assert (arr.in_1d == np.ones((16,))).all()
            assert (arr.in_2d == np.ones((4, 4))).all()
            assert (arr.in_1d_binned == np.ones((4,))).all()
            assert (arr.in_2d_binned == np.ones((2, 2))).all()
            assert arr.pixel_scale == 0.1
            assert arr.geometry.central_pixel_coordinates == (0.5, 0.5)
            assert arr.geometry.shape_arcsec == pytest.approx((0.2, 0.2))
            assert arr.geometry.arc_second_maxima == pytest.approx((0.1, 0.1), 1e-4)
            assert arr.geometry.arc_second_minima == pytest.approx((-0.1, -0.1), 1e-4)

            arr = aa.Array.from_sub_array_2d_pixel_scales_and_sub_size(
                sub_array_2d=np.array([[1.0, 2.0],
                                       [3.0, 4.0],
                                       [5.0, 6.0],
                                       [7.0, 8.0]]), pixel_scales=(0.1, 0.1), sub_size=2, origin=(1.0, 1.0)
            )

            assert arr.in_2d.shape == (4, 2)
            assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
            assert (arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
            assert arr.in_2d_binned.shape == (2, 1)
            assert (arr.in_1d_binned == np.array([2.5, 6.5])).all()
            assert (arr.in_2d_binned == np.array([[2.5], [6.5]])).all()
            assert arr.pixel_scale == 0.1
            assert arr.geometry.central_pixel_coordinates == (0.5, 0.0)
            assert arr.geometry.shape_arcsec == pytest.approx((0.2, 0.1))
            assert arr.geometry.arc_second_maxima == pytest.approx((1.1, 1.05), 1e-4)
            assert arr.geometry.arc_second_minima == pytest.approx((0.9, 0.95), 1e-4)

            arr = aa.Array.from_sub_array_2d_pixel_scales_and_sub_size(
                sub_array_2d=np.ones((3, 3)), pixel_scales=(2.0, 1.0), sub_size=1, origin=(-1.0, -2.0)
            )

            assert arr.in_1d == pytest.approx(np.ones((9,)), 1e-4)
            assert arr.in_2d == pytest.approx(np.ones((3, 3)), 1e-4)
            assert arr.pixel_scales == (2.0, 1.0)
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_arcsec == pytest.approx((6.0, 3.0))
            assert arr.geometry.origin == (-1.0, -2.0)
            assert arr.geometry.arc_second_maxima == pytest.approx((2.0, -0.5), 1e-4)
            assert arr.geometry.arc_second_minima == pytest.approx((-4.0, -3.5), 1e-4)

        def test__constructor_class_method_in_1d(self):
            arr = aa.Array.from_sub_array_1d_shape_2d_pixel_scales_and_sub_size(
                sub_array_1d=np.ones((9,)), shape_2d=(3, 3), pixel_scales=(2.0, 1.0), sub_size=1, origin=(-1.0, -2.0)
            )

            assert arr.in_1d == pytest.approx(np.ones((9,)), 1e-4)
            assert arr.in_2d == pytest.approx(np.ones((3, 3)), 1e-4)
            assert arr.pixel_scales == (2.0, 1.0)
            assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
            assert arr.geometry.shape_arcsec == pytest.approx((6.0, 3.0))
            assert arr.geometry.origin == (-1.0, -2.0)
            assert arr.geometry.arc_second_maxima == pytest.approx((2.0, -0.5), 1e-4)
            assert arr.geometry.arc_second_minima == pytest.approx((-4.0, -3.5), 1e-4)

    class TestNewArrays:

        def test__pad__compare_to_array_util(self):
            array_2d = np.ones((5, 5))
            array_2d[2, 2] = 2.0

            arr = aa.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

            arr = arr.resized_array_from_new_shape(
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

            assert type(arr) == aa.Array
            assert (arr.in_2d == arr_resized_manual).all()
            assert arr.mask.pixel_scale == 1.0

        def test__trim__compare_to_array_util(self):
            array_2d = np.ones((5, 5))
            array_2d[2, 2] = 2.0

            arr = aa.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

            arr = arr.resized_array_from_new_shape(
                new_shape=(3, 3),
            )

            arr_resized_manual = np.array(
                [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
            )

            assert type(arr) == aa.Array
            assert (arr.in_2d == arr_resized_manual).all()
            assert arr.mask.pixel_scale == 1.0

        def test__kernel_trim__trim_edges_where_extra_psf_blurring_is_performed(self):
            array_2d = np.ones((5, 5))
            array_2d[2, 2] = 2.0

            arr = aa.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

            new_arr = arr.trimmed_array_from_kernel_shape(
                kernel_shape=(3, 3)
            )

            assert type(new_arr) == aa.Array
            assert (
                new_arr.in_2d
                == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
            ).all()
            assert new_arr.mask.pixel_scale == 1.0

            new_arr = arr.trimmed_array_from_kernel_shape(
                kernel_shape=(5, 5)
            )

            assert type(new_arr) == aa.Array
            assert (new_arr.in_2d == np.array([[2.0]])).all()
            assert new_arr.mask.pixel_scale == 1.0

            array_2d = np.ones((9, 9))
            array_2d[4, 4] = 2.0

            arr = aa.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

            new_arr = arr.trimmed_array_from_kernel_shape(
                kernel_shape=(7, 7)
            )

            assert type(new_arr) == aa.Array
            assert (
                new_arr.in_2d
                == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
            ).all()
            assert new_arr.mask.pixel_scale == 1.0

        def test__zoomed__2d_array_zoomed__uses_the_limits_of_the_mask(self):
            array_2d = np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            )

            arr = aa.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

            mask = aa.Mask(
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

            arr_zoomed = arr.zoomed_array_from_mask(
                mask=mask, buffer=0
            )
            assert (arr_zoomed.in_2d == np.array([[6.0, 7.0], [10.0, 11.0]])).all()

            mask = aa.Mask(
                mask_2d=np.array(
                    [
                        [True, True, True, True],
                        [True, False, False, True],
                        [True, False, False, False],
                        [True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            arr_zoomed = arr.zoomed_array_from_mask(
                mask=mask, buffer=0
            )
            assert (
                arr_zoomed.in_2d == np.array([[6.0, 7.0, 8.0], [10.0, 11.0, 12.0]])
            ).all()

            mask = aa.Mask(
                mask_2d=np.array(
                    [
                        [True, True, True, True],
                        [True, False, False, True],
                        [True, False, False, True],
                        [True, True, False, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            arr_zoomed = arr.zoomed_array_from_mask(
                mask=mask, buffer=0
            )
            assert (
                arr_zoomed.in_2d
                == np.array([[6.0, 7.0], [10.0, 11.0], [14.0, 15.0]])
            ).all()

            mask = aa.Mask(
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

            arr_zoomed = arr.zoomed_array_from_mask(
                mask=mask, buffer=0
            )
            assert (
                arr_zoomed.in_2d == np.array([[5.0, 6.0, 7.0], [9.0, 10.0, 11.0]])
            ).all()

            mask = aa.Mask(
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

            arr_zoomed = arr.zoomed_array_from_mask(
                mask=mask, buffer=0
            )
            assert (
                arr_zoomed.in_2d
                == np.array([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]])
            ).all()

            mask = aa.Mask(
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

            arr_zoomed = arr.zoomed_array_from_mask(
                mask=mask, buffer=1
            )
            assert (
                arr_zoomed.in_2d
                == np.array(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0],
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

            arr = aa.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(0.1, 0.1))

            arr_binned_util = aa.binning_util.binned_up_array_2d_using_mean_from_array_2d_and_bin_up_factor(
                array_2d=array_2d, bin_up_factor=4
            )
            arr_binned = arr.binned_array_from_bin_up_factor(
                bin_up_factor=4, method="mean"
            )
            assert (arr_binned.in_2d == arr_binned_util).all()
            assert arr_binned.pixel_scale == 0.4

            arr_binned_util = aa.binning_util.binned_array_2d_using_quadrature_from_array_2d_and_bin_up_factor(
                array_2d=array_2d, bin_up_factor=4
            )
            arr_binned = arr.binned_array_from_bin_up_factor(
                bin_up_factor=4, method="quadrature"
            )
            assert (arr_binned.in_2d == arr_binned_util).all()
            assert arr_binned.pixel_scale == 0.4

            arr_binned_util = aa.binning_util.binned_array_2d_using_sum_from_array_2d_and_bin_up_factor(
                array_2d=array_2d, bin_up_factor=4
            )
            arr_binned = arr.binned_array_from_bin_up_factor(
                bin_up_factor=4, method="sum"
            )
            assert (arr_binned.in_2d == arr_binned_util).all()
            assert arr_binned.pixel_scale == 0.4

        def test__binned_up__invalid_method__raises_exception(self):
            array_2d = np.array(
                [
                    [1.0, 6.0, 3.0, 7.0, 3.0, 2.0],
                    [2.0, 5.0, 3.0, 7.0, 7.0, 7.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ]
            )

            array_2d = aa.Array.from_sub_array_2d_pixel_scales_and_sub_size(sub_array_2d=array_2d, sub_size=1, pixel_scales=(0.1, 0.1))
            with pytest.raises(exc.ScaledException):
                array_2d.binned_array_from_bin_up_factor(
                    bin_up_factor=4, method="wrong"
                )