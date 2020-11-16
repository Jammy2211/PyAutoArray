import os
from os import path

import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray import exc

test_data_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "array"
)


class TestConstructorMethods:
    def test__constructor_class_method_in_2d__store_in_1d(self):

        arr = aa.Array.manual_2d(
            array=np.ones((3, 3)), sub_size=1, pixel_scales=(1.0, 1.0), store_in_1d=True
        )

        assert (arr == np.ones((9,))).all()
        assert (arr.in_1d == np.ones((9,))).all()
        assert (arr.in_2d == np.ones((3, 3))).all()
        assert (arr.in_1d_binned == np.ones((9,))).all()
        assert (arr.in_2d_binned == np.ones((3, 3))).all()
        assert (arr.binned == np.ones((9,))).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert arr.geometry.shape_2d_scaled == pytest.approx((3.0, 3.0))
        assert arr.geometry.scaled_maxima == (1.5, 1.5)
        assert arr.geometry.scaled_minima == (-1.5, -1.5)

        arr = aa.Array.manual_2d(
            array=np.ones((4, 4)), sub_size=2, pixel_scales=(0.1, 0.1), store_in_1d=True
        )

        assert (arr == np.ones((16,))).all()
        assert (arr.in_1d == np.ones((16,))).all()
        assert (arr.in_2d == np.ones((4, 4))).all()
        assert (arr.in_1d_binned == np.ones((4,))).all()
        assert (arr.in_2d_binned == np.ones((2, 2))).all()
        assert (arr.binned == np.ones((4,))).all()
        assert arr.pixel_scales == (0.1, 0.1)
        assert arr.geometry.central_pixel_coordinates == (0.5, 0.5)
        assert arr.geometry.shape_2d_scaled == pytest.approx((0.2, 0.2))
        assert arr.geometry.scaled_maxima == pytest.approx((0.1, 0.1), 1e-4)
        assert arr.geometry.scaled_minima == pytest.approx((-0.1, -0.1), 1e-4)

        arr = aa.Array.manual_2d(
            array=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            pixel_scales=(0.1, 0.1),
            sub_size=2,
            origin=(1.0, 1.0),
            store_in_1d=True,
        )

        assert (arr == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
        assert arr.shape_2d == (2, 1)
        assert arr.sub_shape_2d == (4, 2)
        assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
        assert (
            arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert arr.in_2d_binned.shape == (2, 1)
        assert (arr.in_1d_binned == np.array([2.5, 6.5])).all()
        assert (arr.in_2d_binned == np.array([[2.5], [6.5]])).all()
        assert (arr.binned == np.array([2.5, 6.5])).all()
        assert arr.pixel_scales == (0.1, 0.1)
        assert arr.geometry.central_pixel_coordinates == (0.5, 0.0)
        assert arr.geometry.shape_2d_scaled == pytest.approx((0.2, 0.1))
        assert arr.geometry.scaled_maxima == pytest.approx((1.1, 1.05), 1e-4)
        assert arr.geometry.scaled_minima == pytest.approx((0.9, 0.95), 1e-4)

        arr = aa.Array.manual_2d(
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
        arr = aa.Array.manual_1d(
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

        arr = aa.Array.manual_2d(
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
        assert (arr.binned == np.ones((3, 3))).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert arr.geometry.shape_2d_scaled == pytest.approx((3.0, 3.0))
        assert arr.geometry.scaled_maxima == (1.5, 1.5)
        assert arr.geometry.scaled_minima == (-1.5, -1.5)

        arr = aa.Array.manual_2d(
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
        assert (arr.binned == np.ones((2, 2))).all()
        assert arr.pixel_scales == (0.1, 0.1)
        assert arr.geometry.central_pixel_coordinates == (0.5, 0.5)
        assert arr.geometry.shape_2d_scaled == pytest.approx((0.2, 0.2))
        assert arr.geometry.scaled_maxima == pytest.approx((0.1, 0.1), 1e-4)
        assert arr.geometry.scaled_minima == pytest.approx((-0.1, -0.1), 1e-4)

        arr = aa.Array.manual_2d(
            array=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            pixel_scales=(0.1, 0.1),
            sub_size=2,
            origin=(1.0, 1.0),
            store_in_1d=False,
        )

        assert (arr == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
        assert arr.shape_2d == (2, 1)
        assert arr.sub_shape_2d == (4, 2)
        assert (arr.in_1d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
        assert (
            arr.in_2d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert arr.in_2d_binned.shape == (2, 1)
        assert (arr.in_1d_binned == np.array([2.5, 6.5])).all()
        assert (arr.in_2d_binned == np.array([[2.5], [6.5]])).all()
        assert (arr.binned == np.array([[2.5], [6.5]])).all()
        assert arr.pixel_scales == (0.1, 0.1)
        assert arr.geometry.central_pixel_coordinates == (0.5, 0.0)
        assert arr.geometry.shape_2d_scaled == pytest.approx((0.2, 0.1))
        assert arr.geometry.scaled_maxima == pytest.approx((1.1, 1.05), 1e-4)
        assert arr.geometry.scaled_minima == pytest.approx((0.9, 0.95), 1e-4)

        arr = aa.Array.manual_2d(
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
        arr = aa.Array.manual_1d(
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

        arr = aa.Array.manual_2d(array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

        arr = arr.resized_from(new_shape=(7, 7))

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
        assert arr.mask.pixel_scales == (1.0, 1.0)

    def test__trim__compare_to_array_util(self):
        array_2d = np.ones((5, 5))
        array_2d[2, 2] = 2.0

        arr = aa.Array.manual_2d(array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

        arr = arr.resized_from(new_shape=(3, 3))

        arr_resized_manual = np.array(
            [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
        )

        assert type(arr) == aa.Array
        assert (arr.in_2d == arr_resized_manual).all()
        assert arr.mask.pixel_scales == (1.0, 1.0)

    def test__padded_from_kernel_shape__padded_edge_of_zeros_where_extra_psf_blurring_is_performed(
        self,
    ):

        array_2d = np.ones((5, 5))
        array_2d[2, 2] = 2.0

        arr = aa.Array.manual_2d(array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

        new_arr = arr.padded_before_convolution_from(kernel_shape=(3, 3))

        assert type(new_arr) == aa.Array
        assert new_arr.in_2d[0, 0] == 0.0
        assert new_arr.shape_2d == (7, 7)
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

        new_arr = arr.padded_before_convolution_from(kernel_shape=(5, 5))

        assert type(new_arr) == aa.Array
        assert new_arr.in_2d[0, 0] == 0.0
        assert new_arr.shape_2d == (9, 9)
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

        array_2d = np.ones((9, 9))
        array_2d[4, 4] = 2.0

        arr = aa.Array.manual_2d(array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

        new_arr = arr.padded_before_convolution_from(kernel_shape=(7, 7))

        assert type(new_arr) == aa.Array
        assert new_arr.in_2d[0, 0] == 0.0
        assert new_arr.shape_2d == (15, 15)
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

    def test__trimmed_from_kernel_shape__trim_edges_where_extra_psf_blurring_is_performed(
        self,
    ):
        array_2d = np.ones((5, 5))
        array_2d[2, 2] = 2.0

        arr = aa.Array.manual_2d(array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

        new_arr = arr.trimmed_after_convolution_from(kernel_shape=(3, 3))

        assert type(new_arr) == aa.Array
        assert (
            new_arr.in_2d
            == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

        new_arr = arr.trimmed_after_convolution_from(kernel_shape=(5, 5))

        assert type(new_arr) == aa.Array
        assert (new_arr.in_2d == np.array([[2.0]])).all()
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

        array_2d = np.ones((9, 9))
        array_2d[4, 4] = 2.0

        arr = aa.Array.manual_2d(array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

        new_arr = arr.trimmed_after_convolution_from(kernel_shape=(7, 7))

        assert type(new_arr) == aa.Array
        assert (
            new_arr.in_2d
            == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

    def test__zoomed__2d_array_zoomed__uses_the_limits_of_the_mask(self):

        array_2d = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]

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

        arr_masked = aa.Array.manual_mask(array=array_2d, mask=mask)

        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

        assert (arr_zoomed.in_2d == np.array([[6.0, 7.0], [10.0, 11.0]])).all()

        mask = aa.Mask2D.manual(
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

        arr_masked = aa.Array.manual_mask(array=array_2d, mask=mask)
        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
        assert (
            arr_zoomed.in_2d == np.array([[6.0, 7.0, 8.0], [10.0, 11.0, 12.0]])
        ).all()

        mask = aa.Mask2D.manual(
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

        arr_masked = aa.Array.manual_mask(array=array_2d, mask=mask)
        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
        assert (
            arr_zoomed.in_2d == np.array([[6.0, 7.0], [10.0, 11.0], [14.0, 15.0]])
        ).all()

        mask = aa.Mask2D.manual(
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

        arr_masked = aa.Array.manual_mask(array=array_2d, mask=mask)
        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

        assert (
            arr_zoomed.in_2d == np.array([[0.0, 6.0, 7.0], [9.0, 10.0, 11.0]])
        ).all()

        mask = aa.Mask2D.manual(
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

        arr_masked = aa.Array.manual_mask(array=array_2d, mask=mask)
        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
        assert (
            arr_zoomed.in_2d == np.array([[2.0, 0.0], [6.0, 7.0], [10.0, 11.0]])
        ).all()

        mask = aa.Mask2D.manual(
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

        arr_masked = aa.Array.manual_mask(array=array_2d, mask=mask)
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

        mask = aa.Mask2D.manual(
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

        arr_masked = aa.Array.manual_mask(array=array_2d, mask=mask)

        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

        assert arr_zoomed.mask.geometry.origin == (0.0, 0.0)

        array_2d = np.ones(shape=(6, 6))

        mask = aa.Mask2D.manual(
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

        arr_masked = aa.Array.manual_mask(array=array_2d, mask=mask)

        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

        assert arr_zoomed.mask.geometry.origin == (1.0, 1.0)

        mask = aa.Mask2D.manual(
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

        arr_masked = aa.Array.manual_mask(array=array_2d, mask=mask)

        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

        assert arr_zoomed.mask.geometry.origin == (0.0, 1.0)

    def test__zoomed__array_extent__uses_the_limits_of_the_unzoomed_mask(self):

        array_2d = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]

        mask = aa.Mask2D.manual(
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

        arr_masked = aa.Array.manual_mask(array=array_2d, mask=mask)

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

        arr = aa.Array.manual_2d(array=array_2d, sub_size=1, pixel_scales=(0.1, 0.1))

        arr_binned_util = aa.util.binning.bin_array_2d_via_mean(
            array_2d=array_2d, bin_up_factor=4
        )
        arr_binned = arr.binned_up_from(bin_up_factor=4, method="mean")
        assert (arr_binned.in_2d == arr_binned_util).all()
        assert arr_binned.pixel_scales == (0.4, 0.4)

        arr_binned_util = aa.util.binning.bin_array_2d_via_quadrature(
            array_2d=array_2d, bin_up_factor=4
        )
        arr_binned = arr.binned_up_from(bin_up_factor=4, method="quadrature")
        assert (arr_binned.in_2d == arr_binned_util).all()
        assert arr_binned.pixel_scales == (0.4, 0.4)

        arr_binned_util = aa.util.binning.bin_array_2d_via_sum(
            array_2d=array_2d, bin_up_factor=4
        )
        arr_binned = arr.binned_up_from(bin_up_factor=4, method="sum")
        assert (arr_binned.in_2d == arr_binned_util).all()
        assert arr_binned.pixel_scales == (0.4, 0.4)

    def test__binned_up__invalid_method__raises_exception(self):
        array_2d = [
            [1.0, 6.0, 3.0, 7.0, 3.0, 2.0],
            [2.0, 5.0, 3.0, 7.0, 7.0, 7.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ]

        array_2d = aa.Array.manual_2d(
            array=array_2d, sub_size=1, pixel_scales=(0.1, 0.1)
        )
        with pytest.raises(exc.ArrayException):
            array_2d.binned_up_from(bin_up_factor=4, method="wrong")


class TestOutputToFits:
    def test__output_to_fits(self):

        arr = aa.Array.from_fits(
            file_path=path.join(test_data_dir, "3x3_ones.fits"), hdu=0, pixel_scales=1.0
        )

        output_data_dir = path.join(
            "{}".format(path.dirname(path.realpath(__file__))),
            "files",
            "array",
            "output_test",
        )
        if path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=path.join(output_data_dir, "array.fits"))

        array_from_out = aa.Array.from_fits(
            file_path=path.join(output_data_dir, "array.fits"), hdu=0, pixel_scales=1.0
        )

        assert (array_from_out.in_2d == np.ones((3, 3))).all()

    def test__output_to_fits__shapes_of_arrays_are_2d(self):

        arr = aa.Array.from_fits(
            file_path=path.join(test_data_dir, "3x3_ones.fits"), hdu=0, pixel_scales=1.0
        )

        output_data_dir = path.join(
            "{}".format(path.dirname(path.realpath(__file__))),
            "files",
            "array",
            "output_test",
        )

        if path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=path.join(output_data_dir, "array.fits"))

        array_from_out = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(output_data_dir, "array.fits"), hdu=0
        )

        assert (array_from_out == np.ones((3, 3))).all()

        mask = aa.Mask2D.unmasked(shape_2d=(3, 3), pixel_scales=0.1)

        masked_array = aa.Array.manual_mask(array=arr, mask=mask)

        masked_array.output_to_fits(
            file_path=path.join(output_data_dir, "masked_array.fits")
        )

        masked_array_from_out = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(output_data_dir, "masked_array.fits"), hdu=0
        )

        assert (masked_array_from_out == np.ones((3, 3))).all()


class TestExposureInfo:
    def test__exposure_info_has_date_and_time_of_observation__calcs_julian_date(self):

        exposure_info = aa.ExposureInfo(
            date_of_observation="2000-01-01", time_of_observation="00:00:00"
        )

        assert exposure_info.modified_julian_date == 51544.0
