import os
from os import path

import numpy as np
import pytest
import shutil

import autoarray as aa

test_data_dir = path.join(
    "{}".format(os.path.dirname(os.path.realpath(__file__))), "files"
)


def test__constructor():

    mask = aa.Mask2D.all_false(shape_native=(2, 2), pixel_scales=1.0)
    array_2d = aa.Array2D(values=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
    assert (array_2d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 0.0)

    mask = aa.Mask2D(
        mask=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0)
    )
    array_2d = aa.Array2D(values=[1.0, 2.0, 4.0], mask=mask)

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
    assert (array_2d.slim == np.array([1.0, 2.0, 4.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 1.0)

    mask = aa.Mask2D(
        mask=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0)
    )
    array_2d = aa.Array2D(values=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
    assert (array_2d.slim == np.array([1.0, 2.0, 4.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 1.0)

    mask = aa.Mask2D(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)
    array_2d = aa.Array2D(values=[1.0, 2.0, 3.0, 4.0], mask=mask)

    assert type(array_2d) == aa.Array2D
    assert (array_2d == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (
        array_2d.native == np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (array_2d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (array_2d.binned.native == np.array([[2.5], [0.0]])).all()
    assert (array_2d.binned.slim == np.array([2.5])).all()
    assert array_2d.pixel_scales == (2.0, 2.0)
    assert array_2d.origin == (0.0, 0.0)
    assert array_2d.mask.sub_size == 2

    mask = aa.Mask2D(
        mask=[[False, False], [True, False]],
        pixel_scales=1.0,
        origin=(0.0, 1.0),
    )
    array_2d = aa.Array2D(values=[[1.0, 2.0], [3.0, 4.0]], mask=mask, store_native=True)

    assert (array_2d == np.array([[1.0, 2.0], [0.0, 4.0]])).all()


def test__no_mask():

    array_2d = aa.Array2D.no_mask(
        values=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, sub_size=1
    )

    assert type(array_2d) == aa.Array2D
    assert (array_2d == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (array_2d.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
    assert (array_2d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 0.0)
    assert array_2d.mask.sub_size == 1

    array_2d = aa.Array2D.no_mask(
        values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        pixel_scales=(0.1, 0.1),
        sub_size=2,
        origin=(1.0, 1.0),
    )

    assert (array_2d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
    assert array_2d.shape_native == (2, 1)
    assert array_2d.sub_shape_native == (4, 2)
    assert (array_2d.slim == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
    assert (
        array_2d.native == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    ).all()
    assert array_2d.binned.native.shape == (2, 1)
    assert (array_2d.binned == np.array([2.5, 6.5])).all()
    assert (array_2d.binned.native == np.array([[2.5], [6.5]])).all()
    assert (array_2d.binned == np.array([2.5, 6.5])).all()
    assert array_2d.pixel_scales == (0.1, 0.1)
    assert array_2d.geometry.central_pixel_coordinates == (0.5, 0.0)
    assert array_2d.geometry.shape_native_scaled == pytest.approx((0.2, 0.1))
    assert array_2d.geometry.scaled_maxima == pytest.approx((1.1, 1.05), 1e-4)
    assert array_2d.geometry.scaled_minima == pytest.approx((0.9, 0.95), 1e-4)

    array_2d = aa.Array2D.no_mask(
        values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        pixel_scales=(0.1, 0.1),
        sub_size=2,
        origin=(1.0, 1.0),
    )

    assert (array_2d == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
    assert array_2d.shape_native == (2, 1)
    assert array_2d.sub_shape_native == (4, 2)
    assert (array_2d.slim == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
    assert (
        array_2d.native == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    ).all()
    assert array_2d.binned.native.shape == (2, 1)
    assert (array_2d.binned == np.array([2.5, 6.5])).all()
    assert (array_2d.binned.native == np.array([[2.5], [6.5]])).all()
    assert (array_2d.binned == np.array([2.5, 6.5])).all()
    assert array_2d.pixel_scales == (0.1, 0.1)
    assert array_2d.geometry.central_pixel_coordinates == (0.5, 0.0)
    assert array_2d.geometry.shape_native_scaled == pytest.approx((0.2, 0.1))
    assert array_2d.geometry.scaled_maxima == pytest.approx((1.1, 1.05), 1e-4)
    assert array_2d.geometry.scaled_minima == pytest.approx((0.9, 0.95), 1e-4)

    array_2d = aa.Array2D.no_mask(
        values=np.ones((9,)),
        shape_native=(3, 3),
        pixel_scales=(2.0, 1.0),
        sub_size=1,
        origin=(-1.0, -2.0),
    )

    assert array_2d == pytest.approx(np.ones((9,)), 1e-4)
    assert array_2d.slim == pytest.approx(np.ones((9,)), 1e-4)
    assert array_2d.native == pytest.approx(np.ones((3, 3)), 1e-4)
    assert array_2d.pixel_scales == (2.0, 1.0)
    assert array_2d.geometry.central_pixel_coordinates == (1.0, 1.0)
    assert array_2d.geometry.shape_native_scaled == pytest.approx((6.0, 3.0))
    assert array_2d.origin == (-1.0, -2.0)
    assert array_2d.geometry.scaled_maxima == pytest.approx((2.0, -0.5), 1e-4)
    assert array_2d.geometry.scaled_minima == pytest.approx((-4.0, -3.5), 1e-4)

    array_2d = aa.Array2D.no_mask(
        values=np.ones((9,)),
        shape_native=(3, 3),
        pixel_scales=(2.0, 1.0),
        sub_size=1,
        origin=(-1.0, -2.0),
    )

    assert array_2d == pytest.approx(np.ones((9,)), 1e-4)
    assert array_2d.slim == pytest.approx(np.ones((9,)), 1e-4)
    assert array_2d.native == pytest.approx(np.ones((3, 3)), 1e-4)
    assert array_2d.pixel_scales == (2.0, 1.0)
    assert array_2d.geometry.central_pixel_coordinates == (1.0, 1.0)
    assert array_2d.geometry.shape_native_scaled == pytest.approx((6.0, 3.0))
    assert array_2d.origin == (-1.0, -2.0)
    assert array_2d.geometry.scaled_maxima == pytest.approx((2.0, -0.5), 1e-4)
    assert array_2d.geometry.scaled_minima == pytest.approx((-4.0, -3.5), 1e-4)


def test__apply_mask():

    mask = aa.Mask2D(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)
    array_2d = aa.Array2D.no_mask(
        values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        shape_native=(2, 1),
        pixel_scales=2.0,
        sub_size=2,
    )
    array_2d = array_2d.apply_mask(mask=mask)

    assert (
        array_2d.native == np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()


def test__full():

    array_2d = aa.Array2D.full(
        fill_value=2.0, shape_native=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0)
    )

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
    assert (array_2d.slim == np.array([2.0, 2.0, 2.0, 2.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 1.0)

    array_2d = aa.Array2D.full(
        fill_value=2.0,
        shape_native=(1, 1),
        pixel_scales=1.0,
        sub_size=2,
        origin=(0.0, 1.0),
    )

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
    assert (array_2d.slim == np.array([2.0, 2.0, 2.0, 2.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 1.0)
    assert array_2d.mask.sub_size == 2


def test__ones():

    array_2d = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
    assert (array_2d.slim == np.array([1.0, 1.0, 1.0, 1.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 0.0)

    array_2d = aa.Array2D.ones(
        shape_native=(1, 1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0)
    )

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
    assert (array_2d.slim == np.array([1.0, 1.0, 1.0, 1.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 1.0)
    assert array_2d.mask.sub_size == 2


def test__zeros():

    array_2d = aa.Array2D.zeros(shape_native=(2, 2), pixel_scales=1.0)

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
    assert (array_2d.slim == np.array([0.0, 0.0, 0.0, 0.0])).all()

    array_2d = aa.Array2D.zeros(
        shape_native=(1, 1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0)
    )

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
    assert (array_2d.slim == np.array([0.0, 0.0, 0.0, 0.0])).all()
    assert array_2d.pixel_scales == (1.0, 1.0)
    assert array_2d.origin == (0.0, 1.0)
    assert array_2d.mask.sub_size == 2


def test__from_fits():

    array_2d = aa.Array2D.from_fits(
        file_path=path.join(test_data_dir, "4x3_ones.fits"), hdu=0, pixel_scales=1.0
    )

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == np.ones((4, 3))).all()
    assert (array_2d.slim == np.ones((12,))).all()


def test__from_fits__loads_and_stores_header_info():

    array_2d = aa.Array2D.from_fits(
        file_path=path.join(test_data_dir, "3x3_ones.fits"), hdu=0, pixel_scales=1.0
    )

    assert array_2d.header.header_sci_obj["BITPIX"] == -64
    assert array_2d.header.header_hdu_obj["BITPIX"] == -64

    array_2d = aa.Array2D.from_fits(
        file_path=path.join(test_data_dir, "4x3_ones.fits"), hdu=0, pixel_scales=1.0
    )

    assert array_2d.header.header_sci_obj["BITPIX"] == -64
    assert array_2d.header.header_hdu_obj["BITPIX"] == -64


def test__from_yx_and_values():

    array_2d = aa.Array2D.from_yx_and_values(
        y=[0.5, 0.5, -0.5, -0.5],
        x=[-0.5, 0.5, -0.5, 0.5],
        values=[1.0, 2.0, 3.0, 4.0],
        shape_native=(2, 2),
        pixel_scales=1.0,
    )

    assert (array_2d.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

    array_2d = aa.Array2D.from_yx_and_values(
        y=[0.0, 1.0, -1.0, 0.0, -1.0, 1.0],
        x=[-0.5, 0.5, 0.5, 0.5, -0.5, -0.5],
        values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    assert (array_2d.native == np.array([[3.0, 2.0], [6.0, 4.0], [5.0, 1.0]])).all()


def test__output_to_fits():

    array_2d = aa.Array2D.from_fits(
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

    array_2d.output_to_fits(file_path=path.join(output_data_dir, "array.fits"))

    array_from_out = aa.Array2D.from_fits(
        file_path=path.join(output_data_dir, "array.fits"), hdu=0, pixel_scales=1.0
    )

    assert (array_from_out.native == np.ones((3, 3))).all()


def test__manual_native__exception_raised_if_input_array_is_2d_and_not_sub_shape_of_mask():
    with pytest.raises(aa.exc.ArrayException):
        mask = aa.Mask2D.all_false(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)
        aa.Array2D(values=[[1.0], [3.0]], mask=mask)

    with pytest.raises(aa.exc.ArrayException):
        mask = aa.Mask2D.all_false(shape_native=(2, 2), pixel_scales=1.0, sub_size=2)
        aa.Array2D(values=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

    with pytest.raises(aa.exc.ArrayException):
        mask = aa.Mask2D.all_false(shape_native=(2, 2), pixel_scales=1.0, sub_size=2)
        aa.Array2D(values=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], mask=mask)


def test__manual_mask__exception_raised_if_input_array_is_1d_and_not_number_of_masked_sub_pixels():
    with pytest.raises(aa.exc.ArrayException):
        mask = aa.Mask2D(
            mask=[[False, False], [True, False]], pixel_scales=1.0, sub_size=1
        )
        aa.Array2D(values=[1.0, 2.0, 3.0, 4.0], mask=mask)

    with pytest.raises(aa.exc.ArrayException):
        mask = aa.Mask2D(
            mask=[[False, False], [True, False]], pixel_scales=1.0, sub_size=1
        )
        aa.Array2D(values=[1.0, 2.0], mask=mask)

    with pytest.raises(aa.exc.ArrayException):
        mask = aa.Mask2D(
            mask=[[False, True], [True, True]], pixel_scales=1.0, sub_size=2
        )
        aa.Array2D(values=[1.0, 2.0, 4.0], mask=mask)

    with pytest.raises(aa.exc.ArrayException):
        mask = aa.Mask2D(
            mask=[[False, True], [True, True]], pixel_scales=1.0, sub_size=2
        )
        aa.Array2D(values=[1.0, 2.0, 3.0, 4.0, 5.0], mask=mask)


def test__resized_from():

    array_2d = np.ones((5, 5))
    array_2d[2, 2] = 2.0

    array_2d = aa.Array2D.no_mask(values=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

    array_2d = array_2d.resized_from(new_shape=(7, 7))

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

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == arr_resized_manual).all()
    assert array_2d.mask.pixel_scales == (1.0, 1.0)

    array_2d = np.ones((5, 5))
    array_2d[2, 2] = 2.0

    array_2d = aa.Array2D.no_mask(values=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

    array_2d = array_2d.resized_from(new_shape=(3, 3))

    arr_resized_manual = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])

    assert type(array_2d) == aa.Array2D
    assert (array_2d.native == arr_resized_manual).all()
    assert array_2d.mask.pixel_scales == (1.0, 1.0)


def test__padded_before_convolution_from():
    array_2d = np.ones((5, 5))
    array_2d[2, 2] = 2.0

    array_2d = aa.Array2D.no_mask(values=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

    new_arr = array_2d.padded_before_convolution_from(kernel_shape=(3, 3))

    assert type(new_arr) == aa.Array2D
    assert new_arr.native[0, 0] == 0.0
    assert new_arr.shape_native == (7, 7)
    assert new_arr.mask.pixel_scales == (1.0, 1.0)

    new_arr = array_2d.padded_before_convolution_from(kernel_shape=(5, 5))

    assert type(new_arr) == aa.Array2D
    assert new_arr.native[0, 0] == 0.0
    assert new_arr.shape_native == (9, 9)
    assert new_arr.mask.pixel_scales == (1.0, 1.0)

    array_2d = np.ones((9, 9))
    array_2d[4, 4] = 2.0

    array_2d = aa.Array2D.no_mask(values=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

    new_arr = array_2d.padded_before_convolution_from(kernel_shape=(7, 7))

    assert type(new_arr) == aa.Array2D
    assert new_arr.native[0, 0] == 0.0
    assert new_arr.shape_native == (15, 15)
    assert new_arr.mask.pixel_scales == (1.0, 1.0)


def test__trimmed_after_convolution_from():
    array_2d = np.ones((5, 5))
    array_2d[2, 2] = 2.0

    array_2d = aa.Array2D.no_mask(values=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

    new_arr = array_2d.trimmed_after_convolution_from(kernel_shape=(3, 3))

    assert type(new_arr) == aa.Array2D
    assert (
        new_arr.native == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    ).all()
    assert new_arr.mask.pixel_scales == (1.0, 1.0)

    new_arr = array_2d.trimmed_after_convolution_from(kernel_shape=(5, 5))

    assert type(new_arr) == aa.Array2D
    assert (new_arr.native == np.array([[2.0]])).all()
    assert new_arr.mask.pixel_scales == (1.0, 1.0)

    array_2d = np.ones((9, 9))
    array_2d[4, 4] = 2.0

    array_2d = aa.Array2D.no_mask(values=array_2d, sub_size=1, pixel_scales=(1.0, 1.0))

    new_arr = array_2d.trimmed_after_convolution_from(kernel_shape=(7, 7))

    assert type(new_arr) == aa.Array2D
    assert (
        new_arr.native == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    ).all()
    assert new_arr.mask.pixel_scales == (1.0, 1.0)


def test__zoomed_around_mask():
    array_2d = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]

    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    arr_masked = aa.Array2D(values=array_2d, mask=mask)

    arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

    assert (arr_zoomed.native == np.array([[6.0, 7.0], [10.0, 11.0]])).all()

    mask = aa.Mask2D(
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

    arr_masked = aa.Array2D(values=array_2d, mask=mask)
    arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

    assert (arr_zoomed.native == np.array([[0.0, 6.0, 7.0], [9.0, 10.0, 11.0]])).all()

    mask = aa.Mask2D(
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

    arr_masked = aa.Array2D(values=array_2d, mask=mask)
    arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
    assert (arr_zoomed.native == np.array([[2.0, 0.0], [6.0, 7.0], [10.0, 11.0]])).all()


def test__zoomed_around_mask__origin_updated():
    array_2d = np.ones(shape=(4, 4))

    mask = aa.Mask2D(
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

    arr_masked = aa.Array2D(values=array_2d, mask=mask)

    arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

    assert arr_zoomed.mask.origin == (0.0, 0.0)

    array_2d = np.ones(shape=(6, 6))

    mask = aa.Mask2D(
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

    arr_masked = aa.Array2D(values=array_2d, mask=mask)

    arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

    assert arr_zoomed.mask.origin == (0.0, 1.0)


def test__extent_of_zoomed_array():
    array_2d = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]

    mask = aa.Mask2D(
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

    arr_masked = aa.Array2D(values=array_2d, mask=mask)

    extent = arr_masked.extent_of_zoomed_array(buffer=1)

    print(extent)
    print(arr_masked.origin)

    assert extent == pytest.approx(np.array([-4.0, 6.0, -2.0, 3.0]), 1.0e-4)


def test__binned_across_columns():

    array = aa.Array2D.no_mask(values=np.ones((4, 3)), pixel_scales=1.0)

    assert (array.binned_across_columns == np.array([1.0, 1.0, 1.0, 1.0])).all()

    array = aa.Array2D.no_mask(values=np.ones((3, 4)), pixel_scales=1.0)

    assert (array.binned_across_columns == np.array([1.0, 1.0, 1.0])).all()

    array = aa.Array2D.no_mask(
        values=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]]),
        pixel_scales=1.0,
    )

    assert (array.binned_across_columns == np.array([2.0, 6.0, 9.0])).all()

    mask = aa.Mask2D(
        mask=[[False, False, True], [False, False, False], [False, False, False]],
        pixel_scales=1.0,
    )

    array = aa.Array2D(
        values=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]]),
        mask=mask,
    )

    assert (array.binned_across_columns == np.array([1.5, 6.0, 9.0])).all()


def test__binned_across_rows():

    array = aa.Array2D.no_mask(values=np.ones((4, 3)), pixel_scales=1.0)

    assert (array.binned_across_rows == np.array([1.0, 1.0, 1.0])).all()

    array = aa.Array2D.no_mask(values=np.ones((3, 4)), pixel_scales=1.0)

    assert (array.binned_across_rows == np.array([1.0, 1.0, 1.0, 1.0])).all()

    array = aa.Array2D.no_mask(
        values=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]]),
        pixel_scales=1.0,
    )

    assert (array.binned_across_rows == np.array([2.0, 6.0, 9.0])).all()

    mask = aa.Mask2D(
        mask=[[False, False, False], [False, False, False], [True, False, False]],
        pixel_scales=1.0,
    )

    array = aa.Array2D(
        values=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]]),
        mask=mask,
    )

    assert (array.binned_across_rows == np.array([1.5, 6.0, 9.0])).all()


def test__header__modified_julian_date():

    header_sci_obj = {"DATE-OBS": "2000-01-01", "TIME-OBS": "00:00:00"}

    header = aa.Header(header_sci_obj=header_sci_obj, header_hdu_obj=None)

    assert header.modified_julian_date == 51544.0


def test__array_2d__recursive_shape_storage():

    array_2d = aa.Array2D.no_mask(
        values=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, sub_size=1
    )

    assert (array_2d.native.slim.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
    assert (array_2d.slim.native.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
