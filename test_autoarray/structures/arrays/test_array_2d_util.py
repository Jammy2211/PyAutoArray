from autoarray import util

import os
import numpy as np
import pytest

from astropy.io import fits

test_data_path = os.path.join(
    "{}".format(os.path.dirname(os.path.realpath(__file__))), "files"
)


def test__resized_array_2d_from__trimming():
    array = np.ones((7, 7))
    array[3, 3] = 2.0

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(3, 3)
    )

    assert (
        resized_array_2d
        == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    ).all()

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(4, 4)
    )

    assert (
        resized_array_2d
        == np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
    ).all()

    array = np.ones((6, 6))
    array[2:4, 2:4] = 2.0

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(4, 4)
    )

    assert (
        resized_array_2d
        == np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 2.0, 2.0, 1.0],
                [1.0, 2.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
    ).all()

    array = np.ones((6, 6))
    array[2:4, 2:4] = 2.0

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(3, 3)
    )

    assert (
        resized_array_2d
        == np.array([[2.0, 2.0, 1.0], [2.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    ).all()

    array = np.ones((5, 4))
    array[2, 1:3] = 2.0

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(3, 2)
    )

    assert (resized_array_2d == np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]])).all()

    array = np.ones((4, 5))
    array[1:3, 2] = 2.0

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(2, 3)
    )

    assert (resized_array_2d == np.array([[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]])).all()


def test__resized_array_2d_from__trimming_with_new_origin():
    array = np.ones((7, 7))
    array[4, 4] = 2.0
    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(3, 3), origin=(4, 4)
    )
    assert (
        resized_array_2d
        == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    ).all()

    array = np.ones((6, 6))
    array[3, 4] = 2.0
    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(3, 3), origin=(3, 4)
    )
    assert (
        resized_array_2d
        == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    ).all()

    array = np.ones((9, 8))
    array[4, 3] = 2.0
    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(3, 3), origin=(4, 3)
    )
    assert (
        resized_array_2d
        == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    ).all()

    array = np.ones((8, 9))
    array[3, 5] = 2.0
    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(3, 3), origin=(3, 5)
    )
    assert (
        resized_array_2d
        == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    ).all()


def test__resized_array_2d_from__padding():
    array = np.ones((3, 3))
    array[1, 1] = 2.0

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(5, 5)
    )

    assert (
        resized_array_2d
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(4, 4)
    )

    assert (
        resized_array_2d
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
            ]
        )
    ).all()

    array = np.ones((4, 4))
    array[1:3, 1:3] = 2.0

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(6, 6)
    )

    assert (
        resized_array_2d
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    array = np.ones((4, 4))
    array[1:3, 1:3] = 2.0

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(5, 5)
    )

    assert (
        resized_array_2d
        == np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 2.0, 2.0, 1.0, 0.0],
                [1.0, 2.0, 2.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    array = np.ones((3, 2))
    array[1, 0:2] = 2.0

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(5, 4)
    )

    assert (
        resized_array_2d
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    array = np.ones((2, 3))
    array[0:2, 1] = 2.0

    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(4, 5)
    )

    assert (
        resized_array_2d
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()


def test__resized_array_2d_from__padding_with_new_origin():
    array = np.ones((3, 3))
    array[2, 2] = 2.0
    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(5, 5), origin=(2, 2)
    )

    assert (
        resized_array_2d
        == np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    array = np.ones((2, 3))
    array[0, 0] = 2.0
    resized_array_2d = util.array_2d.resized_array_2d_from(
        array_2d=array, resized_shape=(4, 5), origin=(0, 1)
    )

    assert (
        resized_array_2d
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
            ]
        )
    ).all()


def test__numpy_array_2d_via_fits_from():
    arr = util.array_2d.numpy_array_2d_via_fits_from(
        file_path=os.path.join(test_data_path, "3x3_ones.fits"), hdu=0
    )

    assert (arr == np.ones((3, 3))).all()

    arr = util.array_2d.numpy_array_2d_via_fits_from(
        file_path=os.path.join(test_data_path, "4x3_ones.fits"), hdu=0
    )

    assert (arr == np.ones((4, 3))).all()


def test__numpy_array_2d_to_fits():
    file_path = os.path.join(test_data_path, "array_out.fits")

    if os.path.exists(file_path):
        os.remove(file_path)

    arr = np.array([[10.0, 30.0, 40.0], [92.0, 19.0, 20.0]])

    util.array_2d.numpy_array_2d_to_fits(arr, file_path=file_path)

    array_load = util.array_2d.numpy_array_2d_via_fits_from(file_path=file_path, hdu=0)

    assert (arr == array_load).all()


def test__numpy_array_2d_to_fits__header_dict():
    file_path = os.path.join(test_data_path, "array_out.fits")

    if os.path.exists(file_path):
        os.remove(file_path)

    arr = np.array([[10.0, 30.0, 40.0], [92.0, 19.0, 20.0]])

    util.array_2d.numpy_array_2d_to_fits(arr, file_path=file_path, header_dict={"A": 1})

    header = util.array_2d.header_obj_from(file_path=file_path, hdu=0)

    assert header["A"] == 1


def test__header_obj_from():
    header_obj = util.array_2d.header_obj_from(
        file_path=os.path.join(test_data_path, "3x3_ones.fits"), hdu=0
    )

    assert isinstance(header_obj, fits.header.Header)
    assert header_obj["BITPIX"] == -64


def test__replace_noise_map_2d_values_where_image_2d_values_are_negative():
    image_2d = np.ones(shape=(2, 2))

    noise_map_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

    noise_map_2d = (
        util.array_2d.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=1.0
        )
    )

    assert (noise_map_2d == noise_map_2d).all()

    image_2d = -1.0 * np.ones(shape=(2, 2))

    noise_map_2d = np.array([[1.0, 0.5], [0.25, 0.125]])

    noise_map_2d = (
        util.array_2d.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=10.0
        )
    )

    assert (noise_map_2d == noise_map_2d).all()

    noise_map_2d = (
        util.array_2d.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=4.0
        )
    )

    assert (noise_map_2d == np.array([[1.0, 0.5], [0.25, 0.25]])).all()

    noise_map_2d = np.array([[1.0, 0.5], [0.25, 0.125]])

    noise_map_2d = (
        util.array_2d.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=2.0
        )
    )

    assert (noise_map_2d == np.array([[1.0, 0.5], [0.5, 0.5]])).all()

    noise_map_2d = np.array([[1.0, 0.5], [0.25, 0.125]])

    noise_map_2d = (
        util.array_2d.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=1.0
        )
    )

    assert (noise_map_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

    noise_map_2d = np.array([[1.0, 0.5], [0.25, 0.125]])

    noise_map_2d = (
        util.array_2d.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=0.5
        )
    )

    assert (noise_map_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()


def test__same_as_above__image_not_all_negative():
    image_2d = np.array([[1.0, -2.0], [5.0, -4.0]])

    noise_map_2d = np.array([[3.0, 1.0], [4.0, 8.0]])

    noise_map_2d = (
        util.array_2d.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=1.0
        )
    )

    assert (noise_map_2d == np.array([[3.0, 2.0], [4.0, 8.0]])).all()

    image_2d = np.array([[-10.0, -20.0], [100.0, -30.0]])

    noise_map_2d = np.array([[1.0, 2.0], [40.0, 3.0]])

    noise_map_2d = (
        util.array_2d.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=5.0
        )
    )

    assert (noise_map_2d == np.array([[2.0, 4.0], [40.0, 6.0]])).all()


def test__index_2d_for_index_slim_from():
    indexes_1d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    index_2d_for_index_1d = util.array_2d.index_2d_for_index_slim_from(
        indexes_slim=indexes_1d, shape_native=(3, 3)
    )

    assert (
        index_2d_for_index_1d
        == np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )
    ).all()

    indexes_1d = np.array([0, 1, 2, 3, 4, 5])

    index_2d_for_index_1d = util.array_2d.index_2d_for_index_slim_from(
        indexes_slim=indexes_1d, shape_native=(2, 3)
    )

    assert (
        index_2d_for_index_1d
        == np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    ).all()

    indexes_2d = util.array_2d.index_2d_for_index_slim_from(
        indexes_slim=indexes_1d, shape_native=(3, 2)
    )

    assert (
        indexes_2d == np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
    ).all()

    indexes_1d = np.array([1, 4, 7, 8, 0, 2, 3, 5, 6])

    index_2d_for_index_1d = util.array_2d.index_2d_for_index_slim_from(
        indexes_slim=indexes_1d, shape_native=(3, 3)
    )

    assert (
        index_2d_for_index_1d
        == np.array(
            [[0, 1], [1, 1], [2, 1], [2, 2], [0, 0], [0, 2], [1, 0], [1, 2], [2, 0]]
        )
    ).all()


def test__index_slim_for_index_2d_from():
    indexes_2d = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    index_slim_for_index_2d = util.array_2d.index_slim_for_index_2d_from(
        indexes_2d=indexes_2d, shape_native=(3, 3)
    )

    assert (index_slim_for_index_2d == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    indexes_2d = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])

    index_slim_for_index_2d = util.array_2d.index_slim_for_index_2d_from(
        indexes_2d=indexes_2d, shape_native=(2, 3)
    )

    assert (index_slim_for_index_2d == np.array([0, 1, 2, 3, 4, 5])).all()

    indexes_2d = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])

    index_slim_for_index_2d = util.array_2d.index_slim_for_index_2d_from(
        indexes_2d=indexes_2d, shape_native=(3, 2)
    )

    assert (index_slim_for_index_2d == np.array([0, 1, 2, 3, 4, 5])).all()

    indexes_2d = np.array(
        [[0, 1], [1, 1], [2, 1], [2, 2], [0, 0], [0, 2], [1, 0], [1, 2], [2, 0]]
    )

    index_slim_for_index_2d = util.array_2d.index_slim_for_index_2d_from(
        indexes_2d=indexes_2d, shape_native=(3, 3)
    )

    assert (index_slim_for_index_2d == np.array([1, 4, 7, 8, 0, 2, 3, 5, 6])).all()


def test__array_2d_slim_from__sub_size_1():
    array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    array_2d_slim = util.array_2d.array_2d_slim_from(
        mask_2d=mask, array_2d_native=array_2d, sub_size=1
    )

    assert (array_2d_slim == np.array([5])).all()

    array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    mask = np.array([[True, False, True], [False, False, False], [True, False, True]])

    array_2d_slim = util.array_2d.array_2d_slim_from(
        mask_2d=mask, array_2d_native=array_2d, sub_size=1
    )

    assert (array_2d_slim == np.array([2, 4, 5, 6, 8])).all()

    array_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    mask = np.array(
        [
            [True, False, True, True],
            [False, False, False, True],
            [True, False, True, False],
        ]
    )

    array_2d_slim = util.array_2d.array_2d_slim_from(
        mask_2d=mask, array_2d_native=array_2d, sub_size=1
    )

    assert (array_2d_slim == np.array([2, 5, 6, 7, 10, 12])).all()

    array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    mask = np.array(
        [
            [True, False, True],
            [False, False, False],
            [True, False, True],
            [True, True, True],
        ]
    )

    array_2d_slim = util.array_2d.array_2d_slim_from(
        mask_2d=mask, array_2d_native=array_2d, sub_size=1
    )

    assert (array_2d_slim == np.array([2, 4, 5, 6, 8])).all()


def test__array_2d_slim_from__sub_size_2_or_3():
    sub_array_2d = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
        ]
    )

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    sub_array_1d = util.array_2d.array_2d_slim_from(
        array_2d_native=sub_array_2d, mask_2d=mask, sub_size=2
    )

    assert (sub_array_1d == np.array([15, 16, 3, 4])).all()

    mask = np.array([[True, False, True], [True, False, True], [True, True, False]])

    sub_array_1d = util.array_2d.array_2d_slim_from(
        array_2d_native=sub_array_2d, mask_2d=mask, sub_size=2
    )

    assert (sub_array_1d == np.array([3, 4, 9, 10, 15, 16, 3, 4, 11, 12, 17, 18])).all()

    sub_array_2d = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 7, 7],
            [7, 8, 9, 10, 11, 12, 7, 7, 7],
            [13, 14, 15, 16, 17, 18, 7, 7, 7],
            [1, 2, 3, 4, 5, 6, 7, 7, 7],
            [7, 8, 9, 10, 11, 12, 7, 7, 7],
            [13, 14, 15, 16, 17, 18, 7, 7, 7],
        ]
    )

    mask = np.array(
        [
            [True, False, True, True],
            [False, False, False, True],
            [True, False, True, False],
        ]
    )

    sub_array_1d = util.array_2d.array_2d_slim_from(
        array_2d_native=sub_array_2d, mask_2d=mask, sub_size=2
    )

    assert (
        sub_array_1d
        == np.array(
            [
                3,
                4,
                9,
                10,
                13,
                14,
                1,
                2,
                15,
                16,
                3,
                4,
                17,
                18,
                5,
                6,
                9,
                10,
                15,
                16,
                7,
                7,
                7,
                7,
            ]
        )
    ).all()

    sub_array_2d = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7],
        ]
    )

    mask = np.array(
        [
            [True, False, True],
            [False, False, False],
            [True, False, True],
            [True, True, True],
        ]
    )

    sub_array_1d = util.array_2d.array_2d_slim_from(
        array_2d_native=sub_array_2d, mask_2d=mask, sub_size=2
    )

    assert (
        sub_array_1d
        == np.array(
            [3, 4, 9, 10, 13, 14, 1, 2, 15, 16, 3, 4, 17, 18, 5, 6, 9, 10, 15, 16]
        )
    ).all()

    sub_array_2d = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
        ]
    )

    mask = np.array([[False, True], [True, False]])

    sub_array_1d = util.array_2d.array_2d_slim_from(
        array_2d_native=sub_array_2d, mask_2d=mask, sub_size=3
    )

    assert (
        sub_array_1d
        == np.array([1, 2, 3, 7, 8, 9, 13, 14, 15, 4, 5, 6, 10, 11, 12, 16, 17, 18])
    ).all()


def test__array_2d_slim_from__complex_array():
    array_2d = np.array(
        [
            [1 + 1j, 2 + 2j, 3 + 3],
            [4 + 4j, 5 + 5j, 6 + 6j],
            [7 + 7j, 8 + 8j, 9 + 9j],
        ]
    )

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    array_2d_slim = util.array_2d.array_2d_slim_complex_from(
        mask=mask, array_2d_native=array_2d, sub_size=1
    )

    assert (array_2d_slim == np.array([5 + 5j])).all()


def test__array_2d_native_from__sub_size_1():
    array_2d_slim = np.array([1.0, 2.0, 3.0, 4.0])

    mask = np.full(fill_value=False, shape=(2, 2))

    array_2d = util.array_2d.array_2d_native_from(
        array_2d_slim=array_2d_slim, mask_2d=mask, sub_size=1
    )

    assert (array_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

    array_2d_slim = np.array([1.0, 2.0, 3.0])

    mask = np.array([[False, False], [False, True]])

    array_2d = util.array_2d.array_2d_native_from(
        array_2d_slim=array_2d_slim, mask_2d=mask, sub_size=1
    )

    assert (array_2d == np.array([[1.0, 2.0], [3.0, 0.0]])).all()

    array_2d_slim = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])

    mask = np.array(
        [
            [False, False, True, True],
            [False, True, True, True],
            [False, False, True, False],
        ]
    )

    array_2d = util.array_2d.array_2d_native_from(
        array_2d_slim=array_2d_slim, mask_2d=mask, sub_size=1
    )

    assert (
        array_2d
        == np.array(
            [[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [-1.0, -2.0, 0.0, -3.0]]
        )
    ).all()


def test__array_2d_native_from__sub_size_2():
    array_2d_slim = np.array(
        [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]
    )

    mask = np.array([[False, False], [False, True]])

    array_2d = util.array_2d.array_2d_native_from(
        array_2d_slim=array_2d_slim, mask_2d=mask, sub_size=2
    )

    assert (
        array_2d
        == np.array(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 0.0],
            ]
        )
    ).all()


def test__array_2d_native_from__compelx_array():
    array_2d_slim = np.array(
        [1.0 + 1j, 2.0 + 2j, 3.0 + 3j, 4.0 + 4j], dtype="complex128"
    )

    array_2d = util.array_2d.array_2d_native_complex_via_indexes_from(
        array_2d_slim=array_2d_slim,
        sub_shape_native=(2, 2),
        native_index_for_slim_index_2d=np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype="int"
        ),
    )

    assert (array_2d == np.array([[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]])).all()
