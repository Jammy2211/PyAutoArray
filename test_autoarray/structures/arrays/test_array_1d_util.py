import autoarray as aa
import numpy as np
import pytest

import os
from os import path

test_data_path = os.path.join(
    "{}".format(os.path.dirname(os.path.realpath(__file__))), "files"
)


def test__array_1d_slim_from():
    mask_1d = np.array([False, False, False, False])

    array_1d_native = np.array([1.0, 2.0, 3.0, 4.0])

    array_1d_slim = aa.util.array_1d.array_1d_slim_from(
        array_1d_native=array_1d_native, mask_1d=mask_1d, sub_size=1
    )

    assert (array_1d_slim == np.array([1.0, 2.0, 3.0, 4.0])).all()

    mask_1d = np.array([True, False, False, True, False, False])

    array_1d_native = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    array_1d_slim = aa.util.array_1d.array_1d_slim_from(
        array_1d_native=array_1d_native, mask_1d=mask_1d, sub_size=1
    )

    assert (array_1d_slim == np.array([2.0, 3.0, 5.0, 6.0])).all()

    mask_1d = np.array([True, True, False, True, False, False])

    array_1d_native = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )

    array_1d_slim = aa.util.array_1d.array_1d_slim_from(
        array_1d_native=array_1d_native, mask_1d=mask_1d, sub_size=2
    )

    assert (array_1d_slim == np.array([5.0, 6.0, 9.0, 10.0, 11.0, 12.0])).all()


def test__array_1d_native_from():
    mask_1d = np.array([False, False, False, False])

    array_1d_slim = np.array([1.0, 2.0, 3.0, 4.0])

    array_1d_native = aa.util.array_1d.array_1d_native_from(
        array_1d_slim=array_1d_slim, mask_1d=mask_1d, sub_size=1
    )

    assert (array_1d_native == np.array([1.0, 2.0, 3.0, 4.0])).all()

    mask_1d = np.array([False, False, True, True, False, False])

    array_1d_slim = np.array([1.0, 2.0, 3.0, 4.0])

    array_1d_native = aa.util.array_1d.array_1d_native_from(
        array_1d_slim=array_1d_slim, mask_1d=mask_1d, sub_size=1
    )

    assert (array_1d_native == np.array([1.0, 2.0, 0.0, 0.0, 3.0, 4.0])).all()

    mask_1d = np.array([True, False, False, True, False, False])

    array_1d_slim = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    array_1d_native = aa.util.array_1d.array_1d_native_from(
        array_1d_slim=array_1d_slim, mask_1d=mask_1d, sub_size=2
    )

    assert (
        array_1d_native
        == np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0])
    ).all()


def test__numpy_array_1d_via_fits_from():
    arr = aa.util.array_1d.numpy_array_1d_via_fits_from(
        file_path=path.join(test_data_path, "3_ones.fits"), hdu=0
    )

    assert (arr == np.ones((3))).all()


def test__numpy_array_1d_to_fits__output_and_load():
    file_path = path.join(test_data_path, "array_out.fits")

    if path.exists(file_path):
        os.remove(file_path)

    arr = np.array([10.0, 30.0, 40.0, 92.0, 19.0, 20.0])

    aa.util.array_1d.numpy_array_1d_to_fits(
        arr, file_path=file_path, header_dict={"A": 1}
    )

    array_load = aa.util.array_1d.numpy_array_1d_via_fits_from(
        file_path=file_path,
        hdu=0,
    )

    assert (arr == array_load).all()

    header_load = aa.util.array_2d.header_obj_from(file_path=file_path, hdu=0)

    assert header_load["A"] == 1
