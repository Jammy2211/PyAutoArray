from astropy.io import fits
from os import path
import os
import numpy as np
import shutil

import autoarray as aa

fits_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "array_1d"
)

test_data_dir = path.join(
    "{}".format(os.path.dirname(os.path.realpath(__file__))), "files"
)

output_data_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "array", "output_test"
)


def create_fits(
    fits_path,
):

    if path.exists(fits_path):
        shutil.rmtree(fits_path)

    os.makedirs(fits_path)

    hdu_list = fits.HDUList()
    hdu_list.append(fits.ImageHDU(np.ones(3)))
    hdu_list[0].header.set("BITPIX", -64, "")
    hdu_list.writeto(path.join(fits_path, "3_ones.fits"))

    hdu_list = fits.HDUList()
    hdu_list.append(fits.ImageHDU(np.ones(4)))
    hdu_list[0].header.set("BITPIX", -64, "")
    hdu_list.writeto(path.join(fits_path, "4_ones.fits"))


def clean_fits(fits_path):

    if path.exists(fits_path):
        shutil.rmtree(fits_path)


def test__manual_slim():

    array_1d = aa.Array1D.manual_slim(array=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

    assert type(array_1d) == aa.Array1D
    assert (array_1d.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (array_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (array_1d.grid_radial == np.array(([0.0, 1.0, 2.0, 3.0]))).all()
    assert array_1d.pixel_scale == 1.0
    assert array_1d.pixel_scales == (1.0,)
    assert array_1d.origin == (0.0,)


def test__manual_native():

    array_1d = aa.Array1D.manual_native(array=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

    assert type(array_1d) == aa.Array1D
    assert (array_1d.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (array_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (array_1d.grid_radial == np.array(([0.0, 1.0, 2.0, 3.0]))).all()
    assert array_1d.pixel_scale == 1.0
    assert array_1d.pixel_scales == (1.0,)
    assert array_1d.origin == (0.0,)


def test__manual_mask():

    mask = aa.Mask1D.manual(
        mask=[True, False, False, True, False, False], pixel_scales=1.0, sub_size=1
    )

    array_1d = aa.Array1D.manual_mask(
        array=[100.0, 1.0, 2.0, 100.0, 3.0, 4.0], mask=mask
    )

    assert type(array_1d) == aa.Array1D
    assert (array_1d.native == np.array([0.0, 1.0, 2.0, 0.0, 3.0, 4.0])).all()
    assert (array_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert array_1d.pixel_scale == 1.0
    assert array_1d.pixel_scales == (1.0,)
    assert array_1d.origin == (0.0,)


def test__full():

    array_1d = aa.Array1D.full(fill_value=1.0, shape_native=4, pixel_scales=1.0)

    assert type(array_1d) == aa.Array1D
    assert (array_1d.native == np.array([1.0, 1.0, 1.0, 1.0])).all()
    assert (array_1d.slim == np.array([1.0, 1.0, 1.0, 1.0])).all()
    assert array_1d.pixel_scale == 1.0
    assert array_1d.pixel_scales == (1.0,)
    assert array_1d.origin == (0.0,)

    array_1d = aa.Array1D.full(
        fill_value=2.0, shape_native=3, pixel_scales=3.0, sub_size=2, origin=(4.0,)
    )

    assert type(array_1d) == aa.Array1D
    print(array_1d.native)
    assert (array_1d.native == np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])).all()
    assert (array_1d.slim == np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])).all()
    assert array_1d.pixel_scale == 3.0
    assert array_1d.pixel_scales == (3.0,)
    assert array_1d.origin == (4.0,)


def test__ones():

    array_1d = aa.Array1D.ones(
        shape_native=3, pixel_scales=3.0, sub_size=2, origin=(4.0,)
    )

    assert type(array_1d) == aa.Array1D
    assert (array_1d.native == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()
    assert (array_1d.slim == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])).all()
    assert array_1d.pixel_scale == 3.0
    assert array_1d.pixel_scales == (3.0,)
    assert array_1d.origin == (4.0,)


def test__zeros():

    array_1d = aa.Array1D.zeros(
        shape_native=3, pixel_scales=3.0, sub_size=2, origin=(4.0,)
    )

    assert type(array_1d) == aa.Array1D
    assert (array_1d.native == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])).all()
    assert (array_1d.slim == np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])).all()
    assert array_1d.pixel_scale == 3.0
    assert array_1d.pixel_scales == (3.0,)
    assert array_1d.origin == (4.0,)


def test__from_fits():

    create_fits(fits_path=fits_path)

    arr = aa.Array1D.from_fits(
        file_path=path.join(fits_path, "3_ones.fits"), hdu=0, pixel_scales=1.0
    )

    assert type(arr) == aa.Array1D
    assert (arr.native == np.ones((3,))).all()
    assert (arr.slim == np.ones(3)).all()

    arr = aa.Array1D.from_fits(
        file_path=path.join(fits_path, "4_ones.fits"), hdu=0, pixel_scales=1.0
    )

    assert type(arr) == aa.Array1D
    assert (arr == np.ones((4,))).all()
    assert (arr.native == np.ones((4,))).all()
    assert (arr.slim == np.ones((4,))).all()

    clean_fits(fits_path=fits_path)


def test__from_fits__loads_and_stores_header_info():

    create_fits(fits_path=fits_path)

    arr = aa.Array1D.from_fits(
        file_path=path.join(fits_path, "3_ones.fits"), hdu=0, pixel_scales=1.0
    )

    assert arr.header.header_sci_obj["BITPIX"] == -64
    assert arr.header.header_hdu_obj["BITPIX"] == -64

    arr = aa.Array1D.from_fits(
        file_path=path.join(fits_path, "4_ones.fits"), hdu=0, pixel_scales=1.0
    )

    assert arr.header.header_sci_obj["BITPIX"] == -64
    assert arr.header.header_hdu_obj["BITPIX"] == -64

    clean_fits(fits_path=fits_path)


def test__output_to_fits():

    arr = aa.Array1D.ones(shape_native=(3,), pixel_scales=1.0)

    if path.exists(output_data_dir):
        shutil.rmtree(output_data_dir)

    os.makedirs(output_data_dir)

    arr.output_to_fits(file_path=path.join(output_data_dir, "array.fits"))

    array_from_out = aa.Array1D.from_fits(
        file_path=path.join(output_data_dir, "array.fits"), hdu=0, pixel_scales=1.0
    )

    assert (array_from_out.native == np.ones((3,))).all()


def test__recursive_shape_storage():

    array_1d = aa.Array1D.manual_slim(array=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

    assert (array_1d.native.slim.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (array_1d.slim.native.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()

    mask = aa.Mask1D.manual(
        mask=[True, False, False, True, False, False], pixel_scales=1.0, sub_size=1
    )

    array_1d = aa.Array1D.manual_mask(
        array=[100.0, 1.0, 2.0, 100.0, 3.0, 4.0], mask=mask
    )

    assert (
        array_1d.native.slim.native == np.array([0.0, 1.0, 2.0, 0.0, 3.0, 4.0])
    ).all()
    assert (array_1d.slim.native.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
