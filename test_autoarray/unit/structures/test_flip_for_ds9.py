from astropy.io import fits

import os
from os import path
import numpy as np
from autoconf import conf
import autoarray as aa

test_path = "{}".format(path.dirname(path.realpath(__file__)))


def create_fits(fits_path, array):

    if path.exists(fits_path):
        os.remove(fits_path)

    hdu_list = fits.HDUList()

    hdu_list.append(fits.ImageHDU(array))

    hdu_list.writeto(f"{fits_path}")


def test__from_fits__all_imaging_data_structures_are_flipped_for_ds9():

    conf.instance.push(new_path=path.join(test_path, "files", "config_flip"))

    fits_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")

    arr = np.array([[1.0, 0.0], [0.0, 0.0]])
    array_path = path.join(fits_path, "array.fits")
    create_fits(fits_path=array_path, array=arr)

    arr = aa.Array.from_fits(file_path=array_path, hdu=0, pixel_scales=1.0)
    assert (arr.in_2d == np.array([[0.0, 0.0], [1.0, 0.0]])).all()

    arr.output_to_fits(file_path=array_path, overwrite=True)

    hdu_list = fits.open(array_path)
    arr = np.array(hdu_list[0].data).astype("float64")
    assert (arr == np.array([[1.0, 0.0], [0.0, 0.0]])).all()

    frame = np.array([[2.0, 0.0], [0.0, 0.0]])
    frame_path = path.join(fits_path, "frame.fits")
    create_fits(fits_path=frame_path, array=frame)

    frame = aa.Frame.from_fits(file_path=frame_path, hdu=0, pixel_scales=1.0)
    assert (frame.in_2d == np.array([[0.0, 0.0], [2.0, 0.0]])).all()

    frame.output_to_fits(file_path=frame_path, overwrite=True)
    hdu_list = fits.open(frame_path)
    frame = np.array(hdu_list[0].data).astype("float64")
    assert (frame == np.array([[2.0, 0.0], [0.0, 0.0]])).all()

    kernel = np.array([[3.0, 0.0], [0.0, 0.0]])
    kernel_path = path.join(fits_path, "kernel.fits")
    create_fits(fits_path=kernel_path, array=kernel)

    kernel = aa.Kernel.from_fits(file_path=kernel_path, hdu=0, pixel_scales=1.0)
    assert (kernel.in_2d == np.array([[0.0, 0.0], [3.0, 0.0]])).all()

    kernel.output_to_fits(file_path=kernel_path, overwrite=True)
    hdu_list = fits.open(kernel_path)
    kernel = np.array(hdu_list[0].data).astype("float64")
    assert (kernel == np.array([[3.0, 0.0], [0.0, 0.0]])).all()
