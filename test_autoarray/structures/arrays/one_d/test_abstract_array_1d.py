import os
from os import path

import numpy as np
import pytest
import shutil

import autoarray as aa

output_data_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "array", "output_test"
)


class TestOutputToFits:
    def test__output_to_fits(self):

        arr = aa.Array1D.ones(shape_native=(3,), pixel_scales=1.0)

        if path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=path.join(output_data_dir, "array.fits"))

        array_from_out = aa.Array1D.from_fits(
            file_path=path.join(output_data_dir, "array.fits"), hdu=0, pixel_scales=1.0
        )

        assert (array_from_out.native == np.ones((3,))).all()

    def test__output_to_fits__shapes_of_arrays_are_1d(self):

        arr = aa.Array1D.ones(shape_native=(3,), pixel_scales=1.0)

        if path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=path.join(output_data_dir, "array.fits"))

        array_from_out = aa.util.array_1d.numpy_array_1d_from_fits(
            file_path=path.join(output_data_dir, "array.fits"), hdu=0
        )

        assert (array_from_out == np.ones((3,))).all()

        mask = aa.Mask1D.unmasked(shape_slim=(3,), pixel_scales=0.1)

        masked_array = aa.Array1D.manual_mask(array=arr, mask=mask)

        masked_array.output_to_fits(
            file_path=path.join(output_data_dir, "masked_array.fits")
        )

        masked_array_from_out = aa.util.array_1d.numpy_array_1d_from_fits(
            file_path=path.join(output_data_dir, "masked_array.fits"), hdu=0
        )

        assert (masked_array_from_out == np.ones((3,))).all()
