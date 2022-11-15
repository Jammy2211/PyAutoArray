import numpy as np
import os
from os import path
import shutil

import autoarray as aa

from autoarray.operators import transformer

test_data_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))),
    "files",
)


def test__dirty_properties(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    sub_mask_2d_7x7,
):

    interferometer = aa.Interferometer(
        visibilities=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=sub_mask_2d_7x7,
    )

    assert interferometer.dirty_image.shape_native == (7, 7)
    assert (
        interferometer.transformer.image_from(visibilities=interferometer.visibilities)
    ).all()

    assert interferometer.dirty_noise_map.shape_native == (7, 7)
    assert (
        interferometer.transformer.image_from(visibilities=interferometer.noise_map)
    ).all()

    assert interferometer.dirty_signal_to_noise_map.shape_native == (7, 7)
    assert (
        interferometer.transformer.image_from(
            visibilities=interferometer.signal_to_noise_map
        )
    ).all()

    assert interferometer.dirty_inverse_noise_map.shape_native == (7, 7)
    assert (
        interferometer.transformer.image_from(
            visibilities=interferometer.inverse_noise_map
        )
    ).all()


def test__from_fits__all_files_in_one_fits__load_using_different_hdus(sub_mask_2d_7x7):

    interferometer = aa.Interferometer.from_fits(
        real_space_mask=sub_mask_2d_7x7,
        visibilities_path=path.join(test_data_dir, "3x2_multiple_hdu.fits"),
        visibilities_hdu=0,
        noise_map_path=path.join(test_data_dir, "3x2_multiple_hdu.fits"),
        noise_map_hdu=1,
        uv_wavelengths_path=path.join(test_data_dir, "3x2_multiple_hdu.fits"),
        uv_wavelengths_hdu=2,
    )

    assert (
        interferometer.visibilities == np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])
    ).all()
    assert (
        interferometer.noise_map == np.array([2.0 + 2.0j, 2.0 + 2.0j, 2.0 + 2.0j])
    ).all()
    assert (interferometer.uv_wavelengths[:, 0] == 3.0 * np.ones(3)).all()
    assert (interferometer.uv_wavelengths[:, 1] == 3.0 * np.ones(3)).all()


def test__output_all_arrays(sub_mask_2d_7x7):

    interferometer = aa.Interferometer.from_fits(
        real_space_mask=sub_mask_2d_7x7,
        visibilities_path=path.join(test_data_dir, "3x2_ones_twos.fits"),
        noise_map_path=path.join(test_data_dir, "3x2_threes_fours.fits"),
        uv_wavelengths_path=path.join(test_data_dir, "3x2_fives_sixes.fits"),
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

    interferometer.output_to_fits(
        visibilities_path=path.join(output_data_dir, "visibilities.fits"),
        noise_map_path=path.join(output_data_dir, "noise_map.fits"),
        uv_wavelengths_path=path.join(output_data_dir, "uv_wavelengths.fits"),
        overwrite=True,
    )

    interferometer = aa.Interferometer.from_fits(
        real_space_mask=sub_mask_2d_7x7,
        visibilities_path=path.join(output_data_dir, "visibilities.fits"),
        noise_map_path=path.join(output_data_dir, "noise_map.fits"),
        uv_wavelengths_path=path.join(output_data_dir, "uv_wavelengths.fits"),
    )

    assert (
        interferometer.visibilities == np.array([1.0 + 2.0j, 1.0 + 2.0j, 1.0 + 2.0j])
    ).all()
    assert (
        interferometer.noise_map == np.array([3.0 + 4.0j, 3.0 + 4.0j, 3.0 + 4.0j])
    ).all()
    assert (interferometer.uv_wavelengths[:, 0] == 5.0 * np.ones(3)).all()
    assert (interferometer.uv_wavelengths[:, 1] == 6.0 * np.ones(3)).all()


def test__transformer(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    sub_mask_2d_7x7,
):

    interferometer_7 = aa.Interferometer(
        visibilities=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=sub_mask_2d_7x7,
        settings=aa.SettingsInterferometer(
            transformer_class=transformer.TransformerDFT
        ),
    )

    assert type(interferometer_7.transformer) == transformer.TransformerDFT

    interferometer_7 = aa.Interferometer(
        visibilities=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=sub_mask_2d_7x7,
        settings=aa.SettingsInterferometer(
            transformer_class=transformer.TransformerNUFFT
        ),
    )

    assert type(interferometer_7.transformer) == transformer.TransformerNUFFT


def test__different_interferometer_without_mock_objects__customize_constructor_inputs(
    sub_mask_2d_7x7,
):

    interferometer = aa.Interferometer(
        visibilities=aa.Visibilities.ones(shape_slim=(19,)),
        noise_map=2.0 * aa.Visibilities.ones(shape_slim=(19,)),
        uv_wavelengths=3.0 * np.ones((19, 2)),
        real_space_mask=sub_mask_2d_7x7,
    )

    real_space_mask = aa.Mask2D.unmasked(
        shape_native=(19, 19), pixel_scales=1.0, invert=True, sub_size=8
    )
    real_space_mask[9, 9] = False

    assert (interferometer.visibilities == 1.0 + 1.0j * np.ones((19,))).all()
    assert (interferometer.noise_map == 2.0 + 2.0j * np.ones((19,))).all()
    assert (interferometer.uv_wavelengths == 3.0 * np.ones((19, 2))).all()