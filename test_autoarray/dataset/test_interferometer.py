import os
from os import path
import shutil

import numpy as np
import pytest

import autoarray as aa
from autoarray.operators import transformer

test_data_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "interferometer"
)


class TestInterferometer:
    def test__dirty_properties(
        self,
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
            interferometer.transformer.image_from(
                visibilities=interferometer.visibilities
            )
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

    def test__signal_to_noise_limit_below_max_signal_to_noise__signal_to_noise_map_capped_to_limit(
        self, sub_mask_2d_7x7, uv_wavelengths_7x2
    ):

        interferometer = aa.Interferometer(
            real_space_mask=sub_mask_2d_7x7,
            visibilities=aa.Visibilities(visibilities=np.array([1 + 1j, 1 + 1j])),
            noise_map=aa.VisibilitiesNoiseMap(
                visibilities=np.array([1 + 0.25j, 1 + 0.25j])
            ),
            uv_wavelengths=uv_wavelengths_7x2,
        )

        interferometer_capped = interferometer.signal_to_noise_limited_from(
            signal_to_noise_limit=2.0
        )

        assert (
            interferometer_capped.visibilities == np.array([1.0 + 1.0j, 1.0 + 1.0j])
        ).all()
        assert (
            interferometer_capped.noise_map == np.array([1.0 + 0.5j, 1.0 + 0.5j])
        ).all()
        assert (
            interferometer_capped.signal_to_noise_map == np.array([1.0 + 2.0j])
        ).all()

        interferometer_capped = interferometer.signal_to_noise_limited_from(
            signal_to_noise_limit=0.25
        )

        assert (
            interferometer_capped.visibilities == np.array([1.0 + 1.0j, 1.0 + 1.0j])
        ).all()
        assert (
            interferometer_capped.noise_map == np.array([4.0 + 4.0j, 4.0 + 4.0j])
        ).all()
        assert (
            interferometer_capped.signal_to_noise_map == np.array([0.25 + 0.25j])
        ).all()

    def test__from_fits__all_files_in_one_fits__load_using_different_hdus(
        self, sub_mask_2d_7x7
    ):

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
            interferometer.visibilities
            == np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])
        ).all()
        assert (
            interferometer.noise_map == np.array([2.0 + 2.0j, 2.0 + 2.0j, 2.0 + 2.0j])
        ).all()
        assert (interferometer.uv_wavelengths[:, 0] == 3.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 1] == 3.0 * np.ones(3)).all()

    def test__output_all_arrays(self, sub_mask_2d_7x7):

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
            interferometer.visibilities
            == np.array([1.0 + 2.0j, 1.0 + 2.0j, 1.0 + 2.0j])
        ).all()
        assert (
            interferometer.noise_map == np.array([3.0 + 4.0j, 3.0 + 4.0j, 3.0 + 4.0j])
        ).all()
        assert (interferometer.uv_wavelengths[:, 0] == 5.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 1] == 6.0 * np.ones(3)).all()

    def test__transformer(
        self,
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
        self, sub_mask_2d_7x7
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


class TestSimulatorInterferometer:
    def test__from_image__setup_with_all_features_off(
        self, uv_wavelengths_7x2, transformer_7x7_7, mask_2d_7x7
    ):

        image = aa.Array2D.manual_native(
            array=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]],
            pixel_scales=transformer_7x7_7.grid.pixel_scales,
        )

        simulator = aa.SimulatorInterferometer(
            exposure_time=1.0,
            transformer_class=type(transformer_7x7_7),
            uv_wavelengths=uv_wavelengths_7x2,
            noise_sigma=None,
        )

        interferometer = simulator.from_image(image=image)

        transformer = simulator.transformer_class(
            uv_wavelengths=uv_wavelengths_7x2,
            real_space_mask=aa.Mask2D.unmasked(
                shape_native=(3, 3), pixel_scales=image.pixel_scales
            ),
        )

        visibilities = transformer.visibilities_from(image=image)

        assert interferometer.visibilities == pytest.approx(visibilities, 1.0e-4)

    def test__setup_with_noise(self, uv_wavelengths_7x2, transformer_7x7_7):

        image = aa.Array2D.manual_native(
            array=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]],
            pixel_scales=transformer_7x7_7.grid.pixel_scales,
        )

        simulator = aa.SimulatorInterferometer(
            exposure_time=20.0,
            transformer_class=type(transformer_7x7_7),
            uv_wavelengths=uv_wavelengths_7x2,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.from_image(image=image)

        assert interferometer.visibilities[0] == pytest.approx(
            -0.005364 - 2.36682j, 1.0e-4
        )

        assert (interferometer.noise_map == 0.1 + 0.1j * np.ones((7,))).all()
