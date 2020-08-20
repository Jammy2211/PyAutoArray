import os
import shutil

import numpy as np
import pytest

import autoarray as aa
from autoarray.structures import kernel as kern
from autoarray.operators import transformer

test_data_dir = "{}/files/interferometer/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestInterferometer:
    def test__new_interferometer_with_with_modified_visibilities(self):

        interferometer = aa.Interferometer(
            visibilities=np.array([[1, 1]]), noise_map=1, uv_wavelengths=2
        )

        interferometer = interferometer.modified_visibilities_from_visibilities(
            visibilities=np.array([[2, 2]])
        )

        assert (interferometer.visibilities == np.array([[2, 2]])).all()
        assert interferometer.noise_map == 1
        assert interferometer.uv_wavelengths == 2

    def test__new_interferometer_with_signal_to_noise_limit_below_max_signal_to_noise__signal_to_noise_map_capped_to_limit(
        self
    ):

        interferometer = aa.Interferometer(
            visibilities=aa.Visibilities(visibilities_1d=np.array([[1, 1]])),
            noise_map=aa.VisibilitiesNoiseMap(visibilities_1d=np.array([[1, 5]])),
            uv_wavelengths=2,
        )

        interferometer_capped = interferometer.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=0.5
        )

        assert (interferometer_capped.visibilities == np.array([[1.0, 1.0]])).all()

        print(interferometer_capped.noise_map)

        assert (interferometer_capped.noise_map == np.array([[2.0, 5.0]])).all()

        assert (
            interferometer_capped.signal_to_noise_map == np.array([[0.5, 0.2]])
        ).all()

    def test__from_fits__all_files_in_one_fits__load_using_different_hdus(self):

        interferometer = aa.Interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_multiple_hdu.fits",
            visibilities_hdu=0,
            noise_map_path=test_data_dir + "3x2_multiple_hdu.fits",
            noise_map_hdu=1,
            uv_wavelengths_path=test_data_dir + "3x2_multiple_hdu.fits",
            uv_wavelengths_hdu=2,
        )

        assert (interferometer.visibilities.real == np.ones(3)).all()
        assert (interferometer.visibilities.imag == np.ones(3)).all()
        assert (interferometer.noise_map.real == 2.0 * np.ones(3)).all()
        assert (interferometer.noise_map.imag == 2.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 0] == 3.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 1] == 3.0 * np.ones(3)).all()

    def test__output_all_arrays(self):

        interferometer = aa.Interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_ones_twos.fits",
            noise_map_path=test_data_dir + "3x2_threes_fours.fits",
            uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
        )

        output_data_dir = "{}/files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        interferometer.output_to_fits(
            visibilities_path=output_data_dir + "visibilities.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            uv_wavelengths_path=output_data_dir + "uv_wavelengths.fits",
            overwrite=True,
        )

        interferometer = aa.Interferometer.from_fits(
            visibilities_path=output_data_dir + "visibilities.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            uv_wavelengths_path=output_data_dir + "uv_wavelengths.fits",
        )

        assert (interferometer.visibilities.real == np.ones(3)).all()
        assert (interferometer.visibilities.imag == 2.0 * np.ones(3)).all()
        assert (interferometer.noise_map.real == 3.0 * np.ones((3,))).all()
        assert (interferometer.noise_map.imag == 4.0 * np.ones((3,))).all()
        assert (interferometer.uv_wavelengths[:, 0] == 5.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 1] == 6.0 * np.ones(3)).all()


class TestSettingsMaskedInterferometer:
    def test__transformer_tag(self):
        settings = aa.SettingsMaskedInterferometer(transformer_class=aa.TransformerDFT)
        assert settings.transformer_tag == "__dft"
        settings = aa.SettingsMaskedInterferometer(
            transformer_class=aa.TransformerNUFFT
        )
        assert settings.transformer_tag == "__nufft"
        settings = aa.SettingsMaskedInterferometer(transformer_class=None)
        assert settings.transformer_tag == ""


class TestMaskedInterferometer:
    def test__masked_dataset(
        self, interferometer_7, sub_mask_7x7, visibilities_7x2, noise_map_7x2
    ):

        masked_interferometer_7 = aa.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=np.full(fill_value=False, shape=(7, 2)),
            real_space_mask=sub_mask_7x7,
        )

        assert (
            masked_interferometer_7.visibilities == interferometer_7.visibilities
        ).all()
        assert (masked_interferometer_7.visibilities == visibilities_7x2).all()

        assert (masked_interferometer_7.noise_map == noise_map_7x2).all()

        assert (
            masked_interferometer_7.visibilities_mask
            == np.full(fill_value=False, shape=(7, 2))
        ).all()

        assert (
            masked_interferometer_7.interferometer.uv_wavelengths
            == interferometer_7.uv_wavelengths
        ).all()
        assert (
            masked_interferometer_7.interferometer.uv_wavelengths[0, 0]
            == -55636.4609375
        )

    def test__transformer(self, interferometer_7, sub_mask_7x7):

        visibilities_mask = np.full(fill_value=False, shape=(7, 2))

        masked_interferometer_7 = aa.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask,
            real_space_mask=sub_mask_7x7,
            settings=aa.SettingsMaskedInterferometer(
                transformer_class=transformer.TransformerDFT
            ),
        )

        assert type(masked_interferometer_7.transformer) == transformer.TransformerDFT

        masked_interferometer_7 = aa.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask,
            real_space_mask=sub_mask_7x7,
            settings=aa.SettingsMaskedInterferometer(
                transformer_class=transformer.TransformerNUFFT
            ),
        )

        assert type(masked_interferometer_7.transformer) == transformer.TransformerNUFFT

    def test__different_interferometer_without_mock_objects__customize_constructor_inputs(
        self
    ):

        interferometer = aa.Interferometer(
            visibilities=aa.Visibilities.ones(shape_1d=(19,)),
            noise_map=2.0 * aa.Visibilities.ones(shape_1d=(19,)),
            uv_wavelengths=3.0 * np.ones((19, 2)),
        )

        visibilities_mask = np.full(fill_value=False, shape=(19, 2))

        real_space_mask = aa.Mask.unmasked(
            shape_2d=(19, 19), pixel_scales=1.0, invert=True, sub_size=8
        )
        real_space_mask[9, 9] = False

        masked_interferometer_7 = aa.MaskedInterferometer(
            interferometer=interferometer,
            visibilities_mask=visibilities_mask,
            real_space_mask=real_space_mask,
        )

        assert (masked_interferometer_7.visibilities == np.ones((19, 2))).all()
        assert (masked_interferometer_7.noise_map == 2.0 * np.ones((19, 2))).all()
        assert (
            masked_interferometer_7.interferometer.uv_wavelengths
            == 3.0 * np.ones((19, 2))
        ).all()

    def test__modified_noise_map(
        self, noise_map_7x2, interferometer_7, sub_mask_7x7, visibilities_mask_7x2
    ):

        masked_interferometer_7 = aa.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=sub_mask_7x7,
            settings=aa.SettingsMaskedInterferometer(
                transformer_class=aa.TransformerDFT
            ),
        )

        noise_map_7x2[0, 0] = 10.0

        masked_interferometer_7 = masked_interferometer_7.modify_noise_map(
            noise_map=noise_map_7x2
        )

        assert masked_interferometer_7.noise_map[0, 0] == 10.0


class TestSimulatorInterferometer:
    def test__from_image__setup_with_all_features_off(
        self, uv_wavelengths_7x2, transformer_7x7_7, mask_7x7
    ):

        image = aa.Array.manual_2d(
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]],
            pixel_scales=transformer_7x7_7.grid.pixel_scales,
        )

        exposure_time_map = aa.Array.full(
            fill_value=1.0, pixel_scales=0.1, shape_2d=image.shape_2d
        )

        simulator = aa.SimulatorInterferometer(
            exposure_time_map=exposure_time_map,
            transformer_class=type(transformer_7x7_7),
            uv_wavelengths=uv_wavelengths_7x2,
            noise_sigma=None,
        )

        interferometer = simulator.from_image(image=image)

        transformer = simulator.transformer_class(
            uv_wavelengths=uv_wavelengths_7x2,
            real_space_mask=aa.Mask.unmasked(
                shape_2d=(3, 3), pixel_scales=image.pixel_scales
            ),
        )

        visibilities = transformer.visibilities_from_image(image=image)

        assert interferometer.visibilities == pytest.approx(visibilities, 1.0e-4)

    def test__setup_with_background_sky_on__noise_off__no_noise_in_image__noise_map_is_noise_value(
        self, uv_wavelengths_7x2, transformer_7x7_7
    ):
        image = aa.Array.manual_2d(
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]],
            pixel_scales=transformer_7x7_7.grid.pixel_scales,
        )

        exposure_time_map = aa.Array.full(
            fill_value=1.0,
            pixel_scales=transformer_7x7_7.grid.pixel_scales,
            shape_2d=image.shape_2d,
        )

        background_sky_map = aa.Array.full(
            fill_value=2.0,
            pixel_scales=transformer_7x7_7.grid.pixel_scales,
            shape_2d=image.shape_2d,
        )

        simulator = aa.SimulatorInterferometer(
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            transformer_class=type(transformer_7x7_7),
            uv_wavelengths=uv_wavelengths_7x2,
            noise_sigma=None,
            noise_if_add_noise_false=0.2,
        )

        interferometer = simulator.from_image(image=image)

        transformer = simulator.transformer_class(
            uv_wavelengths=uv_wavelengths_7x2,
            real_space_mask=aa.Mask.unmasked(
                shape_2d=(3, 3), pixel_scales=image.pixel_scales
            ),
        )

        visibilities = transformer.visibilities_from_image(
            image=image + background_sky_map
        )

        assert interferometer.visibilities == pytest.approx(visibilities, 1.0e-4)

        assert (interferometer.noise_map == 0.2 * np.ones((7, 2))).all()

    def test__setup_with_noise(self, uv_wavelengths_7x2, transformer_7x7_7):

        image = aa.Array.manual_2d(
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]],
            pixel_scales=transformer_7x7_7.grid.pixel_scales,
        )

        exposure_time_map = aa.Array.full(
            fill_value=20.0, pixel_scales=0.1, shape_2d=image.shape_2d
        )

        simulator = aa.SimulatorInterferometer(
            exposure_time_map=exposure_time_map,
            transformer_class=type(transformer_7x7_7),
            uv_wavelengths=uv_wavelengths_7x2,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.from_image(image=image)

        transformer = simulator.transformer_class(
            uv_wavelengths=uv_wavelengths_7x2,
            real_space_mask=aa.Mask.unmasked(
                shape_2d=(3, 3), pixel_scales=image.pixel_scales
            ),
        )

        visibilities = transformer.visibilities_from_image(image=image)

        assert interferometer.visibilities[0, :] == pytest.approx(
            [-0.005364, -2.36682], 1.0e-4
        )

        assert (interferometer.noise_map == 0.1 * np.ones((7, 2))).all()
