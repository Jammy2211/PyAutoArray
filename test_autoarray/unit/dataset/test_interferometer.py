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
    def test__new_interferometer_with_resized_primary_beam(self):

        interferometer = aa.Interferometer(
            visibilities=aa.Visibilities.manual_1d(visibilities=[[1, 1]]),
            primary_beam=aa.Kernel.zeros(shape_2d=(5, 5), pixel_scales=1.0),
            noise_map=1,
            uv_wavelengths=2,
        )

        interferometer = interferometer.resized_primary_beam_from_new_shape_2d(
            new_shape_2d=(1, 1)
        )

        assert (interferometer.visibilities == np.array([[1, 1]])).all()
        assert (interferometer.primary_beam.in_2d == np.zeros((1, 1))).all()
        assert interferometer.noise_map == 1
        assert interferometer.uv_wavelengths == 2

    def test__new_interferometer_with_with_modified_visibilities(self):

        interferometer = aa.Interferometer(
            visibilities=np.array([[1, 1]]),
            primary_beam=aa.Kernel.zeros(shape_2d=(5, 5), pixel_scales=1.0),
            noise_map=1,
            uv_wavelengths=2,
        )

        interferometer = interferometer.modified_visibilities_from_visibilities(
            visibilities=np.array([[2, 2]])
        )

        assert (interferometer.visibilities == np.array([[2, 2]])).all()
        assert (interferometer.primary_beam.in_2d == np.zeros((1, 1))).all()
        assert interferometer.noise_map == 1
        assert interferometer.uv_wavelengths == 2

    def test__from_fits__loads_arrays_and_primary_beam_renormalized(self):

        interferometer = aa.Interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_ones_twos.fits",
            noise_map_path=test_data_dir + "3x2_threes_fours.fits",
            uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            positions_path=test_data_dir + "positions.dat",
        )

        assert (interferometer.visibilities.real == np.ones(3)).all()
        assert (interferometer.visibilities.imag == 2.0 * np.ones(3)).all()
        assert (interferometer.noise_map.real == 3.0 * np.ones(3)).all()
        assert (interferometer.noise_map.imag == 4.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 0] == 5.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 1] == 6.0 * np.ones(3)).all()
        assert interferometer.amplitudes == pytest.approx(
            np.sqrt(5) * np.ones(3), 1.0e-4
        )
        assert interferometer.phases == pytest.approx(1.10714 * np.ones(3), 1.0e-4)
        assert interferometer.uv_distances == pytest.approx(
            np.sqrt(61) * np.ones(3), 1.0e-4
        )
        assert (
            interferometer.primary_beam.in_2d == (1.0 / 9.0) * np.ones((3, 3))
        ).all()
        assert interferometer.positions.in_list == [[(1.0, 1.0), (2.0, 2.0)]]

    def test__from_fits__all_files_in_one_fits__load_using_different_hdus(self):

        interferometer = aa.Interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_multiple_hdu.fits",
            visibilities_hdu=0,
            noise_map_path=test_data_dir + "3x2_multiple_hdu.fits",
            noise_map_hdu=1,
            uv_wavelengths_path=test_data_dir + "3x2_multiple_hdu.fits",
            uv_wavelengths_hdu=2,
            primary_beam_path=test_data_dir + "3x3_multiple_hdu.fits",
            primary_beam_hdu=3,
        )

        assert (interferometer.visibilities.real == np.ones(3)).all()
        assert (interferometer.visibilities.imag == np.ones(3)).all()
        assert (interferometer.noise_map.real == 2.0 * np.ones(3)).all()
        assert (interferometer.noise_map.imag == 2.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 0] == 3.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 1] == 3.0 * np.ones(3)).all()
        assert (
            interferometer.primary_beam.in_2d == (1.0 / 9.0) * np.ones((3, 3))
        ).all()

    def test__output_all_arrays(self):

        interferometer = aa.Interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_ones_twos.fits",
            noise_map_path=test_data_dir + "3x2_threes_fours.fits",
            uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
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
            primary_beam_path=output_data_dir + "primary_beam.fits",
            uv_wavelengths_path=output_data_dir + "uv_wavelengths.fits",
            overwrite=True,
        )

        interferometer = aa.Interferometer.from_fits(
            visibilities_path=output_data_dir + "visibilities.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            primary_beam_path=output_data_dir + "primary_beam.fits",
            uv_wavelengths_path=output_data_dir + "uv_wavelengths.fits",
        )

        assert (interferometer.visibilities.real == np.ones(3)).all()
        assert (interferometer.visibilities.imag == 2.0 * np.ones(3)).all()
        assert (interferometer.noise_map.real == 3.0 * np.ones((3,))).all()
        assert (interferometer.noise_map.imag == 4.0 * np.ones((3,))).all()
        assert (interferometer.uv_wavelengths[:, 0] == 5.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 1] == 6.0 * np.ones(3)).all()
        assert (
            interferometer.primary_beam.in_2d == (1.0 / 9.0) * np.ones((3, 3))
        ).all()


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
            masked_interferometer_7.primary_beam.in_2d == (1.0 / 9.0) * np.ones((3, 3))
        ).all()
        assert masked_interferometer_7.primary_beam_shape_2d == (3, 3)

        assert (
            masked_interferometer_7.interferometer.uv_wavelengths
            == interferometer_7.uv_wavelengths
        ).all()
        assert (
            masked_interferometer_7.interferometer.uv_wavelengths[0, 0]
            == -55636.4609375
        )

    def test__primary_beam_and_transformer(self, interferometer_7, sub_mask_7x7):

        visibilities_mask = np.full(fill_value=False, shape=(7, 2))

        masked_interferometer_7 = aa.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask,
            real_space_mask=sub_mask_7x7,
            transformer_class=transformer.TransformerDFT,
        )

        assert type(masked_interferometer_7.primary_beam) == kern.Kernel
        assert type(masked_interferometer_7.transformer) == transformer.TransformerDFT

        masked_interferometer_7 = aa.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask,
            real_space_mask=sub_mask_7x7,
            transformer_class=transformer.TransformerFFT,
        )

        assert type(masked_interferometer_7.primary_beam) == kern.Kernel
        assert type(masked_interferometer_7.transformer) == transformer.TransformerFFT

        masked_interferometer_7 = aa.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask,
            real_space_mask=sub_mask_7x7,
            transformer_class=transformer.TransformerNUFFT,
        )

        assert type(masked_interferometer_7.primary_beam) == kern.Kernel
        assert type(masked_interferometer_7.transformer) == transformer.TransformerNUFFT

    def test__different_interferometer_without_mock_objects__customize_constructor_inputs(
        self
    ):
        primary_beam = aa.Kernel.ones(shape_2d=(7, 7), pixel_scales=1.0)

        interferometer = aa.Interferometer(
            visibilities=aa.Visibilities.ones(shape_1d=(19,)),
            primary_beam=primary_beam,
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
            primary_beam_shape_2d=(7, 7),
        )

        assert (masked_interferometer_7.visibilities == np.ones((19, 2))).all()
        assert (masked_interferometer_7.noise_map == 2.0 * np.ones((19, 2))).all()
        assert (
            masked_interferometer_7.interferometer.uv_wavelengths
            == 3.0 * np.ones((19, 2))
        ).all()
        assert (
            masked_interferometer_7.primary_beam.in_2d == (1.0 / 49.0) * np.ones((7, 7))
        ).all()

        assert masked_interferometer_7.primary_beam_shape_2d == (7, 7)

    def test__modified_noise_map(
        self, noise_map_7x2, interferometer_7, sub_mask_7x7, visibilities_mask_7x2
    ):

        masked_interferometer_7 = aa.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=sub_mask_7x7,
            transformer_class=aa.TransformerDFT,
        )

        noise_map_7x2[0, 0] = 10.0

        masked_interferometer_7 = masked_interferometer_7.modify_noise_map(
            noise_map=noise_map_7x2
        )

        assert masked_interferometer_7.noise_map[0, 0] == 10.0


class TestSimulatorInterferometer:
    def test__from_image__setup_with_all_features_off(
        self, uv_wavelengths_7x2, transformer_7x7_7
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
            grid=image.mask.geometry.unmasked_grid.in_1d_binned.in_radians,
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
            grid=image.mask.geometry.unmasked_grid.in_1d_binned.in_radians,
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
            grid=image.mask.geometry.unmasked_grid.in_1d_binned.in_radians,
        )

        visibilities = transformer.visibilities_from_image(image=image)

        assert interferometer.visibilities[0, :] == pytest.approx(
            [-0.005364, -2.36682], 1.0e-4
        )

        assert (interferometer.noise_map == 0.1 * np.ones((7, 2))).all()
