import autoarray as aa

import pytest
import numpy as np
import os


class TestSimulatorInterferometer:
    def test__simulator_grid_is_uniform_grid_with_same_inputs(self):

        grid = aa.Grid.uniform(
            shape_2d=(31, 31), pixel_scales=0.05, sub_size=1, origin=(0.1, 0.1)
        )

        simulator = aa.SimulatorInterferometer(
            real_space_shape_2d=(31, 31),
            real_space_pixel_scales=0.05,
            uv_wavelengths=np.ones((7, 2)),
            sub_size=1,
            origin=(0.1, 0.1),
            exposure_time=20.0,
            background_level=10.0,
        )

        assert (simulator.grid == grid).all()

    def test__constructor_and_specific_instrument_class_methods(self):

        sma = aa.SimulatorInterferometer.sma()

        uv_wavelengths_path = "{}/dataset/sma_uv_wavelengths.fits".format(
            os.path.dirname(os.path.realpath(__file__))
        )

        sma_uv_wavelengths = aa.util.array.numpy_array_1d_from_fits(
            file_path=uv_wavelengths_path, hdu=0
        )

        assert sma.real_space_shape_2d == (151, 151)
        assert sma.real_space_pixel_scales == (0.05, 0.05)
        assert (sma.uv_wavelengths == sma_uv_wavelengths).all()
        assert sma.uv_wavelengths[0] == pytest.approx(
            [184584.953125, -16373.30566406], 1.0e-4
        )
        assert sma.exposure_time == 100.0
        assert sma.background_level == 1.0

    def test__from_real_space_image_same_as_manual_image_input(self):

        primary_beam = aa.Kernel.manual_2d(
            array=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
            pixel_scales=1.0,
        )

        real_space_image = aa.Array.manual_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        simulator = aa.SimulatorInterferometer(
            real_space_shape_2d=(2, 3),
            real_space_pixel_scales=0.05,
            uv_wavelengths=np.ones((7, 2)),
            sub_size=1,
            primary_beam=primary_beam,
            exposure_time=10000.0,
            background_level=100.0,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer_simulated = simulator.from_image(image=real_space_image)

        interferometer_manual = aa.Interferometer.from_image(
            real_space_image=real_space_image,
            real_space_pixel_scales=0.05,
            transformer=simulator.transformer,
            exposure_time=10000.0,
            primary_beam=primary_beam,
            background_level=100.0,
            noise_sigma=0.1,
            noise_seed=1,
        )

        assert (
            interferometer_simulated.visibilities == interferometer_manual.visibilities
        ).all()
        assert (
            interferometer_simulated.noise_map == interferometer_manual.noise_map
        ).all()
        assert (
            interferometer_simulated.primary_beam.in_2d
            == interferometer_manual.primary_beam.in_2d
        ).all()
