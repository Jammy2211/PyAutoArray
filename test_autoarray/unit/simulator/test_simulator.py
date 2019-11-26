import autoarray as aa

import pytest
import numpy as np
import os

test_data_dir = "{}/../test_files/arrays/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestImagingSimulator:
    def test__simulator_grid_is_uniform_grid_with_same_inputs(self):

        grid = aa.grid.uniform(
            shape_2d=(31, 31), pixel_scales=0.05, sub_size=1, origin=(0.1, 0.1)
        )

        simulator = aa.simulator.imaging(
            shape_2d=(31, 31),
            pixel_scales=0.05,
            sub_size=1,
            origin=(0.1, 0.1),
            psf=None,
            exposure_time=20.0,
            background_level=10.0,
        )

        assert (simulator.grid == grid).all()

    def test__constructor_and_specific_instrument_class_methods(self):

        psf = aa.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=0.1)

        simulator = aa.simulator.imaging(
            shape_2d=(51, 51),
            pixel_scales=0.1,
            sub_size=1,
            psf=psf,
            exposure_time=20.0,
            background_level=10.0,
        )

        assert simulator.shape_2d == (51, 51)
        assert simulator.pixel_scales == (0.1, 0.1)
        assert simulator.psf == psf
        assert simulator.exposure_time == 20.0
        assert simulator.background_level == 10.0

        lsst = aa.simulator.imaging.lsst()

        lsst_psf = aa.kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.5, pixel_scales=0.2
        )

        assert lsst.shape_2d == (101, 101)
        assert lsst.pixel_scales == (0.2, 0.2)
        assert lsst.psf == lsst_psf
        assert lsst.exposure_time == 100.0
        assert lsst.background_level == 1.0

        euclid = aa.simulator.imaging.euclid()

        euclid_psf = aa.kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.1, pixel_scales=0.1
        )

        assert euclid.shape_2d == (151, 151)
        assert euclid.pixel_scales == (0.1, 0.1)
        assert euclid.psf == euclid_psf
        assert euclid.exposure_time == 565.0
        assert euclid.background_level == 1.0

        hst = aa.simulator.imaging.hst()

        hst_psf = aa.kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.05, pixel_scales=0.05
        )

        assert hst.shape_2d == (251, 251)
        assert hst.pixel_scales == (0.05, 0.05)
        assert hst.psf == hst_psf
        assert hst.exposure_time == 2000.0
        assert hst.background_level == 1.0

        hst_up_sampled = aa.simulator.imaging.hst_up_sampled()

        hst_up_sampled_psf = aa.kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.05, pixel_scales=0.03
        )

        assert hst_up_sampled.shape_2d == (401, 401)
        assert hst_up_sampled.pixel_scales == (0.03, 0.03)
        assert hst_up_sampled.psf == hst_up_sampled_psf
        assert hst_up_sampled.exposure_time == 2000.0
        assert hst_up_sampled.background_level == 1.0

        adaptive_optics = aa.simulator.imaging.keck_adaptive_optics()

        adaptive_optics_psf = aa.kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.025, pixel_scales=0.01
        )

        assert adaptive_optics.shape_2d == (751, 751)
        assert adaptive_optics.pixel_scales == (0.01, 0.01)
        assert adaptive_optics.psf == adaptive_optics_psf
        assert adaptive_optics.exposure_time == 1000.0
        assert adaptive_optics.background_level == 1.0

    def test__from_image_same_as_manual_image_input(self):

        psf = aa.kernel.manual_2d(
            array=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
            pixel_scales=1.0,
        )

        image = aa.array.manual_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        simulator = aa.simulator.imaging(
            shape_2d=(20, 20),
            pixel_scales=0.05,
            sub_size=1,
            psf=psf,
            exposure_time=10000.0,
            background_level=100.0,
            add_noise=True,
            noise_seed=1,
        )

        imaging_simulated = simulator.from_image(image=image)

        imaging_manual = aa.imaging.simulate(
            image=image,
            exposure_time=10000.0,
            psf=psf,
            background_level=100.0,
            add_noise=True,
            noise_seed=1,
        )

        assert (imaging_simulated.image.in_2d == imaging_manual.image.in_2d).all()
        assert (imaging_simulated.psf == imaging_manual.psf).all()
        assert (imaging_simulated.noise_map == imaging_manual.noise_map).all()
        assert (
            imaging_simulated.background_sky_map == imaging_manual.background_sky_map
        ).all()
        assert (
            imaging_simulated.exposure_time_map == imaging_manual.exposure_time_map
        ).all()


class TestInterferometerSimulator:
    def test__simulator_grid_is_uniform_grid_with_same_inputs(self):

        grid = aa.grid.uniform(
            shape_2d=(31, 31), pixel_scales=0.05, sub_size=1, origin=(0.1, 0.1)
        )

        simulator = aa.simulator.interferometer(
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

        sma = aa.simulator.interferometer.sma()

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

        primary_beam = aa.kernel.manual_2d(
            array=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
            pixel_scales=1.0,
        )

        real_space_image = aa.array.manual_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        simulator = aa.simulator.interferometer(
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

        interferometer_simulated = simulator.from_real_space_image(
            real_space_image=real_space_image
        )

        interferometer_manual = aa.interferometer.simulate(
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
