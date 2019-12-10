import os
import shutil

import numpy as np
import pytest

import autoarray as aa
from autoarray.dataset import interferometer
from autoarray import exc

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestInterferometerMethods(object):
    def test__data_with_resized_primary_beam(self):

        interferometer = aa.interferometer.manual(
            visibilities=aa.visibilities.manual_1d(visibilities=[[1, 1]]),
            primary_beam=aa.kernel.zeros(shape_2d=(5, 5), pixel_scales=1.0),
            noise_map=aa.visibilities.manual_1d(visibilities=[[2, 2]]),
            exposure_time_map=1,
            uv_wavelengths=1,
        )

        interferometer = interferometer.resized_primary_beam_from_new_shape_2d(
            new_shape_2d=(1, 1)
        )

        assert (interferometer.primary_beam.in_2d == np.zeros((1, 1))).all()

    def test__data_with_modified_visibilities(self):

        interferometer = aa.interferometer.manual(
            visibilities=np.array([[1, 1]]),
            primary_beam=aa.kernel.zeros(shape_2d=(5, 5), pixel_scales=1.0),
            noise_map=1,
            exposure_time_map=2,
            uv_wavelengths=3,
        )

        interferometer = interferometer.modified_visibilities_from_visibilities(
            visibilities=np.array([[2, 2]])
        )

        assert (interferometer.visibilities == np.array([[2, 2]])).all()
        assert (interferometer.primary_beam.in_2d == np.zeros((1, 1))).all()
        assert interferometer.noise_map == 1
        assert interferometer.exposure_time_map == 2
        assert interferometer.uv_wavelengths == 3


class TestSimulateInterferometer(object):
    def test__setup_with_all_features_off(self, transformer_7x7_7):
        image = aa.array.manual_2d([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])

        exposure_time_map = aa.array.full(
            fill_value=1.0, pixel_scales=0.1, shape_2d=image.shape_2d
        )

        interferometer_simulated = aa.interferometer.simulate(
            real_space_image=image,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            real_space_pixel_scales=0.1,
            transformer=transformer_7x7_7,
            noise_sigma=None,
        )

        simulated_visibilities = transformer_7x7_7.visibilities_from_image(image=image)

        assert interferometer_simulated.visibilities == pytest.approx(
            simulated_visibilities, 1.0e-4
        )
        assert interferometer_simulated.real_space_pixel_scales == (0.1, 0.1)

    def test__setup_with_background_sky_on__noise_off__no_noise_in_image__noise_map_is_noise_value(
        self, transformer_7x7_7
    ):
        image = aa.array.manual_2d([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])

        exposure_time_map = aa.array.full(
            fill_value=1.0, pixel_scales=0.1, shape_2d=image.shape_2d
        )

        background_sky_map = aa.array.full(
            fill_value=2.0, pixel_scales=0.1, shape_2d=image.shape_2d
        )

        interferometer_simulated = aa.interferometer.simulate(
            real_space_image=image,
            real_space_pixel_scales=0.1,
            exposure_time=1.0,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            transformer=transformer_7x7_7,
            noise_sigma=None,
            noise_if_add_noise_false=0.2,
            noise_seed=1,
        )

        simulated_visibilities = transformer_7x7_7.visibilities_from_image(
            image=image + background_sky_map
        )

        assert interferometer_simulated.visibilities == pytest.approx(
            simulated_visibilities, 1.0e-4
        )
        assert (
            interferometer_simulated.exposure_time_map.in_2d == 1.0 * np.ones((3, 3))
        ).all()

        assert (interferometer_simulated.noise_map == 0.2 * np.ones((7, 2))).all()
        assert interferometer_simulated.real_space_pixel_scales == (0.1, 0.1)

    def test__setup_with_noise(self, transformer_7x7_7):

        image = aa.array.manual_2d([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]])

        exposure_time_map = aa.array.full(
            fill_value=20.0, pixel_scales=0.1, shape_2d=image.shape_2d
        )

        interferometer_simulated = aa.interferometer.simulate(
            real_space_image=image,
            real_space_pixel_scales=0.1,
            exposure_time=20.0,
            exposure_time_map=exposure_time_map,
            transformer=transformer_7x7_7,
            noise_sigma=0.1,
            noise_seed=1,
        )

        simulated_visibilities = transformer_7x7_7.visibilities_from_image(image=image)

        assert (
            interferometer_simulated.exposure_time_map.in_2d == 20.0 * np.ones((3, 3))
        ).all()
        assert interferometer_simulated.real_space_pixel_scales == (0.1, 0.1)

        assert interferometer_simulated.visibilities[0, :] == pytest.approx(
            [-0.005364, -2.36682], 1.0e-4
        )
        noise_map_realization = (
            interferometer_simulated.visibilities - simulated_visibilities
        )

        assert noise_map_realization == pytest.approx(
            interferometer_simulated.noise_map_realization, 1.0e-4
        )

        assert (interferometer_simulated.noise_map == 0.1 * np.ones((7, 2))).all()

    class TestCreateGaussianNoiseMap(object):
        def test__gaussian_noise_sigma_0__gaussian_noise_map_all_0__image_is_identical_to_input(
            self
        ):
            simulate_gaussian_noise = interferometer.gaussian_noise_map_from_shape_and_sigma(
                shape=(9,), sigma=0.0, noise_seed=1
            )

            assert (simulate_gaussian_noise == np.zeros((9,))).all()

        def test__gaussian_noise_sigma_1__gaussian_noise_map_all_non_0__image_has_noise_added(
            self
        ):
            simulate_gaussian_noise = interferometer.gaussian_noise_map_from_shape_and_sigma(
                shape=(9,), sigma=1.0, noise_seed=1
            )

            # Use seed to give us a known gaussian noises map we'll test_autoarray for

            assert simulate_gaussian_noise == pytest.approx(
                np.array([1.62, -0.61, -0.53, -1.07, 0.87, -2.30, 1.74, -0.76, 0.32]),
                1e-2,
            )


class TestInterferometerFromFits(object):
    def test__no_settings_just_pass_fits(self):

        interferometer = aa.interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_ones_twos.fits",
            noise_map_path=test_data_dir + "3x2_threes_fours.fits",
            uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            renormalize_primary_beam=False,
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
        assert (interferometer.primary_beam.in_2d == 5.0 * np.ones((3, 3))).all()

    def test__optional_array_paths_included__loads_optional_array(self):

        interferometer = aa.interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_ones_twos.fits",
            noise_map_path=test_data_dir + "3x2_threes_fours.fits",
            uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            renormalize_primary_beam=False,
        )

        assert (interferometer.primary_beam.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (interferometer.exposure_time_map == 6.0 * np.ones((3,))).all()

    def test__all_files_in_one_fits__load_using_different_hdus(self):

        interferometer = aa.interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_multiple_hdu.fits",
            visibilities_hdu=0,
            noise_map_path=test_data_dir + "3x2_multiple_hdu.fits",
            noise_map_hdu=1,
            uv_wavelengths_path=test_data_dir + "3x2_multiple_hdu.fits",
            uv_wavelengths_hdu=2,
            primary_beam_path=test_data_dir + "3x3_multiple_hdu.fits",
            primary_beam_hdu=3,
            exposure_time_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            exposure_time_map_hdu=4,
            renormalize_primary_beam=False,
        )

        assert (interferometer.visibilities.real == np.ones(3)).all()
        assert (interferometer.visibilities.imag == np.ones(3)).all()
        assert (interferometer.noise_map.real == 2.0 * np.ones(3)).all()
        assert (interferometer.noise_map.imag == 2.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 0] == 3.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 1] == 3.0 * np.ones(3)).all()
        assert (interferometer.primary_beam.in_2d == 4.0 * np.ones((3, 3))).all()
        assert (interferometer.exposure_time_map == 5.0 * np.ones((3, 3))).all()

    def test__exposure_time_included__creates_exposure_time_map_using_exposure_time(
        self
    ):

        interferometer = aa.interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_ones_twos.fits",
            noise_map_path=test_data_dir + "3x2_threes_fours.fits",
            uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
            primary_beam_path=test_data_dir + "3x3_ones.fits",
            exposure_time_map_from_single_value=3.0,
        )

        assert (interferometer.exposure_time_map == 3.0 * np.ones((3,))).all()

    def test__pad_shape_of_primary_beam(self):

        interferometer = aa.interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_ones_twos.fits",
            noise_map_path=test_data_dir + "3x2_threes_fours.fits",
            uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            resized_primary_beam_shape_2d=(9, 9),
            renormalize_primary_beam=False,
        )

        primary_beam_padded_array = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        assert (interferometer.primary_beam.in_2d == primary_beam_padded_array).all()

    def test__trim_shape_of_primary_beam(self):

        interferometer = aa.interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_ones_twos.fits",
            noise_map_path=test_data_dir + "3x2_threes_fours.fits",
            uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            resized_primary_beam_shape_2d=(1, 1),
            renormalize_primary_beam=False,
        )

        trimmed_array = np.array([[1.0]])

        assert (interferometer.primary_beam.in_2d == 5.0 * trimmed_array).all()

    def test__primary_beam_renormalized_false__does_not_renormalize_primary_beam(self):

        interferometer = aa.interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_ones_twos.fits",
            noise_map_path=test_data_dir + "3x2_threes_fours.fits",
            uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            renormalize_primary_beam=False,
        )

        assert (interferometer.primary_beam.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (interferometer.exposure_time_map == 6.0 * np.ones((3,))).all()

    def test__primary_beam_renormalized_true__renormalized_primary_beam(self):

        interferometer = aa.interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_ones_twos.fits",
            noise_map_path=test_data_dir + "3x2_threes_fours.fits",
            uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            renormalize_primary_beam=True,
        )

        assert interferometer.primary_beam.in_2d == pytest.approx(
            (1.0 / 9.0) * np.ones((3, 3)), 1e-2
        )
        assert (interferometer.exposure_time_map == 6.0 * np.ones((3,))).all()

    def test__exposure_time_and_exposure_time_map_included__raies_imaging_error(self):

        with pytest.raises(exc.DataException):
            aa.interferometer.from_fits(
                visibilities_path=test_data_dir + "3x2_ones_twos.fits",
                noise_map_path=test_data_dir + "3x3_threes.fits",
                uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
                exposure_time_map_path=test_data_dir + "3x3_ones.fits",
                exposure_time_map_from_single_value=1.0,
            )

    def test__output_all_arrays(self):

        interferometer = aa.interferometer.from_fits(
            visibilities_path=test_data_dir + "3x2_ones_twos.fits",
            noise_map_path=test_data_dir + "3x2_threes_fours.fits",
            uv_wavelengths_path=test_data_dir + "3x2_fives_sixes.fits",
            primary_beam_path=test_data_dir + "3x3_fives.fits",
            exposure_time_map_path=test_data_dir + "3_sixes.fits",
            renormalize_primary_beam=False,
        )

        output_data_dir = "{}/../test_files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        interferometer.output_to_fits(
            visibilities_path=output_data_dir + "visibilities.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            primary_beam_path=output_data_dir + "primary_beam.fits",
            exposure_time_map_path=output_data_dir + "exposure_time_map.fits",
            uv_wavelengths_path=output_data_dir + "uv_wavelengths.fits",
            overwrite=True,
        )

        interferometer = aa.interferometer.from_fits(
            visibilities_path=output_data_dir + "visibilities.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
            primary_beam_path=output_data_dir + "primary_beam.fits",
            exposure_time_map_path=output_data_dir + "exposure_time_map.fits",
            uv_wavelengths_path=output_data_dir + "uv_wavelengths.fits",
            renormalize_primary_beam=False,
        )

        assert (interferometer.visibilities.real == np.ones(3)).all()
        assert (interferometer.visibilities.imag == 2.0 * np.ones(3)).all()
        assert (interferometer.noise_map.real == 3.0 * np.ones((3,))).all()
        assert (interferometer.noise_map.imag == 4.0 * np.ones((3,))).all()
        assert (interferometer.uv_wavelengths[:, 0] == 5.0 * np.ones(3)).all()
        assert (interferometer.uv_wavelengths[:, 1] == 6.0 * np.ones(3)).all()
        assert (interferometer.primary_beam.in_2d == 5.0 * np.ones((3, 3))).all()
        assert (interferometer.exposure_time_map == 6.0 * np.ones((3,))).all()
