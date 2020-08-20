import os

import numpy as np
import pytest
import shutil

import autoarray as aa

test_data_dir = "{}/files/imaging/".format(os.path.dirname(os.path.realpath(__file__)))


class TestImaging:
    def test__new_imaging_binned(self):

        image = aa.Array.manual_2d(array=np.ones((6, 6)), pixel_scales=1.0)
        image[21] = 2.0
        image[27] = 2.0
        image[33] = 2.0

        binned_image_util = aa.util.binning.bin_array_2d_via_mean(
            array_2d=image.in_2d, bin_up_factor=2
        )

        noise_map_array = aa.Array.ones(shape_2d=(6, 6), pixel_scales=1.0)
        noise_map_array[21:24] = 3.0
        binned_noise_map_util = aa.util.binning.bin_array_2d_via_quadrature(
            array_2d=noise_map_array.in_2d, bin_up_factor=2
        )

        psf = aa.Kernel.ones(shape_2d=(3, 5), pixel_scales=1.0)
        psf_util = psf.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.5, renormalize=False
        )

        imaging = aa.Imaging(image=image, psf=psf, noise_map=noise_map_array)

        imaging = imaging.binned_from_bin_up_factor(bin_up_factor=2)

        assert (imaging.image.in_2d == binned_image_util).all()
        assert (imaging.psf == psf_util).all()

        assert (imaging.noise_map.in_2d == binned_noise_map_util).all()

        assert imaging.image.pixel_scales == (2.0, 2.0)
        assert imaging.psf.pixel_scales == pytest.approx((1.0, 1.66666666666), 1.0e-4)
        assert imaging.noise_map.pixel_scales == (2.0, 2.0)

        assert imaging.image.geometry.origin == (0.0, 0.0)

    def test__new_imaging_with_signal_to_noise_limit__limit_above_max_signal_to_noise__signal_to_noise_map_unchanged(
        self
    ):
        image = aa.Array.full(fill_value=20.0, shape_2d=(2, 2), store_in_1d=True)
        image[3] = 5.0

        noise_map_array = aa.Array.full(
            fill_value=5.0, shape_2d=(2, 2), store_in_1d=True
        )
        noise_map_array[3] = 2.0

        imaging = aa.Imaging(
            image=image, psf=aa.Kernel.zeros(shape_2d=(3, 3)), noise_map=noise_map_array
        )

        imaging = imaging.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=100.0
        )

        assert (imaging.image == np.array([20.0, 20.0, 20.0, 5.0])).all()

        assert (imaging.noise_map == np.array([5.0, 5.0, 5.0, 2.0])).all()

        assert (imaging.signal_to_noise_map == np.array([4.0, 4.0, 4.0, 2.5])).all()

        assert (imaging.psf.in_2d == np.zeros((3, 3))).all()

        image = aa.Array.full(fill_value=20.0, shape_2d=(2, 2), store_in_1d=False)
        image[1, 1] = 5.0

        noise_map_array = aa.Array.full(
            fill_value=5.0, shape_2d=(2, 2), store_in_1d=False
        )
        noise_map_array[1, 1] = 2.0

        imaging = aa.Imaging(image=image, noise_map=noise_map_array)

        imaging = imaging.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=100.0
        )

        assert (imaging.image == np.array([[20.0, 20.0], [20.0, 5.0]])).all()

        assert (imaging.noise_map == np.array([[5.0, 5.0], [5.0, 2.0]])).all()

        assert (imaging.signal_to_noise_map == np.array([[4.0, 4.0], [4.0, 2.5]])).all()

    def test__new_imaging_with_signal_to_noise_limit_below_max_signal_to_noise__signal_to_noise_map_capped_to_limit(
        self
    ):
        image = aa.Array.full(fill_value=20.0, shape_2d=(2, 2))
        image[3] = 5.0

        noise_map_array = aa.Array.full(fill_value=5.0, shape_2d=(2, 2))
        noise_map_array[3] = 2.0

        imaging = aa.Imaging(
            image=image, psf=aa.Kernel.zeros(shape_2d=(3, 3)), noise_map=noise_map_array
        )

        imaging_capped = imaging.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=2.0
        )

        assert (
            imaging_capped.image.in_2d == np.array([[20.0, 20.0], [20.0, 5.0]])
        ).all()

        assert (
            imaging_capped.noise_map.in_2d == np.array([[10.0, 10.0], [10.0, 2.5]])
        ).all()

        assert (
            imaging_capped.signal_to_noise_map.in_2d
            == np.array([[2.0, 2.0], [2.0, 2.0]])
        ).all()

        assert (imaging_capped.psf.in_2d == np.zeros((3, 3))).all()

        imaging_capped = imaging.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=3.0
        )

        assert (
            imaging_capped.image.in_2d == np.array([[20.0, 20.0], [20.0, 5.0]])
        ).all()

        assert (
            imaging_capped.noise_map.in_2d
            == np.array([[(20.0 / 3.0), (20.0 / 3.0)], [(20.0 / 3.0), 2.0]])
        ).all()

        assert (
            imaging_capped.signal_to_noise_map.in_2d
            == np.array([[3.0, 3.0], [3.0, 2.5]])
        ).all()

        assert (imaging_capped.psf.in_2d == np.zeros((3, 3))).all()

    def test__from_fits__loads_arrays_and_psf_is_renormalized(self):
        imaging = aa.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=test_data_dir + "3x3_ones.fits",
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
            positions_path=test_data_dir + "positions.dat",
        )

        assert (imaging.image.in_2d == np.ones((3, 3))).all()
        assert (imaging.psf.in_2d == (1.0 / 9.0) * np.ones((3, 3))).all()
        assert (imaging.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()
        assert imaging.positions.in_list == [[(1.0, 1.0), (2.0, 2.0)]]

        assert imaging.pixel_scales == (0.1, 0.1)
        assert imaging.psf.mask.pixel_scales == (0.1, 0.1)
        assert imaging.noise_map.mask.pixel_scales == (0.1, 0.1)

    def test__from_fits__all_files_in_one_fits__load_using_different_hdus(self):

        imaging = aa.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=test_data_dir + "3x3_multiple_hdu.fits",
            image_hdu=0,
            psf_path=test_data_dir + "3x3_multiple_hdu.fits",
            psf_hdu=1,
            noise_map_path=test_data_dir + "3x3_multiple_hdu.fits",
            noise_map_hdu=2,
        )

        assert (imaging.image.in_2d == np.ones((3, 3))).all()
        assert (imaging.psf.in_2d == (1.0 / 9.0) * np.ones((3, 3))).all()
        assert (imaging.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()

        assert imaging.pixel_scales == (0.1, 0.1)
        assert imaging.psf.mask.pixel_scales == (0.1, 0.1)
        assert imaging.noise_map.mask.pixel_scales == (0.1, 0.1)

    def test__output_to_fits__outputs_all_imaging_arrays(self):

        imaging = aa.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=test_data_dir + "3x3_ones.fits",
            psf_path=test_data_dir + "3x3_twos.fits",
            noise_map_path=test_data_dir + "3x3_threes.fits",
        )

        output_data_dir = "{}/files/array/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        imaging.output_to_fits(
            image_path=output_data_dir + "image.fits",
            psf_path=output_data_dir + "psf.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
        )

        imaging = aa.Imaging.from_fits(
            pixel_scales=0.1,
            image_path=output_data_dir + "image.fits",
            psf_path=output_data_dir + "psf.fits",
            noise_map_path=output_data_dir + "noise_map.fits",
        )

        assert (imaging.image.in_2d == np.ones((3, 3))).all()
        assert (imaging.psf.in_2d == (1.0 / 9.0) * np.ones((3, 3))).all()
        assert (imaging.noise_map.in_2d == 3.0 * np.ones((3, 3))).all()

        assert imaging.pixel_scales == (0.1, 0.1)
        assert imaging.psf.mask.pixel_scales == (0.1, 0.1)
        assert imaging.noise_map.mask.pixel_scales == (0.1, 0.1)


class TestSettingsMaskedImaging:
    def test__bin_up_factor_tag(self):

        settings = aa.SettingsMaskedImaging(bin_up_factor=None)
        assert settings.bin_up_factor_tag == ""
        settings = aa.SettingsMaskedImaging(bin_up_factor=1)
        assert settings.bin_up_factor_tag == ""
        settings = aa.SettingsMaskedImaging(bin_up_factor=2)
        assert settings.bin_up_factor_tag == "__bin_2"

    def test__psf_shape_2d_tag(self):

        settings = aa.SettingsMaskedImaging(psf_shape_2d=None)
        assert settings.psf_shape_tag == ""
        settings = aa.SettingsMaskedImaging(psf_shape_2d=(2, 2))
        assert settings.psf_shape_tag == "__psf_2x2"
        settings = aa.SettingsMaskedImaging(psf_shape_2d=(3, 4))
        assert settings.psf_shape_tag == "__psf_3x4"


class TestMaskedImaging:
    def test__masked_dataset(self, imaging_7x7, sub_mask_7x7):

        masked_imaging_7x7 = aa.MaskedImaging(imaging=imaging_7x7, mask=sub_mask_7x7)

        assert (masked_imaging_7x7.image.in_1d == np.ones(9)).all()

        assert (
            masked_imaging_7x7.image.in_2d == np.ones((7, 7)) * np.invert(sub_mask_7x7)
        ).all()

        assert (masked_imaging_7x7.noise_map.in_1d == 2.0 * np.ones(9)).all()
        assert (
            masked_imaging_7x7.noise_map.in_2d
            == 2.0 * np.ones((7, 7)) * np.invert(sub_mask_7x7)
        ).all()

        assert (masked_imaging_7x7.psf.in_1d == (1.0 / 9.0) * np.ones(9)).all()
        assert (masked_imaging_7x7.psf.in_2d == (1.0 / 9.0) * np.ones((3, 3))).all()

    def test__grid(
        self,
        imaging_7x7,
        sub_mask_7x7,
        grid_7x7,
        sub_grid_7x7,
        blurring_grid_7x7,
        grid_iterate_7x7,
    ):
        masked_imaging_7x7 = aa.MaskedImaging(
            imaging=imaging_7x7,
            mask=sub_mask_7x7,
            settings=aa.SettingsMaskedImaging(grid_class=aa.Grid),
        )

        assert isinstance(masked_imaging_7x7.grid, aa.Grid)
        assert (masked_imaging_7x7.grid.in_1d_binned == grid_7x7).all()
        assert (masked_imaging_7x7.grid.in_1d == sub_grid_7x7).all()
        assert isinstance(masked_imaging_7x7.blurring_grid, aa.Grid)
        assert (masked_imaging_7x7.blurring_grid.in_1d == blurring_grid_7x7).all()

        masked_imaging_7x7 = aa.MaskedImaging(
            imaging=imaging_7x7,
            mask=sub_mask_7x7,
            settings=aa.SettingsMaskedImaging(grid_class=aa.GridIterate),
        )

        assert isinstance(masked_imaging_7x7.grid, aa.GridIterate)
        assert (masked_imaging_7x7.grid.in_1d_binned == grid_iterate_7x7).all()
        assert isinstance(masked_imaging_7x7.blurring_grid, aa.GridIterate)
        assert (masked_imaging_7x7.blurring_grid.in_1d == blurring_grid_7x7).all()

        masked_imaging_7x7 = aa.MaskedImaging(
            imaging=imaging_7x7,
            mask=sub_mask_7x7,
            settings=aa.SettingsMaskedImaging(
                grid_class=aa.GridInterpolate, pixel_scales_interp=1.0
            ),
        )

        grid = aa.GridInterpolate.from_mask(mask=sub_mask_7x7, pixel_scales_interp=1.0)

        blurring_grid = grid.blurring_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        assert isinstance(masked_imaging_7x7.grid, aa.GridInterpolate)
        assert (masked_imaging_7x7.grid == grid).all()
        assert (masked_imaging_7x7.grid.vtx == grid.vtx).all()
        assert (masked_imaging_7x7.grid.wts == grid.wts).all()

        assert isinstance(masked_imaging_7x7.blurring_grid, aa.GridInterpolate)
        assert (masked_imaging_7x7.blurring_grid == blurring_grid).all()
        assert (masked_imaging_7x7.blurring_grid.vtx == blurring_grid.vtx).all()
        assert (masked_imaging_7x7.blurring_grid.wts == blurring_grid.wts).all()

    def test__psf_and_convolvers(self, imaging_7x7, sub_mask_7x7):

        masked_imaging_7x7 = aa.MaskedImaging(imaging=imaging_7x7, mask=sub_mask_7x7)

        assert type(masked_imaging_7x7.psf) == aa.Kernel
        assert type(masked_imaging_7x7.convolver) == aa.Convolver

    def test__masked_imaging__uses_bin_up_factor(self, imaging_7x7, mask_7x7_1_pix):

        masked_imaging_7x7 = aa.MaskedImaging(
            imaging=imaging_7x7,
            mask=mask_7x7_1_pix,
            settings=aa.SettingsMaskedImaging(grid_class=aa.Grid, bin_up_factor=2),
        )

        binned_up_imaging = imaging_7x7.binned_from_bin_up_factor(bin_up_factor=2)
        binned_up_mask = mask_7x7_1_pix.binned_mask_from_bin_up_factor(bin_up_factor=2)

        assert (
            masked_imaging_7x7.image.in_2d
            == binned_up_imaging.image.in_2d * np.invert(binned_up_mask)
        ).all()

        assert (masked_imaging_7x7.psf == (1.0 / 9.0) * binned_up_imaging.psf).all()
        assert (
            masked_imaging_7x7.noise_map.in_2d
            == binned_up_imaging.noise_map.in_2d * np.invert(binned_up_mask)
        ).all()

        assert (masked_imaging_7x7.mask == binned_up_mask).all()

    def test__masked_imaging__uses_signal_to_noise_limit(
        self, imaging_7x7, mask_7x7_1_pix
    ):

        masked_imaging_7x7 = aa.MaskedImaging(
            imaging=imaging_7x7,
            mask=mask_7x7_1_pix,
            settings=aa.SettingsMaskedImaging(
                grid_class=aa.Grid, signal_to_noise_limit=1.0
            ),
        )

        imaging_snr_limit = imaging_7x7.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=1.0
        )

        assert (
            masked_imaging_7x7.image.in_2d
            == imaging_snr_limit.image.in_2d * np.invert(mask_7x7_1_pix)
        ).all()
        assert (
            masked_imaging_7x7.noise_map.in_2d
            == imaging_snr_limit.noise_map.in_2d * np.invert(mask_7x7_1_pix)
        ).all()

    def test__different_imaging_without_mock_objects__customize_constructor_inputs(
        self
    ):

        psf = aa.Kernel.ones(shape_2d=(7, 7), pixel_scales=3.0)
        imaging = aa.Imaging(
            image=aa.Array.ones(shape_2d=(19, 19), pixel_scales=3.0),
            psf=psf,
            noise_map=aa.Array.full(
                fill_value=2.0, shape_2d=(19, 19), pixel_scales=3.0
            ),
        )
        mask = aa.Mask.unmasked(
            shape_2d=(19, 19), pixel_scales=1.0, invert=True, sub_size=8
        )
        mask[9, 9] = False

        masked_imaging = aa.MaskedImaging(
            imaging=imaging,
            mask=mask,
            settings=aa.SettingsMaskedImaging(psf_shape_2d=(7, 7)),
        )

        assert (masked_imaging.imaging.image.in_2d == np.ones((19, 19))).all()
        assert (masked_imaging.imaging.noise_map.in_2d == 2.0 * np.ones((19, 19))).all()
        assert (masked_imaging.psf.in_2d == (1.0 / 49.0) * np.ones((7, 7))).all()
        assert masked_imaging.convolver.kernel.shape_2d == (7, 7)
        assert (masked_imaging.image == np.array([1.0])).all()
        assert (masked_imaging.noise_map == np.array([2.0])).all()

    def test__modified_image_and_noise_map(
        self, image_7x7, noise_map_7x7, imaging_7x7, sub_mask_7x7
    ):

        masked_imaging_7x7 = aa.MaskedImaging(imaging=imaging_7x7, mask=sub_mask_7x7)

        image_7x7[0] = 10.0
        noise_map_7x7[0] = 11.0

        masked_imaging_7x7 = masked_imaging_7x7.modify_image_and_noise_map(
            image=image_7x7, noise_map=noise_map_7x7
        )

        assert masked_imaging_7x7.image.in_1d[0] == 10.0
        assert masked_imaging_7x7.image.in_2d[0, 0] == 10.0
        assert masked_imaging_7x7.noise_map.in_1d[0] == 11.0
        assert masked_imaging_7x7.noise_map.in_2d[0, 0] == 11.0


class TestSimulatorImaging:
    def test__from_image__all_features_off(self):

        image = aa.Array.manual_2d(
            array=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            pixel_scales=0.1,
        )

        exposure_time_map = aa.Array.ones(shape_2d=image.shape_2d)

        simulator = aa.SimulatorImaging(
            exposure_time_map=exposure_time_map, add_noise=False
        )

        imaging = simulator.from_image(image=image)

        assert (
            imaging.image.in_2d
            == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()
        assert imaging.pixel_scales == (0.1, 0.1)

    def test__from_image__noise_off___noise_map_is_noise_value(self):

        image = aa.Array.manual_2d(
            array=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        )

        exposure_time_map = aa.Array.ones(shape_2d=image.shape_2d)

        simulator = aa.SimulatorImaging(
            exposure_time_map=exposure_time_map,
            add_noise=False,
            noise_if_add_noise_false=0.2,
            noise_seed=1,
        )

        imaging = simulator.from_image(image=image)

        assert (
            imaging.image.in_2d
            == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()
        assert (imaging.noise_map.in_2d == 0.2 * np.ones((3, 3))).all()

    def test__from_image__psf_blurs_image_with_edge_trimming(self):

        image = aa.Array.manual_2d(
            array=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )

        psf = aa.Kernel.manual_2d(
            array=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
        )

        exposure_time_map = aa.Array.ones(shape_2d=image.shape_2d)

        simulator = aa.SimulatorImaging(
            exposure_time_map=exposure_time_map,
            psf=psf,
            add_noise=False,
            renormalize_psf=False,
        )

        imaging = simulator.from_image(image=image)

        assert (
            imaging.image.in_2d
            == np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
        ).all()

    def test__setup_with_noise(self):

        image = aa.Array.manual_2d(
            array=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        )

        exposure_time_map = aa.Array.full(fill_value=20.0, shape_2d=image.shape_2d)

        simulator = aa.SimulatorImaging(
            exposure_time_map=exposure_time_map, add_noise=True, noise_seed=1
        )

        imaging = simulator.from_image(image=image)

        assert imaging.image.in_2d == pytest.approx(
            np.array([[0.0, 0.0, 0.0], [0.0, 1.05, 0.0], [0.0, 0.0, 0.0]]), 1e-2
        )

        # Because of the value is 1.05, the estimated Poisson noise_map_1d is:
        # sqrt((1.05 * 20))/20 = 0.2291

        assert imaging.noise_map.in_2d == pytest.approx(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.2291, 0.0], [0.0, 0.0, 0.0]]), 1e-2
        )

    def test__from_image__background_sky_on__noise_on_so_background_adds_noise_to_image(
        self
    ):

        image = aa.Array.manual_2d(
            array=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        )

        exposure_time_map = aa.Array.ones(shape_2d=image.shape_2d)

        background_sky_map = aa.Array.full(fill_value=16.0, shape_2d=image.shape_2d)

        simulator = aa.SimulatorImaging(
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            add_noise=True,
            noise_seed=1,
        )

        imaging = simulator.from_image(image=image)

        assert (
            imaging.image.in_2d
            == np.array([[1.0, 5.0, 4.0], [1.0, 2.0, 1.0], [5.0, 2.0, 7.0]])
        ).all()

        assert imaging.noise_map.in_2d[0, 0] == pytest.approx(4.12310, 1.0e-4)

    def test__from_image__psf_and_noise__noise_added_to_blurred_image(self):
        image = aa.Array.manual_2d(
            array=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        )

        psf = aa.Kernel.manual_2d(
            array=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
        )

        exposure_time_map = aa.Array.full(fill_value=20.0, shape_2d=image.shape_2d)

        simulator = aa.SimulatorImaging(
            exposure_time_map=exposure_time_map,
            psf=psf,
            add_noise=True,
            noise_seed=1,
            renormalize_psf=False,
        )

        imaging = simulator.from_image(image=image)

        assert imaging.image.in_2d == pytest.approx(
            np.array([[0.0, 1.05, 0.0], [1.3, 2.35, 1.05], [0.0, 1.05, 0.0]]), 1e-2
        )

    def test__modified_noise_map(self, noise_map_7x7, imaging_7x7, mask_7x7):

        masked_imaging_7x7 = aa.MaskedImaging(imaging=imaging_7x7, mask=mask_7x7)

        noise_map_7x7[0] = 11.0

        masked_imaging_7x7 = masked_imaging_7x7.modify_noise_map(
            noise_map=noise_map_7x7
        )

        assert masked_imaging_7x7.noise_map[0] == 11.0
