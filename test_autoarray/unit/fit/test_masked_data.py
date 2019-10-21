import autoarray as aa
from autoarray.operators import convolution, fourier_transform
from autoarray.fit import masked_data as md
import numpy as np
import pytest


class TestAbstractMaskedData(object):
    
    def test__grids_are_setup_if_input_mask_has_pixel_scale(
        self, imaging_7x7, sub_mask_7x7, grid_7x7, sub_grid_7x7, blurring_grid_7x7
    ):
        
        masked_data = md.AbstractMaskedData(mask=sub_mask_7x7)

        assert (masked_data.grid.in_1d_binned == grid_7x7).all()
        assert (masked_data.grid.in_1d == sub_grid_7x7).all()

        sub_mask_7x7.pixel_scales = None

        masked_data = md.AbstractMaskedData(mask=sub_mask_7x7)

        assert masked_data.grid is None

    def test__pixel_scale_interpolation_grid_input__grids_nclude_interpolators(
        self, sub_mask_7x7
    ):

        masked_imaging_7x7 = md.AbstractMaskedData(
            mask=sub_mask_7x7, pixel_scale_interpolation_grid=1.0
        )

        grid = aa.masked_grid.from_mask(mask=sub_mask_7x7)
        new_grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=1.0)

        assert (masked_imaging_7x7.grid == new_grid).all()
        assert (
            masked_imaging_7x7.grid.interpolator.vtx == new_grid.interpolator.vtx
        ).all()
        assert (
            masked_imaging_7x7.grid.interpolator.wts == new_grid.interpolator.wts
        ).all()

        masked_imaging_7x7 = md.AbstractMaskedData(
            mask=sub_mask_7x7,
        )

        assert masked_imaging_7x7.grid.interpolator is None

    def test__inversion_pixel_limit(self, sub_mask_7x7):
        masked_imaging_7x7 = md.AbstractMaskedData(
            mask=sub_mask_7x7, inversion_pixel_limit=2
        )

        assert masked_imaging_7x7.inversion_pixel_limit == 2

        masked_imaging_7x7 = md.AbstractMaskedData(
            mask=sub_mask_7x7, inversion_pixel_limit=5
        )

        assert masked_imaging_7x7.inversion_pixel_limit == 5

    def test__inversion_uses_border(self, sub_mask_7x7):
        masked_imaging_7x7 = md.AbstractMaskedData(
            mask=sub_mask_7x7, inversion_uses_border=True
        )

        assert masked_imaging_7x7.inversion_uses_border == True

        masked_imaging_7x7 = md.AbstractMaskedData(
            mask=sub_mask_7x7, inversion_uses_border=False
        )

        assert masked_imaging_7x7.inversion_uses_border == False

    def test__hyper_noise_map_max(self, sub_mask_7x7):
        masked_imaging_7x7 = md.AbstractMaskedData(
            mask=sub_mask_7x7, hyper_noise_map_max=10.0
        )

        assert masked_imaging_7x7.hyper_noise_map_max == 10.0

        masked_imaging_7x7 = md.AbstractMaskedData(
            mask=sub_mask_7x7, hyper_noise_map_max=20.0
        )

        assert masked_imaging_7x7.hyper_noise_map_max == 20.0


class TestMaskedImaging(object):
    def test__masked_data(self, imaging_7x7, sub_mask_7x7):

        masked_imaging_7x7 = aa.masked_imaging.manual(
            imaging=imaging_7x7,
            mask=sub_mask_7x7,
        )

        assert (
            masked_imaging_7x7.image.in_1d
            == np.ones(9,)
        ).all()

        assert (
            masked_imaging_7x7.image.in_2d
            == np.ones((7, 7)) * np.invert(sub_mask_7x7)
        ).all()

        assert (
            masked_imaging_7x7.noise_map.in_1d
            == 2.0 * np.ones(9,)
        ).all()
        assert (
            masked_imaging_7x7.noise_map.in_2d
            == 2.0 * np.ones((7, 7)) * np.invert(sub_mask_7x7)
        ).all()

        assert (masked_imaging_7x7.psf.in_1d == np.ones(9,)).all()
        assert (masked_imaging_7x7.psf.in_2d == np.ones((3, 3))).all()
        assert masked_imaging_7x7.trimmed_psf_shape_2d == (3, 3)

    def test__blurring_grid(
        self, imaging_7x7, sub_mask_7x7, grid_7x7, sub_grid_7x7, blurring_grid_7x7
    ):
        masked_imaging_7x7 = aa.masked_imaging.manual(
            imaging=imaging_7x7,
            mask=sub_mask_7x7,
        )

        assert (masked_imaging_7x7.grid.in_1d_binned == grid_7x7).all()
        assert (masked_imaging_7x7.grid.in_1d == sub_grid_7x7).all()
        assert (masked_imaging_7x7.blurring_grid.in_1d == blurring_grid_7x7).all()

    def test__pixel_scale_interpolation_grid_input__grids_include_interpolator_on_blurring_grid(
        self, imaging_7x7, sub_mask_7x7
    ):

        masked_imaging_7x7 = aa.masked_imaging.manual(
            imaging=imaging_7x7,
            mask=sub_mask_7x7,
            pixel_scale_interpolation_grid=1.0,
        )

        grid = aa.masked_grid.from_mask(mask=sub_mask_7x7)
        new_grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=1.0)

        blurring_grid = grid.blurring_grid_from_kernel_shape(
            kernel_shape=(3, 3)
        )
        new_blurring_grid = blurring_grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=1.0
        )

        assert (masked_imaging_7x7.grid == new_grid).all()
        assert (
            masked_imaging_7x7.grid.interpolator.vtx == new_grid.interpolator.vtx
        ).all()
        assert (
            masked_imaging_7x7.grid.interpolator.wts == new_grid.interpolator.wts
        ).all()

        assert (masked_imaging_7x7.blurring_grid == new_blurring_grid).all()
        assert (
            masked_imaging_7x7.blurring_grid.interpolator.vtx
            == new_blurring_grid.interpolator.vtx
        ).all()
        assert (
            masked_imaging_7x7.blurring_grid.interpolator.wts
            == new_blurring_grid.interpolator.wts
        ).all()

    def test__convolvers(self, imaging_7x7, sub_mask_7x7):

        masked_imaging_7x7 = aa.masked_imaging.manual(
            imaging=imaging_7x7,
            mask=sub_mask_7x7)

        assert type(masked_imaging_7x7.convolver) == convolution.Convolver

    def test__different_imaging_without_mock_objects__customize_constructor_inputs(
        self
    ):

        psf = aa.kernel.ones(shape_2d=(7,7), pixel_scales=3.0)
        imaging = aa.imaging(
            image=aa.array.ones(shape_2d=(19, 19), pixel_scales=3.0),
            psf=psf,
            noise_map=aa.array.full(fill_value=2.0, shape_2d=(19, 19), pixel_scales=3.0),
        )
        mask = aa.mask.unmasked(
            shape_2d=(19, 19), pixel_scales=1.0, invert=True, sub_size=8
        )
        mask[9, 9] = False

        masked_imaging = aa.masked_imaging.manual(
            imaging=imaging,
            mask=mask,
            trimmed_psf_shape_2d=(7, 7),
        )

        assert (
            masked_imaging.imaging.image.in_2d
            == np.ones((19, 19))
        ).all()
        assert (
            masked_imaging.imaging.noise_map.in_2d
            == 2.0 * np.ones((19, 19))
        ).all()
        assert (masked_imaging.psf.in_2d == np.ones((7, 7))).all()
        assert masked_imaging.convolver.kernel.shape == (7, 7)
        assert (masked_imaging.image == np.array([1.0])).all()
        assert (masked_imaging.noise_map == np.array([2.0])).all()

    def test__masked_imaging_6x6_with_binned_up_imaging(
        self,imaging_6x6, mask_6x6
    ):

        masked_imaging_6x6 = aa.masked_imaging.manual(imaging=imaging_6x6, mask=mask_6x6)

        binned_mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        binned_up_psf = masked_imaging_6x6.imaging.psf.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.5,
        )

        masked_imaging = masked_imaging_6x6.binned_from_bin_up_factor(
            bin_up_factor=2
        )

        assert (masked_imaging.mask == binned_mask).all()
        assert (masked_imaging.psf == binned_up_psf).all()

        assert (
            masked_imaging.image.in_2d
            == np.ones((3, 3)) * np.invert(binned_mask)
        ).all()

        assert (
            masked_imaging.noise_map.in_2d
            == np.ones((3, 3)) * np.invert(binned_mask)
        ).all()

        assert (masked_imaging.image.in_1d == np.ones((1))).all()
        assert (masked_imaging.noise_map.in_1d == np.ones((1))).all()

    def test__masked_imaging_7x7_with_signal_to_noise_limit(
        self, imaging_7x7, mask_7x7
    ):

        masked_imaging_7x7 = aa.masked_imaging.manual(
            imaging=imaging_7x7,
            mask=mask_7x7)

        masked_data_snr_limit = masked_imaging_7x7.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=0.25
        )

        assert (
            masked_data_snr_limit.image.in_2d
            == np.ones((7, 7)) * np.invert(mask_7x7)
        ).all()

        assert (
            masked_data_snr_limit.noise_map.in_2d
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        assert (masked_data_snr_limit.psf.in_2d == np.ones((3, 3))).all()
        assert masked_data_snr_limit.trimmed_psf_shape_2d == (3, 3)

        assert (masked_data_snr_limit.image.in_1d == np.ones(9)).all()
        assert (masked_data_snr_limit.noise_map.in_1d == 4.0 * np.ones(9)).all()


class TestMaskedInterferometer(object):
    def test__masked_data(self, interferometer_7, sub_mask_7x7):

        masked_interferometer_7 = aa.masked_interferometer.manual(
        interferometer=interferometer_7,
            mask=sub_mask_7x7,
        )

        assert (
            masked_interferometer_7.visibilities == interferometer_7.visibilities
        ).all()
        assert (masked_interferometer_7.visibilities == np.ones((7, 2))).all()

        assert (masked_interferometer_7.noise_map == 2.0 * np.ones((7, 2))).all()

        assert (
            masked_interferometer_7.visibilities_mask
            == np.full(fill_value=False, shape=(7, 2))
        ).all()

        assert (masked_interferometer_7.primary_beam.in_2d == np.ones((3, 3))).all()
        assert masked_interferometer_7.trimmed_primary_beam_shape_2d == (3, 3)

        assert (
            masked_interferometer_7.interferometer.uv_wavelengths
            == interferometer_7.uv_wavelengths
        ).all()
        assert masked_interferometer_7.interferometer.uv_wavelengths[0, 0] == -55636.4609375

    def test__transformer(self, interferometer_7, sub_mask_7x7):

        masked_interferometer_7 = aa.masked_interferometer.manual(
        interferometer=interferometer_7,
            mask=sub_mask_7x7,
        )

        assert type(masked_interferometer_7.transformer) == fourier_transform.Transformer

    def test__different_interferometer_without_mock_objects__customize_constructor_inputs(
        self
    ):
        primary_beam = aa.kernel.ones(shape_2d=(7, 7), pixel_scales=1.0)

        interferometer = aa.interferometer(
            shape_2d=(2, 2),
            visibilities=np.ones((19, 2)),
            pixel_scales=3.0,
            primary_beam=primary_beam,
            noise_map=2.0 * np.ones((19,)),
            uv_wavelengths=3.0 * np.ones((19, 2)),
        )
        mask = aa.mask.unmasked(
            shape_2d=(19, 19), pixel_scales=1.0, invert=True, sub_size=8
        )
        mask[9, 9] = False

        masked_interferometer_7 = aa.masked_interferometer.manual(
            interferometer=interferometer,
            mask=mask,
            trimmed_primary_beam_shape_2d=(7, 7),
        )

        assert (masked_interferometer_7.visibilities == np.ones((19, 2))).all()
        assert (masked_interferometer_7.noise_map == 2.0 * np.ones((19,2))).all()
        assert (
            masked_interferometer_7.interferometer.uv_wavelengths == 3.0 * np.ones((19, 2))
        ).all()
        assert (masked_interferometer_7.primary_beam.in_2d == np.ones((7, 7))).all()

        assert masked_interferometer_7.mask.sub_size == 8
        assert masked_interferometer_7.trimmed_primary_beam_shape_2d == (7, 7)