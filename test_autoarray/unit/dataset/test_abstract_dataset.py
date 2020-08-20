import logging

import autoarray as aa
from autoarray.dataset import abstract_dataset
import numpy as np

logger = logging.getLogger(__name__)


class TestInverseNoiseMap:
    def test__inverse_noise_is_one_over_noise(self):
        array = aa.Array.manual_2d([[1.0, 2.0], [3.0, 4.0]])
        noise_map = aa.Array.manual_2d([[1.0, 2.0], [4.0, 8.0]])

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.inverse_noise_map.in_2d == np.array([[1.0, 0.5], [0.25, 0.125]])
        ).all()


class TestSignalToNoise:
    def test__image_and_noise_are_values__signal_to_noise_is_ratio_of_each(self):
        array = aa.Array.manual_2d([[1.0, 2.0], [3.0, 4.0]])
        noise_map = aa.Array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.signal_to_noise_map.in_2d == np.array([[0.1, 0.2], [0.1, 1.0]])
        ).all()
        assert dataset.signal_to_noise_max == 1.0

    def test__same_as_above__but_image_has_negative_values__replaced_with_zeros(self):
        array = aa.Array.manual_2d([[-1.0, 2.0], [3.0, -4.0]])

        noise_map = aa.Array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.signal_to_noise_map.in_2d == np.array([[0.0, 0.2], [0.1, 0.0]])
        ).all()
        assert dataset.signal_to_noise_max == 0.2


class TestAbsoluteSignalToNoise:
    def test__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(
        self
    ):
        array = aa.Array.manual_2d([[-1.0, 2.0], [3.0, -4.0]])

        noise_map = aa.Array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.absolute_signal_to_noise_map.in_2d
            == np.array([[0.1, 0.2], [0.1, 1.0]])
        ).all()
        assert dataset.absolute_signal_to_noise_max == 1.0


class TestPotentialChiSquaredMap:
    def test__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(
        self
    ):
        array = aa.Array.manual_2d([[-1.0, 2.0], [3.0, -4.0]])
        noise_map = aa.Array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.potential_chi_squared_map.in_2d
            == np.array([[0.1 ** 2.0, 0.2 ** 2.0], [0.1 ** 2.0, 1.0 ** 2.0]])
        ).all()
        assert dataset.potential_chi_squared_max == 1.0


class TestAbstractMaskedDatasetTags:
    def test__grids__sub_size_tags(self):

        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.GridIterate, sub_size=1
        )
        assert settings.grid_sub_size_tag == ""
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.Grid, sub_size=1
        )
        assert settings.grid_sub_size_tag == "sub_1"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.Grid, sub_size=2
        )
        assert settings.grid_sub_size_tag == "sub_2"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.Grid, sub_size=4
        )
        assert settings.grid_sub_size_tag == "sub_4"

        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.GridIterate, sub_size=1
        )
        assert settings.grid_inversion_sub_size_tag == ""
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.Grid, sub_size=1
        )
        assert settings.grid_inversion_sub_size_tag == "sub_1"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.Grid, sub_size=2
        )
        assert settings.grid_inversion_sub_size_tag == "sub_2"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.Grid, sub_size=4
        )
        assert settings.grid_inversion_sub_size_tag == "sub_4"

    def test__grids__fractional_accuracy_tags(self):

        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.Grid, fractional_accuracy=1
        )
        assert settings.grid_fractional_accuracy_tag == ""
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.GridIterate, fractional_accuracy=0.5
        )
        assert settings.grid_fractional_accuracy_tag == "facc_0.5"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.GridIterate, fractional_accuracy=0.71
        )
        assert settings.grid_fractional_accuracy_tag == "facc_0.71"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.GridIterate, fractional_accuracy=0.999999
        )
        assert settings.grid_fractional_accuracy_tag == "facc_0.999999"

        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.Grid, fractional_accuracy=1
        )
        assert settings.grid_inversion_fractional_accuracy_tag == ""
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.GridIterate, fractional_accuracy=0.5
        )
        assert settings.grid_inversion_fractional_accuracy_tag == "facc_0.5"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.GridIterate, fractional_accuracy=0.71
        )
        assert settings.grid_inversion_fractional_accuracy_tag == "facc_0.71"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.GridIterate, fractional_accuracy=0.999999
        )
        assert settings.grid_inversion_fractional_accuracy_tag == "facc_0.999999"

    def test__grid__pixel_scales_interp_tag(self):

        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.Grid, pixel_scales_interp=0.1
        )
        assert settings.grid_pixel_scales_interp_tag == ""
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.GridInterpolate, pixel_scales_interp=None
        )
        assert settings.grid_pixel_scales_interp_tag == ""
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.GridInterpolate, pixel_scales_interp=0.5
        )
        assert settings.grid_pixel_scales_interp_tag == "interp_0.500"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.GridInterpolate, pixel_scales_interp=0.25
        )
        assert settings.grid_pixel_scales_interp_tag == "interp_0.250"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.GridInterpolate, pixel_scales_interp=0.234
        )
        assert settings.grid_pixel_scales_interp_tag == "interp_0.234"

        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.Grid, pixel_scales_interp=0.1
        )
        assert settings.grid_inversion_pixel_scales_interp_tag == ""
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.GridInterpolate, pixel_scales_interp=None
        )
        assert settings.grid_inversion_pixel_scales_interp_tag == ""
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.GridInterpolate, pixel_scales_interp=0.5
        )
        assert settings.grid_inversion_pixel_scales_interp_tag == "interp_0.500"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.GridInterpolate, pixel_scales_interp=0.25
        )
        assert settings.grid_inversion_pixel_scales_interp_tag == "interp_0.250"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_inversion_class=aa.GridInterpolate, pixel_scales_interp=0.234
        )
        assert settings.grid_inversion_pixel_scales_interp_tag == "interp_0.234"

    def test__grid_tags(self):

        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.Grid,
            sub_size=1,
            grid_inversion_class=aa.GridIterate,
            fractional_accuracy=0.5,
        )
        assert settings.grid_tag_no_inversion == "__grid_sub_1"
        assert settings.grid_tag_with_inversion == "__grid_sub_1_inv_facc_0.5"

        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.GridInterpolate,
            grid_inversion_class=aa.GridInterpolate,
            pixel_scales_interp=0.5,
        )
        assert settings.grid_tag_no_inversion == "__grid_interp_0.500"
        assert (
            settings.grid_tag_with_inversion == "__grid_interp_0.500_inv_interp_0.500"
        )

        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            grid_class=aa.GridIterate,
            fractional_accuracy=0.8,
            grid_inversion_class=aa.Grid,
            sub_size=2,
        )
        assert settings.grid_tag_no_inversion == "__grid_facc_0.8"
        assert settings.grid_tag_with_inversion == "__grid_facc_0.8_inv_sub_2"

    def test__signal_to_noise_limit_tag(self):

        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            signal_to_noise_limit=None
        )
        assert settings.signal_to_noise_limit_tag == ""
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            signal_to_noise_limit=1
        )
        assert settings.signal_to_noise_limit_tag == "__snr_1"
        settings = abstract_dataset.AbstractSettingsMaskedDataset(
            signal_to_noise_limit=2
        )
        assert settings.signal_to_noise_limit_tag == "__snr_2"


class TestAbstractMaskedData:
    def test__grid(
        self,
        imaging_7x7,
        sub_mask_7x7,
        grid_7x7,
        sub_grid_7x7,
        blurring_grid_7x7,
        grid_iterate_7x7,
    ):
        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7,
            mask=sub_mask_7x7,
            settings=abstract_dataset.AbstractSettingsMaskedDataset(grid_class=aa.Grid),
        )

        assert isinstance(masked_imaging_7x7.grid, aa.Grid)
        assert (masked_imaging_7x7.grid.in_1d_binned == grid_7x7).all()
        assert (masked_imaging_7x7.grid.in_1d == sub_grid_7x7).all()

        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7,
            mask=sub_mask_7x7,
            settings=abstract_dataset.AbstractSettingsMaskedDataset(
                grid_class=aa.GridIterate
            ),
        )

        assert isinstance(masked_imaging_7x7.grid, aa.GridIterate)
        assert (masked_imaging_7x7.grid.in_1d_binned == grid_iterate_7x7).all()

        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7,
            mask=sub_mask_7x7,
            settings=abstract_dataset.AbstractSettingsMaskedDataset(
                grid_class=aa.GridInterpolate, pixel_scales_interp=1.0
            ),
        )

        grid = aa.GridInterpolate.from_mask(mask=sub_mask_7x7, pixel_scales_interp=1.0)

        assert isinstance(masked_imaging_7x7.grid, aa.GridInterpolate)
        assert (masked_imaging_7x7.grid == grid).all()
        assert (masked_imaging_7x7.grid.grid_interp == grid.grid_interp).all()
        assert (masked_imaging_7x7.grid.vtx == grid.vtx).all()
        assert (masked_imaging_7x7.grid.wts == grid.wts).all()

    def test__grid_inversion(
        self, imaging_7x7, sub_mask_7x7, grid_7x7, sub_grid_7x7, blurring_grid_7x7
    ):
        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7,
            mask=sub_mask_7x7,
            settings=abstract_dataset.AbstractSettingsMaskedDataset(
                grid_inversion_class=aa.Grid
            ),
        )

        assert isinstance(masked_imaging_7x7.grid_inversion, aa.Grid)
        assert (masked_imaging_7x7.grid_inversion.in_1d_binned == grid_7x7).all()
        assert (masked_imaging_7x7.grid_inversion.in_1d == sub_grid_7x7).all()

        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7,
            mask=sub_mask_7x7,
            settings=abstract_dataset.AbstractSettingsMaskedDataset(
                grid_inversion_class=aa.GridInterpolate, pixel_scales_interp=1.0
            ),
        )

        grid = aa.GridInterpolate.from_mask(mask=sub_mask_7x7, pixel_scales_interp=1.0)

        assert isinstance(masked_imaging_7x7.grid_inversion, aa.GridInterpolate)
        assert (masked_imaging_7x7.grid_inversion == grid).all()
        assert (masked_imaging_7x7.grid_inversion.vtx == grid.vtx).all()
        assert (masked_imaging_7x7.grid_inversion.wts == grid.wts).all()

    def test__mask_changes_sub_size_using_settings(self, imaging_7x7):
        # If an input mask is supplied we use mask input.

        mask_input = aa.Mask.circular(
            shape_2d=imaging_7x7.shape_2d, pixel_scales=1, sub_size=1, radius=1.5
        )

        masked_dataset = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7,
            mask=mask_input,
            settings=abstract_dataset.AbstractSettingsMaskedDataset(sub_size=1),
        )

        assert (masked_dataset.mask == mask_input).all()
        assert masked_dataset.mask.sub_size == 1
        assert masked_dataset.mask.pixel_scales == mask_input.pixel_scales

        masked_dataset = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7,
            mask=mask_input,
            settings=abstract_dataset.AbstractSettingsMaskedDataset(sub_size=2),
        )

        assert (masked_dataset.mask == mask_input).all()
        assert masked_dataset.mask.sub_size == 2
        assert masked_dataset.mask.pixel_scales == mask_input.pixel_scales
