import logging

import autoarray as aa
from autoarray.dataset import abstract_dataset
import numpy as np

logger = logging.getLogger(__name__)


class TestInversionNoiseMap:

    def test__inverse_noise_is_one_over_noise(self):
        array = aa.Array.manual_2d([[1.0, 2.0], [3.0, 4.0]])
        noise_map = aa.Array.manual_2d([[1.0, 2.0], [4.0, 8.0]])

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (dataset.inverse_noise_map.in_2d == np.array([[1.0, 0.5], [0.25, 0.125]])).all()


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
            dataset=imaging_7x7, mask=sub_mask_7x7, grid_class=aa.Grid
        )

        assert isinstance(masked_imaging_7x7.grid, aa.Grid)
        assert (masked_imaging_7x7.grid.in_1d_binned == grid_7x7).all()
        assert (masked_imaging_7x7.grid.in_1d == sub_grid_7x7).all()

        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7, mask=sub_mask_7x7, grid_class=aa.GridIterate
        )

        assert isinstance(masked_imaging_7x7.grid, aa.GridIterate)
        assert (masked_imaging_7x7.grid.in_1d_binned == grid_iterate_7x7).all()

        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7,
            mask=sub_mask_7x7,
            grid_class=aa.GridInterpolate,
            pixel_scales_interp=1.0,
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
            dataset=imaging_7x7, mask=sub_mask_7x7, grid_inversion_class=aa.Grid
        )

        assert isinstance(masked_imaging_7x7.grid_inversion, aa.Grid)
        assert (masked_imaging_7x7.grid_inversion.in_1d_binned == grid_7x7).all()
        assert (masked_imaging_7x7.grid_inversion.in_1d == sub_grid_7x7).all()

        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7,
            mask=sub_mask_7x7,
            grid_inversion_class=aa.GridInterpolate,
            pixel_scales_interp=1.0,
        )

        grid = aa.GridInterpolate.from_mask(mask=sub_mask_7x7, pixel_scales_interp=1.0)

        assert isinstance(masked_imaging_7x7.grid_inversion, aa.Grid)
        assert (masked_imaging_7x7.grid_inversion == grid).all()
        assert (masked_imaging_7x7.grid_inversion.vtx == grid.vtx).all()
        assert (masked_imaging_7x7.grid_inversion.wts == grid.wts).all()

    def test__inversion_pixel_limit(self, imaging_7x7, sub_mask_7x7):
        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7, mask=sub_mask_7x7, inversion_pixel_limit=2
        )

        assert masked_imaging_7x7.inversion_pixel_limit == 2

        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7, mask=sub_mask_7x7, inversion_pixel_limit=5
        )

        assert masked_imaging_7x7.inversion_pixel_limit == 5

    def test__inversion_uses_border(self, imaging_7x7, sub_mask_7x7):
        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7, mask=sub_mask_7x7, inversion_uses_border=True
        )

        assert masked_imaging_7x7.inversion_uses_border == True

        masked_imaging_7x7 = abstract_dataset.AbstractMaskedDataset(
            dataset=imaging_7x7, mask=sub_mask_7x7, inversion_uses_border=False
        )

        assert masked_imaging_7x7.inversion_uses_border == False
