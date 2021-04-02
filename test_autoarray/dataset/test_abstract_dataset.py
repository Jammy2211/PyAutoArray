import logging

import autoarray as aa
from autoarray.dataset import abstract_dataset
import numpy as np

logger = logging.getLogger(__name__)


class TestAbstractDataset:
    def test__dataset_takes_structures_of_different_formats(self):

        array = aa.Array1D.manual_native([1.0, 2.0], pixel_scales=1.0)
        noise_map = aa.Array1D.manual_native([1.0, 3.0], pixel_scales=1.0)

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (dataset.data.native == np.array([1.0, 2.0])).all()
        assert (dataset.noise_map.native == np.array([1.0, 3.0])).all()

        array = aa.Array2D.manual_native([[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
        noise_map = aa.Array2D.manual_native([[1.0, 2.0], [3.0, 5.0]], pixel_scales=1.0)

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (dataset.data.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (dataset.noise_map.native == np.array([[1.0, 2.0], [3.0, 5.0]])).all()

    def test__inverse_noise_is_one_over_noise(self):

        array = aa.Array2D.manual_native([[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
        noise_map = aa.Array2D.manual_native([[1.0, 2.0], [4.0, 8.0]], pixel_scales=1.0)

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.inverse_noise_map.native == np.array([[1.0, 0.5], [0.25, 0.125]])
        ).all()

    def test__signal_to_noise_map__image_and_noise_are_values__signal_to_noise_is_ratio_of_each(
        self,
    ):
        array = aa.Array2D.manual_native([[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
        noise_map = aa.Array2D.manual_native(
            [[10.0, 10.0], [30.0, 4.0]], pixel_scales=1.0
        )

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.signal_to_noise_map.native == np.array([[0.1, 0.2], [0.1, 1.0]])
        ).all()
        assert dataset.signal_to_noise_max == 1.0

    def test__signal_to_noise_map__same_as_above__but_image_has_negative_values__replaced_with_zeros(
        self,
    ):
        array = aa.Array2D.manual_native([[-1.0, 2.0], [3.0, -4.0]], pixel_scales=1.0)

        noise_map = aa.Array2D.manual_native(
            [[10.0, 10.0], [30.0, 4.0]], pixel_scales=1.0
        )

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.signal_to_noise_map.native == np.array([[0.0, 0.2], [0.1, 0.0]])
        ).all()
        assert dataset.signal_to_noise_max == 0.2

    def test__absolute_signal_to_noise_map__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(
        self,
    ):
        array = aa.Array2D.manual_native([[-1.0, 2.0], [3.0, -4.0]], pixel_scales=1.0)

        noise_map = aa.Array2D.manual_native(
            [[10.0, 10.0], [30.0, 4.0]], pixel_scales=1.0
        )

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.absolute_signal_to_noise_map.native
            == np.array([[0.1, 0.2], [0.1, 1.0]])
        ).all()
        assert dataset.absolute_signal_to_noise_max == 1.0

    def test__potential_chi_squared_map__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(
        self,
    ):
        array = aa.Array2D.manual_native([[-1.0, 2.0], [3.0, -4.0]], pixel_scales=1.0)
        noise_map = aa.Array2D.manual_native(
            [[10.0, 10.0], [30.0, 4.0]], pixel_scales=1.0
        )

        dataset = abstract_dataset.AbstractDataset(data=array, noise_map=noise_map)

        assert (
            dataset.potential_chi_squared_map.native
            == np.array([[0.1 ** 2.0, 0.2 ** 2.0], [0.1 ** 2.0, 1.0 ** 2.0]])
        ).all()
        assert dataset.potential_chi_squared_max == 1.0


class TestAbstractDatasetApplyMask:
    def test__grid(
        self,
        image_7x7,
        noise_map_7x7,
        sub_mask_7x7,
        grid_7x7,
        sub_grid_7x7,
        grid_iterate_7x7,
    ):

        masked_imaging_7x7 = abstract_dataset.AbstractDataset(
            data=aa.Array1D.manual_native(array=[1.0], pixel_scales=1.0),
            noise_map=aa.Array1D.manual_native(array=[1.0], pixel_scales=1.0),
            settings=abstract_dataset.AbstractSettingsDataset(),
        )

        assert isinstance(masked_imaging_7x7.grid, aa.Grid1D)

        masked_image_7x7 = aa.Array2D.manual_mask(
            array=image_7x7.native, mask=sub_mask_7x7.mask_sub_1
        )

        masked_noise_map_7x7 = aa.Array2D.manual_mask(
            array=noise_map_7x7.native, mask=sub_mask_7x7.mask_sub_1
        )

        masked_imaging_7x7 = abstract_dataset.AbstractDataset(
            data=masked_image_7x7,
            noise_map=masked_noise_map_7x7,
            settings=abstract_dataset.AbstractSettingsDataset(),
        )

        assert isinstance(masked_imaging_7x7.grid, aa.Grid2D)
        assert (masked_imaging_7x7.grid.binned == grid_7x7).all()
        assert (masked_imaging_7x7.grid.slim == sub_grid_7x7).all()

        masked_imaging_7x7 = abstract_dataset.AbstractDataset(
            data=masked_image_7x7,
            noise_map=masked_noise_map_7x7,
            settings=abstract_dataset.AbstractSettingsDataset(
                grid_class=aa.Grid2DIterate
            ),
        )

        assert isinstance(masked_imaging_7x7.grid, aa.Grid2DIterate)
        assert (masked_imaging_7x7.grid.binned == grid_iterate_7x7).all()

        masked_imaging_7x7 = abstract_dataset.AbstractDataset(
            data=masked_image_7x7,
            noise_map=masked_noise_map_7x7,
            settings=abstract_dataset.AbstractSettingsDataset(
                grid_class=aa.Grid2DInterpolate, pixel_scales_interp=1.0
            ),
        )

        grid = aa.Grid2DInterpolate.from_mask(
            mask=sub_mask_7x7, pixel_scales_interp=1.0
        )

        assert isinstance(masked_imaging_7x7.grid, aa.Grid2DInterpolate)
        assert (masked_imaging_7x7.grid == grid).all()
        assert (masked_imaging_7x7.grid.grid_interp == grid.grid_interp).all()
        assert (masked_imaging_7x7.grid.vtx == grid.vtx).all()
        assert (masked_imaging_7x7.grid.wts == grid.wts).all()

    def test__grid_inversion(
        self, image_7x7, noise_map_7x7, sub_mask_7x7, grid_7x7, sub_grid_7x7
    ):

        masked_imaging_7x7 = abstract_dataset.AbstractDataset(
            data=aa.Array1D.manual_native(array=[1.0], pixel_scales=1.0),
            noise_map=aa.Array1D.manual_native(array=[1.0], pixel_scales=1.0),
            settings=abstract_dataset.AbstractSettingsDataset(),
        )

        assert isinstance(masked_imaging_7x7.grid, aa.Grid1D)

        masked_image_7x7 = aa.Array2D.manual_mask(
            array=image_7x7.native, mask=sub_mask_7x7.mask_sub_1
        )

        masked_noise_map_7x7 = aa.Array2D.manual_mask(
            array=noise_map_7x7.native, mask=sub_mask_7x7.mask_sub_1
        )

        masked_imaging_7x7 = abstract_dataset.AbstractDataset(
            data=masked_image_7x7,
            noise_map=masked_noise_map_7x7,
            settings=abstract_dataset.AbstractSettingsDataset(
                grid_inversion_class=aa.Grid2D, sub_size_inversion=2
            ),
        )

        assert masked_imaging_7x7.grid_inversion.sub_size == 2
        assert (masked_imaging_7x7.grid_inversion.binned == grid_7x7).all()
        assert (masked_imaging_7x7.grid_inversion.slim == sub_grid_7x7).all()

        masked_imaging_7x7 = abstract_dataset.AbstractDataset(
            data=masked_image_7x7,
            noise_map=masked_noise_map_7x7,
            settings=abstract_dataset.AbstractSettingsDataset(
                grid_inversion_class=aa.Grid2D, sub_size=2, sub_size_inversion=4
            ),
        )

        assert isinstance(masked_imaging_7x7.grid_inversion, aa.Grid2D)
        assert masked_imaging_7x7.grid_inversion.sub_size == 4

        masked_imaging_7x7 = abstract_dataset.AbstractDataset(
            data=masked_image_7x7,
            noise_map=masked_noise_map_7x7,
            settings=abstract_dataset.AbstractSettingsDataset(
                grid_inversion_class=aa.Grid2DInterpolate, pixel_scales_interp=1.0
            ),
        )

        grid = aa.Grid2DInterpolate.from_mask(
            mask=sub_mask_7x7, pixel_scales_interp=1.0
        )

        assert isinstance(masked_imaging_7x7.grid_inversion, aa.Grid2DInterpolate)
        assert (masked_imaging_7x7.grid_inversion == grid).all()
        assert (masked_imaging_7x7.grid_inversion.vtx == grid.vtx).all()
        assert (masked_imaging_7x7.grid_inversion.wts == grid.wts).all()

    def test__grids_change_sub_size_using_settings(self, image_7x7, noise_map_7x7):

        masked_dataset = abstract_dataset.AbstractDataset(
            data=image_7x7,
            noise_map=noise_map_7x7,
            settings=abstract_dataset.AbstractSettingsDataset(
                sub_size=1, sub_size_inversion=1
            ),
        )

        assert masked_dataset.grid.mask.sub_size == 1
        assert masked_dataset.grid_inversion.mask.sub_size == 1

        masked_dataset = abstract_dataset.AbstractDataset(
            data=image_7x7,
            noise_map=noise_map_7x7,
            settings=abstract_dataset.AbstractSettingsDataset(
                sub_size=2, sub_size_inversion=2
            ),
        )

        assert masked_dataset.grid.mask.sub_size == 2
        assert masked_dataset.grid_inversion.mask.sub_size == 2


class TestMethods:
    def test__new_imaging_with_arrays_trimmed_via_kernel_shape(self):
        data = aa.Array2D.full(fill_value=20.0, shape_native=(3, 3), pixel_scales=1.0)
        data[4] = 5.0

        noise_map_array = aa.Array2D.full(
            fill_value=5.0, shape_native=(3, 3), pixel_scales=1.0
        )
        noise_map_array[4] = 2.0

        dataset = abstract_dataset.AbstractDataset(data=data, noise_map=noise_map_array)

        dataset_trimmed = dataset.trimmed_after_convolution_from(kernel_shape=(3, 3))

        assert (dataset_trimmed.data.native == np.array([[5.0]])).all()

        assert (dataset_trimmed.noise_map.native == np.array([[2.0]])).all()
