import logging
import numpy as np

import autoarray as aa

from autoarray.dataset.abstract import dataset as ds

logger = logging.getLogger(__name__)


def test__dataset_takes_structures_of_different_formats():
    array = aa.Array1D.no_mask([1.0, 2.0], pixel_scales=1.0)
    noise_map = aa.Array1D.no_mask([1.0, 3.0], pixel_scales=1.0)

    dataset = ds.AbstractDataset(data=array, noise_map=noise_map)

    assert (dataset.data.native == np.array([1.0, 2.0])).all()
    assert (dataset.noise_map.native == np.array([1.0, 3.0])).all()

    array = aa.Array2D.no_mask([[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
    noise_map = aa.Array2D.no_mask([[1.0, 2.0], [3.0, 5.0]], pixel_scales=1.0)

    dataset = ds.AbstractDataset(data=array, noise_map=noise_map)

    assert (dataset.data.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
    assert (dataset.noise_map.native == np.array([[1.0, 2.0], [3.0, 5.0]])).all()


def test__signal_to_noise_map():
    array = aa.Array2D.no_mask([[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
    noise_map = aa.Array2D.no_mask([[10.0, 10.0], [30.0, 4.0]], pixel_scales=1.0)

    dataset = ds.AbstractDataset(data=array, noise_map=noise_map)

    assert (
        dataset.signal_to_noise_map.native == np.array([[0.1, 0.2], [0.1, 1.0]])
    ).all()
    assert dataset.signal_to_noise_max == 1.0

    array = aa.Array2D.no_mask([[-1.0, 2.0], [3.0, -4.0]], pixel_scales=1.0)

    noise_map = aa.Array2D.no_mask([[10.0, 10.0], [30.0, 4.0]], pixel_scales=1.0)

    dataset = ds.AbstractDataset(data=array, noise_map=noise_map)

    assert (
        dataset.signal_to_noise_map.native == np.array([[0.0, 0.2], [0.1, 0.0]])
    ).all()
    assert dataset.signal_to_noise_max == 0.2


def test__grid__uses_mask_and_settings(
    image_7x7,
    noise_map_7x7,
    sub_mask_2d_7x7,
    grid_2d_7x7,
    sub_grid_2d_7x7,
    grid_2d_iterate_7x7,
):
    dataset_1d = ds.AbstractDataset(
        data=aa.Array1D.no_mask(values=[1.0], pixel_scales=1.0),
        noise_map=aa.Array1D.no_mask(values=[1.0], pixel_scales=1.0),
        settings=ds.AbstractSettingsDataset(),
    )

    assert isinstance(dataset_1d.grid, aa.Grid1D)

    masked_image_7x7 = aa.Array2D(
        values=image_7x7.native, mask=sub_mask_2d_7x7.derive_mask.sub_1
    )

    masked_noise_map_7x7 = aa.Array2D(
        values=noise_map_7x7.native, mask=sub_mask_2d_7x7.derive_mask.sub_1
    )

    masked_imaging_7x7 = ds.AbstractDataset(
        data=masked_image_7x7,
        noise_map=masked_noise_map_7x7,
        settings=ds.AbstractSettingsDataset(sub_size=2),
    )

    assert isinstance(masked_imaging_7x7.grid, aa.Grid2D)
    assert (masked_imaging_7x7.grid.binned == grid_2d_7x7).all()
    assert (masked_imaging_7x7.grid.slim == sub_grid_2d_7x7).all()

    masked_imaging_7x7 = ds.AbstractDataset(
        data=masked_image_7x7,
        noise_map=masked_noise_map_7x7,
        settings=ds.AbstractSettingsDataset(grid_class=aa.Grid2DIterate),
    )

    assert isinstance(masked_imaging_7x7.grid, aa.Grid2DIterate)
    assert (masked_imaging_7x7.grid.binned == grid_2d_iterate_7x7).all()


def test__grid_pixelization__uses_mask_and_settings(
    image_7x7, noise_map_7x7, sub_mask_2d_7x7, grid_2d_7x7, sub_grid_2d_7x7
):
    masked_dataset = ds.AbstractDataset(
        data=aa.Array1D.no_mask(values=[1.0], pixel_scales=1.0),
        noise_map=aa.Array1D.no_mask(values=[1.0], pixel_scales=1.0),
        settings=ds.AbstractSettingsDataset(),
    )

    assert isinstance(masked_dataset.grid, aa.Grid1D)

    masked_image_7x7 = aa.Array2D(
        values=image_7x7.native, mask=sub_mask_2d_7x7.derive_mask.sub_1
    )

    masked_noise_map_7x7 = aa.Array2D(
        values=noise_map_7x7.native, mask=sub_mask_2d_7x7.derive_mask.sub_1
    )

    masked_imaging_7x7 = ds.AbstractDataset(
        data=masked_image_7x7,
        noise_map=masked_noise_map_7x7,
        settings=ds.AbstractSettingsDataset(
            grid_pixelization_class=aa.Grid2D, sub_size_pixelization=2
        ),
    )

    assert masked_imaging_7x7.grid_pixelization.sub_size == 2
    assert (masked_imaging_7x7.grid_pixelization.binned == grid_2d_7x7).all()
    assert (masked_imaging_7x7.grid_pixelization.slim == sub_grid_2d_7x7).all()

    masked_imaging_7x7 = ds.AbstractDataset(
        data=masked_image_7x7,
        noise_map=masked_noise_map_7x7,
        settings=ds.AbstractSettingsDataset(
            grid_pixelization_class=aa.Grid2D, sub_size=2, sub_size_pixelization=4
        ),
    )

    assert isinstance(masked_imaging_7x7.grid_pixelization, aa.Grid2D)
    assert masked_imaging_7x7.grid_pixelization.sub_size == 4


def test__grid_settings__sub_size(image_7x7, noise_map_7x7):
    dataset_7x7 = ds.AbstractDataset(
        data=image_7x7,
        noise_map=noise_map_7x7,
        settings=ds.AbstractSettingsDataset(sub_size=1, sub_size_pixelization=1),
    )

    assert dataset_7x7.grid.mask.sub_size == 1
    assert dataset_7x7.grid_pixelization.mask.sub_size == 1

    dataset_7x7 = ds.AbstractDataset(
        data=image_7x7,
        noise_map=noise_map_7x7,
        settings=ds.AbstractSettingsDataset(sub_size=2, sub_size_pixelization=2),
    )

    assert dataset_7x7.grid.mask.sub_size == 2
    assert dataset_7x7.grid_pixelization.mask.sub_size == 2


def test__new_imaging_with_arrays_trimmed_via_kernel_shape():
    data = aa.Array2D.full(fill_value=20.0, shape_native=(3, 3), pixel_scales=1.0)
    data[4] = 5.0

    noise_map_array = aa.Array2D.full(
        fill_value=5.0, shape_native=(3, 3), pixel_scales=1.0
    )
    noise_map_array[4] = 2.0

    dataset = ds.AbstractDataset(data=data, noise_map=noise_map_array)

    dataset_trimmed = dataset.trimmed_after_convolution_from(kernel_shape=(3, 3))

    assert (dataset_trimmed.data.native == np.array([[5.0]])).all()

    assert (dataset_trimmed.noise_map.native == np.array([[2.0]])).all()
