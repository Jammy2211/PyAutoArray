import logging
import numpy as np
import pytest

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

    assert dataset.signal_to_noise_map.native == pytest.approx(
        np.array([[0.1, 0.2], [0.1, 1.0]]), 1.0e-4
    )
    assert dataset.signal_to_noise_max == 1.0

    array = aa.Array2D.no_mask([[-1.0, 2.0], [3.0, -4.0]], pixel_scales=1.0)

    noise_map = aa.Array2D.no_mask([[10.0, 10.0], [30.0, 4.0]], pixel_scales=1.0)

    dataset = ds.AbstractDataset(data=array, noise_map=noise_map)

    assert dataset.signal_to_noise_map.native == pytest.approx(
        np.array([[0.0, 0.2], [0.1, 0.0]]), 1.0e-4
    )
    assert dataset.signal_to_noise_max == 0.2


def test__grid__uses_mask_and_settings(
    image_7x7,
    noise_map_7x7,
    mask_2d_7x7,
    grid_2d_7x7,
):
    masked_image_7x7 = aa.Array2D(
        values=image_7x7.native,
        mask=mask_2d_7x7,
    )

    masked_noise_map_7x7 = aa.Array2D(values=noise_map_7x7.native, mask=mask_2d_7x7)

    masked_imaging_7x7 = ds.AbstractDataset(
        data=masked_image_7x7,
        noise_map=masked_noise_map_7x7,
        over_sample_size_lp=2,
    )

    assert isinstance(masked_imaging_7x7.grids.lp, aa.Grid2D)
    assert (masked_imaging_7x7.grids.lp == grid_2d_7x7).all()
    assert (masked_imaging_7x7.grids.lp.slim == grid_2d_7x7).all()


def test__grids_pixelization__uses_mask_and_settings(
    image_7x7,
    noise_map_7x7,
    mask_2d_7x7,
    grid_2d_7x7,
):
    masked_image_7x7 = aa.Array2D(values=image_7x7.native, mask=mask_2d_7x7)

    masked_noise_map_7x7 = aa.Array2D(values=noise_map_7x7.native, mask=mask_2d_7x7)

    masked_imaging_7x7 = ds.AbstractDataset(
        data=masked_image_7x7,
        noise_map=masked_noise_map_7x7,
    )

    assert (masked_imaging_7x7.grids.pixelization == grid_2d_7x7).all()
    assert (masked_imaging_7x7.grids.pixelization.slim == grid_2d_7x7).all()

    masked_imaging_7x7 = ds.AbstractDataset(
        data=masked_image_7x7,
        noise_map=masked_noise_map_7x7,
        over_sample_size_lp=2,
        over_sample_size_pixelization=4,
    )

    assert isinstance(masked_imaging_7x7.grids.pixelization, aa.Grid2D)
    assert masked_imaging_7x7.grids.over_sample_size_pixelization[0] == 4


def test__grid_settings__sub_size(image_7x7, noise_map_7x7):
    dataset_7x7 = ds.AbstractDataset(
        data=image_7x7,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
        over_sample_size_pixelization=4,
    )

    assert dataset_7x7.grids.over_sample_size_lp[0] == 2
    assert dataset_7x7.grids.over_sample_size_pixelization[0] == 4


def test__new_imaging_with_arrays_trimmed_via_kernel_shape():

    data = aa.Array2D.no_mask(
        values=[[20.0, 20.0, 20.0], [20.0, 5.0, 20.0], [20.0, 20.0, 20.0]],
        pixel_scales=1.0,
    )

    noise_map = aa.Array2D.no_mask(
        values=[[20.0, 20.0, 20.0], [20.0, 2.0, 20.0], [20.0, 20.0, 20.0]],
        pixel_scales=1.0,
    )

    dataset = ds.AbstractDataset(data=data, noise_map=noise_map)

    dataset_trimmed = dataset.trimmed_after_convolution_from(kernel_shape=(3, 3))

    assert (dataset_trimmed.data.native == np.array([[5.0]])).all()

    assert (dataset_trimmed.noise_map.native == np.array([[2.0]])).all()


def test__apply_over_sampling(image_7x7, noise_map_7x7):
    dataset_7x7 = aa.Imaging(
        data=image_7x7,
        noise_map=noise_map_7x7,
        over_sample_size_lp=2,
        over_sample_size_pixelization=2,
    )

    # The grid and grid_pixelizaiton are a cached_property which needs to be reset,
    # Which the code below tests.

    grid_sub_2 = dataset_7x7.grids.lp
    grids_pixelization_sub_2 = dataset_7x7.grids.pixelization

    dataset_7x7.grids.__dict__["lp"][0][0] = 100.0
    dataset_7x7.grids.__dict__["pixelization"][0][0] = 100.0

    assert dataset_7x7.grids.lp[0][0] == pytest.approx(100.0, 1.0e-4)
    assert dataset_7x7.grids.pixelization[0][0] == pytest.approx(100.0, 1.0e-4)

    dataset_7x7 = dataset_7x7.apply_over_sampling(
        over_sample_size_lp=2,
        over_sample_size_pixelization=4,
    )

    assert dataset_7x7.over_sample_size_lp.slim[0] == 2
    assert dataset_7x7.over_sample_size_pixelization.slim[0] == 4

    assert dataset_7x7.grids.lp[0][0] == pytest.approx(3.0, 1.0e-4)
    assert dataset_7x7.grids.pixelization[0][0] == pytest.approx(3.0, 1.0e-4)
