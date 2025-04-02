import copy
import os
from os import path

import numpy as np
import pytest
import shutil

import autoarray as aa

from autoarray import exc

test_data_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))),
    "files",
)


@pytest.fixture(name="test_data_path")
def make_test_data_path():
    test_data_path = path.join(
        "{}".format(os.path.dirname(os.path.realpath(__file__))),
        "files",
        "array",
        "output_test",
    )

    if os.path.exists(test_data_path):
        shutil.rmtree(test_data_path)

    os.makedirs(test_data_path)

    return test_data_path


def test__psf_and_mask_hit_edge__automatically_pads_image_and_noise_map():
    image = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    noise_map = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    psf = aa.Kernel2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    dataset = aa.Imaging(
        data=image, noise_map=noise_map, psf=psf, pad_for_psf=False
    )

    assert dataset.data.shape_native == (3, 3)
    assert dataset.noise_map.shape_native == (3, 3)

    dataset = aa.Imaging(
        data=image, noise_map=noise_map, psf=psf, pad_for_psf=True
    )

    assert dataset.data.shape_native == (5, 5)
    assert dataset.noise_map.shape_native == (5, 5)
    assert dataset.data.mask[0, 0] == True
    assert dataset.data.mask[1, 1] == False


def test__noise_covariance_input__noise_map_uses_diag():
    image = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    noise_covariance_matrix = np.ones(shape=(9, 9))

    dataset = aa.Imaging(data=image, noise_covariance_matrix=noise_covariance_matrix)

    noise_map = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    assert (dataset.noise_map.native == noise_map.native).all()


def test__no_noise_map__raises_exception():
    image = aa.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    with pytest.raises(aa.exc.DatasetException):
        aa.Imaging(data=image)


def test__from_fits():
    dataset = aa.Imaging.from_fits(
        pixel_scales=0.1,
        data_path=path.join(test_data_path, "3x3_ones.fits"),
        psf_path=path.join(test_data_path, "3x3_twos.fits"),
        noise_map_path=path.join(test_data_path, "3x3_threes.fits"),
    )

    assert (dataset.data.native == np.ones((3, 3))).all()
    assert (dataset.psf.native == (1.0 / 9.0) * np.ones((3, 3))).all()
    assert (dataset.noise_map.native == 3.0 * np.ones((3, 3))).all()

    assert dataset.pixel_scales == (0.1, 0.1)
    assert dataset.psf.mask.pixel_scales == (0.1, 0.1)
    assert dataset.noise_map.mask.pixel_scales == (0.1, 0.1)

    dataset = aa.Imaging.from_fits(
        pixel_scales=0.1,
        data_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
        data_hdu=0,
        psf_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
        psf_hdu=1,
        noise_map_path=path.join(test_data_path, "3x3_multiple_hdu.fits"),
        noise_map_hdu=2,
    )

    assert (dataset.data.native == np.ones((3, 3))).all()
    assert (dataset.psf.native == (1.0 / 9.0) * np.ones((3, 3))).all()
    assert (dataset.noise_map.native == 3.0 * np.ones((3, 3))).all()

    assert dataset.pixel_scales == (0.1, 0.1)
    assert dataset.psf.mask.pixel_scales == (0.1, 0.1)
    assert dataset.noise_map.mask.pixel_scales == (0.1, 0.1)


def test__output_to_fits(imaging_7x7, test_data_path):
    imaging_7x7.output_to_fits(
        data_path=path.join(test_data_path, "data.fits"),
        psf_path=path.join(test_data_path, "psf.fits"),
        noise_map_path=path.join(test_data_path, "noise_map.fits"),
    )

    dataset = aa.Imaging.from_fits(
        pixel_scales=0.1,
        data_path=path.join(test_data_path, "data.fits"),
        psf_path=path.join(test_data_path, "psf.fits"),
        noise_map_path=path.join(test_data_path, "noise_map.fits"),
    )

    assert (dataset.data.native == np.ones((7, 7))).all()
    assert dataset.psf.native[1, 1] == pytest.approx(0.33333, 1.0e-4)
    assert (dataset.noise_map.native == 2.0 * np.ones((7, 7))).all()
    assert dataset.pixel_scales == (0.1, 0.1)


def test__apply_mask(imaging_7x7, mask_2d_7x7, psf_3x3):
    masked_imaging_7x7 = imaging_7x7.apply_mask(mask=mask_2d_7x7)

    assert (masked_imaging_7x7.data.slim == np.ones(9)).all()

    assert (
        masked_imaging_7x7.data.native == np.ones((7, 7)) * np.invert(mask_2d_7x7)
    ).all()

    assert (masked_imaging_7x7.noise_map.slim == 2.0 * np.ones(9)).all()
    assert (
        masked_imaging_7x7.noise_map.native
        == 2.0 * np.ones((7, 7)) * np.invert(mask_2d_7x7)
    ).all()

    assert (masked_imaging_7x7.psf.slim == (1.0 / 3.0) * psf_3x3.slim).all()

    assert type(masked_imaging_7x7.psf) == aa.Kernel2D
    assert masked_imaging_7x7.w_tilde.curvature_preload.shape == (35,)
    assert masked_imaging_7x7.w_tilde.indexes.shape == (35,)
    assert masked_imaging_7x7.w_tilde.lengths.shape == (9,)


def test__apply_noise_scaling(imaging_7x7, mask_2d_7x7):
    masked_imaging_7x7 = imaging_7x7.apply_noise_scaling(
        mask=mask_2d_7x7, noise_value=1e5
    )

    assert masked_imaging_7x7.data.native[4, 4] == 0.0
    assert masked_imaging_7x7.noise_map.native[4, 4] == 1e5


def test__apply_noise_scaling__use_signal_to_noise_value(imaging_7x7, mask_2d_7x7):
    imaging_7x7 = copy.copy(imaging_7x7)

    imaging_7x7.data[24] = 2.0

    masked_imaging_7x7 = imaging_7x7.apply_noise_scaling(
        mask=mask_2d_7x7, signal_to_noise_value=0.1, should_zero_data=False
    )

    assert masked_imaging_7x7.data.native[3, 4] == 1.0
    assert masked_imaging_7x7.noise_map.native[3, 4] == 10.0
    assert masked_imaging_7x7.data.native[3, 3] == 2.0
    assert masked_imaging_7x7.noise_map.native[3, 3] == 10.0


def test__apply_mask__noise_covariance_matrix():
    image = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=(1.0, 1.0))

    noise_covariance_matrix = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 4.0],
        ]
    )

    mask = np.array(
        [
            [False, True],
            [True, False],
        ]
    )

    mask_2d = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    dataset = aa.Imaging(data=image, noise_covariance_matrix=noise_covariance_matrix)

    masked_dataset = dataset.apply_mask(mask=mask_2d)

    assert masked_dataset.noise_covariance_matrix == pytest.approx(
        np.array([[1.0, 1.0], [4.0, 4.0]]), 1.0e-4
    )


def test__different_imaging_without_mock_objects__customize_constructor_inputs():
    psf = aa.Kernel2D.ones(shape_native=(7, 7), pixel_scales=3.0)

    dataset = aa.Imaging(
        data=aa.Array2D.ones(shape_native=(19, 19), pixel_scales=3.0),
        psf=psf,
        noise_map=aa.Array2D.full(
            fill_value=2.0, shape_native=(19, 19), pixel_scales=3.0
        ),
    )
    mask = aa.Mask2D.all_false(
        shape_native=(19, 19),
        pixel_scales=1.0,
        invert=True,
    )
    mask[9, 9] = False

    masked_dataset = dataset.apply_mask(mask=mask)

    assert masked_dataset.psf.native == pytest.approx(
        (1.0 / 49.0) * np.ones((7, 7)), 1.0e-4
    )
    assert (masked_dataset.data == np.array([1.0])).all()
    assert (masked_dataset.noise_map == np.array([2.0])).all()


def test__noise_map_unmasked_has_zeros_or_negative__raises_exception():
    array = aa.Array2D.no_mask([[1.0, 2.0]], pixel_scales=1.0)

    noise_map = aa.Array2D.no_mask([[0.0, 3.0]], pixel_scales=1.0)

    with pytest.raises(aa.exc.DatasetException):
        aa.Imaging(data=array, noise_map=noise_map)

    noise_map = aa.Array2D.no_mask([[-1.0, 3.0]], pixel_scales=1.0)

    with pytest.raises(aa.exc.DatasetException):
        aa.Imaging(data=array, noise_map=noise_map)

def test__psf_not_odd_x_odd_kernel__raises_error():

    with pytest.raises(exc.KernelException):
        aa.Kernel2D.no_mask(values=[[0.0, 1.0], [1.0, 2.0]], pixel_scales=1.0)