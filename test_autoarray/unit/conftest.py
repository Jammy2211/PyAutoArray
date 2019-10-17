from os import path

import numpy as np
import pytest

import autoarray as aa
from autoarray import conf

from test_autoarray.mock import mock_mask
from test_autoarray.mock import mock_data
from test_autoarray.mock import mock_grids
from test_autoarray.mock import mock_convolution

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "test_files/config"), path.join(directory, "output")
    )


# MASK #


@pytest.fixture(name="mask_7x7")
def make_mask_7x7():
    mask_2d = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return mock_mask.MockMask(mask_2d=mask_2d, pixel_scales=(1.0, 1.0), sub_size=1)


@pytest.fixture(name="sub_mask_7x7")
def make_sub_mask_7x7():
    mask_2d = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return mock_mask.MockMask(mask_2d=mask_2d, sub_size=2)


@pytest.fixture(name="mask_7x7_1_pix")
def make_mask_7x7_1_pix():
    mask_2d = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return mock_mask.MockMask(mask_2d=mask_2d)


@pytest.fixture(name="blurring_mask_7x7")
def make_blurring_mask_7x7():
    blurring_mask_2d = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, False, False, False, False, False, True],
            [True, False, True, True, True, False, True],
            [True, False, True, True, True, False, True],
            [True, False, True, True, True, False, True],
            [True, False, False, False, False, False, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return mock_mask.MockMask(blurring_mask_2d=blurring_mask_2d)


@pytest.fixture(name="mask_6x6")
def make_mask_6x6():
    mask_2d = np.array(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
            [True, True, False, False, True, True],
            [True, True, False, False, True, True],
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
        ]
    )

    return mock_mask.MockMask(mask_2d=mask_2d)


# GRIDS #


@pytest.fixture(name="grid_7x7")
def make_grid_7x7(mask_7x7):
    return aa.grid_masked.from_mask(mask=mask_7x7)


@pytest.fixture(name="sub_grid_7x7")
def make_sub_grid_7x7(sub_mask_7x7):
    return aa.grid_masked.from_mask(mask=sub_mask_7x7)


@pytest.fixture(name="blurring_grid_7x7")
def make_blurring_grid_7x7(blurring_mask_7x7):
    return aa.grid_masked.from_mask(mask=blurring_mask_7x7)


# CONVOLVERS #


@pytest.fixture(name="convolver_7x7")
def make_convolver_7x7(mask_7x7, blurring_mask_7x7, psf_3x3):
    return mock_convolution.MockConvolver(
        mask=mask_7x7, blurring_mask=blurring_mask_7x7, kernel=psf_3x3
    )

@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return mock_data.MockImage(shape=(7, 7), value=1.0)


@pytest.fixture(name="psf_3x3")
def make_psf_3x3():
    return mock_data.MockPSF(shape=(3, 3), value=1.0)


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return mock_data.MockNoiseMap(shape=(7, 7), value=2.0)


@pytest.fixture(name="background_noise_map_7x7")
def make_background_noise_map_7x7():
    return mock_data.MockBackgroundNoiseMap(shape=(7, 7), value=3.0)


@pytest.fixture(name="poisson_noise_map_7x7")
def make_poisson_noise_map_7x7():
    return mock_data.MockPoissonNoiseMap(shape=(7, 7), value=4.0)


@pytest.fixture(name="exposure_time_map_7x7")
def make_exposure_time_map_7x7():
    return mock_data.MockExposureTimeMap(shape=(7, 7), value=5.0)


@pytest.fixture(name="background_sky_map_7x7")
def make_background_sky_map_7x7():
    return mock_data.MockBackgrondSkyMap(shape=(7, 7), value=6.0)


@pytest.fixture(name="positions_7x7")
def make_positions_7x7():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))


@pytest.fixture(name="imaging_data_7x7")
def make_imaging_data_7x7(
    image_7x7,
    psf_3x3,
    noise_map_7x7,
    background_noise_map_7x7,
    poisson_noise_map_7x7,
    exposure_time_map_7x7,
    background_sky_map_7x7,
):
    return mock_data.MockImaging(
        image=image_7x7,
        pixel_scales=image_7x7.pixel_scales,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        background_noise_map=background_noise_map_7x7,
        poisson_noise_map=poisson_noise_map_7x7,
        exposure_time_map=exposure_time_map_7x7,
        background_sky_map=background_sky_map_7x7,
        name="mock_imaging_data_7x7",
    )


@pytest.fixture(name="imaging_data_6x6")
def make_imaging_data_6x6():
    image = mock_data.MockImage(shape=(6, 6), value=1.0)
    psf = mock_data.MockPSF(shape=(3, 3), value=1.0)
    noise_map = mock_data.MockNoiseMap(shape=(6, 6), value=2.0)
    background_noise_map = mock_data.MockBackgroundNoiseMap(shape=(6, 6), value=3.0)
    poisson_noise_map = mock_data.MockPoissonNoiseMap(shape=(6, 6), value=4.0)
    exposure_time_map = mock_data.MockExposureTimeMap(shape=(6, 6), value=5.0)
    background_sky_map = mock_data.MockBackgrondSkyMap(shape=(6, 6), value=6.0)

    return mock_data.MockImaging(
        image=image,
        pixel_scales=1.0,
        psf=psf,
        noise_map=noise_map,
        background_noise_map=background_noise_map,
        poisson_noise_map=poisson_noise_map,
        exposure_time_map=exposure_time_map,
        background_sky_map=background_sky_map,
        name="mock_imaging_data_6x6",
    )


@pytest.fixture(name="visibilities_7")
def make_visibilities_7():
    return mock_data.MockVisibilities(shape=7, value=1.0)


@pytest.fixture(name="visibilities_noise_map_7")
def make_visibilities_noise_map_7():
    return mock_data.MockVisibilitiesNoiseMap(shape=7, value=2.0)


@pytest.fixture(name="primary_beam_3x3")
def make_primary_beam_3x3():
    return mock_data.MockPrimaryBeam(shape=(3, 3), value=1.0)


@pytest.fixture(name="uv_wavelengths_7")
def make_uv_wavelengths_7():
    return mock_data.MockUVWavelengths(shape=7, value=3.0)


@pytest.fixture(name="uv_plane_data_7")
def make_uv_plane_data_7(
    visibilities_7, visibilities_noise_map_7, primary_beam_3x3, uv_wavelengths_7
):
    return mock_data.MockInterferometer(
        shape_2d=(7, 7),
        visibilities=visibilities_7,
        pixel_scales=1.0,
        noise_map=visibilities_noise_map_7,
        primary_beam=primary_beam_3x3,
        uv_wavelengths=uv_wavelengths_7,
    )


@pytest.fixture(name="transformer_7x7_7")
def make_transformer_7x7_7(uv_wavelengths_7, grid_7x7):
    return mock_data.MockTransformer(
        uv_wavelengths=uv_wavelengths_7,
        grid_radians=grid_7x7.mask.geometry.masked_grid.in_radians,
    )