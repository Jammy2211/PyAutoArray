from os import path

import numpy as np
import pytest

import autoarray as aa
from autoarray import conf
from autoarray.fit import fit
from test_autoarray.mock import mock_mask
from test_autoarray.mock import mock_data
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

    return mock_mask.MockMask(mask_2d=blurring_mask_2d)


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
    return aa.masked.grid.from_mask(mask=mask_7x7)


@pytest.fixture(name="sub_grid_7x7")
def make_sub_grid_7x7(sub_mask_7x7):
    return aa.masked.grid.from_mask(mask=sub_mask_7x7)


@pytest.fixture(name="sub_grid_7x7_simple")
def make_sub_grid_7x7_simple(mask_7x7, sub_grid_7x7):
    sub_grid_7x7[0] = np.array([1.0, 1.0])
    sub_grid_7x7[1] = np.array([1.0, 0.0])
    sub_grid_7x7[2] = np.array([1.0, 1.0])
    sub_grid_7x7[3] = np.array([1.0, 0.0])
    return sub_grid_7x7


@pytest.fixture(name="blurring_grid_7x7")
def make_blurring_grid_7x7(blurring_mask_7x7):
    return aa.masked.grid.from_mask(mask=blurring_mask_7x7)


# CONVOLVERS #


@pytest.fixture(name="convolver_7x7")
def make_convolver_7x7(mask_7x7, blurring_mask_7x7, psf_3x3):
    return mock_convolution.MockConvolver(mask=mask_7x7, kernel=psf_3x3)


@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return mock_data.MockImage(shape_2d=(7, 7), value=1.0)


@pytest.fixture(name="psf_3x3")
def make_psf_3x3():
    return mock_data.MockPSF(shape_2d=(3, 3), value=1.0)


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return mock_data.MockNoiseMap(shape_2d=(7, 7), value=2.0)


@pytest.fixture(name="background_noise_map_7x7")
def make_background_noise_map_7x7():
    return mock_data.MockBackgroundNoiseMap(shape_2d=(7, 7), value=3.0)


@pytest.fixture(name="poisson_noise_map_7x7")
def make_poisson_noise_map_7x7():
    return mock_data.MockPoissonNoiseMap(shape_2d=(7, 7), value=4.0)


@pytest.fixture(name="exposure_time_map_7x7")
def make_exposure_time_map_7x7():
    return mock_data.MockExposureTimeMap(shape_2d=(7, 7), value=5.0)


@pytest.fixture(name="background_sky_map_7x7")
def make_background_sky_map_7x7():
    return mock_data.MockBackgrondSkyMap(shape_2d=(7, 7), value=6.0)


@pytest.fixture(name="positions_7x7")
def make_positions_7x7():
    positions = [[[0.1, 0.1], [0.2, 0.2]], [[0.3, 0.3]]]
    return list(map(lambda position_set: np.asarray(position_set), positions))


@pytest.fixture(name="imaging_7x7")
def make_imaging_7x7(
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
        name="mock_imaging_7x7",
    )


@pytest.fixture(name="imaging_6x6")
def make_imaging_6x6():
    image = mock_data.MockImage(shape_2d=(6, 6), value=1.0)
    psf = mock_data.MockPSF(shape_2d=(3, 3), value=1.0)
    noise_map = mock_data.MockNoiseMap(shape_2d=(6, 6), value=2.0)
    background_noise_map = mock_data.MockBackgroundNoiseMap(shape_2d=(6, 6), value=3.0)
    poisson_noise_map = mock_data.MockPoissonNoiseMap(shape_2d=(6, 6), value=4.0)
    exposure_time_map = mock_data.MockExposureTimeMap(shape_2d=(6, 6), value=5.0)
    background_sky_map = mock_data.MockBackgrondSkyMap(shape_2d=(6, 6), value=6.0)

    return mock_data.MockImaging(
        image=image,
        pixel_scales=1.0,
        psf=psf,
        noise_map=noise_map,
        background_noise_map=background_noise_map,
        poisson_noise_map=poisson_noise_map,
        exposure_time_map=exposure_time_map,
        background_sky_map=background_sky_map,
        name="mock_imaging_6x6",
    )


@pytest.fixture(name="visibilities_7x2")
def make_visibilities_7():
    return mock_data.MockVisibilities(shape_1d=7, value=1.0)


@pytest.fixture(name="noise_map_7x2")
def make_noise_map_7():
    return mock_data.MockVisibilitiesNoiseMap(shape_1d=7, value=2.0)


@pytest.fixture(name="primary_beam_3x3")
def make_primary_beam_3x3():
    return mock_data.MockPrimaryBeam(shape_2d=(3, 3), value=1.0)


@pytest.fixture(name="uv_wavelengths_7x2")
def make_uv_wavelengths_7():
    return mock_data.MockUVWavelengths(shape=(7, 2), value=3.0)


@pytest.fixture(name="interferometer_7")
def make_interferometer_7(
    visibilities_7x2, noise_map_7x2, primary_beam_3x3, uv_wavelengths_7x2
):
    return mock_data.MockInterferometer(
        visibilities=visibilities_7x2,
        noise_map=noise_map_7x2,
        uv_wavelengths=uv_wavelengths_7x2,
        primary_beam=primary_beam_3x3,
    )


@pytest.fixture(name="transformer_7x7_7")
def make_transformer_7x7_7(uv_wavelengths_7x2, grid_7x7):
    return mock_data.MockTransformer(
        uv_wavelengths=uv_wavelengths_7x2,
        grid_radians=grid_7x7.mask.geometry.masked_grid.in_radians,
    )


### MASKED DATA ###


@pytest.fixture(name="masked_imaging_7x7")
def make_masked_imaging_7x7(imaging_7x7, sub_mask_7x7):
    return aa.masked.imaging.manual(imaging=imaging_7x7, mask=sub_mask_7x7)


@pytest.fixture(name="masked_interferometer_7")
def make_masked_interferometer_7(
    interferometer_7, mask_7x7, sub_grid_7x7, transformer_7x7_7
):
    return aa.masked.interferometer.manual(
        interferometer=interferometer_7, real_space_mask=mask_7x7
    )


@pytest.fixture(name="fit_imaging_7x7")
def make_masked_imaging_fit_x1_plane_7x7(masked_imaging_7x7):
    return fit.ImagingFit(
        mask=masked_imaging_7x7.mask,
        image=masked_imaging_7x7.image,
        noise_map=masked_imaging_7x7.noise_map,
        model_image=5.0 * masked_imaging_7x7.image,
    )

@pytest.fixture(name="fit_interferometer_7")
def make_masked_interferometer_fit_x1_plane_7(masked_interferometer_7):
    return fit.InterferometerFit(
        visibilities_mask=masked_interferometer_7.visibilities_mask,
        visibilities=masked_interferometer_7.visibilities,
        noise_map=masked_interferometer_7.noise_map,
        model_visibilities=5.0 * masked_interferometer_7.visibilities,
    )
