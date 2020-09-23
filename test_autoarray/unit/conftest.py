from os import path

import numpy as np
import pytest

import autoarray as aa
from autoconf import conf
from autoarray.fit import fit

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        config_path=path.join(directory, "config"),
        output_path=path.join(directory, "output"),
    )


# MASK #


@pytest.fixture(name="mask_7x7")
def make_mask_7x7():
    mask = np.array(
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

    return aa.Mask.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=1)


@pytest.fixture(name="sub_mask_7x7")
def make_sub_mask_7x7():
    mask = np.array(
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

    return aa.Mask.manual(mask=mask, sub_size=2, pixel_scales=(1.0, 1.0))


@pytest.fixture(name="mask_7x7_1_pix")
def make_mask_7x7_1_pix():
    mask = np.array(
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

    return aa.Mask.manual(mask=mask, pixel_scales=(1.0, 1.0))


@pytest.fixture(name="blurring_mask_7x7")
def make_blurring_mask_7x7():
    blurring_mask = np.array(
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

    return aa.Mask.manual(mask=blurring_mask, pixel_scales=(1.0, 1.0))


@pytest.fixture(name="mask_6x6")
def make_mask_6x6():
    mask = np.array(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
            [True, True, False, False, True, True],
            [True, True, False, False, True, True],
            [True, True, True, True, True, True],
            [True, True, True, True, True, True],
        ]
    )

    return aa.Mask.manual(mask=mask, pixel_scales=(1.0, 1.0))


# GRIDS #


@pytest.fixture(name="grid_7x7")
def make_grid_7x7(mask_7x7):
    return aa.Grid.from_mask(mask=mask_7x7)


@pytest.fixture(name="sub_grid_7x7")
def make_sub_grid_7x7(sub_mask_7x7):
    return aa.Grid.from_mask(mask=sub_mask_7x7)


@pytest.fixture(name="grid_iterate_7x7")
def make_grid_iterate_7x7(mask_7x7):
    return aa.GridIterate.from_mask(
        mask=mask_7x7, fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
    )


@pytest.fixture(name="sub_grid_7x7_simple")
def make_sub_grid_7x7_simple(mask_7x7, sub_grid_7x7):
    sub_grid_7x7[0] = np.array([1.0, 1.0])
    sub_grid_7x7[1] = np.array([1.0, 0.0])
    sub_grid_7x7[2] = np.array([1.0, 1.0])
    sub_grid_7x7[3] = np.array([1.0, 0.0])
    return sub_grid_7x7


@pytest.fixture(name="blurring_grid_7x7")
def make_blurring_grid_7x7(blurring_mask_7x7):
    return aa.Grid.from_mask(mask=blurring_mask_7x7)


# CONVOLVERS #


@pytest.fixture(name="convolver_7x7")
def make_convolver_7x7(mask_7x7, blurring_mask_7x7, psf_3x3):
    return aa.Convolver(mask=mask_7x7, kernel=psf_3x3)


@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return aa.Array.ones(shape_2d=(7, 7), pixel_scales=(1.0, 1.0))


@pytest.fixture(name="psf_3x3")
def make_psf_3x3():
    return aa.Kernel.ones(shape_2d=(3, 3), pixel_scales=(1.0, 1.0))


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return aa.Array.full(fill_value=2.0, shape_2d=(7, 7), pixel_scales=(1.0, 1.0))


@pytest.fixture(name="positions_7x7")
def make_positions_7x7():
    return aa.GridCoordinates(coordinates=[[(0.1, 0.1), (0.2, 0.2)], [(0.3, 0.3)]])


@pytest.fixture(name="imaging_7x7")
def make_imaging_7x7(image_7x7, psf_3x3, noise_map_7x7):
    return aa.Imaging(
        image=image_7x7, psf=psf_3x3, noise_map=noise_map_7x7, name="mock_imaging_7x7"
    )


@pytest.fixture(name="imaging_6x6")
def make_imaging_6x6():

    image = aa.Array.full(shape_2d=(6, 6), fill_value=1.0)
    psf = aa.Kernel.full(shape_2d=(3, 3), fill_value=1.0)
    noise_map = aa.Array.full(shape_2d=(6, 6), fill_value=2.0)

    return aa.Imaging(
        image=image, psf=psf, noise_map=noise_map, name="mock_imaging_6x6"
    )


@pytest.fixture(name="visibilities_mask_7x2")
def make_visibilities_mask_7x2():
    return np.full(fill_value=False, shape=(7, 2))


@pytest.fixture(name="visibilities_7x2")
def make_visibilities_7():
    visibilities = aa.Visibilities.full(shape_1d=(7,), fill_value=1.0)
    visibilities[6, 0] = -1.0
    visibilities[6, 1] = -1.0
    return visibilities


@pytest.fixture(name="noise_map_7x2")
def make_noise_map_7():
    return aa.VisibilitiesNoiseMap.full(shape_1d=(7,), fill_value=2.0)


@pytest.fixture(name="uv_wavelengths_7x2")
def make_uv_wavelengths_7():
    return np.array(
        [
            [-55636.4609375, 171376.90625],
            [-6903.21923828, 51155.578125],
            [-63488.4140625, 4141.28369141],
            [55502.828125, 47016.7265625],
            [54160.75390625, -99354.1796875],
            [-9327.66308594, -95212.90625],
            [0.0, 0.0],
        ]
    )


@pytest.fixture(name="interferometer_7")
def make_interferometer_7(visibilities_7x2, noise_map_7x2, uv_wavelengths_7x2):
    return aa.Interferometer(
        visibilities=visibilities_7x2,
        noise_map=noise_map_7x2,
        uv_wavelengths=uv_wavelengths_7x2,
    )


@pytest.fixture(name="transformer_7x7_7")
def make_transformer_7x7_7(uv_wavelengths_7x2, mask_7x7):
    return aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2, real_space_mask=mask_7x7
    )


### MASKED DATA ###


@pytest.fixture(name="masked_imaging_7x7")
def make_masked_imaging_7x7(imaging_7x7, sub_mask_7x7):
    return aa.MaskedImaging(
        imaging=imaging_7x7,
        mask=sub_mask_7x7,
        settings=aa.SettingsMaskedImaging(sub_size=1),
    )


@pytest.fixture(name="masked_interferometer_7")
def make_masked_interferometer_7(
    interferometer_7, visibilities_mask_7x2, mask_7x7, sub_grid_7x7, transformer_7x7_7
):
    return aa.MaskedInterferometer(
        interferometer=interferometer_7,
        visibilities_mask=visibilities_mask_7x2,
        real_space_mask=mask_7x7,
        settings=aa.SettingsMaskedInterferometer(
            sub_size=1, transformer_class=aa.TransformerDFT
        ),
    )


@pytest.fixture(name="fit_imaging_7x7")
def make_masked_imaging_fit_x1_plane_7x7(masked_imaging_7x7):
    return fit.FitImaging(
        masked_imaging=masked_imaging_7x7, model_image=5.0 * masked_imaging_7x7.image
    )


@pytest.fixture(name="fit_interferometer_7")
def make_masked_interferometer_fit_x1_plane_7(masked_interferometer_7):
    fit_interferometer = fit.FitInterferometer(
        masked_interferometer=masked_interferometer_7,
        model_visibilities=5.0 * masked_interferometer_7.visibilities,
    )
    fit_interferometer.masked_dataset = masked_interferometer_7
    return fit_interferometer


@pytest.fixture(name="rectangular_pixelization_grid_3x3")
def make_rectangular_pixelization_grid_3x3(grid_7x7):
    return aa.GridRectangular.overlay_grid(grid=grid_7x7, shape_2d=(3, 3))


@pytest.fixture(name="rectangular_mapper_7x7_3x3")
def make_rectangular_mapper_7x7_3x3(grid_7x7, rectangular_pixelization_grid_3x3):
    return aa.Mapper(grid=grid_7x7, pixelization_grid=rectangular_pixelization_grid_3x3)


@pytest.fixture(name="voronoi_pixelization_grid_9")
def make_voronoi_pixelization_grid_9(grid_7x7):
    grid_9 = aa.Grid.manual_1d(
        grid=[
            [0.6, -0.3],
            [0.5, -0.8],
            [0.2, 0.1],
            [0.0, 0.5],
            [-0.3, -0.8],
            [-0.6, -0.5],
            [-0.4, -1.1],
            [-1.2, 0.8],
            [-1.5, 0.9],
        ],
        shape_2d=(3, 3),
        pixel_scales=1.0,
    )
    return aa.GridVoronoi(
        grid=grid_9,
        nearest_pixelization_1d_index_for_mask_1d_index=np.zeros(
            shape=grid_7x7.shape_1d, dtype="int"
        ),
    )


@pytest.fixture(name="voronoi_mapper_9_3x3")
def make_voronoi_mapper_9_3x3(grid_7x7, voronoi_pixelization_grid_9):
    return aa.Mapper(grid=grid_7x7, pixelization_grid=voronoi_pixelization_grid_9)


@pytest.fixture(name="rectangular_inversion_7x7_3x3")
def make_rectangular_inversion_7x7_3x3(masked_imaging_7x7, rectangular_mapper_7x7_3x3):
    regularization = aa.reg.Constant(coefficient=1.0)

    return aa.Inversion(
        masked_dataset=masked_imaging_7x7,
        mapper=rectangular_mapper_7x7_3x3,
        regularization=regularization,
    )


@pytest.fixture(name="voronoi_inversion_9_3x3")
def make_voronoi_inversion_9_3x3(masked_imaging_7x7, voronoi_mapper_9_3x3):
    regularization = aa.reg.Constant(coefficient=1.0)
    return aa.Inversion(
        masked_dataset=masked_imaging_7x7,
        mapper=voronoi_mapper_9_3x3,
        regularization=regularization,
    )


### EUCLID DATA ####


@pytest.fixture(name="euclid_data")
def make_euclid_data():
    return np.zeros((2086, 2119))


### ACS DATA ####


@pytest.fixture(name="acs_ccd")
def make_acs_ccd():
    return np.zeros((2068, 4144))


@pytest.fixture(name="acs_quadrant")
def make_acs_quadrant():
    return np.zeros((2068, 2072))
