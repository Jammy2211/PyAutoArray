import numpy as np

import autoarray as aa


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

    return aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=1)


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

    return aa.Mask2D.manual(mask=mask, sub_size=2, pixel_scales=(1.0, 1.0))


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

    return aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))


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

    return aa.Mask2D.manual(mask=blurring_mask, pixel_scales=(1.0, 1.0))


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

    return aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))


# GRIDS #


def make_grid_7x7():
    return aa.Grid.from_mask(mask=make_mask_7x7())


def make_sub_grid_7x7():
    return aa.Grid.from_mask(mask=make_sub_mask_7x7())


def make_grid_iterate_7x7():
    return aa.GridIterate.from_mask(
        mask=make_mask_7x7(), fractional_accuracy=0.9999, sub_steps=[2, 4, 8, 16]
    )


def make_sub_grid_7x7_simple():
    sub_grid_7x7 = make_sub_grid_7x7()
    sub_grid_7x7[0] = np.array([1.0, 1.0])
    sub_grid_7x7[1] = np.array([1.0, 0.0])
    sub_grid_7x7[2] = np.array([1.0, 1.0])
    sub_grid_7x7[3] = np.array([1.0, 0.0])
    return sub_grid_7x7


def make_blurring_grid_7x7():
    return aa.Grid.from_mask(mask=make_blurring_mask_7x7())


# CONVOLVERS #


def make_convolver_7x7():
    return aa.Convolver(mask=make_mask_7x7(), kernel=make_psf_3x3())


def make_image_7x7():
    return aa.Array.ones(shape_2d=(7, 7), pixel_scales=(1.0, 1.0))


def make_psf_3x3():
    return aa.Kernel.ones(shape_2d=(3, 3), pixel_scales=(1.0, 1.0))


def make_noise_map_7x7():
    return aa.Array.full(fill_value=2.0, shape_2d=(7, 7), pixel_scales=(1.0, 1.0))


def make_positions_7x7():
    return aa.GridIrregularGrouped(grid=[[(0.1, 0.1), (0.2, 0.2)], [(0.3, 0.3)]])


def make_imaging_7x7():
    return aa.Imaging(
        image=make_image_7x7(),
        psf=make_psf_3x3(),
        noise_map=make_noise_map_7x7(),
        name="mock_imaging_7x7",
    )


def make_imaging_6x6():
    image = aa.Array.full(shape_2d=(6, 6), fill_value=1.0)
    psf = aa.Kernel.full(shape_2d=(3, 3), fill_value=1.0)
    noise_map = aa.Array.full(shape_2d=(6, 6), fill_value=2.0)

    return aa.Imaging(
        image=image, psf=psf, noise_map=noise_map, name="mock_imaging_6x6"
    )


def make_visibilities_mask_7():
    return np.full(fill_value=False, shape=(7,))


def make_visibilities_7():
    visibilities = aa.Visibilities.full(shape_1d=(7,), fill_value=1.0)
    visibilities[6] = -1.0 - 1.0j
    return visibilities


def make_visibilities_noise_map_7():
    return aa.VisibilitiesNoiseMap.full(shape_1d=(7,), fill_value=2.0)


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


def make_interferometer_7():
    return aa.Interferometer(
        visibilities=make_visibilities_7(),
        noise_map=make_visibilities_noise_map_7(),
        uv_wavelengths=make_uv_wavelengths_7(),
    )


def make_transformer_7x7_7():
    return aa.TransformerDFT(
        uv_wavelengths=make_uv_wavelengths_7(), real_space_mask=make_mask_7x7()
    )


### MASKED DATA ###


def make_masked_imaging_7x7():
    return aa.MaskedImaging(
        imaging=make_imaging_7x7(),
        mask=make_sub_mask_7x7(),
        settings=aa.SettingsMaskedImaging(sub_size=1),
    )


def make_masked_interferometer_7():
    return aa.MaskedInterferometer(
        interferometer=make_interferometer_7(),
        visibilities_mask=make_visibilities_mask_7(),
        real_space_mask=make_mask_7x7(),
        settings=aa.SettingsMaskedInterferometer(
            sub_size=1, transformer_class=aa.TransformerDFT
        ),
    )


def make_masked_imaging_fit_x1_plane_7x7():
    return aa.FitImaging(
        masked_imaging=make_masked_imaging_7x7(),
        model_image=5.0 * make_masked_imaging_7x7().image,
        use_mask_in_fit=False,
    )


def make_masked_interferometer_fit_x1_plane_7():
    masked_interferometer_7 = make_masked_interferometer_7()
    fit_interferometer = aa.FitInterferometer(
        masked_interferometer=masked_interferometer_7,
        model_visibilities=5.0 * masked_interferometer_7.visibilities,
        use_mask_in_fit=False,
    )
    fit_interferometer.masked_dataset = masked_interferometer_7
    return fit_interferometer


def make_rectangular_pixelization_grid_3x3():
    return aa.GridRectangular.overlay_grid(grid=make_grid_7x7(), shape_2d=(3, 3))


def make_rectangular_mapper_7x7_3x3():
    return aa.Mapper(
        grid=make_grid_7x7(), pixelization_grid=make_rectangular_pixelization_grid_3x3()
    )


def make_voronoi_pixelization_grid_9():
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
            shape=make_grid_7x7().shape_1d, dtype="int"
        ),
    )


def make_voronoi_mapper_9_3x3():
    return aa.Mapper(
        grid=make_grid_7x7(), pixelization_grid=make_voronoi_pixelization_grid_9()
    )


def make_rectangular_inversion_7x7_3x3():
    regularization = aa.reg.Constant(coefficient=1.0)

    return aa.Inversion(
        masked_dataset=make_masked_imaging_7x7(),
        mapper=make_rectangular_mapper_7x7_3x3(),
        regularization=regularization,
    )


def make_voronoi_inversion_9_3x3():
    regularization = aa.reg.Constant(coefficient=1.0)
    return aa.Inversion(
        masked_dataset=make_masked_imaging_7x7(),
        mapper=make_voronoi_mapper_9_3x3(),
        regularization=regularization,
    )


### EUCLID DATA ####


def make_euclid_data():
    return np.zeros((2086, 2128))


### ACS DATA ####


def make_acs_ccd():
    return np.zeros((2068, 4144))


def make_acs_quadrant():
    return np.zeros((2068, 2072))
