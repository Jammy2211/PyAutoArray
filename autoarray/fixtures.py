import numpy as np

import autoarray as aa


def make_mask_1d_7():
    mask = np.array([True, True, False, False, False, True, True])

    return aa.Mask1D(mask=mask, pixel_scales=(1.0,))


def make_mask_2d_7x7():
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

    return aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))


def make_mask_2d_7x7_1_pix():
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

    return aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))


def make_blurring_mask_2d_7x7():
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

    return aa.Mask2D(mask=blurring_mask, pixel_scales=(1.0, 1.0))


### arrays ###


def make_array_1d_7():
    return aa.Array1D.ones(shape_native=(7,), pixel_scales=(1.0,))


def make_array_2d_7x7():
    return aa.Array2D.ones(shape_native=(7, 7), pixel_scales=(1.0, 1.0))


def make_layout_2d_7x7():
    return aa.Layout2D(
        shape_2d=(7, 7),
        original_roe_corner=(1, 0),
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


# GRIDS #


def make_grid_1d_7():
    return aa.Grid1D.from_mask(mask=make_mask_1d_7())


def make_grid_2d_7x7():
    return aa.Grid2D.from_mask(mask=make_mask_2d_7x7())


def make_grid_2d_sub_1_7x7():
    return aa.Grid2D.from_mask(mask=make_mask_2d_7x7(), over_sampling_size=1)


def make_grid_2d_sub_2_7x7():
    return aa.Grid2D.from_mask(mask=make_mask_2d_7x7(), over_sampling_size=2)


def make_grid_2d_7x7_simple():
    grid_2d_7x7 = make_grid_2d_7x7()
    grid_2d_7x7[0] = np.array([1.0, 1.0])
    grid_2d_7x7[1] = np.array([1.0, 0.0])
    grid_2d_7x7[2] = np.array([1.0, 1.0])
    grid_2d_7x7[3] = np.array([1.0, 0.0])
    return grid_2d_7x7


def make_blurring_grid_2d_7x7():
    return aa.Grid2D.from_mask(mask=make_blurring_mask_2d_7x7())


# CONVOLVERS #


def make_convolver_7x7():
    return aa.Convolver(mask=make_mask_2d_7x7(), kernel=make_psf_3x3())


def make_image_7x7():
    return aa.Array2D.ones(shape_native=(7, 7), pixel_scales=(1.0, 1.0))


def make_psf_3x3():
    psf = np.array([[0.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 0.0]])

    return aa.Kernel2D.no_mask(values=psf, pixel_scales=(1.0, 1.0))


def make_psf_3x3_no_blur():
    return aa.Kernel2D.no_blur(pixel_scales=(1.0, 1.0))


def make_noise_map_7x7():
    return aa.Array2D.full(fill_value=2.0, shape_native=(7, 7), pixel_scales=(1.0, 1.0))


def make_noise_covariance_matrix_7x7():
    noise_covariance_matrix_7x7 = np.eye(N=49, M=49)

    noise_covariance_matrix_7x7[:, 24] = 1.0
    noise_covariance_matrix_7x7[24, :] = 1.0

    return noise_covariance_matrix_7x7


def make_grid_2d_irregular_7x7():
    return aa.Grid2DIrregular(values=[(0.1, 0.1), (0.2, 0.2)])


def make_grid_2d_irregular_7x7_list():
    return [
        aa.Grid2DIrregular(values=[(0.1, 0.1), (0.2, 0.2)]),
        aa.Grid2DIrregular(values=[(0.3, 0.3)]),
    ]


def make_imaging_7x7():
    return aa.Imaging(
        data=make_image_7x7(),
        psf=make_psf_3x3(),
        noise_map=make_noise_map_7x7(),
        over_sampling=aa.OverSamplingDataset(lp=1),
    )


def make_imaging_7x7_sub_2():
    return aa.Imaging(
        data=make_image_7x7(),
        psf=make_psf_3x3(),
        noise_map=make_noise_map_7x7(),
        over_sampling=aa.OverSamplingDataset(lp=2),
    )


def make_imaging_covariance_7x7():
    return aa.Imaging(
        data=make_image_7x7(),
        psf=make_psf_3x3(),
        noise_covariance_matrix=make_noise_covariance_matrix_7x7(),
        over_sampling=aa.OverSamplingDataset(lp=1),
    )


def make_imaging_7x7_no_blur():
    return aa.Imaging(
        data=make_image_7x7(),
        psf=make_psf_3x3_no_blur(),
        noise_map=make_noise_map_7x7(),
        over_sampling=aa.OverSamplingDataset(lp=1),
    )


def make_imaging_7x7_no_blur_sub_2():
    return aa.Imaging(
        data=make_image_7x7(),
        psf=make_psf_3x3_no_blur(),
        noise_map=make_noise_map_7x7(),
        over_sampling=aa.OverSamplingDataset(lp=2),
    )


def make_visibilities_7():
    return aa.Visibilities.full(shape_slim=(7,), fill_value=1.0)


def make_visibilities_noise_map_7():
    return aa.VisibilitiesNoiseMap.full(shape_slim=(7,), fill_value=2.0)


def make_uv_wavelengths_7x2():
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


def make_uv_wavelengths_7x2_no_fft():
    return np.ones(shape=(7, 2))


def make_interferometer_7():
    return aa.Interferometer(
        data=make_visibilities_7(),
        noise_map=make_visibilities_noise_map_7(),
        uv_wavelengths=make_uv_wavelengths_7x2(),
        real_space_mask=make_mask_2d_7x7(),
        transformer_class=aa.TransformerDFT,
    )


def make_interferometer_7_no_fft():
    return aa.Interferometer(
        data=make_visibilities_7(),
        noise_map=make_visibilities_noise_map_7(),
        uv_wavelengths=make_uv_wavelengths_7x2_no_fft(),
        real_space_mask=make_mask_2d_7x7(),
        transformer_class=aa.TransformerDFT,
    )


def make_interferometer_7_grid():
    return aa.Interferometer(
        data=make_visibilities_7(),
        noise_map=make_visibilities_noise_map_7(),
        uv_wavelengths=make_uv_wavelengths_7x2(),
        real_space_mask=make_mask_2d_7x7(),
        transformer_class=aa.TransformerDFT,
    )


def make_interferometer_7_lop():
    return aa.Interferometer(
        data=make_visibilities_7(),
        noise_map=make_visibilities_noise_map_7(),
        uv_wavelengths=make_uv_wavelengths_7x2(),
        real_space_mask=make_mask_2d_7x7(),
        transformer_class=aa.TransformerNUFFT,
    )


def make_transformer_7x7_7():
    return aa.TransformerDFT(
        uv_wavelengths=make_uv_wavelengths_7x2(), real_space_mask=make_mask_2d_7x7()
    )


### MASKED DATA ###


def make_masked_imaging_7x7():
    imaging_7x7 = make_imaging_7x7()

    return imaging_7x7.apply_mask(mask=make_mask_2d_7x7())


def make_masked_imaging_covariance_7x7():
    imaging_7x7 = make_imaging_covariance_7x7()

    return imaging_7x7.apply_mask(mask=make_mask_2d_7x7())


def make_masked_imaging_7x7_no_blur():
    imaging_7x7 = make_imaging_7x7_no_blur()

    return imaging_7x7.apply_mask(mask=make_mask_2d_7x7())


def make_masked_imaging_7x7_no_blur_sub_2():
    imaging_7x7 = make_imaging_7x7_no_blur_sub_2()

    return imaging_7x7.apply_mask(mask=make_mask_2d_7x7())


def make_model_image_7x7():
    imaging_7x7 = make_masked_imaging_7x7()

    return 5.0 * imaging_7x7.data


def make_imaging_fit_x1_plane_7x7():
    imaging_7x7 = make_masked_imaging_7x7()

    model_data = 5.0 * imaging_7x7.data

    return aa.m.MockFitImaging(
        dataset=imaging_7x7, use_mask_in_fit=False, model_data=model_data
    )


def make_fit_interferometer_7():
    interferometer_7 = make_interferometer_7()

    model_data = 5.0 * interferometer_7.data

    return aa.m.MockFitInterferometer(
        dataset=interferometer_7, use_mask_in_fit=False, model_data=model_data
    )


def make_regularization_constant():
    return aa.reg.Constant(coefficient=1.0)


def make_regularization_constant_split():
    return aa.reg.ConstantSplit(coefficient=1.0)


def make_regularization_adaptive_brightness():
    return aa.reg.AdaptiveBrightness(
        inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.5
    )


def make_regularization_adaptive_brightness_split():
    return aa.reg.AdaptiveBrightnessSplit(
        inner_coefficient=0.1, outer_coefficient=10.0, signal_scale=0.5
    )


def make_regularization_gaussian_kernel():
    return aa.reg.GaussianKernel(coefficient=1.0, scale=0.5)


def make_regularization_exponential_kernel():
    return aa.reg.ExponentialKernel(coefficient=1.0, scale=0.5)


def make_regularization_matern_kernel():
    return aa.reg.MaternKernel(coefficient=1.0, scale=0.5, nu=0.7)


def make_rectangular_mesh_grid_3x3():
    return aa.Mesh2DRectangular.overlay_grid(
        grid=make_grid_2d_sub_2_7x7().grid_over_sampled, shape_native=(3, 3)
    )


def make_delaunay_mesh_grid_9():
    grid_9 = aa.Grid2D.no_mask(
        values=[
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
        shape_native=(3, 3),
        pixel_scales=1.0,
    )

    return aa.Mesh2DDelaunay(values=grid_9)


def make_voronoi_mesh_grid_9():
    grid_9 = aa.Grid2D.no_mask(
        values=[
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
        shape_native=(3, 3),
        pixel_scales=1.0,
    )

    return aa.Mesh2DVoronoi(
        values=grid_9,
    )


def make_over_sampler_2d_7x7():
    return aa.OverSampler(mask=make_mask_2d_7x7(), sub_size=2)


def make_border_relocator_2d_7x7():
    return aa.BorderRelocator(
        mask=make_mask_2d_7x7(), sub_size=np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])
    )


def make_rectangular_mapper_7x7_3x3():
    mapper_grids = aa.MapperGrids(
        mask=make_mask_2d_7x7(),
        source_plane_data_grid=make_grid_2d_sub_2_7x7(),
        source_plane_mesh_grid=make_rectangular_mesh_grid_3x3(),
        image_plane_mesh_grid=None,
        adapt_data=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
    )

    return aa.MapperRectangular(
        mapper_grids=mapper_grids,
        border_relocator=make_border_relocator_2d_7x7(),
        regularization=make_regularization_constant(),
    )


def make_delaunay_mapper_9_3x3():
    mapper_grids = aa.MapperGrids(
        mask=make_mask_2d_7x7(),
        source_plane_data_grid=make_grid_2d_sub_2_7x7(),
        source_plane_mesh_grid=make_delaunay_mesh_grid_9(),
        image_plane_mesh_grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1),
        adapt_data=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
    )

    return aa.MapperDelaunay(
        mapper_grids=mapper_grids,
        border_relocator=make_border_relocator_2d_7x7(),
        regularization=make_regularization_constant(),
    )


def make_voronoi_mapper_9_3x3():
    mapper_grids = aa.MapperGrids(
        mask=make_mask_2d_7x7(),
        source_plane_data_grid=make_grid_2d_sub_2_7x7(),
        source_plane_mesh_grid=make_voronoi_mesh_grid_9(),
        image_plane_mesh_grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1),
        adapt_data=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
    )

    return aa.MapperVoronoi(
        mapper_grids=mapper_grids,
        border_relocator=make_border_relocator_2d_7x7(),
        regularization=make_regularization_constant(),
    )


def make_rectangular_inversion_7x7_3x3():
    return aa.Inversion(
        dataset=make_masked_imaging_7x7(),
        linear_obj_list=[make_rectangular_mapper_7x7_3x3()],
    )


def make_delaunay_inversion_9_3x3():
    return aa.Inversion(
        dataset=make_masked_imaging_7x7(),
        linear_obj_list=[make_delaunay_mapper_9_3x3()],
    )


def make_voronoi_inversion_9_3x3():
    return aa.Inversion(
        dataset=make_masked_imaging_7x7(), linear_obj_list=[make_voronoi_mapper_9_3x3()]
    )


### EUCLID DATA ####


def make_euclid_data():
    return np.zeros((2086, 2128))


### ACS DATA ####


def make_acs_ccd():
    return np.zeros((2068, 4144))


def make_acs_quadrant():
    return np.zeros((2068, 2072))
