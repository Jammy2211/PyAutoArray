import numpy as np

from autoarray.mask import mask_2d
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures.grids.two_d import grid_2d_iterate
from autoarray.structures.grids.two_d import grid_2d_irregular
from autoarray.structures.grids.two_d import grid_2d_pixelization
from autoarray.structures.frames import abstract_frame
from autoarray.structures.frames import frames
from autoarray.structures import kernel_2d
from autoarray.structures import visibilities as vis
from autoarray.dataset import imaging
from autoarray.dataset import interferometer
from autoarray.operators import convolver
from autoarray.operators import transformer
from autoarray.fit import fit
from autoarray.inversion import regularization as reg
from autoarray.inversion import mappers
from autoarray.inversion import inversions


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

    return mask_2d.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=1)


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

    return mask_2d.Mask2D.manual(mask=mask, sub_size=2, pixel_scales=(1.0, 1.0))


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

    return mask_2d.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))


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

    return mask_2d.Mask2D.manual(mask=blurring_mask, pixel_scales=(1.0, 1.0))


### arrays ###


def make_array_7x7():
    return array_2d.Array2D.ones(shape_native=(7, 7), pixel_scales=(1.0, 1.0))


def make_scans_7x7():
    return abstract_frame.Scans(
        serial_overscan=(0, 6, 6, 7),
        serial_prescan=(0, 7, 0, 1),
        parallel_overscan=(6, 7, 1, 6),
    )


def make_frame_7x7():
    return frames.Frame2D.ones(
        shape_native=(7, 7), pixel_scales=(1.0, 1.0), scans=make_scans_7x7()
    )


# GRIDS #


def make_grid_7x7():
    return grid_2d.Grid2D.from_mask(mask=make_mask_7x7())


def make_sub_grid_7x7():
    return grid_2d.Grid2D.from_mask(mask=make_sub_mask_7x7())


def make_grid_iterate_7x7():
    return grid_2d_iterate.Grid2DIterate.from_mask(
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
    return grid_2d.Grid2D.from_mask(mask=make_blurring_mask_7x7())


# CONVOLVERS #


def make_convolver_7x7():
    return convolver.Convolver(mask=make_mask_7x7(), kernel=make_psf_3x3())


def make_image_7x7():
    return array_2d.Array2D.ones(shape_native=(7, 7), pixel_scales=(1.0, 1.0))


def make_psf_3x3():
    return kernel_2d.Kernel2D.ones(shape_native=(3, 3), pixel_scales=(1.0, 1.0))


def make_psf_no_blur_3x3():
    return kernel_2d.Kernel2D.no_blur(pixel_scales=(1.0, 1.0))


def make_noise_map_7x7():
    return array_2d.Array2D.full(
        fill_value=2.0, shape_native=(7, 7), pixel_scales=(1.0, 1.0)
    )


def make_grid_irregular_7x7():
    return grid_2d_irregular.Grid2DIrregular(grid=[(0.1, 0.1), (0.2, 0.2)])


def make_grid_irregular_7x7_list():
    return [
        grid_2d_irregular.Grid2DIrregular(grid=[(0.1, 0.1), (0.2, 0.2)]),
        grid_2d_irregular.Grid2DIrregular(grid=[(0.3, 0.3)]),
    ]


def make_imaging_7x7():
    return imaging.Imaging(
        image=make_image_7x7(),
        psf=make_psf_3x3(),
        noise_map=make_noise_map_7x7(),
        name="mock_imaging_7x7",
    )


def make_imaging_no_blur_7x7():
    return imaging.Imaging(
        image=make_image_7x7(),
        psf=make_psf_no_blur_3x3(),
        noise_map=make_noise_map_7x7(),
        name="mock_imaging_7x7",
    )


def make_visibilities_mask_7():
    return np.full(fill_value=False, shape=(7,))


def make_visibilities_7():
    visibilities = vis.Visibilities.full(shape_slim=(7,), fill_value=1.0)
    visibilities[6] = -1.0 - 1.0j
    return visibilities


def make_visibilities_noise_map_7():
    return vis.VisibilitiesNoiseMap.full(shape_slim=(7,), fill_value=2.0)


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
    return interferometer.Interferometer(
        visibilities=make_visibilities_7(),
        noise_map=make_visibilities_noise_map_7(),
        uv_wavelengths=make_uv_wavelengths_7(),
    )


def make_transformer_7x7_7():
    return transformer.TransformerDFT(
        uv_wavelengths=make_uv_wavelengths_7(), real_space_mask=make_mask_7x7()
    )


### MASKED DATA ###


def make_masked_imaging_7x7():
    return imaging.MaskedImaging(
        imaging=make_imaging_7x7(),
        mask=make_sub_mask_7x7(),
        settings=imaging.SettingsMaskedImaging(sub_size=1),
    )


def make_masked_imaging_no_blur_7x7():
    return imaging.MaskedImaging(
        imaging=make_imaging_no_blur_7x7(),
        mask=make_sub_mask_7x7(),
        settings=imaging.SettingsMaskedImaging(sub_size=1),
    )


def make_masked_interferometer_7():
    return interferometer.MaskedInterferometer(
        interferometer=make_interferometer_7(),
        visibilities_mask=make_visibilities_mask_7(),
        real_space_mask=make_mask_7x7(),
        settings=interferometer.SettingsMaskedInterferometer(
            sub_size=1, transformer_class=transformer.TransformerDFT
        ),
    )


def make_masked_imaging_fit_x1_plane_7x7():
    return fit.FitImaging(
        masked_imaging=make_masked_imaging_7x7(),
        model_image=5.0 * make_masked_imaging_7x7().image,
        use_mask_in_fit=False,
    )


def make_masked_interferometer_fit_x1_plane_7():
    masked_interferometer_7 = make_masked_interferometer_7()
    fit_interferometer = fit.FitInterferometer(
        masked_interferometer=masked_interferometer_7,
        model_visibilities=5.0 * masked_interferometer_7.visibilities,
        use_mask_in_fit=False,
    )
    fit_interferometer.masked_dataset = masked_interferometer_7
    return fit_interferometer


def make_rectangular_pixelization_grid_3x3():
    return grid_2d_pixelization.Grid2DRectangular.overlay_grid(
        grid=make_grid_7x7(), shape_native=(3, 3)
    )


def make_rectangular_mapper_7x7_3x3():
    return mappers.mapper(
        source_grid_slim=make_grid_7x7(),
        source_pixelization_grid=make_rectangular_pixelization_grid_3x3(),
    )


def make_voronoi_pixelization_grid_9():

    grid_9 = grid_2d.Grid2D.manual_slim(
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
        shape_native=(3, 3),
        pixel_scales=1.0,
    )

    return grid_2d_pixelization.Grid2DVoronoi(
        grid=grid_9,
        nearest_pixelization_index_for_slim_index=np.zeros(
            shape=make_grid_7x7().shape_slim, dtype="int"
        ),
    )


def make_voronoi_mapper_9_3x3():
    return mappers.mapper(
        source_grid_slim=make_grid_7x7(),
        source_pixelization_grid=make_voronoi_pixelization_grid_9(),
        data_pixelization_grid=grid_2d.Grid2D.uniform(
            shape_native=(2, 2), pixel_scales=0.1
        ),
    )


def make_rectangular_inversion_7x7_3x3():
    regularization = reg.Constant(coefficient=1.0)

    return inversions.inversion(
        masked_dataset=make_masked_imaging_7x7(),
        mapper=make_rectangular_mapper_7x7_3x3(),
        regularization=regularization,
    )


def make_voronoi_inversion_9_3x3():

    regularization = reg.Constant(coefficient=1.0)

    return inversions.inversion(
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
