import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.mappers.voronoi import MapperVoronoi


def test__inversion_matrices__linear_eqns_mapping__rectangular_mapper():

    mask = aa.Mask2D.manual(
        mask=[
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        pixel_scales=2.0,
        sub_size=2,
    )

    # Assume a 2x2 sub-grid, so each of our 5 masked_image-pixels are split into 4.
    # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
    # happen for a real lens calculation. This is to make a mapping_matrix matrix which explicitly tests the
    # sub-grid.
    grid = aa.Grid2D.manual_mask(
        grid=[
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        mask=mask,
    )

    pix = aa.pix.Rectangular(shape=(3, 3))

    mapper = pix.mapper_from(
        grid=grid, sparse_grid=None, settings=aa.SettingsPixelization(use_border=False)
    )

    assert mapper.data_pixelization_grid == None
    assert mapper.source_pixelization_grid.shape_native_scaled == pytest.approx(
        (2.0, 2.0), 1.0e-4
    )
    assert mapper.source_pixelization_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

    assert (
        mapper.mapping_matrix
        == np.array(
            [
                [0.75, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.75],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()
    assert mapper.shape_native == (3, 3)

    reg = aa.reg.Constant(coefficient=1.0)
    regularization_matrix = reg.regularization_matrix_from(mapper=mapper)

    assert (
        regularization_matrix
        == np.array(
            [
                [2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001],
            ]
        )
    ).all()

    image = aa.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
    noise_map = aa.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
    psf = aa.Kernel2D.no_blur(pixel_scales=1.0)

    imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

    masked_imaging = imaging.apply_mask(mask=mask)

    inversion = aa.Inversion(
        dataset=masked_imaging,
        mapper_list=[mapper],
        regularization_list=[reg],
        settings=aa.SettingsInversion(check_solution=False),
    )

    assert (inversion.regularization_matrix == regularization_matrix).all()

    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(5), 1.0e-4)


def test__inversion_matrices__linear_eqns_mapping__voronoi_mapper():

    mask = aa.Mask2D.manual(
        mask=[
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, False, False, False, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
        ],
        pixel_scales=1.0,
        sub_size=2,
    )

    grid = np.array(
        [
            [1.01, 0.0],
            [1.01, 0.0],
            [1.01, 0.0],
            [0.01, 0.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.01, 0.0],
            [0.0, 1.01],
            [0.0, 1.01],
            [0.0, 1.01],
            [0.01, 0.0],
            [-1.01, 0.0],
            [-1.01, 0.0],
            [-1.01, 0.0],
            [0.01, 0.0],
        ]
    )

    grid = aa.Grid2D.manual_mask(grid=grid, mask=mask)

    pix = aa.pix.VoronoiMagnification(shape=(3, 3))
    sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        grid=grid, unmasked_sparse_shape=pix.shape
    )

    mapper = pix.mapper_from(
        grid=grid,
        sparse_grid=sparse_grid,
        settings=aa.SettingsPixelization(use_border=False),
    )

    assert mapper.source_grid_slim.shape_native_scaled == pytest.approx(
        (2.02, 2.01), 1.0e-4
    )
    assert (mapper.source_pixelization_grid == sparse_grid).all()
    #    assert mapper.pixelization_grid.origin == pytest.approx((0.0, 0.005), 1.0e-4)

    assert isinstance(mapper, MapperVoronoi)

    assert (
        mapper.mapping_matrix
        == np.array(
            [
                [0.75, 0.0, 0.25, 0.0, 0.0],
                [0.0, 0.75, 0.25, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.25, 0.75, 0.0],
                [0.0, 0.0, 0.25, 0.0, 0.75],
            ]
        )
    ).all()

    reg = aa.reg.Constant(coefficient=1.0)
    regularization_matrix = reg.regularization_matrix_from(mapper=mapper)

    assert (
        regularization_matrix
        == np.array(
            [
                [3.00000001, -1.0, -1.0, -1.0, 0.0],
                [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                [0.0, -1.0, -1.0, -1.0, 3.00000001],
            ]
        )
    ).all()

    image = aa.Array2D.ones(shape_native=(5, 5), pixel_scales=1.0)
    noise_map = aa.Array2D.ones(shape_native=(5, 5), pixel_scales=1.0)
    psf = aa.Kernel2D.no_blur(pixel_scales=1.0)

    imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

    masked_imaging = imaging.apply_mask(mask=mask)

    inversion = aa.Inversion(
        dataset=masked_imaging,
        mapper_list=[mapper],
        regularization_list=[reg],
        settings=aa.SettingsInversion(check_solution=False),
    )

    assert (inversion.regularization_matrix == regularization_matrix).all()
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(5), 1.0e-4)


def test__inversion_matrices__linear_eqns_w_tilde__identical_values_as_linear_eqns_mapping():

    mask = aa.Mask2D.manual(
        mask=[
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        pixel_scales=1.0,
        sub_size=1,
    )

    grid = aa.Grid2D.manual_mask(
        grid=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]], mask=mask
    )

    pix = aa.pix.Rectangular(shape=(3, 3))

    mapper = pix.mapper_from(
        grid=grid, sparse_grid=None, settings=aa.SettingsPixelization(use_border=False)
    )

    reg = aa.reg.Constant(coefficient=1.0)

    image = aa.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
    noise_map = aa.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
    psf = aa.Kernel2D.manual_native(
        array=[[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]], pixel_scales=1.0
    )

    imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

    masked_imaging = imaging.apply_mask(mask=mask)

    inversion_w_tilde = aa.Inversion(
        dataset=masked_imaging,
        mapper_list=[mapper],
        regularization_list=[reg],
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    inversion_mapping_matrices = aa.Inversion(
        dataset=masked_imaging,
        mapper_list=[mapper],
        regularization_list=[reg],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    assert (inversion_w_tilde.data == inversion_mapping_matrices.data).all()
    assert (inversion_w_tilde.noise_map == inversion_mapping_matrices.noise_map).all()
    assert inversion_w_tilde.mapper_list == inversion_mapping_matrices.mapper_list
    assert (
        inversion_w_tilde.regularization_list
        == inversion_mapping_matrices.regularization_list
    )
    assert (
        inversion_w_tilde.regularization_matrix
        == inversion_mapping_matrices.regularization_matrix
    ).all()
    assert (
        inversion_w_tilde.curvature_matrix
        == inversion_mapping_matrices.curvature_matrix
    ).all()
    assert (
        inversion_w_tilde.curvature_reg_matrix
        == inversion_mapping_matrices.curvature_reg_matrix
    ).all()
    assert inversion_w_tilde.reconstruction == pytest.approx(
        inversion_mapping_matrices.reconstruction, 1.0e-4
    )
    assert inversion_w_tilde.mapped_reconstructed_image == pytest.approx(
        inversion_mapping_matrices.mapped_reconstructed_image, 1.0e-4
    )


# def test__inversion_matrices__linear_eqns_x2_mapping():
#
#     mask = aa.Mask2D.manual(
#         mask=[
#             [True, True, True, True, True, True, True],
#             [True, True, True, True, True, True, True],
#             [True, True, True, False, True, True, True],
#             [True, True, False, False, False, True, True],
#             [True, True, True, False, True, True, True],
#             [True, True, True, True, True, True, True],
#             [True, True, True, True, True, True, True],
#         ],
#         pixel_scales=2.0,
#         sub_size=2,
#     )
#
#     # Assume a 2x2 sub-grid, so each of our 5 masked_image-pixels are split into 4.
#     # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
#     # happen for a real lens calculation. This is to make a mapping_matrix matrix which explicitly tests the
#     # sub-grid.
#     grid = aa.Grid2D.manual_mask(
#         grid=[
#             [1.0, -1.0],
#             [1.0, -1.0],
#             [1.0, -1.0],
#             [1.0, 1.0],
#             [1.0, 1.0],
#             [1.0, 1.0],
#             [-1.0, -1.0],
#             [-1.0, -1.0],
#             [-1.0, -1.0],
#             [-1.0, 1.0],
#             [-1.0, 1.0],
#             [-1.0, 1.0],
#             [0.0, 0.0],
#             [0.0, 0.0],
#             [0.0, 0.0],
#             [0.0, 0.0],
#             [0.0, 0.0],
#             [0.0, 0.0],
#             [0.0, 0.0],
#             [0.0, 0.0],
#         ],
#         mask=mask,
#     )
#
#     pix = aa.pix.Rectangular(shape=(3, 3))
#     pix_1 = aa.pix.Rectangular(shape=(4, 4))
#
#     mapper = pix.mapper_from(
#         grid=grid, sparse_grid=None, settings=aa.SettingsPixelization(use_border=False)
#     )
#
#     mapper_1 = pix_1.mapper_from(
#         grid=grid, sparse_grid=None, settings=aa.SettingsPixelization(use_border=False)
#     )
#
#     reg = aa.reg.Constant(coefficient=1.0)
#
#     image = aa.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
#     noise_map = aa.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
#     psf = aa.Kernel2D.no_blur(pixel_scales=1.0)
#
#     imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)
#
#     masked_imaging = imaging.apply_mask(mask=mask)
#
#     inversion = aa.Inversion(
#         dataset=masked_imaging,
#         mapper_list=[mapper, mapper_1],
#         regularization_list=[reg, reg],
#         settings=aa.SettingsInversion(check_solution=False),
#     )
#
#     curvature_matrix_true = aa.util.linear_eqn.curvature_matrix_via_mapping_matrix_from(
#         mapping_matrix=inversion.operated_mapping_matrix,
#         noise_map=inversion.noise_map
#     )
#
#     curvature_matrix_top_left = aa.util.linear_eqn.curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
#         curvature_preload=inversion.linear_eqn_list[0].w_tilde.curvature_preload,
#         curvature_indexes=inversion.linear_eqn_list[0].w_tilde.indexes,
#         curvature_lengths=inversion.linear_eqn_list[0].w_tilde.lengths,
#         data_to_pix_unique=inversion.mapper_list[0].data_unique_mappings.data_to_pix_unique,
#         data_weights=inversion.mapper_list[0].data_unique_mappings.data_weights,
#         pix_lengths=inversion.mapper_list[0].data_unique_mappings.pix_lengths,
#         pix_pixels=inversion.mapper_list[0].pixels,
#     )
#
#     curvature_matrix_diagonal = aa.util.linear_eqn.curvature_matrix_off_diags_via_w_tilde_curvature_preload_imaging_from(
#         curvature_preload=inversion.linear_eqn_list[0].w_tilde.curvature_preload,
#         curvature_indexes=inversion.linear_eqn_list[0].w_tilde.indexes,
#         curvature_lengths=inversion.linear_eqn_list[0].w_tilde.lengths,
#         data_to_pix_unique_0=inversion.mapper_list[0].data_unique_mappings.data_to_pix_unique,
#         data_weights_0=inversion.mapper_list[0].data_unique_mappings.data_weights,
#         pix_lengths_0=inversion.mapper_list[0].data_unique_mappings.pix_lengths,
#         pix_pixels_0=inversion.mapper_list[0].pixels,
#         data_to_pix_unique_1=inversion.mapper_list[1].data_unique_mappings.data_to_pix_unique,
#         data_weights_1=inversion.mapper_list[1].data_unique_mappings.data_weights,
#         pix_lengths_1=inversion.mapper_list[1].data_unique_mappings.pix_lengths,
#         pix_pixels_1=inversion.mapper_list[1].pixels,
#     )
#
#    # print(curvature_matrix_diagonal)
#
#     curvature_matrix_inversion = inversion.curvature_matrix_diag
#     curvature_matrix_inversion[0:9, 9:25] = curvature_matrix_diagonal
#
#     print(curvature_matrix_inversion.shape)
#
#     for i in range(curvature_matrix_inversion.shape[0]):
#         for j in range(curvature_matrix_inversion.shape[1]):
#             curvature_matrix_inversion[j, i] = curvature_matrix_inversion[i, j]
#
#     print(curvature_matrix_true - curvature_matrix_inversion)
#     stop
#
#
#     curvature_matrix = curvature_matrix_inversion + curvature_matrix_diagonal
#
#     blurred_mapping_matrix = masked_imaging.convolver.convolve_mapping_matrix(mapper.mapping_matrix)
#
#     # assert (inversion.operated_mapping_matrix[0:5, 0:9] == blurred_mapping_matrix).all()
#     # assert (inversion.operated_mapping_matrix[0:5, 9:18] == blurred_mapping_matrix).all()
#
#     curvature_matrix = aa.util.linear_eqn.curvature_matrix_via_mapping_matrix_from(
#         mapping_matrix=inversion.operated_mapping_matrix,
#         noise_map=masked_imaging.noise_map
#     )
#
#     # print(curvature_matrix - inversion.curvature_matrix)
#     #
#     # assert (inversion.curvature_matrix == curvature_matrix).all()
#
#     regularization_matrix_of_reg = reg.regularization_matrix_from(mapper=mapper)
#
#     assert (inversion.regularization_matrix[0:9, 0:9] == regularization_matrix_of_reg).all()
#     assert (inversion.regularization_matrix[9:18, 9:18] == regularization_matrix_of_reg).all()
#
#     print(inversion.mapped_reconstructed_data_of_mappers)
#     print(inversion.mapped_reconstructed_image)
#
#     assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(5), 1.0e-4)
#

def test__inversion_matrices__linear_eqns_mapping__rectangular_mapper__matrix_formalism():

    real_space_mask = aa.Mask2D.unmasked(
        shape_native=(7, 7), pixel_scales=0.1, sub_size=1
    )

    grid = aa.Grid2D.from_mask(mask=real_space_mask)

    pix = aa.pix.Rectangular(shape=(7, 7))

    mapper = pix.mapper_from(
        grid=grid, sparse_grid=None, settings=aa.SettingsPixelization(use_border=False)
    )

    reg = aa.reg.Constant(coefficient=0.0)

    visibilities = aa.Visibilities.manual_slim(
        visibilities=[
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
        ]
    )
    noise_map = aa.VisibilitiesNoiseMap.ones(shape_slim=(7,))
    uv_wavelengths = np.ones(shape=(7, 2))

    interferometer = aa.Interferometer(
        visibilities=visibilities,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
    )

    inversion = aa.Inversion(
        dataset=interferometer,
        mapper_list=[mapper],
        regularization_list=[reg],
        settings=aa.SettingsInversion(check_solution=False),
    )

    assert inversion.mapped_reconstructed_data == pytest.approx(
        1.0 + 0.0j * np.ones(shape=(7,)), 1.0e-4
    )
    assert (np.imag(inversion.mapped_reconstructed_data) < 0.0001).all()
    assert (np.imag(inversion.mapped_reconstructed_data) > 0.0).all()


def test__inversion_matirces__linear_eqns_mapping__voronoi_mapper__matrix_formalism():

    real_space_mask = aa.Mask2D.unmasked(
        shape_native=(7, 7), pixel_scales=0.1, sub_size=1
    )

    grid = aa.Grid2D.from_mask(mask=real_space_mask)

    pix = aa.pix.VoronoiMagnification(shape=(7, 7))

    sparse_grid = pix.sparse_grid_from_grid(grid=grid)

    mapper = pix.mapper_from(
        grid=grid,
        sparse_grid=sparse_grid,
        settings=aa.SettingsPixelization(use_border=False),
    )

    reg = aa.reg.Constant(coefficient=0.0)

    visibilities = aa.Visibilities.manual_slim(
        visibilities=[
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
        ]
    )
    noise_map = aa.VisibilitiesNoiseMap.ones(shape_slim=(7,))
    uv_wavelengths = np.ones(shape=(7, 2))

    interferometer = aa.Interferometer(
        visibilities=visibilities,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
    )

    inversion = aa.Inversion(
        dataset=interferometer,
        mapper_list=[mapper],
        regularization_list=[reg],
        settings=aa.SettingsInversion(check_solution=False),
    )

    assert inversion.mapped_reconstructed_data == pytest.approx(
        1.0 + 0.0j * np.ones(shape=(7,)), 1.0e-4
    )
    assert (np.imag(inversion.mapped_reconstructed_data) < 0.0001).all()
    assert (np.imag(inversion.mapped_reconstructed_data) > 0.0).all()


def test__inversion_linear_operator__linear_eqns_linear_operator_formalism():

    real_space_mask = aa.Mask2D.unmasked(
        shape_native=(7, 7), pixel_scales=0.1, sub_size=1
    )

    grid = aa.Grid2D.from_mask(mask=real_space_mask)

    pix = aa.pix.Rectangular(shape=(7, 7))

    mapper = pix.mapper_from(
        grid=grid, sparse_grid=None, settings=aa.SettingsPixelization(use_border=False)
    )

    reg = aa.reg.Constant(coefficient=0.0)

    visibilities = aa.Visibilities.manual_slim(
        visibilities=[
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
        ]
    )
    noise_map = aa.VisibilitiesNoiseMap.ones(shape_slim=(7,))
    uv_wavelengths = np.ones(shape=(7, 2))

    interferometer = aa.Interferometer(
        visibilities=visibilities,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        settings=aa.SettingsInterferometer(transformer_class=aa.TransformerNUFFT),
    )

    inversion = aa.Inversion(
        dataset=interferometer,
        mapper_list=[mapper],
        regularization_list=[reg],
        settings=aa.SettingsInversion(use_linear_operators=True, check_solution=False),
    )

    assert inversion.mapped_reconstructed_data == pytest.approx(
        1.0 + 0.0j * np.ones(shape=(7,)), 1.0e-4
    )
    assert (np.imag(inversion.mapped_reconstructed_data) < 0.0001).all()
    assert (np.imag(inversion.mapped_reconstructed_data) > 0.0).all()
