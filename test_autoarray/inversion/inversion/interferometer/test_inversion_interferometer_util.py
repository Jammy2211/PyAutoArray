import autoarray as aa
import numpy as np
import pytest


def test__data_vector_via_transformed_mapping_matrix_from():
    mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    data_real = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    noise_map_real = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

    data_vector_real_via_blurred = (
        aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=mapping_matrix,
            image=data_real,
            noise_map=noise_map_real,
        )
    )

    data_imag = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    noise_map_imag = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

    data_vector_imag_via_blurred = (
        aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=mapping_matrix,
            image=data_imag,
            noise_map=noise_map_imag,
        )
    )

    data_vector_complex_via_blurred = (
        data_vector_real_via_blurred + data_vector_imag_via_blurred
    )

    transformed_mapping_matrix = np.array(
        [
            [1.0 + 1.0j, 1.0 + 1.0j, 0.0 + 0.0j],
            [1.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 1.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 1.0j, 1.0 + 1.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ]
    )

    data = np.array(
        [4.0 + 4.0j, 1.0 + 1.0j, 1.0 + 1.0j, 16.0 + 16.0j, 1.0 + 1.0j, 1.0 + 1.0j]
    )
    noise_map = np.array(
        [2.0 + 2.0j, 1.0 + 1.0j, 1.0 + 1.0j, 4.0 + 4.0j, 1.0 + 1.0j, 1.0 + 1.0j]
    )

    data_vector_via_transformed = aa.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=data,
        noise_map=noise_map,
    )

    assert (data_vector_complex_via_blurred == data_vector_via_transformed).all()


def test__curvature_matrix_via_psf_precision_operator_from():
    noise_map = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    uv_wavelengths = np.array(
        [[0.0001, 2.0, 3000.0, 50000.0, 200000.0], [3000.0, 2.0, 0.0001, 10.0, 5000.0]]
    )

    grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.0005)

    mapping_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    nufft_precision_operator = aa.util.inversion_interferometer.nufft_precision_operator_from(
        noise_map_real=noise_map,
        uv_wavelengths=uv_wavelengths,
        shape_masked_pixels_2d=(3, 3),
        grid_radians_2d=np.array(grid.native),
    )

    native_index_for_slim_index = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    psf_weighted_noise = (
        aa.util.inversion_interferometer.nufft_weighted_noise_via_sparse_operator_from(
            translation_invariant_kernel=nufft_precision_operator,
            native_index_for_slim_index=native_index_for_slim_index,
        )
    )

    curvature_matrix_via_nufft_weighted_noise = (
        aa.util.inversion.curvature_matrix_diag_via_psf_weighted_noise_from(
            psf_weighted_noise=psf_weighted_noise, mapping_matrix=mapping_matrix
        )
    )

    pix_indexes_for_sub_slim_index = np.array(
        [[0], [2], [1], [1], [2], [2], [0], [2], [0]]
    )

    pix_weights_for_sub_slim_index = np.ones(shape=(9, 1))

    sparse_operator = aa.InterferometerSparseOperator.from_nufft_precision_operator(
        nufft_precision_operator=nufft_precision_operator,
        dirty_image=None,
    )

    curvature_matrix_via_preload = (
        sparse_operator.curvature_matrix_via_sparse_operator_from(
            pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
            fft_index_for_masked_pixel=grid.mask.fft_index_for_masked_pixel,
            pix_pixels=3,
        )
    )

    assert curvature_matrix_via_nufft_weighted_noise == pytest.approx(
        curvature_matrix_via_preload, 1.0e-4
    )
