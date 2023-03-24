import autoarray as aa
import numpy as np
import pytest


def test__curvature_matrix_from_w_tilde():

    w_tilde = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 2.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )

    mapping_matrix = np.array(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )

    curvature_matrix = aa.util.inversion.curvature_matrix_via_w_tilde_from(
        w_tilde=w_tilde, mapping_matrix=mapping_matrix
    )

    assert (
        curvature_matrix
        == np.array([[6.0, 8.0, 0.0], [8.0, 8.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()


def test__curvature_matrix_via_sparse_preload():

    blurred_mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    (
        curvature_matrix_preload,
        curvature_matrix_counts,
    ) = aa.util.inversion.curvature_matrix_preload_from(
        mapping_matrix=blurred_mapping_matrix
    )

    curvature_matrix = aa.util.inversion.curvature_matrix_via_sparse_preload_from(
        mapping_matrix=blurred_mapping_matrix,
        noise_map=noise_map,
        curvature_matrix_preload=curvature_matrix_preload.astype("int"),
        curvature_matrix_counts=curvature_matrix_counts.astype("int"),
    )

    assert (
        curvature_matrix
        == np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])
    ).all()

    blurred_mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.25],
            [0.0, 1.0, 0.6, 0.75],
            [0.0, 1.0, 1.0, 0.1],
            [0.0, 0.0, 0.3, 1.0],
            [0.0, 0.0, 0.5, 0.7],
        ]
    )

    noise_map = np.array([2.0, 1.0, 10.0, 0.5, 3.0, 7.0])

    curvature_matrix_via_mapping_matrix = (
        aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
        )
    )

    (
        curvature_matrix_preload,
        curvature_matrix_counts,
    ) = aa.util.inversion.curvature_matrix_preload_from(
        mapping_matrix=blurred_mapping_matrix
    )

    curvature_matrix = aa.util.inversion.curvature_matrix_via_sparse_preload_from(
        mapping_matrix=blurred_mapping_matrix,
        noise_map=noise_map,
        curvature_matrix_preload=curvature_matrix_preload.astype("int"),
        curvature_matrix_counts=curvature_matrix_counts.astype("int"),
    )

    assert curvature_matrix_via_mapping_matrix == pytest.approx(
        curvature_matrix, 1.0e-4
    )


def test__curvature_matrix_via_mapping_matrix_from():

    blurred_mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
    )

    assert (
        curvature_matrix
        == np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])
    ).all()

    blurred_mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    noise_map = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
    )

    assert (
        curvature_matrix
        == np.array([[1.25, 0.25, 0.0], [0.25, 2.25, 1.0], [0.0, 1.0, 1.0]])
    ).all()


def test__reconstruction_positive_negative_from():

    data_vector = np.array([1.0, 1.0, 2.0])

    curvature_reg_matrix = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])

    reconstruction = aa.util.inversion.reconstruction_positive_negative_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        mapper_param_range_list=[[0, 3]],
    )

    assert reconstruction == pytest.approx(np.array([1.0, -1.0, 3.0]), 1.0e-4)


def test__reconstruction_positive_negative_from__check_solution_raises_error_cause_all_values_identical():

    data_vector = np.array([1.0, 1.0, 1.0])

    curvature_reg_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # reconstruction = np.array([1.0, 1.0, 1.0])

    with pytest.raises(aa.exc.InversionException):

        aa.util.inversion.reconstruction_positive_negative_from(
            data_vector=data_vector,
            curvature_reg_matrix=curvature_reg_matrix,
            mapper_param_range_list=[[0, 3]],
            force_check_reconstruction=True,
        )


def test__mapped_reconstructed_data_via_mapping_matrix_from():

    mapping_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_data = (
        aa.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix, reconstruction=reconstruction
        )
    )

    assert (mapped_reconstructed_data == np.array([1.0, 1.0, 2.0])).all()

    mapping_matrix = np.array([[0.25, 0.50, 0.25], [0.0, 1.0, 0.0], [0.0, 0.25, 0.75]])

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_data = (
        aa.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix, reconstruction=reconstruction
        )
    )

    assert (mapped_reconstructed_data == np.array([1.25, 1.0, 1.75])).all()


def test__mapped_reconstructed_data_via_image_to_pix_unique_from():

    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    pix_indexes_for_sub_slim_index_sizes = np.array([1, 1, 1]).astype("int")
    pix_weights_for_sub_slim_index = np.array([[1.0], [1.0], [1.0]])

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper.data_slim_to_pixelization_unique_from(
        data_pixels=3,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_indexes_for_sub_slim_index_sizes,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        sub_size=1,
    )

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_data = (
        aa.util.inversion.mapped_reconstructed_data_via_image_to_pix_unique_from(
            data_to_pix_unique=data_to_pix_unique.astype("int"),
            data_weights=data_weights,
            pix_lengths=pix_lengths.astype("int"),
            reconstruction=reconstruction,
        )
    )

    assert (mapped_reconstructed_data == np.array([1.0, 1.0, 2.0])).all()

    pix_indexes_for_sub_slim_index = np.array(
        [[0], [1], [1], [2], [1], [1], [1], [1], [1], [2], [2], [2]]
    )
    pix_indexes_for_sub_slim_index_sizes = np.ones(shape=(12,)).astype("int")
    pix_weights_for_sub_slim_index = np.ones(shape=(12, 1))

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper.data_slim_to_pixelization_unique_from(
        data_pixels=3,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_indexes_for_sub_slim_index_sizes,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        sub_size=2,
    )

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_data = (
        aa.util.inversion.mapped_reconstructed_data_via_image_to_pix_unique_from(
            data_to_pix_unique=data_to_pix_unique.astype("int"),
            data_weights=data_weights,
            pix_lengths=pix_lengths.astype("int"),
            reconstruction=reconstruction,
        )
    )

    assert (mapped_reconstructed_data == np.array([1.25, 1.0, 1.75])).all()


def test__preconditioner_matrix_via_mapping_matrix_from():

    mapping_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=1.0,
            regularization_matrix=np.zeros((3, 3)),
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    ).all()

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=2.0,
            regularization_matrix=np.zeros((3, 3)),
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    ).all()

    regularization_matrix = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=2.0,
            regularization_matrix=regularization_matrix,
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[5.0, 2.0, 3.0], [4.0, 9.0, 6.0], [7.0, 8.0, 13.0]])
    ).all()


def test__inversion_residual_map_from():

    pixelization_values = np.ones(3)
    reconstructed_data_1d = np.ones(9)
    slim_index_for_sub_slim_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    sub_slim_indexes_for_pix_index = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    pixelization_residuals = aa.util.inversion.inversion_residual_map_from(
        reconstruction=pixelization_values,
        data=reconstructed_data_1d,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_slim_indexes_for_pix_index=sub_slim_indexes_for_pix_index,
    )

    assert (pixelization_residuals == np.zeros(3)).all()

    pixelization_values = np.ones(3)
    reconstructed_data_1d = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    slim_index_for_sub_slim_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    sub_slim_indexes_for_pix_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    pixelization_residuals = aa.util.inversion.inversion_residual_map_from(
        reconstruction=pixelization_values,
        data=reconstructed_data_1d,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_slim_indexes_for_pix_index=sub_slim_indexes_for_pix_index,
    )

    assert (pixelization_residuals == np.array([0.0, 1.0, 2.0])).all()


def test__inversion_normalized_residual_map_from():

    pixelization_values = np.ones(3)
    reconstructed_data_1d = np.ones(9)
    noise_map_1d = np.ones(9)
    slim_index_for_sub_slim_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    sub_slim_indexes_for_pix_index = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    pixelization_normalized_residuals = (
        aa.util.inversion.inversion_normalized_residual_map_from(
            reconstruction=pixelization_values,
            data=reconstructed_data_1d,
            noise_map_1d=noise_map_1d,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            sub_slim_indexes_for_pix_index=sub_slim_indexes_for_pix_index,
        )
    )

    assert (pixelization_normalized_residuals == np.zeros(3)).all()

    pixelization_values = np.ones(3)
    reconstructed_data_1d = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    noise_map_1d = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    slim_index_for_sub_slim_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    sub_slim_indexes_for_pix_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    pixelization_normalized_residuals = (
        aa.util.inversion.inversion_normalized_residual_map_from(
            reconstruction=pixelization_values,
            data=reconstructed_data_1d,
            noise_map_1d=noise_map_1d,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            sub_slim_indexes_for_pix_index=sub_slim_indexes_for_pix_index,
        )
    )

    assert (pixelization_normalized_residuals == np.array([0.0, 1.0, 1.0])).all()


def test__inversion_chi_squared_map_from():

    pixelization_values = np.ones(3)
    reconstructed_data_1d = np.ones(9)
    noise_map_1d = np.ones(9)
    slim_index_for_sub_slim_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    sub_slim_indexes_for_pix_index = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    pixelization_chi_squareds = aa.util.inversion.inversion_chi_squared_map_from(
        reconstruction=pixelization_values,
        data=reconstructed_data_1d,
        noise_map_1d=noise_map_1d,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_slim_indexes_for_pix_index=sub_slim_indexes_for_pix_index,
    )

    assert (pixelization_chi_squareds == np.zeros(3)).all()

    pixelization_values = np.ones(3)
    reconstructed_data_1d = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    noise_map_1d = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 4.0, 4.0, 4.0])
    slim_index_for_sub_slim_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    sub_slim_indexes_for_pix_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    pixelization_chi_squareds = aa.util.inversion.inversion_chi_squared_map_from(
        reconstruction=pixelization_values,
        data=reconstructed_data_1d,
        noise_map_1d=noise_map_1d,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_slim_indexes_for_pix_index=sub_slim_indexes_for_pix_index,
    )

    assert (pixelization_chi_squareds == np.array([0.0, 4.0, 0.25])).all()
