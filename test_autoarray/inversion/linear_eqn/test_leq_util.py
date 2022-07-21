import autoarray as aa
import numpy as np
import pytest


class TestMappedReconstructedDataFrom:
    def test__mapped_reconstructed_data_via_mapping_matrix_from(self):

        mapping_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        reconstruction = np.array([1.0, 1.0, 2.0])

        mapped_reconstructed_data = aa.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix, reconstruction=reconstruction
        )

        assert (mapped_reconstructed_data == np.array([1.0, 1.0, 2.0])).all()

        mapping_matrix = np.array(
            [[0.25, 0.50, 0.25], [0.0, 1.0, 0.0], [0.0, 0.25, 0.75]]
        )

        reconstruction = np.array([1.0, 1.0, 2.0])

        mapped_reconstructed_data = aa.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix, reconstruction=reconstruction
        )

        assert (mapped_reconstructed_data == np.array([1.25, 1.0, 1.75])).all()

    def test__mapped_reconstructed_data_via_image_to_pix_unique_from(self):

        pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
        pix_indexes_for_sub_slim_index_sizes = np.array([1, 1, 1]).astype("int")
        pix_weights_for_sub_slim_index = np.array([[1.0], [1.0], [1.0]])

        data_to_pix_unique, data_weights, pix_lengths = aa.util.mapper.data_slim_to_pixelization_unique_from(
            data_pixels=3,
            pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
            pix_sizes_for_sub_slim_index=pix_indexes_for_sub_slim_index_sizes,
            pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
            sub_size=1,
        )

        reconstruction = np.array([1.0, 1.0, 2.0])

        mapped_reconstructed_data = aa.util.inversion.mapped_reconstructed_data_via_image_to_pix_unique_from(
            data_to_pix_unique=data_to_pix_unique.astype("int"),
            data_weights=data_weights,
            pix_lengths=pix_lengths.astype("int"),
            reconstruction=reconstruction,
        )

        assert (mapped_reconstructed_data == np.array([1.0, 1.0, 2.0])).all()

        pix_indexes_for_sub_slim_index = np.array(
            [[0], [1], [1], [2], [1], [1], [1], [1], [1], [2], [2], [2]]
        )
        pix_indexes_for_sub_slim_index_sizes = np.ones(shape=(12,)).astype("int")
        pix_weights_for_sub_slim_index = np.ones(shape=(12, 1))

        data_to_pix_unique, data_weights, pix_lengths = aa.util.mapper.data_slim_to_pixelization_unique_from(
            data_pixels=3,
            pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
            pix_sizes_for_sub_slim_index=pix_indexes_for_sub_slim_index_sizes,
            pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
            sub_size=2,
        )

        reconstruction = np.array([1.0, 1.0, 2.0])

        mapped_reconstructed_data = aa.util.inversion.mapped_reconstructed_data_via_image_to_pix_unique_from(
            data_to_pix_unique=data_to_pix_unique.astype("int"),
            data_weights=data_weights,
            pix_lengths=pix_lengths.astype("int"),
            reconstruction=reconstruction,
        )

        assert (mapped_reconstructed_data == np.array([1.25, 1.0, 1.75])).all()
