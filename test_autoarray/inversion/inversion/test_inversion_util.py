import autoarray as aa
import numpy as np


class TestPreconditionerMatrix:
    def test__simple_calculations(self):

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

        preconditioner_matrix = aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=1.0,
            regularization_matrix=np.zeros((3, 3)),
        )

        assert (
            preconditioner_matrix
            == np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        ).all()

        preconditioner_matrix = aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=2.0,
            regularization_matrix=np.zeros((3, 3)),
        )

        assert (
            preconditioner_matrix
            == np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
        ).all()

        regularization_matrix = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )

        preconditioner_matrix = aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=2.0,
            regularization_matrix=regularization_matrix,
        )

        assert (
            preconditioner_matrix
            == np.array([[5.0, 2.0, 3.0], [4.0, 9.0, 6.0], [7.0, 8.0, 13.0]])
        ).all()


class TestPixelizationQuantity:
    def test__residuals(self,):

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

    def test__normalized_residuals__pixelization_perfectly_reconstructed_data__quantities_like_residuals_all_zeros(
        self,
    ):

        pixelization_values = np.ones(3)
        reconstructed_data_1d = np.ones(9)
        noise_map_1d = np.ones(9)
        slim_index_for_sub_slim_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        sub_slim_indexes_for_pix_index = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        pixelization_normalized_residuals = aa.util.inversion.inversion_normalized_residual_map_from(
            reconstruction=pixelization_values,
            data=reconstructed_data_1d,
            noise_map_1d=noise_map_1d,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            sub_slim_indexes_for_pix_index=sub_slim_indexes_for_pix_index,
        )

        assert (pixelization_normalized_residuals == np.zeros(3)).all()

        pixelization_values = np.ones(3)
        reconstructed_data_1d = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        noise_map_1d = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        slim_index_for_sub_slim_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        sub_slim_indexes_for_pix_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        pixelization_normalized_residuals = aa.util.inversion.inversion_normalized_residual_map_from(
            reconstruction=pixelization_values,
            data=reconstructed_data_1d,
            noise_map_1d=noise_map_1d,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            sub_slim_indexes_for_pix_index=sub_slim_indexes_for_pix_index,
        )

        assert (pixelization_normalized_residuals == np.array([0.0, 1.0, 1.0])).all()

    def test__chi_squared__pixelization_perfectly_reconstructed_data__quantities_like_residuals_all_zeros(
        self,
    ):

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
