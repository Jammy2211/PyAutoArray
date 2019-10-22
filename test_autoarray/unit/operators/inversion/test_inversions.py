import autoarray as aa
from autoarray import exc
import numpy as np
import pytest

from test_autoarray.mock import mock_inversion


class TestRegularizationTerm:
    def test__solution_all_1s__regularization_matrix_simple(self):

        matrix_shape = (3, 3)

        inv = aa.inversions.InversionImaging.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape=matrix_shape),
            regularization=mock_inversion.MockRegularization(matrix_shape),
        )

        inv.reconstruction = np.array([1.0, 1.0, 1.0])

        inv.regularization_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

        # G_l = s_T * H * s

        # Matrix multiplication:

        # s_T * H = [1.0, 1.0, 1.0] * [1.0, 1.0, 1.0] = [(1.0*1.0) + (1.0*0.0) + (1.0*0.0)] = [1.0, 1.0, 1.0]
        #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*1.0) + (1.0*0.0)]
        #                             [1.0, 1.0, 1.0]   [(1.0*0.0) + (1.0*0.0) + (1.0*1.0)]

        # (s_T * H) * s = [1.0, 1.0, 1.0] * [1.0] = 3.0
        #                                   [1.0]
        #                                   [1.0]

        assert inv.regularization_term == 3.0

    def test__solution_and_regularization_matrix_range_of_values(self):

        matrix_shape = (3, 3)

        inv = aa.inversions.InversionImaging.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape),
            regularization=mock_inversion.MockRegularization(matrix_shape),
        )

        # G_l term, Warren & Dye 2003 / Nightingale /2015 2018

        # G_l = s_T * H * s

        # Matrix multiplication:

        # s_T * H = [2.0, 3.0, 5.0] * [2.0,  -1.0,  0.0] = [(2.0* 2.0) + (3.0*-1.0) + (5.0 *0.0)] = [1.0, -1.0, 7.0]
        #                             [-1.0,  2.0, -1.0]   [(2.0*-1.0) + (3.0* 2.0) + (5.0*-1.0)]
        #                             [ 0.0, -1.0,  2.0]   [(2.0* 0.0) + (3.0*-1.0) + (5.0 *2.0)]

        # (s_T * H) * s = [1.0, -1.0, 7.0] * [2.0] = 34.0
        #                                    [3.0]
        #                                    [5.0]

        inv.reconstruction = np.array([2.0, 3.0, 5.0])

        inv.regularization_matrix = np.array(
            [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
        )

        assert inv.regularization_term == 34.0


class TestLogDetMatrix:
    def test__determinant_of_positive_definite_matrix_via_cholesky(self):

        matrix_shape = (3, 3)

        inv = aa.inversions.InversionImaging.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape),
            regularization=mock_inversion.MockRegularization(matrix_shape),
        )

        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(
            inv.log_determinant_of_matrix_cholesky(matrix), 1e-4
        )

    def test__determinant_of_positive_definite_matrix_2_via_cholesky(self):

        matrix_shape = (3, 3)

        inv = aa.inversions.InversionImaging.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape),
            regularization=mock_inversion.MockRegularization(matrix_shape),
        )

        matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(
            inv.log_determinant_of_matrix_cholesky(matrix), 1e-4
        )

    def test__matrix_not_positive_definite__raises_reconstruction_exception(self):

        matrix_shape = (3, 3)

        inv = aa.inversions.InversionImaging.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape),
            regularization=mock_inversion.MockRegularization(matrix_shape),
        )

        matrix = np.array([[2.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 0.0]])

        with pytest.raises(exc.InversionException):
            assert pytest.approx(inv.log_determinant_of_matrix_cholesky(matrix), 1e-4)


class TestReconstructedDataVectorAndImage:
    def test__solution_all_1s__simple_blurred_mapping_matrix__correct_reconstructed_image(
        self
    ):

        matrix_shape = (3, 3)

        mask = aa.mask.manual(
            mask_2d=np.array(
                [[True, True, True], [False, False, False], [True, True, True]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.masked_grid.from_mask(mask=mask)

        inv = aa.inversions.InversionImaging.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape=matrix_shape, grid=grid),
            regularization=mock_inversion.MockRegularization(matrix_shape),
        )

        inv.reconstruction = np.array([1.0, 1.0, 1.0, 1.0])

        inv.blurred_mapping_matrix = np.array(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]]
        )
        # Imaging pixel 0 maps to 4 pixs pixxels -> value is 4.0
        # Imaging pixel 1 maps to 3 pixs pixxels -> value is 3.0
        # Imaging pixel 2 maps to 1 pixs pixxels -> value is 1.0

        assert (inv.mapped_reconstructed_image == np.array([4.0, 3.0, 1.0])).all()
        assert (inv.mapped_reconstructed_image.in_2d == np.array(
            [[0.0, 0.0, 0.0], [4.0, 3.0, 1.0], [0.0, 0.0, 0.0]]
        )).all()

        assert inv.errors_with_covariance == pytest.approx(
            np.array([[0.7, -0.3, -0.3], [-0.3, 0.7, -0.3], [-0.3, -0.3, 0.7]]), 1.0e-4
        )
        assert inv.errors == pytest.approx(
            np.array([0.7, 0.7, 0.7]), 1.0e-4
        )

    def test__solution_different_values__simple_blurred_mapping_matrix__correct_reconstructed_image(
        self
    ):

        matrix_shape = (3, 3)

        mask = aa.mask.manual(
            mask_2d=np.array(
                [[True, True, True], [False, False, False], [True, True, True]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.masked_grid.from_mask(mask=mask)

        inv = aa.inversions.InversionImaging.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock_inversion.MockConvolver(matrix_shape),
            mapper=mock_inversion.MockMapper(matrix_shape=matrix_shape, grid=grid),
            regularization=mock_inversion.MockRegularization(matrix_shape),
        )

        inv.reconstruction = np.array([1.0, 2.0, 3.0, 4.0])

        inv.blurred_mapping_matrix = np.array(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]]
        )

        # # Imaging pixel 0 maps to 4 pixs pixxels -> value is 1.0 + 2.0 + 3.0 + 4.0 = 10.0
        # # Imaging pixel 1 maps to 3 pixs pixxels -> value is 1.0 + 3.0 + 4.0
        # # Imaging pixel 2 maps to 1 pixs pixxels -> value is 1.0

        assert (inv.mapped_reconstructed_image == np.array([10.0, 8.0, 1.0])).all()
        assert (inv.mapped_reconstructed_image.in_2d == np.array(
            [[0.0, 0.0, 0.0], [10.0, 8.0, 1.0], [0.0, 0.0, 0.0]]
        )).all()

        assert inv.errors_with_covariance == pytest.approx(
            np.array([[0.7, -0.3, -0.3], [-0.3, 0.7, -0.3], [-0.3, -0.3, 0.7]]), 1.0e-4
        )
        assert inv.errors == pytest.approx(
            np.array([0.7, 0.7, 0.7]), 1.0e-4
        )


#
# class TestPixelizationQuantities:
#
#     def test__compare_to_inversion_utils(self):
#
#         matrix_shape = (3,3)
#
#         mask = aa.mask.manual(array=np.array([[True, True, True],
#                                         [False, False, False],
#                                         [True, True, True]]), pixel_scales=1.0)
#
#         grid = aa.grid_stack_from_mask_sub_size_and_psf_shape(
#             mask=mask, sub_size=1, psf_shape=(1,1))
#
#         inv = aa.inversions.Inversion(
#             image_1d=np.ones(9), noise_map_1d=np.ones(9), convolver=mock_inversion.MockConvolver(matrix_shape),
#             mapper=mock_inversion.MockMapper(matrix_shape, grid),
#             regularization=mock_inversion.MockRegularization(matrix_shape))
#
#         inv.reconstruction = np.array([1.0, 1.0, 1.0, 1.0])
#
#         inv.blurred_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
#                                                [1.0, 0.0, 1.0, 1.0],
#                                                [1.0, 0.0, 0.0, 0.0]])
#
#         pixelization_residuals_util = \
#             am.util.inversion.pixelization_residuals_from_pixelization_values_reconstructed_data_1d_and_mapping_quantities(
#                 pixelization_values=inv.reconstruction, reconstructed_data_1d=inv.reconstructed_data_1d,
#                 mask_1d_index_for_sub_mask_1d_index=inv.mapper.mask_1d_index_for_sub_mask_1d_index, all_sub_mask_1d_indexes_for_pixelization_1d_index=inv.mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index)
