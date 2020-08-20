import autoarray as aa
from autoarray.inversion import inversions
from autoarray import exc
import numpy as np

from test_autoarray import mock


class TestSettingsInversion:
    def test__inversion_uses_linear_operator_tag(self):

        settings = aa.SettingsInversion(use_linear_operators=True)
        assert settings.use_linear_operators_tag == "__lop"

        settings = aa.SettingsInversion(use_linear_operators=False)
        assert settings.use_linear_operators_tag == ""


class TestLogDetMatrixCholesky:
    def test__determinant_of_positive_definite_matrix_via_cholesky(self):

        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(
            inversions.log_determinant_of_matrix_cholesky(matrix), 1e-4
        )

    def test__determinant_of_positive_definite_matrix_2_via_cholesky(self):

        matrix = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

        log_determinant = np.log(np.linalg.det(matrix))

        assert log_determinant == pytest.approx(
            inversions.log_determinant_of_matrix_cholesky(matrix), 1e-4
        )

    def test__matrix_not_positive_definite__raises_reconstruction_exception(self):

        matrix = np.array([[2.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 0.0]])

        with pytest.raises(exc.InversionException):
            assert pytest.approx(
                inversions.log_determinant_of_matrix_cholesky(matrix), 1e-4
            )


class TestAbstractInversion:
    def test__regularization_term__solution_all_1s__regularization_matrix_simple(self):

        matrix_shape = (3, 3)

        inversion = inversions.InversionImagingMatrix.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock.MockConvolver(matrix_shape),
            mapper=mock.MockMapper(matrix_shape=matrix_shape),
            regularization=mock.MockRegularization(matrix_shape),
            settings=aa.SettingsInversion(check_solution=False),
        )

        inversion.reconstruction = np.array([1.0, 1.0, 1.0])

        inversion.regularization_matrix = np.array(
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

        assert inversion.regularization_term == 3.0

    def test__regularization_term__solution_and_regularization_matrix_range_of_values(
        self
    ):

        matrix_shape = (3, 3)

        inversion = inversions.InversionImagingMatrix.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock.MockConvolver(matrix_shape),
            mapper=mock.MockMapper(matrix_shape),
            regularization=mock.MockRegularization(matrix_shape),
            settings=aa.SettingsInversion(check_solution=False),
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

        inversion.reconstruction = np.array([2.0, 3.0, 5.0])

        inversion.regularization_matrix = np.array(
            [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
        )

        assert inversion.regularization_term == 34.0

    def test__brightest_reconstruction_pixel_and_centre(self):

        matrix_shape = (3, 3)

        inversion = inversions.InversionImagingMatrix.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock.MockConvolver(matrix_shape),
            mapper=mock.MockMapper(
                matrix_shape,
                pixelization_grid=aa.GridVoronoi.manual_1d(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 0.0]]
                ),
            ),
            regularization=mock.MockRegularization(matrix_shape),
            settings=aa.SettingsInversion(check_solution=False),
        )

        inversion.reconstruction = np.array([2.0, 3.0, 5.0, 0.0])

        assert inversion.brightest_reconstruction_pixel == 2
        assert inversion.brightest_reconstruction_pixel_centre.in_list == [[(5.0, 6.0)]]

    def test__interpolated_reconstruction__config_is_image_grid__grid_as_mapper_with_good_interp(
        self
    ):

        conf.instance = conf.Config(
            path.join(directory, "files/inversion_image_grid"),
            path.join(directory, "output"),
        )

        matrix_shape = (3, 3)

        mask = aa.Mask.manual(
            mask=np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, True, True],
                    [True, True, True, True, True],
                ]
            ),
            pixel_scales=1.0,
            sub_size=2,
        )

        grid = aa.Grid.from_mask(mask=mask)

        pixelization_grid = aa.Grid.uniform(
            shape_2d=(3, 3), pixel_scales=1.0, sub_size=1
        )

        inversion = inversions.InversionImagingMatrix.from_data_mapper_and_regularization(
            image=np.ones(25),
            noise_map=np.ones(25),
            convolver=mock.MockConvolver(matrix_shape),
            mapper=mock.MockMapper(
                matrix_shape=matrix_shape,
                grid=grid,
                pixelization_grid=pixelization_grid,
            ),
            regularization=mock.MockRegularization(matrix_shape),
            settings=aa.SettingsInversion(check_solution=False),
        )

        inversion.reconstruction = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

        interpolated_reconstruction = (
            inversion.interpolated_reconstruction_from_shape_2d()
        )

        assert interpolated_reconstruction.shape_2d == (5, 5)

        assert interpolated_reconstruction.in_1d == pytest.approx(
            np.ones(shape=(25,)), 1.0e-4
        )
        assert interpolated_reconstruction.in_2d == pytest.approx(
            np.ones(shape=(5, 5)), 1.0e-4
        )
        assert interpolated_reconstruction.pixel_scales == pytest.approx(
            (1.0, 1.0), 1.0e-4
        )

    def test__interpolated_errors__also_on_image_grid__interpolates_values(self):

        conf.instance = conf.Config(
            path.join(directory, "files/inversion_image_grid"),
            path.join(directory, "output"),
        )

        matrix_shape = (3, 3)

        mask = aa.Mask.manual(
            mask=np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, True, True],
                    [True, True, True, True, True],
                ]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.Grid.from_mask(mask=mask)

        pixelization_grid = aa.Grid.uniform(
            shape_2d=(3, 3), pixel_scales=1.0, sub_size=1
        )

        inversion = inversions.InversionImagingMatrix.from_data_mapper_and_regularization(
            image=np.ones(25),
            noise_map=np.ones(25),
            convolver=mock.MockConvolver(matrix_shape),
            mapper=mock.MockMapper(
                matrix_shape=matrix_shape,
                grid=grid,
                pixelization_grid=pixelization_grid,
            ),
            regularization=mock.MockRegularization(matrix_shape),
            settings=aa.SettingsInversion(check_solution=False),
        )

        inversion.reconstruction = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

        inversion.curvature_reg_matrix = np.eye(N=9)

        interpolated_errors = inversion.interpolated_errors_from_shape_2d()

        assert interpolated_errors.shape_2d == (5, 5)

        assert interpolated_errors.in_1d == pytest.approx(np.ones(shape=(25,)), 1.0e-4)
        assert interpolated_errors.in_2d == pytest.approx(np.ones(shape=(5, 5)), 1.0e-4)
        assert interpolated_errors.pixel_scales == pytest.approx((1.0, 1.0), 1.0e-4)

    def test__interpolated_reconstruction__config_is_source_grid__grid_is_zoomed_as_uses_mapper_grid(
        self
    ):
        conf.instance = conf.Config(
            path.join(directory, "files/inversion_source_grid"),
            path.join(directory, "output"),
        )

        matrix_shape = (3, 3)

        mask = aa.Mask.manual(
            mask=np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, True, True],
                    [True, True, True, True, True],
                ]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.Grid.from_mask(mask=mask)

        pixelization_grid = aa.Grid.uniform(
            shape_2d=(3, 3), pixel_scales=1.0, sub_size=1
        )

        inversion = inversions.InversionImagingMatrix.from_data_mapper_and_regularization(
            image=np.ones(25),
            noise_map=np.ones(25),
            convolver=mock.MockConvolver(matrix_shape),
            mapper=mock.MockMapper(
                matrix_shape=matrix_shape,
                grid=grid,
                pixelization_grid=pixelization_grid,
            ),
            regularization=mock.MockRegularization(matrix_shape),
            settings=aa.SettingsInversion(check_solution=False),
        )

        inversion.reconstruction = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

        interpolated_reconstruction = (
            inversion.interpolated_reconstruction_from_shape_2d()
        )

        assert (
            interpolated_reconstruction.in_1d
            == np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ).all()
        assert (
            interpolated_reconstruction.in_2d
            == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()
        assert interpolated_reconstruction.pixel_scales == pytest.approx(
            (0.66666, 0.66666), 1.0e-4
        )

        interpolated_reconstruction = inversion.interpolated_reconstruction_from_shape_2d(
            shape_2d=(2, 2)
        )

        assert (
            interpolated_reconstruction.in_1d == np.array([1.0, 1.0, 1.0, 1.0])
        ).all()
        assert (
            interpolated_reconstruction.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])
        ).all()
        assert interpolated_reconstruction.pixel_scales == pytest.approx(
            (1.0, 1.0), 1.0e-4
        )

        inversion.reconstruction = np.array(
            [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0]
        )

        interpolated_reconstruction = inversion.interpolated_reconstruction_from_shape_2d(
            shape_2d=(2, 2)
        )

        assert (
            interpolated_reconstruction.in_1d == np.array([3.0, 3.0, 3.0, 3.0])
        ).all()
        assert (
            interpolated_reconstruction.in_2d == np.array([[3.0, 3.0], [3.0, 3.0]])
        ).all()
        assert interpolated_reconstruction.pixel_scales == (1.0, 1.0)

    def test__interp__manual_shape_2d__uses_input_shape_2d(self):

        matrix_shape = (3, 3)

        mask = aa.Mask.manual(
            mask=np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, True, True],
                    [True, True, True, True, True],
                ]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.Grid.from_mask(mask=mask)

        pixelization_grid = aa.Grid.uniform(
            shape_2d=(3, 3), pixel_scales=1.0, sub_size=1
        )

        inversion = inversions.InversionImagingMatrix.from_data_mapper_and_regularization(
            image=np.ones(25),
            noise_map=np.ones(25),
            convolver=mock.MockConvolver(matrix_shape),
            mapper=mock.MockMapper(
                matrix_shape=matrix_shape,
                grid=grid,
                pixelization_grid=pixelization_grid,
            ),
            regularization=mock.MockRegularization(matrix_shape),
            settings=aa.SettingsInversion(check_solution=False),
        )

        inversion.reconstruction = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

        interpolated_reconstruction = inversion.interpolated_reconstruction_from_shape_2d(
            shape_2d=(2, 2)
        )

        assert (
            interpolated_reconstruction.in_1d == np.array([1.0, 1.0, 1.0, 1.0])
        ).all()
        assert (
            interpolated_reconstruction.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])
        ).all()
        assert interpolated_reconstruction.pixel_scales == pytest.approx(
            (1.0, 1.0), 1.0e-4
        )

        inversion.reconstruction = np.array(
            [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0]
        )

        interpolated_reconstruction = inversion.interpolated_reconstruction_from_shape_2d(
            shape_2d=(2, 2)
        )

        assert (
            interpolated_reconstruction.in_1d == np.array([3.0, 3.0, 3.0, 3.0])
        ).all()
        assert (
            interpolated_reconstruction.in_2d == np.array([[3.0, 3.0], [3.0, 3.0]])
        ).all()
        assert interpolated_reconstruction.pixel_scales == (1.0, 1.0)


class TestInversionImagingMatrix:
    def test__reconstruction_all_1s__simple_blurred_mapping_matrix__correct_reconstructed_image(
        self
    ):

        matrix_shape = (3, 3)

        mask = aa.Mask.manual(
            mask=np.array(
                [[True, True, True], [False, False, False], [True, True, True]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.Grid.from_mask(mask=mask)

        inversion = inversions.InversionImagingMatrix.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock.MockConvolver(matrix_shape),
            mapper=mock.MockMapper(matrix_shape=matrix_shape, grid=grid),
            regularization=mock.MockRegularization(matrix_shape),
            settings=aa.SettingsInversion(check_solution=False),
        )

        inversion.reconstruction = np.array([1.0, 1.0, 1.0, 1.0])

        inversion.blurred_mapping_matrix = np.array(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]]
        )
        # Imaging pixel 0 maps to 4 pixs pixxels -> value is 4.0
        # Imaging pixel 1 maps to 3 pixs pixxels -> value is 3.0
        # Imaging pixel 2 maps to 1 pixs pixxels -> value is 1.0

        assert (inversion.mapped_reconstructed_image == np.array([4.0, 3.0, 1.0])).all()
        assert (
            inversion.mapped_reconstructed_image.in_2d
            == np.array([[0.0, 0.0, 0.0], [4.0, 3.0, 1.0], [0.0, 0.0, 0.0]])
        ).all()

        assert inversion.errors_with_covariance == pytest.approx(
            np.array([[0.7, -0.3, -0.3], [-0.3, 0.7, -0.3], [-0.3, -0.3, 0.7]]), 1.0e-4
        )
        assert inversion.errors == pytest.approx(np.array([0.7, 0.7, 0.7]), 1.0e-4)

    def test__reconstruction_different_values__simple_blurred_mapping_matrix__correct_reconstructed_image(
        self
    ):

        matrix_shape = (3, 3)

        mask = aa.Mask.manual(
            mask=np.array(
                [[True, True, True], [False, False, False], [True, True, True]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.Grid.from_mask(mask=mask)

        inversion = inversions.InversionImagingMatrix.from_data_mapper_and_regularization(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock.MockConvolver(matrix_shape),
            mapper=mock.MockMapper(matrix_shape=matrix_shape, grid=grid),
            regularization=mock.MockRegularization(matrix_shape),
            settings=aa.SettingsInversion(check_solution=False),
        )

        inversion.reconstruction = np.array([1.0, 2.0, 3.0, 4.0])

        inversion.blurred_mapping_matrix = np.array(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]]
        )

        # # Imaging pixel 0 maps to 4 pixs pixxels -> value is 1.0 + 2.0 + 3.0 + 4.0 = 10.0
        # # Imaging pixel 1 maps to 3 pixs pixxels -> value is 1.0 + 3.0 + 4.0
        # # Imaging pixel 2 maps to 1 pixs pixxels -> value is 1.0

        assert (
            inversion.mapped_reconstructed_image == np.array([10.0, 8.0, 1.0])
        ).all()
        assert (
            inversion.mapped_reconstructed_image.in_2d
            == np.array([[0.0, 0.0, 0.0], [10.0, 8.0, 1.0], [0.0, 0.0, 0.0]])
        ).all()

        assert inversion.errors_with_covariance == pytest.approx(
            np.array([[0.7, -0.3, -0.3], [-0.3, 0.7, -0.3], [-0.3, -0.3, 0.7]]), 1.0e-4
        )
        assert inversion.errors == pytest.approx(np.array([0.7, 0.7, 0.7]), 1.0e-4)

    def test__func_testing_range_of_reconstruction(self):

        mask = aa.Mask.manual(
            mask=np.array(
                [[True, True, True], [False, False, False], [True, True, True]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.Grid.from_mask(mask=mask)

        matrix_shape = (3, 3)

        with pytest.raises(exc.InversionException):

            inversions.InversionImagingMatrix.from_data_mapper_and_regularization(
                image=np.ones(9),
                noise_map=np.ones(9),
                convolver=mock.MockConvolver(matrix_shape),
                mapper=mock.MockMapper(matrix_shape=matrix_shape, grid=grid),
                regularization=mock.MockRegularization(matrix_shape),
            )


from autoconf import conf
from os import path
import pytest


directory = path.dirname(path.realpath(__file__))


#
# class TestPixelizationQuantities:
#
#     def test__compare_to_inversion_utils(self):
#
#         matrix_shape = (3,3)
#
#         mask = aa.Mask.manual(array=np.array([[True, True, True],
#                                         [False, False, False],
#                                         [True, True, True]]), pixel_scales=1.0)
#
#         grid = aa.grid_stack_from_mask_sub_size_and_psf_shape(
#             mask=mask, sub_size=1, psf_shape_2d=(1,1))
#
#         inversion = inversions.Inversion(
#             image_1d=np.ones(9), noise_map_1d=np.ones(9), convolver=mock.MockConvolver(matrix_shape),
#             mapper=mock.MockMapper(matrix_shape, grid),
#             regularization=mock.MockRegularization(matrix_shape))
#
#         inversion.reconstruction = np.array([1.0, 1.0, 1.0, 1.0])
#
#         inversion.blurred_mapping_matrix = np.array([[1.0, 1.0, 1.0, 1.0],
#                                                [1.0, 0.0, 1.0, 1.0],
#                                                [1.0, 0.0, 0.0, 0.0]])
#
#         pixelization_residuals_util = \
#             am.util.inversion.pixelization_residuals_from_pixelization_values_reconstructed_data_1d_and_mapping_quantities(
#                 pixelization_values=inversion.reconstruction, reconstructed_data_1d=inversion.reconstructed_data_1d,
#                 mask_1d_index_for_sub_mask_1d_index=inversion.mapper.mask_1d_index_for_sub_mask_1d_index, all_sub_mask_1d_indexes_for_pixelization_1d_index=inversion.mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index)
