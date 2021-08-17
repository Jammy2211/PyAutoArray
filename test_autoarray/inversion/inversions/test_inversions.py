import autoarray as aa
from autoarray.dataset.imaging import WTildeImaging
from autoarray.inversion.inversion.imaging import InversionImagingMatrix
from autoarray import exc
from autoarray.mock import mock

import numpy as np
from os import path
import pytest

directory = path.dirname(path.realpath(__file__))


class TestInversionImagingMatrix:
    def test__blurred_mapping_matrix_property(self, rectangular_inversion_7x7_3x3):

        assert rectangular_inversion_7x7_3x3.blurred_mapping_matrix[
            0, 0
        ] == pytest.approx(0.111111, 1e-4)

    def test__func_testing_range_of_reconstruction(self):

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [False, False, False], [True, True, True]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        matrix_shape = (9, 3)

        with pytest.raises(exc.InversionException):

            InversionImagingMatrix.from_data_via_pixelization_convolution(
                image=np.ones(9),
                noise_map=np.ones(9),
                convolver=mock.MockConvolver(matrix_shape),
                mapper=mock.MockMapper(
                    matrix_shape=matrix_shape, source_grid_slim=grid
                ),
                regularization=mock.MockRegularization(matrix_shape),
            )

    def test__w_tilde_checks_noise_map_and_raises_exception_if_preloads_dont_match_noise_map(
        self
    ):

        matrix_shape = (9, 3)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [False, False, False], [True, True, True]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        w_tilde = WTildeImaging(
            curvature_preload=None, indexes=None, lengths=None, noise_map_value=2.0
        )

        with pytest.raises(exc.InversionException):

            InversionImagingMatrix.from_data_via_w_tilde(
                image=np.ones(9),
                noise_map=np.ones(9),
                convolver=mock.MockConvolver(matrix_shape),
                w_tilde=w_tilde,
                mapper=mock.MockMapper(
                    matrix_shape=matrix_shape, source_grid_slim=grid
                ),
                regularization=mock.MockRegularization(matrix_shape),
                settings=aa.SettingsInversion(check_solution=False),
            )

    def test__preloads(self):

        matrix_shape = (9, 3)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [False, False, False], [True, True, True]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        blurred_mapping_matrix = 2.0 * np.ones(matrix_shape)

        curvature_matrix_sparse_preload, curvature_matrix_preload_counts = aa.util.inversion.curvature_matrix_sparse_preload_via_mapping_matrix_from(
            mapping_matrix=blurred_mapping_matrix
        )

        inversion = InversionImagingMatrix.from_data_via_pixelization_convolution(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock.MockConvolver(matrix_shape),
            mapper=mock.MockMapper(matrix_shape=matrix_shape, source_grid_slim=grid),
            regularization=mock.MockRegularization(matrix_shape),
            settings=aa.SettingsInversion(check_solution=False),
            preloads=aa.Preloads(
                blurred_mapping_matrix=blurred_mapping_matrix,
                curvature_matrix_sparse_preload=curvature_matrix_sparse_preload,
                curvature_matrix_preload_counts=curvature_matrix_preload_counts,
            ),
        )

        assert inversion.reconstruction == pytest.approx(
            np.array([0.16513, 0.16513, 0.16513]), 1.0e-4
        )
