import autoarray as aa
from autoarray.dataset.imaging import WTildeImaging
from autoarray.inversion.inversion.imaging import InversionImagingWTilde
from autoarray.inversion.inversion.imaging import InversionImagingMapping
from autoarray import exc
from autoarray.mock import mock

import numpy as np
from os import path
import pytest

directory = path.dirname(path.realpath(__file__))


class TestInversionImaging:
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

        pixel_neighbors = np.zeros((3, 3)).astype("int")
        pixel_neighbors_sizes = np.array([0, 0, 0]).astype("int")

        source_pixelization_grid = mock.MockPixelizationGrid(
            pixel_neighbors=pixel_neighbors, pixel_neighbors_sizes=pixel_neighbors_sizes
        )

        with pytest.raises(exc.InversionException):

            # noinspection PyTypeChecker
            inversion = InversionImagingMapping(
                image=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
                noise_map=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
                convolver=mock.MockConvolver(matrix_shape),
                mapper=mock.MockMapper(
                    mapping_matrix=np.ones(matrix_shape),
                    source_grid_slim=grid,
                    source_pixelization_grid=source_pixelization_grid,
                ),
                regularization=mock.MockRegularization(matrix_shape),
            )

            # noinspection PyStatementEffect
            inversion.reconstruction

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
            curvature_preload=None,
            indexes=None,
            lengths=None,
            noise_map_value=2.0,
            snr_cut=1.0e-10,
        )

        with pytest.raises(exc.InversionException):

            # noinspection PyTypeChecker
            InversionImagingWTilde(
                image=np.ones(9),
                noise_map=np.ones(9),
                convolver=mock.MockConvolver(matrix_shape),
                w_tilde=w_tilde,
                mapper=mock.MockMapper(
                    mapping_matrix=np.ones(matrix_shape), source_grid_slim=grid
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

        preloads = aa.Preloads(
            blurred_mapping_matrix=blurred_mapping_matrix,
            curvature_matrix_sparse_preload=curvature_matrix_sparse_preload.astype(
                "int"
            ),
            curvature_matrix_preload_counts=curvature_matrix_preload_counts.astype(
                "int"
            ),
        )

        pixel_neighbors = np.zeros((3, 3)).astype("int")
        pixel_neighbors_sizes = np.array([0, 0, 0]).astype("int")

        source_pixelization_grid = mock.MockPixelizationGrid(
            pixel_neighbors=pixel_neighbors, pixel_neighbors_sizes=pixel_neighbors_sizes
        )

        # noinspection PyTypeChecker
        inversion = InversionImagingMapping(
            image=np.ones(9),
            noise_map=np.ones(9),
            convolver=mock.MockConvolver(matrix_shape),
            mapper=mock.MockMapper(
                mapping_matrix=np.ones(matrix_shape),
                source_grid_slim=grid,
                source_pixelization_grid=source_pixelization_grid,
            ),
            regularization=mock.MockRegularization(matrix_shape),
            settings=aa.SettingsInversion(check_solution=False),
            preloads=preloads,
        )

        assert inversion.reconstruction == pytest.approx(
            np.array([0.16513, 0.16513, 0.16513]), 1.0e-4
        )
