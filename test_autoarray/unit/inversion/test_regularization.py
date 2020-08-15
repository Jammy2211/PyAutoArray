import autoarray as aa
import numpy as np

from test_autoarray.mock import MockPixelizationGrid, MockRegMapper


class TestRegularizationinstance:
    def test__regularization_matrix__compare_to_regularization_util(self):

        pixel_neighbors = np.array(
            [
                [1, 3, 7, 2],
                [4, 2, 0, -1],
                [1, 5, 3, -1],
                [4, 6, 0, -1],
                [7, 1, 5, 3],
                [4, 2, 8, -1],
                [7, 3, 0, -1],
                [4, 8, 6, -1],
                [7, 5, -1, -1],
            ]
        )

        pixel_neighbors_size = np.array([4, 3, 3, 3, 4, 3, 3, 3, 2])

        pixelization_grid = MockPixelizationGrid(
            pixel_neighbors=pixel_neighbors, pixel_neighbors_size=pixel_neighbors_size
        )

        mapper = MockRegMapper(pixelization_grid=pixelization_grid)

        reg = aa.reg.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

        regularization_matrix_util = aa.util.regularization.constant_regularization_matrix_from(
            coefficient=1.0,
            pixel_neighbors=pixel_neighbors,
            pixel_neighbors_size=pixel_neighbors_size,
        )

        assert (regularization_matrix == regularization_matrix_util).all()


class TestRegularizationWeighted:
    def test__weights__compare_to_regularization_util(self):

        reg = aa.reg.AdaptiveBrightness(inner_coefficient=10.0, outer_coefficient=15.0)

        pixel_signals = np.array([0.21, 0.586, 0.45])

        mapper = MockRegMapper(pixel_signals=pixel_signals)

        weights = reg.regularization_weights_from_mapper(mapper=mapper)

        weights_util = aa.util.regularization.adaptive_regularization_weights_from(
            inner_coefficient=10.0, outer_coefficient=15.0, pixel_signals=pixel_signals
        )

        assert (weights == weights_util).all()

    def test__regularization_matrix__compare_to_regularization_util(self):

        reg = aa.reg.AdaptiveBrightness(
            inner_coefficient=1.0, outer_coefficient=2.0, signal_scale=1.0
        )

        pixel_neighbors = np.array(
            [
                [1, 4, -1, -1],
                [2, 4, 0, -1],
                [3, 4, 5, 1],
                [5, 2, -1, -1],
                [5, 0, 1, 2],
                [2, 3, 4, -1],
            ]
        )

        pixel_neighbors_size = np.array([2, 3, 4, 2, 4, 3])
        pixel_signals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        pixelization_grid = MockPixelizationGrid(
            pixel_neighbors=pixel_neighbors, pixel_neighbors_size=pixel_neighbors_size
        )

        mapper = MockRegMapper(
            pixelization_grid=pixelization_grid, pixel_signals=pixel_signals
        )

        regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

        regularization_weights = aa.util.regularization.adaptive_regularization_weights_from(
            pixel_signals=pixel_signals, inner_coefficient=1.0, outer_coefficient=2.0
        )

        regularization_matrix_util = aa.util.regularization.weighted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            pixel_neighbors=pixel_neighbors,
            pixel_neighbors_size=pixel_neighbors_size,
        )

        assert (regularization_matrix == regularization_matrix_util).all()
