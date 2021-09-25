import autoarray as aa
import numpy as np

from autoarray.mock.mock import MockPixelizationGrid, MockMapper


def test__weight_list__matches_util():

    reg = aa.reg.AdaptiveBrightness(inner_coefficient=10.0, outer_coefficient=15.0)

    pixel_signals = np.array([0.21, 0.586, 0.45])

    mapper = MockMapper(pixel_signals=pixel_signals)

    weight_list = reg.regularization_weights_from_mapper(mapper=mapper)

    weight_list_util = aa.util.regularization.adaptive_regularization_weight_list_from(
        inner_coefficient=10.0, outer_coefficient=15.0, pixel_signals=pixel_signals
    )

    assert (weight_list == weight_list_util).all()


def test__regularization_matrix__matches_util():

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

    pixel_neighbors_sizes = np.array([2, 3, 4, 2, 4, 3])
    pixel_signals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    pixelization_grid = MockPixelizationGrid(
        pixel_neighbors=pixel_neighbors, pixel_neighbors_sizes=pixel_neighbors_sizes
    )

    mapper = MockMapper(
        source_pixelization_grid=pixelization_grid, pixel_signals=pixel_signals
    )

    regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

    regularization_weight_list = aa.util.regularization.adaptive_regularization_weight_list_from(
        pixel_signals=pixel_signals, inner_coefficient=1.0, outer_coefficient=2.0
    )

    regularization_matrix_util = aa.util.regularization.weighted_regularization_matrix_from(
        regularization_weight_list=regularization_weight_list,
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_sizes=pixel_neighbors_sizes,
    )

    assert (regularization_matrix == regularization_matrix_util).all()
