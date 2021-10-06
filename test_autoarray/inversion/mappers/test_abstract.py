import numpy as np

import autoarray as aa

from autoarray.mock.mock import MockMapper


def test__pixelization_indexes_for_slim_indexes__different_types_of_lists_input():

    mapper = MockMapper(
        pixelization_index_for_sub_slim_index=[0, 0, 0, 0, 0, 0, 0, 0], pixels=9
    )

    full_indexes = mapper.pixelization_indexes_for_slim_indexes(
        pixelization_indexes=[0, 1]
    )

    assert full_indexes == [0, 1, 2, 3, 4, 5, 6, 7]

    mapper = MockMapper(
        pixelization_index_for_sub_slim_index=[0, 0, 0, 0, 3, 4, 4, 7], pixels=9
    )

    full_indexes = mapper.pixelization_indexes_for_slim_indexes(
        pixelization_indexes=[[0], [4]]
    )

    assert full_indexes == [[0, 1, 2, 3], [5, 6]]


def test__adaptive_pixel_signals_from___matches_util(grid_2d_7x7, image_7x7):

    pixels = 6
    signal_scale = 2.0
    pixelization_index_for_sub_slim_index = np.array([1, 1, 4, 0, 0, 3, 0, 0, 3])

    mapper = MockMapper(
        source_grid_slim=grid_2d_7x7,
        pixelization_index_for_sub_slim_index=pixelization_index_for_sub_slim_index,
        hyper_image=image_7x7,
        pixels=pixels,
    )

    pixel_signals = mapper.pixel_signals_from(signal_scale=2.0)

    pixel_signals_util = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=pixels,
        signal_scale=signal_scale,
        pixelization_index_for_sub_slim_index=pixelization_index_for_sub_slim_index,
        slim_index_for_sub_slim_index=grid_2d_7x7.mask.slim_index_for_sub_slim_index,
        hyper_image=image_7x7,
    )

    assert (pixel_signals == pixel_signals_util).all()
