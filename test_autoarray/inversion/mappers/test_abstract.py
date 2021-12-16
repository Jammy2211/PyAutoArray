import numpy as np

import autoarray as aa

from autoarray.mock.mock import MockMapper
from autoarray.inversion.mappers.abstract import PixForSub


def test__pixelization_indexes_for_slim_indexes__different_types_of_lists_input():

    mapper = MockMapper(
        pixelization_indexes_for_sub_slim_index=PixForSub(
            mappings=np.array([[0], [0], [0], [0], [0], [0], [0], [0]]),
            sizes=np.ones(8, dtype='int')),
        pixels=9
    )

    full_indexes = mapper.pixelization_indexes_for_slim_indexes(
        pixelization_indexes=[0, 1]
    )

    assert full_indexes == [0, 1, 2, 3, 4, 5, 6, 7]

    mapper = MockMapper(
        pixelization_indexes_for_sub_slim_index=PixForSub(
            mappings=np.array([[0], [0], [0], [0], [3], [4], [4], [7]]),
            sizes=np.ones(8, dtype='int')),
        pixels=9
    )

    full_indexes = mapper.pixelization_indexes_for_slim_indexes(
        pixelization_indexes=[[0], [4]]
    )

    assert full_indexes == [[0, 1, 2, 3], [5, 6]]


def test__adaptive_pixel_signals_from___matches_util(grid_2d_7x7, image_7x7):

    pixels = 6
    signal_scale = 2.0
    pixelization_indexes_for_sub_slim_index = PixForSub(
            mappings=np.array([[1], [1], [4], [0], [0], [3], [0], [0], [3]]),
            sizes=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
            )
    pixelization_weights_for_sub_slim_index = np.ones((9, 1), dtype='int')

    mapper = MockMapper(
        source_grid_slim=grid_2d_7x7,
        pixelization_indexes_for_sub_slim_index=pixelization_indexes_for_sub_slim_index,
        pixelization_weights_for_sub_slim_index=pixelization_weights_for_sub_slim_index,
        hyper_image=image_7x7,
        pixels=pixels,
    )

    pixel_signals = mapper.pixel_signals_from(signal_scale=2.0)

    pixel_signals_util = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=pixels,
        pixel_weights=pixelization_weights_for_sub_slim_index,
        signal_scale=signal_scale,
        pixelization_indexes_for_sub_slim_index=pixelization_indexes_for_sub_slim_index.mappings,
        pixelization_size_for_sub_slim_index=pixelization_indexes_for_sub_slim_index.sizes,
        slim_index_for_sub_slim_index=grid_2d_7x7.mask.slim_index_for_sub_slim_index,
        hyper_image=image_7x7,
    )

    assert (pixel_signals == pixel_signals_util).all()
