import autoarray as aa
import numpy as np
import pytest


def test__data_to_pix_unique_from():

    image_pixels = 2
    sub_size = 2

    pix_index_for_sub_slim_index = np.array(
        [[0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1]]
    ).astype("int")
    pix_indexes_for_sub_slim_sizes = np.array([1, 1, 1, 1, 1, 1, 1, 1]).astype("int")
    pix_weights_for_sub_slim_index = np.array(
        [
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
        ]
    )

    data_to_pix_unique, data_weights, pix_lengths = aa.util.mapper.data_slim_to_pixelization_unique_from(
        data_pixels=image_pixels,
        pix_indexes_for_sub_slim_index=pix_index_for_sub_slim_index,
        pix_indexes_for_sub_slim_sizes=pix_indexes_for_sub_slim_sizes,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        sub_size=sub_size,
    )

    assert (data_to_pix_unique[0, :] == np.array([0, -1, -1, -1])).all()
    assert (data_to_pix_unique[1, :] == np.array([0, -1, -1, -1])).all()
    assert (data_weights[0, :] == np.array([1.0, 0.0, 0.0, 0.0])).all()
    assert (data_weights[1, :] == np.array([1.0, 0.0, 0.0, 0.0])).all()
    assert (pix_lengths == np.array([1, 1])).all()

    linear_obj = aa.LinearObjFunc(sub_slim_shape=8, sub_size=2)

    assert (
        linear_obj.data_unique_mappings.data_to_pix_unique == data_to_pix_unique
    ).all()
    assert (linear_obj.data_unique_mappings.data_weights == data_weights).all()
    assert (linear_obj.data_unique_mappings.pix_lengths == pix_lengths).all()
