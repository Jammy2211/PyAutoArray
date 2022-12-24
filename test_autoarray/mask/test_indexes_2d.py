import os
from os import path
import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray import exc


def test__native_index_for_slim_index():

    mask = aa.Mask2D.manual(
        mask=[[True, True, True], [True, False, False], [True, True, False]],
        pixel_scales=1.0,
    )

    indexes_2d = aa.Indexes2D(mask=mask)

    sub_native_index_for_sub_slim_index_2d = (
        aa.util.mask_2d.native_index_for_slim_index_2d_from(mask_2d=mask, sub_size=1)
    )

    assert indexes_2d.native_index_for_slim_index == pytest.approx(
        sub_native_index_for_sub_slim_index_2d, 1e-4
    )


def test__unmasked_1d_indexes():
    mask = aa.Mask2D.manual(
        mask=[
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    indexes_2d = aa.Indexes2D(mask=mask)

    unmasked_pixels_util = aa.util.mask_2d.mask_1d_indexes_from(
        mask_2d=mask, return_masked_indexes=False
    )

    assert indexes_2d.unmasked_1d_indexes == pytest.approx(unmasked_pixels_util, 1e-4)


def test__masked_1d_indexes():
    mask = aa.Mask2D.manual(
        mask=[
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    indexes_2d = aa.Indexes2D(mask=mask)

    masked_pixels_util = aa.util.mask_2d.mask_1d_indexes_from(
        mask_2d=mask, return_masked_indexes=True
    )

    assert indexes_2d.masked_1d_indexes == pytest.approx(masked_pixels_util, 1e-4)


def test__edge_1d_indexes():
    mask = aa.Mask2D.manual(
        mask=[
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    indexes_2d = aa.Indexes2D(mask=mask)

    edge_1d_indexes_util = aa.util.mask_2d.edge_1d_indexes_from(mask_2d=mask)

    assert indexes_2d.edge_1d_indexes == pytest.approx(edge_1d_indexes_util, 1e-4)
    assert indexes_2d.edge_1d_indexes.shape[0] == indexes_2d.edge_2d_indexes.shape[0]


def test__edge_2d_indexes():
    mask = aa.Mask2D.manual(
        mask=[
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    indexes_2d = aa.Indexes2D(mask=mask)

    edge_1d_indexes_util = aa.util.mask_2d.edge_1d_indexes_from(mask_2d=mask)

    assert indexes_2d.edge_2d_indexes[0] == pytest.approx(np.array([1, 1]), 1e-4)
    assert indexes_2d.edge_2d_indexes[10] == pytest.approx(np.array([3, 3]), 1e-4)
