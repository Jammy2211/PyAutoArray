import os
import numpy as np
import pytest

import autoarray as aa
from autoarray.structures.mock.mock_decorators import (
    ndarray_1d_from,
    ndarray_2d_from,
)


def test__in_grid_2d__over_sample_uniform__out_ndarray_1d():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    over_sampling = aa.OverSamplingUniform(sub_size=2)

    grid_2d = aa.Grid2D.from_mask(mask=mask, over_sampling=over_sampling)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_1d = obj.ndarray_1d_from(grid=grid_2d)

    over_sample_uniform = aa.OverSamplerUniform(mask=mask, sub_size=2)

    mask_sub_2 = aa.util.over_sample.oversample_mask_2d_from(
        mask=np.array(mask), sub_size=2
    )

    mask_sub_2 = aa.Mask2D(mask=mask_sub_2, pixel_scales=(0.5, 0.5))

    grid = aa.Grid2D(values=over_sample_uniform.over_sampled_grid, mask=mask_sub_2)

    ndarray_1d_via_grid = obj.ndarray_1d_from(grid=grid)

    ndarray_1d_via_grid = aa.Array2D(values=ndarray_1d_via_grid, mask=mask_sub_2)

    ndarray_1d_via_grid = over_sample_uniform.binned_array_2d_from(
        array=ndarray_1d_via_grid,
    )

    assert isinstance(ndarray_1d, aa.Array2D)
    assert (ndarray_1d == ndarray_1d_via_grid).all()
