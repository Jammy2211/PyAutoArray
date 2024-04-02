import os
import numpy as np
import pytest

import autoarray as aa
from autoarray.structures.mock.mock_decorators import (
    ndarray_1d_from,
    ndarray_2d_from,
    ndarray_1d_list_from,
    ndarray_2d_list_from,
    ndarray_2d_yx_from
)



def test__in_grid_2d_out_ndarray_1d():
    grid_2d = aa.Grid2D.uniform(shape_native=(4, 4), pixel_scales=1.0)

    obj = aa.m.MockGrid1DLikeObj()

    maker = aa.StructureMaker(
        func=ndarray_1d_from,
        obj=obj,
        grid=grid_2d
    )

    assert maker.result_type == "array"

    maker = aa.StructureMaker(
        func=ndarray_2d_from,
        obj=obj,
        grid=grid_2d
    )

    assert maker.result_type == "grid"

    maker = aa.StructureMaker(
        func=ndarray_1d_list_from,
        obj=obj,
        grid=grid_2d
    )

    assert maker.result_type == "array"

    maker = aa.StructureMaker(
        func=ndarray_2d_list_from,
        obj=obj,
        grid=grid_2d
    )

    assert maker.result_type == "grid"