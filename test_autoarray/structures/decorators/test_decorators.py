import os
import numpy as np
import pytest

import autoarray as aa
from autoarray.structures.mock.mock_decorators import (
    ndarray_1d_from,
    ndarray_2d_from,
)


pass


# def test__in_grid_2d__over_sample_uniform__out_ndarray_1d():
#     mask = aa.Mask2D(
#         mask=[
#             [True, True, True, True],
#             [True, False, False, True],
#             [True, False, False, True],
#             [True, True, True, True],
#         ],
#         pixel_scales=(1.0, 1.0),
#     )
#
#     over_sample = aa.OverSamplingUniform(sub_size=1)
#
#     grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample=over_sample)
#
#     obj = aa.m.MockGridLikeIteratorObj()
#
#     ndarray_1d = obj.ndarray_1d_from(grid=grid_2d)
#     ndarray_1d_via_grid = obj.ndarray_1d_from(np.array(grid_2d))
#
#     assert isinstance(ndarray_1d, aa.Array2D)
#     assert (ndarray_1d == ndarray_1d_via_grid).all()
#
#     over_sample = aa.OverSamplingUniform(sub_size=2)
#
#     grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample=over_sample)
#
#     obj = aa.m.MockGridLikeIteratorObj()
#
#     ndarray_1d = obj.ndarray_1d_from(grid=grid_2d)
#
#     over_sample_uniform = aa.OverSamplerUniform(mask=mask, sub_size=2)
#
#     ndarray_1d_via_grid = obj.ndarray_1d_from(
#         np.array(over_sample_uniform.oversampled_grid)
#     )
#
#     mask_sub_2 = aa.util.mask_2d.oversample_mask_2d_from(
#         mask=np.array(mask), sub_size=2
#     )
#     mask_sub_2 = aa.Mask2D(mask=mask_sub_2, pixel_scales=(0.5, 0.5))
#     ndarray_1d_via_grid = aa.Array2D(values=ndarray_1d_via_grid, mask=mask_sub_2)
#     ndarray_1d_via_grid = over_sample_uniform.binned_array_2d_from(
#         array=ndarray_1d_via_grid,
#     )
#
#     assert isinstance(ndarray_1d, aa.Array2D)
#     assert (ndarray_1d == ndarray_1d_via_grid).all()
#
#
# def test__in_grid_2d__over_sample_uniform__out_ndarray_1d_list():
#     mask = aa.Mask2D(
#         mask=[
#             [True, True, True, True],
#             [True, False, False, True],
#             [True, False, False, True],
#             [True, True, True, True],
#         ],
#         pixel_scales=(1.0, 1.0),
#     )
#
#     over_sample = aa.OverSamplingUniform(sub_size=1)
#
#     grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample=over_sample)
#
#     obj = aa.m.MockGridLikeIteratorObj()
#
#     ndarray_1d = obj.ndarray_1d_list_from(grid=grid_2d)
#     ndarray_1d_via_grid = obj.ndarray_1d_list_from(np.array(grid_2d))
#
#     assert isinstance(ndarray_1d[0], aa.Array2D)
#     assert (ndarray_1d[0] == ndarray_1d_via_grid[0]).all()
#
#     over_sample = aa.OverSamplingUniform(sub_size=2)
#
#     grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample=over_sample)
#
#     obj = aa.m.MockGridLikeIteratorObj()
#
#     ndarray_1d = obj.ndarray_1d_list_from(grid=grid_2d)
#
#     over_sample_uniform = aa.OverSamplerUniform(mask=mask, sub_size=2)
#     ndarray_1d_via_grid = obj.ndarray_1d_from(
#         np.array(over_sample_uniform.oversampled_grid)
#     )
#
#     mask_sub_2 = aa.util.mask_2d.oversample_mask_2d_from(
#         mask=np.array(mask), sub_size=2
#     )
#     mask_sub_2 = aa.Mask2D(mask=mask_sub_2, pixel_scales=(0.5, 0.5))
#     ndarray_1d_via_grid = aa.Array2D(values=ndarray_1d_via_grid, mask=mask_sub_2)
#     ndarray_1d_via_grid = over_sample_uniform.binned_array_2d_from(
#         array=ndarray_1d_via_grid,
#     )
#
#     assert isinstance(ndarray_1d[0], aa.Array2D)
#     assert (ndarray_1d[0] == ndarray_1d_via_grid).all()
#
#
# def test__in_grid_2d_over_sample_iterate__out_ndarray_1d__values_use_iteration():
#     mask = aa.Mask2D(
#         mask=[
#             [True, True, True, True, True],
#             [True, False, False, False, True],
#             [True, False, False, False, True],
#             [True, False, False, False, True],
#             [True, True, True, True, True],
#         ],
#         pixel_scales=(1.0, 1.0),
#         origin=(0.001, 0.001),
#     )
#
#     over_sample = aa.OverSamplingIterate(fractional_accuracy=1.0, sub_steps=[2, 3])
#
#     grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample=over_sample)
#
#     obj = aa.m.MockGridLikeIteratorObj()
#
#     ndarray_1d = obj.ndarray_1d_from(grid=grid_2d)
#
#     over_sample_uniform = aa.OverSamplerUniform(mask=mask, sub_size=3)
#
#     values_sub_3 = over_sample_uniform.array_via_func_from(
#         func=ndarray_1d_from, cls=object
#     )
#
#     assert ndarray_1d == pytest.approx(values_sub_3, 1.0e-4)
#
#     grid_2d = aa.Grid2D.from_mask(
#         mask=mask,
#         over_sample=aa.OverSamplingIterate(
#             fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
#         ),
#     )
#
#     obj = aa.m.MockGridLikeIteratorObj()
#
#     ndarray_1d = obj.ndarray_1d_from(grid=grid_2d)
#
#     over_sample_uniform = aa.OverSamplerUniform(mask=mask, sub_size=2)
#
#     values_sub_2 = over_sample_uniform.array_via_func_from(
#         func=ndarray_1d_from, cls=object
#     )
#
#     assert ndarray_1d == pytest.approx(values_sub_2, 1.0e-4)
#
#     grid_2d = aa.Grid2D.from_mask(
#         mask=mask,
#         over_sample=aa.OverSamplingIterate(fractional_accuracy=0.5, sub_steps=[2, 4]),
#     )
#
#     iterate_obj = aa.m.MockGridLikeIteratorObj()
#
#     ndarray_1d = iterate_obj.ndarray_1d_from(grid=grid_2d)
#
#     over_sample_uniform = aa.OverSamplerUniform(mask=mask, sub_size=2)
#     values_sub_2 = over_sample_uniform.array_via_func_from(
#         func=ndarray_1d_from, cls=object
#     )
#     over_sample_uniform = aa.OverSamplerUniform(mask=mask, sub_size=4)
#     values_sub_4 = over_sample_uniform.array_via_func_from(
#         func=ndarray_1d_from, cls=object
#     )
#
#     assert ndarray_1d.native[1, 1] == values_sub_2.native[1, 1]
#     assert ndarray_1d.native[2, 2] != values_sub_2.native[2, 2]
#
#     assert ndarray_1d.native[1, 1] != values_sub_4.native[1, 1]
#     assert ndarray_1d.native[2, 2] == values_sub_4.native[2, 2]
