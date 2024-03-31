import os
import numpy as np
import pytest

import autoarray as aa
from autoarray.structures.mock.mock_structure_decorators import (
    ndarray_1d_from,
    ndarray_2d_from,
)


def test__in_grid_1d__out_ndarray_1d():
    grid_1d = aa.Grid1D.no_mask(values=[1.0, 2.0, 3.0], pixel_scales=1.0)

    obj = aa.m.MockGrid1DLikeObj()

    ndarray_1d = obj.ndarray_1d_from(grid=grid_1d)

    assert isinstance(ndarray_1d, aa.Array1D)
    assert (ndarray_1d.native == np.array([1.0, 1.0, 1.0])).all()
    assert ndarray_1d.pixel_scales == (1.0,)

    obj = aa.m.MockGrid1DLikeObj(centre=(1.0, 0.0), angle=45.0)

    ndarray_1d = obj.ndarray_1d_from(grid=grid_1d)

    assert isinstance(ndarray_1d, aa.Array1D)
    assert (ndarray_1d.native == np.array([1.0, 1.0, 1.0])).all()
    assert ndarray_1d.pixel_scales == (1.0,)

    mask_1d = aa.Mask1D(mask=[True, False, False, True], pixel_scales=(1.0,))

    grid_1d = aa.Grid1D.from_mask(mask=mask_1d)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_1d = obj.ndarray_1d_from(grid=grid_1d)

    assert isinstance(ndarray_1d, aa.Array1D)
    assert (ndarray_1d.native == np.array([0.0, 1.0, 1.0, 0.0])).all()


def test__in_grid_1d__out_ndarray_2d():
    mask_1d = aa.Mask1D(mask=[True, False, False, True], pixel_scales=(1.0,))

    grid_1d = aa.Grid1D.from_mask(mask=mask_1d)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_2d = obj.ndarray_2d_from(grid=grid_1d)

    assert isinstance(ndarray_2d, aa.Grid2D)
    assert ndarray_2d.native == pytest.approx(
        np.array([[[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]]]), 1.0e-4
    )


def test__in_grid_1d__out_ndarray_1d_list():
    mask = aa.Mask1D(mask=[True, False, False, True], pixel_scales=(1.0,))

    grid_1d = aa.Grid1D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_1d_list = obj.ndarray_1d_list_from(grid=grid_1d)

    assert isinstance(ndarray_1d_list[0], aa.Array1D)
    assert (ndarray_1d_list[0].native == np.array([[0.0, 1.0, 1.0, 0.0]])).all()

    assert isinstance(ndarray_1d_list[1], aa.Array1D)
    assert (ndarray_1d_list[1].native == np.array([[0.0, 2.0, 2.0, 0.0]])).all()


def test__in_grid_1d__out_ndarray_2d_list():
    mask = aa.Mask1D(mask=[True, False, False, True], pixel_scales=(1.0,))

    grid_1d = aa.Grid1D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_2d_list = obj.ndarray_2d_list_from(grid=grid_1d)

    assert isinstance(ndarray_2d_list[0], aa.Grid2D)
    assert ndarray_2d_list[0].native == pytest.approx(
        np.array([[[0.0, 0.0], [0.0, -0.5], [0.0, 0.5], [0.0, 0.0]]]), 1.0e-4
    )

    assert isinstance(ndarray_2d_list[1], aa.Grid2D)
    assert ndarray_2d_list[1].native == pytest.approx(
        np.array([[[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]]]), 1.0e-4
    )


def test__in_grid_2d_out_ndarray_1d():
    grid_2d = aa.Grid2D.uniform(shape_native=(4, 4), pixel_scales=1.0)

    obj = aa.m.MockGrid1DLikeObj()

    ndarray_1d = obj.ndarray_1d_from(grid=grid_2d)

    assert isinstance(ndarray_1d, aa.Array1D)
    assert (ndarray_1d.native == np.array([1.0])).all()
    assert ndarray_1d.pixel_scales == (1.0,)

    obj = aa.m.MockGrid1DLikeObj(centre=(1.0, 0.0))

    ndarray_1d = obj.ndarray_1d_from(grid=grid_2d)

    assert isinstance(ndarray_1d, aa.Array1D)
    assert (ndarray_1d.native == np.array([1.0, 1.0, 1.0, 1.0])).all()
    assert ndarray_1d.pixel_scales == (1.0,)

    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_1d = obj.ndarray_1d_from(grid=grid_2d)

    assert isinstance(ndarray_1d, aa.Array2D)
    assert (
        ndarray_1d.native
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()


def test__in_grid_2d__out_ndarray_2d():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_2d = obj.ndarray_2d_from(grid=grid_2d)

    assert isinstance(ndarray_2d, aa.Grid2D)
    assert (
        ndarray_2d.native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__in_grid_2d__out_ndarray_1d_list():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_1d_list = obj.ndarray_1d_list_from(grid=grid_2d)

    assert isinstance(ndarray_1d_list[0], aa.Array2D)
    assert (
        ndarray_1d_list[0].native
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    assert isinstance(ndarray_1d_list[1], aa.Array2D)
    assert (
        ndarray_1d_list[1].native
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()


def test__in_grid_2d__out_ndarray_2d_list():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_2d_list = obj.ndarray_2d_list_from(grid=grid_2d)

    assert isinstance(ndarray_2d_list[0], aa.Grid2D)
    assert (
        ndarray_2d_list[0].native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.5, -0.5], [0.5, 0.5], [0.0, 0.0]],
                [[0.0, 0.0], [-0.5, -0.5], [-0.5, 0.5], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()

    assert isinstance(ndarray_2d_list[1], aa.Grid2D)
    assert (
        ndarray_2d_list[1].native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__in_grid_2d__out_ndarray_yx_2d():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_yx_2d = obj.ndarray_yx_2d_from(grid=grid_2d)

    assert isinstance(ndarray_yx_2d, aa.VectorYX2D)
    assert (
        ndarray_yx_2d.native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__in_grid_2d__out_ndarray_yx_2d_list():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_yx_2d = obj.ndarray_yx_2d_list_from(grid=grid_2d)

    assert isinstance(ndarray_yx_2d[0], aa.VectorYX2D)
    assert (
        ndarray_yx_2d[0].native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.5, -0.5], [0.5, 0.5], [0.0, 0.0]],
                [[0.0, 0.0], [-0.5, -0.5], [-0.5, 0.5], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()

    assert isinstance(ndarray_yx_2d[1], aa.VectorYX2D)
    assert (
        ndarray_yx_2d[1].native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__in_grid_2d_irregular__out_ndarray_1d():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d_irregular = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_1d = obj.ndarray_1d_from(grid=grid_2d_irregular)

    assert isinstance(ndarray_1d, aa.ArrayIrregular)
    assert ndarray_1d.in_list == [1.0, 1.0, 1.0]


def test__in_grid_2d_irregular__out_ndarray_2d():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d_irregular = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_2d = obj.ndarray_2d_from(grid=grid_2d_irregular)

    assert ndarray_2d.in_list == [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]


def test__in_grid_2d_irregular__out_ndarray_1d_list():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_1d_list = obj.ndarray_1d_list_from(grid=grid_2d)

    assert ndarray_1d_list[0].in_list == [1.0, 1.0, 1.0]
    assert ndarray_1d_list[1].in_list == [2.0, 2.0, 2.0]


def test__in_grid_2d_irregular__out_ndarray_2d_list():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_2d_list = obj.ndarray_2d_list_from(grid=grid_2d)

    assert ndarray_2d_list[0].in_list == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    assert ndarray_2d_list[1].in_list == [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]


def test__in_grid_2d_irregular__out_ndarray_yx_2d():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_yx_2d = obj.ndarray_yx_2d_from(grid=grid_2d)

    assert isinstance(ndarray_yx_2d, aa.VectorYX2DIrregular)
    assert ndarray_yx_2d.in_list == [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]


def test__in_grid_2d_irregular__out_ndarray_yx_2d_list():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_yx_2d_list = obj.ndarray_yx_2d_list_from(grid=grid_2d)

    assert isinstance(ndarray_yx_2d_list[0], aa.VectorYX2DIrregular)
    assert ndarray_yx_2d_list[0].in_list == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]

    assert isinstance(ndarray_yx_2d_list[1], aa.VectorYX2DIrregular)
    assert ndarray_yx_2d_list[1].in_list == [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]


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

    over_sample = aa.OverSampleUniform(sub_size=1)

    grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample=over_sample)

    obj = aa.m.MockGridLikeIteratorObj()

    ndarray_1d = obj.ndarray_1d_over_sample_from(grid=grid_2d)
    ndarray_1d_via_grid = obj.ndarray_1d_over_sample_from(np.array(grid_2d))

    assert isinstance(ndarray_1d, aa.Array2D)
    assert (ndarray_1d == ndarray_1d_via_grid).all()

    over_sample = aa.OverSampleUniform(sub_size=2)

    grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample=over_sample)

    obj = aa.m.MockGridLikeIteratorObj()

    ndarray_1d = obj.ndarray_1d_over_sample_from(grid=grid_2d)

    over_sample_uniform = aa.OverSampleUniformFunc(
        mask=mask, sub_size=2
    )

    ndarray_1d_via_grid = obj.ndarray_1d_over_sample_from(np.array(over_sample_uniform.oversampled_grid))

    mask_sub_2 = aa.util.mask_2d.oversample_mask_2d_from(
        mask=np.array(mask), sub_size=2
    )
    mask_sub_2 = aa.Mask2D(mask=mask_sub_2, pixel_scales=(0.5, 0.5))
    ndarray_1d_via_grid = aa.Array2D(values=ndarray_1d_via_grid, mask=mask_sub_2)
    ndarray_1d_via_grid = over_sample_uniform.binned_array_2d_from(
        array=ndarray_1d_via_grid,
    )

    assert isinstance(ndarray_1d, aa.Array2D)
    assert (ndarray_1d == ndarray_1d_via_grid).all()


def test__in_grid_2d__over_sample_uniform__out_ndarray_1d_list():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    over_sample = aa.OverSampleUniform(sub_size=1)

    grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample=over_sample)

    obj = aa.m.MockGridLikeIteratorObj()

    ndarray_1d = obj.ndarray_1d_over_sample_list_from(grid=grid_2d)
    ndarray_1d_via_grid = obj.ndarray_1d_over_sample_list_from(np.array(grid_2d))

    assert isinstance(ndarray_1d[0], aa.Array2D)
    assert (ndarray_1d[0] == ndarray_1d_via_grid[0]).all()

    over_sample = aa.OverSampleUniform(sub_size=2)

    grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample=over_sample)

    obj = aa.m.MockGridLikeIteratorObj()

    ndarray_1d = obj.ndarray_1d_over_sample_list_from(grid=grid_2d)

    over_sample_uniform = aa.OverSampleUniformFunc(
        mask=mask, sub_size=2
    )
    ndarray_1d_via_grid = obj.ndarray_1d_over_sample_from(np.array(over_sample_uniform.oversampled_grid))

    mask_sub_2 = aa.util.mask_2d.oversample_mask_2d_from(
        mask=np.array(mask), sub_size=2
    )
    mask_sub_2 = aa.Mask2D(mask=mask_sub_2, pixel_scales=(0.5, 0.5))
    ndarray_1d_via_grid = aa.Array2D(values=ndarray_1d_via_grid, mask=mask_sub_2)
    ndarray_1d_via_grid = over_sample_uniform.binned_array_2d_from(
        array=ndarray_1d_via_grid,
    )

    assert isinstance(ndarray_1d[0], aa.Array2D)
    assert (ndarray_1d[0] == ndarray_1d_via_grid).all()


def test__in_grid_2d_over_sample_iterate__out_ndarray_1d__values_use_iteration():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
        origin=(0.001, 0.001),
    )

    over_sample = aa.OverSampleIterate(fractional_accuracy=1.0, sub_steps=[2, 3])

    grid_2d = aa.Grid2D.from_mask(mask=mask, over_sample=over_sample)

    obj = aa.m.MockGridLikeIteratorObj()

    ndarray_1d = obj.ndarray_1d_over_sample_from(grid=grid_2d)

    over_sample_uniform = aa.OverSampleUniformFunc(
        mask=mask, sub_size=3
    )

    values_sub_3 = over_sample_uniform.evaluated_func_from(
        func=ndarray_1d_from,
    )

    assert ndarray_1d == pytest.approx(values_sub_3, 1.0e-4)

    grid_2d = aa.Grid2D.from_mask(
        mask=mask,
        over_sample=aa.OverSampleIterate(
            fractional_accuracy=0.000001, sub_steps=[2, 4, 8, 16, 32]
        ),
    )

    obj = aa.m.MockGridLikeIteratorObj()

    ndarray_1d = obj.ndarray_1d_over_sample_from(grid=grid_2d)

    over_sample_uniform = aa.OverSampleUniformFunc(
        mask=mask, sub_size=2
    )

    values_sub_2 = over_sample_uniform.evaluated_func_from(
        func=ndarray_1d_from,
    )

    assert ndarray_1d == pytest.approx(values_sub_2, 1.0e-4)

    grid_2d = aa.Grid2D.from_mask(
        mask=mask,
        over_sample=aa.OverSampleIterate(fractional_accuracy=0.5, sub_steps=[2, 4]),
    )

    iterate_obj = aa.m.MockGridLikeIteratorObj()

    ndarray_1d = iterate_obj.ndarray_1d_over_sample_from(grid=grid_2d)

    over_sample_uniform = aa.OverSampleUniformFunc(
        mask=mask, sub_size=2
    )
    values_sub_2 = over_sample_uniform.evaluated_func_from(
        func=ndarray_1d_from,
    )
    over_sample_uniform = aa.OverSampleUniformFunc(
        mask=mask, sub_size=4
    )
    values_sub_4 = over_sample_uniform.evaluated_func_from(
        func=ndarray_1d_from,
    )

    assert ndarray_1d.native[1, 1] == values_sub_2.native[1, 1]
    assert ndarray_1d.native[2, 2] != values_sub_2.native[2, 2]

    assert ndarray_1d.native[1, 1] != values_sub_4.native[1, 1]
    assert ndarray_1d.native[2, 2] == values_sub_4.native[2, 2]
