import numpy as np
import pytest

import autoarray as aa


def test__in_grid_1d__out_ndarray_2d():
    mask_1d = aa.Mask1D(mask=[True, False, False, True], pixel_scales=(1.0,))

    grid_1d = aa.Grid1D.from_mask(mask=mask_1d)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_2d = obj.ndarray_2d_from(grid=grid_1d)

    assert isinstance(ndarray_2d, aa.Grid2D)
    assert ndarray_2d.native == pytest.approx(
        np.array([[[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]]]), abs=1.0e-4
    )


def test__in_dgrid_1d__out_ndarray_2d_list():
    mask = aa.Mask1D(mask=[True, False, False, True], pixel_scales=(1.0,))

    grid_1d = aa.Grid1D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_2d_list = obj.ndarray_2d_list_from(grid=grid_1d)

    assert isinstance(ndarray_2d_list[0], aa.Grid2D)
    assert ndarray_2d_list[0].native == pytest.approx(
        np.array([[[0.0, 0.0], [0.0, -0.5], [0.0, 0.5], [0.0, 0.0]]]), abs=1.0e-4
    )

    assert isinstance(ndarray_2d_list[1], aa.Grid2D)
    assert ndarray_2d_list[1].native == pytest.approx(
        np.array([[[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]]]), abs=1.0e-4
    )


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


def test__in_grid_2d_irregular__out_ndarray_2d():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d_irregular = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_2d = obj.ndarray_2d_from(grid=grid_2d_irregular)

    assert ndarray_2d.in_list == [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]


def test__in_grid_2d_irregular__out_ndarray_2d_list():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_2d_list = obj.ndarray_2d_list_from(grid=grid_2d)

    assert ndarray_2d_list[0].in_list == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    assert ndarray_2d_list[1].in_list == [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]
