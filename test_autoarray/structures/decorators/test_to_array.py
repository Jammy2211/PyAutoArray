import numpy as np

import autoarray as aa


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


def test__in_grid_1d__out_ndarray_1d_list():
    mask = aa.Mask1D(mask=[True, False, False, True], pixel_scales=(1.0,))

    grid_1d = aa.Grid1D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_1d_list = obj.ndarray_1d_list_from(grid=grid_1d)

    assert isinstance(ndarray_1d_list[0], aa.Array1D)
    assert (ndarray_1d_list[0].native == np.array([[0.0, 1.0, 1.0, 0.0]])).all()

    assert isinstance(ndarray_1d_list[1], aa.Array1D)
    assert (ndarray_1d_list[1].native == np.array([[0.0, 2.0, 2.0, 0.0]])).all()


def test__in_grid_2d__out_ndarray_1d():
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


def test__in_grid_2d_irregular__out_ndarray_1d():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d_irregular = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_1d = obj.ndarray_1d_from(grid=grid_2d_irregular)

    assert isinstance(ndarray_1d, aa.ArrayIrregular)
    assert ndarray_1d.in_list == [1.0, 1.0, 1.0]


def test__in_grid_2d_irregular__out_ndarray_1d_list():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_1d_list = obj.ndarray_1d_list_from(grid=grid_2d)

    assert ndarray_1d_list[0].in_list == [1.0, 1.0, 1.0]
    assert ndarray_1d_list[1].in_list == [2.0, 2.0, 2.0]
