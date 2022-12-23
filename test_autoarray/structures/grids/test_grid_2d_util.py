import autoarray as aa
import numpy as np
import pytest


def test__grid_2d_slim_via_mask_from():

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 6.0), sub_size=1
    )

    assert (grid[0] == np.array([0.0, 0.0])).all()

    mask = np.array(
        [
            [True, False, True, True],
            [False, False, False, True],
            [True, False, True, False],
        ]
    )

    grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=1
    )

    assert (
        grid
        == np.array(
            [
                [3.0, -1.5],
                [0.0, -4.5],
                [0.0, -1.5],
                [0.0, 1.5],
                [-3.0, -1.5],
                [-3.0, 4.5],
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, False, True, True],
            [False, False, False, True],
            [True, False, True, False],
        ]
    )

    grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=1, origin=(1.0, 2.0)
    )

    assert grid == pytest.approx(
        np.array(
            [
                [4.0, 0.5],
                [1.0, -2.5],
                [1.0, 0.5],
                [1.0, 3.5],
                [-2.0, 0.5],
                [-2.0, 6.5],
            ]
        ),
        1e-4,
    )

    mask = np.array([[True, True, False], [False, False, False], [True, True, False]])

    grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2
    )

    assert (
        grid
        == np.array(
            [
                [3.75, 2.25],
                [3.75, 3.75],
                [2.25, 2.25],
                [2.25, 3.75],
                [0.75, -3.75],
                [0.75, -2.25],
                [-0.75, -3.75],
                [-0.75, -2.25],
                [0.75, -0.75],
                [0.75, 0.75],
                [-0.75, -0.75],
                [-0.75, 0.75],
                [0.75, 2.25],
                [0.75, 3.75],
                [-0.75, 2.25],
                [-0.75, 3.75],
                [-2.25, 2.25],
                [-2.25, 3.75],
                [-3.75, 2.25],
                [-3.75, 3.75],
            ]
        )
    ).all()

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=3
    )

    assert (
        grid
        == np.array(
            [
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, False],
            [True, False, False, True],
            [False, True, False, True],
        ]
    )

    grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2
    )

    assert (
        grid
        == np.array(
            [
                [3.75, 3.75],
                [3.75, 5.25],
                [2.25, 3.75],
                [2.25, 5.25],
                [0.75, -2.25],
                [0.75, -0.75],
                [-0.75, -2.25],
                [-0.75, -0.75],
                [0.75, 0.75],
                [0.75, 2.25],
                [-0.75, 0.75],
                [-0.75, 2.25],
                [-2.25, -5.25],
                [-2.25, -3.75],
                [-3.75, -5.25],
                [-3.75, -3.75],
                [-2.25, 0.75],
                [-2.25, 2.25],
                [-3.75, 0.75],
                [-3.75, 2.25],
            ]
        )
    ).all()

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 6.0), sub_size=2, origin=(1.0, 1.0)
    )

    assert grid[0:4] == pytest.approx(
        np.array([[1.75, -0.5], [1.75, 2.5], [0.25, -0.5], [0.25, 2.5]]), 1e-4
    )


def test__grid_2d_via_mask_from():

    mask = np.array([[False, True, True], [True, True, False], [True, True, True]])

    grid_2d = aa.util.grid_2d.grid_2d_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 6.0), sub_size=1
    )

    assert (
        grid_2d
        == np.array(
            [
                [[3.0, -6.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 6.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()

    mask = np.array([[False, True], [True, False]])

    grid_2d = aa.util.grid_2d.grid_2d_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 6.0), sub_size=2
    )

    assert (
        grid_2d
        == np.array(
            [
                [[2.25, -4.5], [2.25, -1.5], [0.0, 0.0], [0.0, 0.0]],
                [[0.75, -4.5], [0.75, -1.5], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [-0.75, 1.5], [-0.75, 4.5]],
                [[0.0, 0.0], [0.0, 0.0], [-2.25, 1.5], [-2.25, 4.5]],
            ]
        )
    ).all()


def test__grid_2d_slim_via_shape_native_from():

    grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(2, 3), pixel_scales=(1.0, 1.0), sub_size=1
    )

    assert (
        grid_2d
        == np.array(
            [
                [0.5, -1.0],
                [0.5, 0.0],
                [0.5, 1.0],
                [-0.5, -1.0],
                [-0.5, 0.0],
                [-0.5, 1.0],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(3, 2), pixel_scales=(1.0, 1.0), sub_size=1
    )

    assert (
        grid_2d
        == np.array(
            [
                [1.0, -0.5],
                [1.0, 0.5],
                [0.0, -0.5],
                [0.0, 0.5],
                [-1.0, -0.5],
                [-1.0, 0.5],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(3, 2), pixel_scales=(1.0, 1.0), sub_size=1, origin=(3.0, -2.0)
    )

    assert (
        grid_2d
        == np.array(
            [
                [4.0, -2.5],
                [4.0, -1.5],
                [3.0, -2.5],
                [3.0, -1.5],
                [2.0, -2.5],
                [2.0, -1.5],
            ]
        )
    ).all()

    grid = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(3, 3), pixel_scales=(1.0, 1.0), sub_size=2
    )

    assert (
        grid
        == np.array(
            [
                [1.25, -1.25],
                [1.25, -0.75],
                [0.75, -1.25],
                [0.75, -0.75],
                [1.25, -0.25],
                [1.25, 0.25],
                [0.75, -0.25],
                [0.75, 0.25],
                [1.25, 0.75],
                [1.25, 1.25],
                [0.75, 0.75],
                [0.75, 1.25],
                [0.25, -1.25],
                [0.25, -0.75],
                [-0.25, -1.25],
                [-0.25, -0.75],
                [0.25, -0.25],
                [0.25, 0.25],
                [-0.25, -0.25],
                [-0.25, 0.25],
                [0.25, 0.75],
                [0.25, 1.25],
                [-0.25, 0.75],
                [-0.25, 1.25],
                [-0.75, -1.25],
                [-0.75, -0.75],
                [-1.25, -1.25],
                [-1.25, -0.75],
                [-0.75, -0.25],
                [-0.75, 0.25],
                [-1.25, -0.25],
                [-1.25, 0.25],
                [-0.75, 0.75],
                [-0.75, 1.25],
                [-1.25, 0.75],
                [-1.25, 1.25],
            ]
        )
    ).all()

    sub_grid_shape = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(2, 4), pixel_scales=(2.0, 1.0), sub_size=3, origin=(0.5, 0.6)
    )

    sub_grid_mask = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=np.full(fill_value=False, shape=(2, 4)),
        pixel_scales=(2.0, 1.0),
        sub_size=3,
        origin=(0.5, 0.6),
    )

    assert (sub_grid_shape == sub_grid_mask).all()


def test__grid_2d_via_shape_native_from():

    grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(2, 3), pixel_scales=(1.0, 1.0), sub_size=1
    )

    assert (
        grid_2d
        == np.array(
            [
                [[0.5, -1.0], [0.5, 0.0], [0.5, 1.0]],
                [[-0.5, -1.0], [-0.5, 0.0], [-0.5, 1.0]],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(3, 2), pixel_scales=(1.0, 1.0), sub_size=1
    )

    assert (
        grid_2d
        == np.array(
            [
                [[1.0, -0.5], [1.0, 0.5]],
                [[0.0, -0.5], [0.0, 0.5]],
                [[-1.0, -0.5], [-1.0, 0.5]],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(3, 2), pixel_scales=(1.0, 1.0), sub_size=1, origin=(3.0, -2.0)
    )

    assert (
        grid_2d
        == np.array(
            [
                [[4.0, -2.5], [4.0, -1.5]],
                [[3.0, -2.5], [3.0, -1.5]],
                [[2.0, -2.5], [2.0, -1.5]],
            ]
        )
    ).all()


def test__radial_projected_shape_slim_from():

    shape_slim = aa.util.grid_2d._radial_projected_shape_slim_from(
        extent=np.array([-1.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 0.0),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    assert shape_slim == 2

    shape_slim = aa.util.grid_2d._radial_projected_shape_slim_from(
        extent=np.array([-1.0, 3.0, -1.0, 1.0]),
        centre=(0.0, 0.0),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    assert shape_slim == 4

    shape_slim = aa.util.grid_2d._radial_projected_shape_slim_from(
        extent=np.array([-1.0, 3.0, -1.0, 1.0]),
        centre=(0.0, 1.0),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    assert shape_slim == 3

    shape_slim = aa.util.grid_2d._radial_projected_shape_slim_from(
        extent=np.array([-2.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 1.0),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    assert shape_slim == 4


def test__grid_scaled_2d_slim_radial_projected_from():

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 0.0),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    assert (grid_radii == np.array([[0.0, 0.0], [0.0, 1.0]])).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 3.0, -1.0, 1.0]),
        centre=(0.0, 0.0),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    assert (
        grid_radii == np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    ).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 3.0, -1.0, 1.0]),
        centre=(0.0, 1.0),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    assert (grid_radii == np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-2.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 1.0),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    assert (
        grid_radii == np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
    ).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 1.0),
        pixel_scales=(0.1, 0.5),
        sub_size=1,
    )

    assert (
        grid_radii
        == np.array([[0.0, 1.0], [0.0, 1.5], [0.0, 2.0], [0.0, 2.5], [0.0, 3.0]])
    ).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 0.0),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
        shape_slim=3,
    )

    assert (grid_radii == np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 1.0),
        pixel_scales=(0.1, 1.0),
        sub_size=2,
    )

    assert (
        grid_radii
        == np.array([[0.0, 1.0], [0.0, 1.5], [0.0, 2.0], [0.0, 2.5], [0.0, 3.0]])
    ).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([5.0, 8.0, 99.9, 100.1]),
        centre=(100.0, 7.0),
        pixel_scales=(10.0, 0.25),
        sub_size=1,
    )

    assert (
        grid_radii
        == np.array(
            [
                [100.0, 7.0],
                [100.0, 7.25],
                [100.0, 7.5],
                [100.0, 7.75],
                [100.0, 8.0],
                [100.0, 8.25],
                [100.0, 8.5],
                [100.0, 8.75],
                [100.0, 9.0],
            ]
        )
    ).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -1.0, 3.0]),
        centre=(0.0, 0.0),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    assert (
        grid_radii == np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    ).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -2.0, 1.0]),
        centre=(1.0, 0.0),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    assert (
        grid_radii == np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    ).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -1.0, 1.0]),
        centre=(1.0, 0.0),
        pixel_scales=(0.5, 0.1),
        sub_size=1,
    )

    assert (
        grid_radii
        == np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5], [1.0, 2.0]])
    ).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -1.0, 1.0]),
        centre=(1.0, 0.0),
        pixel_scales=(0.5, 0.1),
        sub_size=2,
    )

    assert (
        grid_radii
        == np.array(
            [
                [1.0, 0.0],
                [1.0, 0.25],
                [1.0, 0.5],
                [1.0, 0.75],
                [1.0, 1.0],
                [1.0, 1.25],
                [1.0, 1.5],
                [1.0, 1.75],
                [1.0, 2.0],
            ]
        )
    ).all()

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([99.9, 100.1, -1.0, 3.0]),
        centre=(-1.0, 100.0),
        pixel_scales=(1.5, 10.0),
        sub_size=1,
    )

    assert (grid_radii == np.array([[-1.0, 100.0], [-1.0, 101.5], [-1.0, 103.0]])).all()


def test__grid_2d_slim_from():

    grid_2d = np.array(
        [
            [[1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6]],
            [[7, 7], [8, 8], [9, 9]],
        ]
    )

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    grid_slim = aa.util.grid_2d.grid_2d_slim_from(
        grid_2d_native=grid_2d, mask=mask, sub_size=1
    )

    assert (grid_slim == np.array([[5, 5]])).all()

    grid_2d = np.array(
        [
            [[1, 1], [2, 2], [3, 3], [4, 4]],
            [[5, 5], [6, 6], [7, 7], [8, 8]],
            [[9, 9], [10, 10], [11, 11], [12, 12]],
        ]
    )

    mask = np.array(
        [
            [True, False, True, True],
            [False, False, False, True],
            [True, False, True, False],
        ]
    )

    grid_slim = aa.util.grid_2d.grid_2d_slim_from(
        grid_2d_native=grid_2d, mask=mask, sub_size=1
    )

    assert (
        grid_slim == np.array([[2, 2], [5, 5], [6, 6], [7, 7], [10, 10], [12, 12]])
    ).all()

    grid_2d = np.array(
        [
            [[1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6]],
            [[7, 7], [8, 8], [9, 9]],
            [[10, 10], [11, 11], [12, 12]],
        ]
    )

    mask = np.array(
        [
            [True, False, True],
            [False, False, False],
            [True, False, True],
            [True, True, True],
        ]
    )

    grid_slim = aa.util.grid_2d.grid_2d_slim_from(
        grid_2d_native=grid_2d, mask=mask, sub_size=1
    )

    assert (grid_slim == np.array([[2, 2], [4, 4], [5, 5], [6, 6], [8, 8]])).all()

    sub_grid_2d = np.array(
        [
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
            [[7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]],
            [[13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18]],
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
            [[7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]],
            [[13, 13], [14, 14], [15, 15], [16, 16], [17, 17], [18, 18]],
        ]
    )

    mask = np.array([[True, False, True], [True, False, True], [True, True, False]])

    sub_array_2d = aa.util.grid_2d.grid_2d_slim_from(
        grid_2d_native=sub_grid_2d, mask=mask, sub_size=2
    )

    assert (
        sub_array_2d
        == np.array(
            [
                [3, 3],
                [4, 4],
                [9, 9],
                [10, 10],
                [15, 15],
                [16, 16],
                [3, 3],
                [4, 4],
                [11, 11],
                [12, 12],
                [17, 17],
                [18, 18],
            ]
        )
    ).all()


def test__grid_2d_native_from():

    grid_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

    mask = np.full(fill_value=False, shape=(2, 2))

    grid_2d = aa.util.grid_2d.grid_2d_native_from(
        grid_2d_slim=grid_slim, mask_2d=mask, sub_size=1
    )

    assert (
        grid_2d == np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
    ).all()

    grid_slim = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
            [-3.0, -3.0],
        ]
    )

    mask = np.array(
        [
            [False, False, True, True],
            [False, True, True, True],
            [False, False, True, False],
        ]
    )

    grid_2d = aa.util.grid_2d.grid_2d_native_from(
        grid_2d_slim=grid_slim, mask_2d=mask, sub_size=1
    )

    assert (
        grid_2d
        == np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                [[3.0, 3.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[-1.0, -1.0], [-2.0, -2.0], [0.0, 0.0], [-3.0, -3.0]],
            ]
        )
    ).all()

    grid_slim = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [3.0, 3.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )

    mask = np.array([[False, False], [False, True]])

    grid_2d = aa.util.grid_2d.grid_2d_native_from(
        grid_2d_slim=grid_slim, mask_2d=mask, sub_size=2
    )

    assert (
        grid_2d
        == np.array(
            [
                [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
                [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
                [[3.0, 3.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0]],
                [[3.0, 3.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__grid_2d_slim_upscaled_from():

    grid_slim = np.array([[1.0, 1.0]])

    grid_upscaled_2d = aa.util.grid_2d.grid_2d_slim_upscaled_from(
        grid_slim=grid_slim, upscale_factor=1, pixel_scales=(2.0, 2.0)
    )

    assert (grid_upscaled_2d == np.array([[1.0, 1.0]])).all()

    grid_upscaled_2d = aa.util.grid_2d.grid_2d_slim_upscaled_from(
        grid_slim=grid_slim, upscale_factor=2, pixel_scales=(2.0, 2.0)
    )

    assert (
        grid_upscaled_2d == np.array([[1.5, 0.5], [1.5, 1.5], [0.5, 0.5], [0.5, 1.5]])
    ).all()

    grid_slim = np.array([[1.0, 1.0], [1.0, 3.0]])

    grid_upscaled_2d = aa.util.grid_2d.grid_2d_slim_upscaled_from(
        grid_slim=grid_slim, upscale_factor=2, pixel_scales=(2.0, 2.0)
    )

    assert (
        grid_upscaled_2d
        == np.array(
            [
                [1.5, 0.5],
                [1.5, 1.5],
                [0.5, 0.5],
                [0.5, 1.5],
                [1.5, 2.5],
                [1.5, 3.5],
                [0.5, 2.5],
                [0.5, 3.5],
            ]
        )
    ).all()

    grid_slim = np.array([[1.0, 1.0], [3.0, 1.0]])

    grid_upscaled_2d = aa.util.grid_2d.grid_2d_slim_upscaled_from(
        grid_slim=grid_slim, upscale_factor=2, pixel_scales=(2.0, 2.0)
    )

    assert (
        grid_upscaled_2d
        == np.array(
            [
                [1.5, 0.5],
                [1.5, 1.5],
                [0.5, 0.5],
                [0.5, 1.5],
                [3.5, 0.5],
                [3.5, 1.5],
                [2.5, 0.5],
                [2.5, 1.5],
            ]
        )
    ).all()

    grid_slim = np.array([[1.0, 1.0]])

    grid_upscaled_2d = aa.util.grid_2d.grid_2d_slim_upscaled_from(
        grid_slim=grid_slim, upscale_factor=2, pixel_scales=(3.0, 2.0)
    )

    assert (
        grid_upscaled_2d
        == np.array([[1.75, 0.5], [1.75, 1.5], [0.25, 0.5], [0.25, 1.5]])
    ).all()

    grid_upscaled_2d = aa.util.grid_2d.grid_2d_slim_upscaled_from(
        grid_slim=grid_slim, upscale_factor=3, pixel_scales=(2.0, 2.0)
    )

    assert grid_upscaled_2d[0] == pytest.approx(np.array([1.666, 0.333]), 1.0e-2)
    assert grid_upscaled_2d[1] == pytest.approx(np.array([1.666, 1.0]), 1.0e-2)
    assert grid_upscaled_2d[2] == pytest.approx(np.array([1.666, 1.666]), 1.0e-2)
    assert grid_upscaled_2d[3] == pytest.approx(np.array([1.0, 0.333]), 1.0e-2)
    assert grid_upscaled_2d[4] == pytest.approx(np.array([1.0, 1.0]), 1.0e-2)
    assert grid_upscaled_2d[5] == pytest.approx(np.array([1.0, 1.666]), 1.0e-2)
    assert grid_upscaled_2d[6] == pytest.approx(np.array([0.333, 0.333]), 1.0e-2)
    assert grid_upscaled_2d[7] == pytest.approx(np.array([0.333, 1.0]), 1.0e-2)
    assert grid_upscaled_2d[8] == pytest.approx(np.array([0.333, 1.666]), 1.0e-2)
