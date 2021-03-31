import autoarray as aa
import numpy as np
import pytest


class TestGrid2dSlimFromMask:
    def test__from_3x3_mask__sub_size_1(self):

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=(3.0, 6.0), sub_size=1
        )

        assert (grid[0] == np.array([0.0, 0.0])).all()

        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=(6.0, 3.0), sub_size=1
        )

        assert (
            grid
            == np.array([[6.0, 0.0], [0.0, -3.0], [0.0, 0.0], [0.0, 3.0], [-6.0, 0.0]])
        ).all()

    def test__from_bigger_and_rectangular_masks__sub_size_1(self):
        mask = np.array(
            [
                [True, False, False, True],
                [False, False, False, True],
                [True, False, False, True],
                [False, False, False, True],
            ]
        )

        grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=(1.0, 1.0), sub_size=1
        )

        assert (
            grid
            == np.array(
                [
                    [1.5, -0.5],
                    [1.5, 0.5],
                    [0.5, -1.5],
                    [0.5, -0.5],
                    [0.5, 0.5],
                    [-0.5, -0.5],
                    [-0.5, 0.5],
                    [-1.5, -1.5],
                    [-1.5, -0.5],
                    [-1.5, 0.5],
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

    def test__same_as_above__include_nonzero_origin(self):
        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=(6.0, 3.0), sub_size=1, origin=(1.0, 1.0)
        )

        assert grid == pytest.approx(
            np.array([[7.0, 1.0], [1.0, -2.0], [1.0, 1.0], [1.0, 4.0], [-5.0, 1.0]]),
            1e-4,
        )

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

    def test__3x3_mask__2x2_sub_grid(self):
        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=(3.0, 6.0), sub_size=2
        )

        assert (
            grid[0:4]
            == np.array([[0.75, -1.5], [0.75, 1.5], [-0.75, -1.5], [-0.75, 1.5]])
        ).all()

        mask = np.array([[True, True, True], [False, False, False], [True, True, True]])

        grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2
        )

        assert (
            grid[0:4]
            == np.array([[0.75, -3.75], [0.75, -2.25], [-0.75, -3.75], [-0.75, -2.25]])
        ).all()

        assert (
            grid[4:8]
            == np.array([[0.75, -0.75], [0.75, 0.75], [-0.75, -0.75], [-0.75, 0.75]])
        ).all()

        assert (
            grid[8:12]
            == np.array([[0.75, 2.25], [0.75, 3.75], [-0.75, 2.25], [-0.75, 3.75]])
        ).all()

        mask = np.array(
            [[True, True, False], [False, False, False], [True, True, False]]
        )

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

        mask = np.array(
            [[True, True, False], [False, False, False], [True, True, False]]
        )

        grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=(0.3, 0.3), sub_size=2
        )

        grid = np.round(grid, decimals=3)

        np.testing.assert_almost_equal(
            grid,
            np.array(
                [
                    [0.375, 0.225],
                    [0.375, 0.375],
                    [0.225, 0.225],
                    [0.225, 0.375],
                    [0.075, -0.375],
                    [0.075, -0.225],
                    [-0.075, -0.375],
                    [-0.075, -0.225],
                    [0.075, -0.075],
                    [0.075, 0.075],
                    [-0.075, -0.075],
                    [-0.075, 0.075],
                    [0.075, 0.225],
                    [0.075, 0.375],
                    [-0.075, 0.225],
                    [-0.075, 0.375],
                    [-0.225, 0.225],
                    [-0.225, 0.375],
                    [-0.375, 0.225],
                    [-0.375, 0.375],
                ]
            ),
        )

    def test__3x3_mask_with_one_pixel__3x3_and_4x4_sub_grids(self):
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
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, False],
            ]
        )

        grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=(2.0, 2.0), sub_size=4
        )

        grid = np.round(grid, decimals=2)

        assert (
            grid
            == np.array(
                [
                    [1.75, -1.75],
                    [1.75, -1.25],
                    [1.75, -0.75],
                    [1.75, -0.25],
                    [1.25, -1.75],
                    [1.25, -1.25],
                    [1.25, -0.75],
                    [1.25, -0.25],
                    [0.75, -1.75],
                    [0.75, -1.25],
                    [0.75, -0.75],
                    [0.75, -0.25],
                    [0.25, -1.75],
                    [0.25, -1.25],
                    [0.25, -0.75],
                    [0.25, -0.25],
                    [1.75, 0.25],
                    [1.75, 0.75],
                    [1.75, 1.25],
                    [1.75, 1.75],
                    [1.25, 0.25],
                    [1.25, 0.75],
                    [1.25, 1.25],
                    [1.25, 1.75],
                    [0.75, 0.25],
                    [0.75, 0.75],
                    [0.75, 1.25],
                    [0.75, 1.75],
                    [0.25, 0.25],
                    [0.25, 0.75],
                    [0.25, 1.25],
                    [0.25, 1.75],
                    [-0.25, -1.75],
                    [-0.25, -1.25],
                    [-0.25, -0.75],
                    [-0.25, -0.25],
                    [-0.75, -1.75],
                    [-0.75, -1.25],
                    [-0.75, -0.75],
                    [-0.75, -0.25],
                    [-1.25, -1.75],
                    [-1.25, -1.25],
                    [-1.25, -0.75],
                    [-1.25, -0.25],
                    [-1.75, -1.75],
                    [-1.75, -1.25],
                    [-1.75, -0.75],
                    [-1.75, -0.25],
                    [-0.25, 0.25],
                    [-0.25, 0.75],
                    [-0.25, 1.25],
                    [-0.25, 1.75],
                    [-0.75, 0.25],
                    [-0.75, 0.75],
                    [-0.75, 1.25],
                    [-0.75, 1.75],
                    [-1.25, 0.25],
                    [-1.25, 0.75],
                    [-1.25, 1.25],
                    [-1.25, 1.75],
                    [-1.75, 0.25],
                    [-1.75, 0.75],
                    [-1.75, 1.25],
                    [-1.75, 1.75],
                    [-2.25, 2.25],
                    [-2.25, 2.75],
                    [-2.25, 3.25],
                    [-2.25, 3.75],
                    [-2.75, 2.25],
                    [-2.75, 2.75],
                    [-2.75, 3.25],
                    [-2.75, 3.75],
                    [-3.25, 2.25],
                    [-3.25, 2.75],
                    [-3.25, 3.25],
                    [-3.25, 3.75],
                    [-3.75, 2.25],
                    [-3.75, 2.75],
                    [-3.75, 3.25],
                    [-3.75, 3.75],
                ]
            )
        ).all()

    def test__rectangular_masks_with_one_pixel__2x2_sub_grid(self):
        mask = np.array(
            [
                [True, True, True],
                [True, False, True],
                [True, False, False],
                [False, True, True],
            ]
        )

        grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2
        )

        assert (
            grid
            == np.array(
                [
                    [2.25, -0.75],
                    [2.25, 0.75],
                    [0.75, -0.75],
                    [0.75, 0.75],
                    [-0.75, -0.75],
                    [-0.75, 0.75],
                    [-2.25, -0.75],
                    [-2.25, 0.75],
                    [-0.75, 2.25],
                    [-0.75, 3.75],
                    [-2.25, 2.25],
                    [-2.25, 3.75],
                    [-3.75, -3.75],
                    [-3.75, -2.25],
                    [-5.25, -3.75],
                    [-5.25, -2.25],
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

    def test__3x3_mask_with_one_pixel__2x2_and_3x3_sub_grid__include_nonzero_origin(
        self,
    ):
        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=(3.0, 6.0), sub_size=2, origin=(1.0, 1.0)
        )

        assert grid[0:4] == pytest.approx(
            np.array([[1.75, -0.5], [1.75, 2.5], [0.25, -0.5], [0.25, 2.5]]), 1e-4
        )

        mask = np.array([[True, True, False], [True, False, True], [True, True, False]])

        grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=3, origin=(1.0, -1.0)
        )

        assert grid == pytest.approx(
            np.array(
                [
                    [5.0, 1.0],
                    [5.0, 2.0],
                    [5.0, 3.0],
                    [4.0, 1.0],
                    [4.0, 2.0],
                    [4.0, 3.0],
                    [3.0, 1.0],
                    [3.0, 2.0],
                    [3.0, 3.0],
                    [2.0, -2.0],
                    [2.0, -1.0],
                    [2.0, 0.0],
                    [1.0, -2.0],
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [0.0, -2.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [-1.0, 1.0],
                    [-1.0, 2.0],
                    [-1.0, 3.0],
                    [-2.0, 1.0],
                    [-2.0, 2.0],
                    [-2.0, 3.0],
                    [-3.0, 1.0],
                    [-3.0, 2.0],
                    [-3.0, 3.0],
                ]
            ),
            1e-4,
        )


class TestGrid2DFromMask:
    def test__same_as_2d_grids(self):

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


class TestGrid2dFromShape:
    def test__array_3x3__sub_grid_1__sets_up_scaledond_grid(self):

        grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
            shape_native=(3, 3), pixel_scales=(2.0, 1.0), sub_size=1
        )

        assert (
            grid_2d
            == np.array(
                [
                    [2.0, -1.0],
                    [2.0, 0.0],
                    [2.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [-2.0, -1.0],
                    [-2.0, 0.0],
                    [-2.0, 1.0],
                ]
            )
        ).all()

        grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
            shape_native=(4, 4), pixel_scales=(0.5, 0.5), sub_size=1
        )

        assert (
            grid_2d
            == np.array(
                [
                    [0.75, -0.75],
                    [0.75, -0.25],
                    [0.75, 0.25],
                    [0.75, 0.75],
                    [0.25, -0.75],
                    [0.25, -0.25],
                    [0.25, 0.25],
                    [0.25, 0.75],
                    [-0.25, -0.75],
                    [-0.25, -0.25],
                    [-0.25, 0.25],
                    [-0.25, 0.75],
                    [-0.75, -0.75],
                    [-0.75, -0.25],
                    [-0.75, 0.25],
                    [-0.75, 0.75],
                ]
            )
        ).all()

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

    def test__array_3x3__input_origin__shifts_grid_by_origin(self):

        grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
            shape_native=(3, 3), pixel_scales=(2.0, 1.0), sub_size=1, origin=(1.0, 1.0)
        )

        assert (
            grid_2d
            == np.array(
                [
                    [3.0, 0.0],
                    [3.0, 1.0],
                    [3.0, 2.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 2.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                    [-1.0, 2.0],
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

    def test__from_shape_3x3_ask_with_one_pixel__2x2_sub_grid(self):

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

    def test__compare_to_mask_manually(self):

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


class TestGrid2DFromShape:
    def test__sets_up_scaledond_grid__sub_size_1(self):

        grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
            shape_native=(3, 3), pixel_scales=(2.0, 1.0), sub_size=1
        )

        assert (
            grid_2d
            == np.array(
                [
                    [[2.0, -1.0], [2.0, 0.0], [2.0, 1.0]],
                    [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]],
                    [[-2.0, -1.0], [-2.0, 0.0], [-2.0, 1.0]],
                ]
            )
        ).all()

        grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
            shape_native=(4, 4), pixel_scales=(0.5, 0.5), sub_size=1
        )

        assert (
            grid_2d
            == np.array(
                [
                    [[0.75, -0.75], [0.75, -0.25], [0.75, 0.25], [0.75, 0.75]],
                    [[0.25, -0.75], [0.25, -0.25], [0.25, 0.25], [0.25, 0.75]],
                    [[-0.25, -0.75], [-0.25, -0.25], [-0.25, 0.25], [-0.25, 0.75]],
                    [[-0.75, -0.75], [-0.75, -0.25], [-0.75, 0.25], [-0.75, 0.75]],
                ]
            )
        ).all()

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

    def test__array_3x3___input_origin__shifts_grid_by_origin(self):

        grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
            shape_native=(3, 3), pixel_scales=(2.0, 1.0), sub_size=1, origin=(1.0, 1.0)
        )

        assert (
            grid_2d
            == np.array(
                [
                    [[3.0, 0.0], [3.0, 1.0], [3.0, 2.0]],
                    [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
                    [[-1.0, 0.0], [-1.0, 1.0], [-1.0, 2.0]],
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


class TestGridRadialProjected:
    def test__grid_radial_projected_from_scaled_2d__vary_all_x_dimension_parameters(
        self
    ):

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
            extent=np.array([5.0, 8.0, 99.9, 100.1]),
            centre=(-10.0, -10.0),
            pixel_scales=(1.0, 10.0),
            sub_size=1,
        )

    def test__grid_radii_from_scaled_2d__vary_all_y_dimension_parameters(self):

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

        assert (
            grid_radii == np.array([[-1.0, 100.0], [-1.0, 101.5], [-1.0, 103.0]])
        ).all()


class TestGridConversions:
    def test__pixel_grid_2d_from_scaled_grid_2d__coordinates_in_origins_of_pixels(self):

        grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels = aa.util.grid_2d.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (
            grid_pixels == np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])
        ).all()

        grid_scaled = np.array(
            [
                [3.0, -6.0],
                [3.0, 0.0],
                [3.0, 6.0],
                [0.0, -6.0],
                [0.0, 0.0],
                [0.0, 6.0],
                [-3.0, -6.0],
                [-3.0, 0.0],
                [-3.0, 6.0],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (
            grid_pixels
            == np.array(
                [
                    [0.5, 0.5],
                    [0.5, 1.5],
                    [0.5, 2.5],
                    [1.5, 0.5],
                    [1.5, 1.5],
                    [1.5, 2.5],
                    [2.5, 0.5],
                    [2.5, 1.5],
                    [2.5, 2.5],
                ]
            )
        ).all()

    def test__same_as_above__pixels__but_coordinates_are_top_left_of_each_pixel(self):

        grid_scaled = np.array([[2.0, -4], [2.0, 0.0], [0.0, -4], [0.0, 0.0]])

        grid_pixels = aa.util.grid_2d.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (grid_pixels == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all()

        grid_scaled = np.array(
            [
                [4.5, -9.0],
                [4.5, -3.0],
                [4.5, 3.0],
                [1.5, -9.0],
                [1.5, -3.0],
                [1.5, 3.0],
                [-1.5, -9.0],
                [-1.5, -3.0],
                [-1.5, 3.0],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (
            grid_pixels
            == np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
            )
        ).all()

    def test__same_as_above___pixels__but_coordinates_are_bottom_right_of_each_pixel(
        self,
    ):

        grid_scaled = np.array([[0.0, 0.0], [0.0, 4.0], [-2.0, 0.0], [-2.0, 4.0]])

        grid_pixels = aa.util.grid_2d.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (grid_pixels == np.array([[1, 1], [1, 2], [2, 1], [2, 2]])).all()

        grid_scaled = np.array(
            [
                [1.5, -3.0],
                [1.5, 3.0],
                [1.5, 9.0],
                [-1.5, -3.0],
                [-1.5, 3.0],
                [-1.5, 9.0],
                [-4.5, -3.0],
                [-4.5, 3.0],
                [-4.5, 9.0],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (
            grid_pixels
            == np.array(
                [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]]
            )
        ).all()

    def test__same_as_above___scaled_to_pixel__but_nonzero_origin(self):

        # -1.0 from all entries for a origin of (-1.0, -1.0)
        grid_scaled = np.array([[-1.0, -1.0], [-1.0, 3.0], [-3.0, -1.0], [-3.0, 3.0]])

        grid_pixels = aa.util.grid_2d.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
            origin=(-1.0, -1.0),
        )

        assert (grid_pixels == np.array([[1, 1], [1, 2], [2, 1], [2, 2]])).all()

        # -1.0, +2.0, for origin of (-1.0, +2.0)
        grid_scaled = np.array(
            [
                [0.5, -1.0],
                [0.5, 5.0],
                [0.5, 11.0],
                [-2.5, -1.0],
                [-2.5, 5.0],
                [-2.5, 11.0],
                [-5.5, -1.0],
                [-5.5, 5.0],
                [-5.5, 11.0],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
            origin=(-1.0, 2.0),
        )

        assert (
            grid_pixels
            == np.array(
                [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]]
            )
        ).all()

    def test__pixel_centre_grid_2d_from_scaled_grid_2d__coordinates_in_origins_of_pixels(
        self,
    ):

        grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (grid_pixels == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all()

        grid_scaled = np.array(
            [
                [3.0, -6.0],
                [3.0, 0.0],
                [3.0, 6.0],
                [0.0, -6.0],
                [0.0, 0.0],
                [0.0, 6.0],
                [-3.0, -6.0],
                [-3.0, 0.0],
                [-3.0, 6.0],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (
            grid_pixels
            == np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
            )
        ).all()

    def test__same_as_above_but_coordinates_are_top_left_of_each_pixel(self):

        grid_scaled = np.array(
            [[1.99, -3.99], [1.99, 0.01], [-0.01, -3.99], [-0.01, 0.01]]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (grid_pixels == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all()

        grid_scaled = np.array(
            [
                [4.49, -8.99],
                [4.49, -2.99],
                [4.49, 3.01],
                [1.49, -8.99],
                [1.49, -2.99],
                [1.49, 3.01],
                [-1.51, -8.99],
                [-1.51, -2.99],
                [-1.51, 3.01],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (
            grid_pixels
            == np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
            )
        ).all()

    def test__same_as_above_but_coordinates_are_bottom_right_of_each_pixel(self):

        grid_scaled = np.array(
            [[0.01, -0.01], [0.01, 3.99], [-1.99, -0.01], [-1.99, 3.99]]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (grid_pixels == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all()

        grid_scaled = np.array(
            [
                [1.51, -3.01],
                [1.51, 2.99],
                [1.51, 8.99],
                [-1.49, -3.01],
                [-1.49, 2.99],
                [-1.49, 8.99],
                [-4.49, -3.01],
                [-4.49, 2.99],
                [-4.49, 8.99],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (
            grid_pixels
            == np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
            )
        ).all()

    def test__same_as_above__scaled_to_pixel_origin__but_nonzero_origin(self):

        # +1.0 for all entries for a origin of (1.0, 1.0)
        grid_scaled = np.array([[2.0, -1.0], [2.0, 3.0], [0.0, -1.0], [0.0, 3.0]])

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
            origin=(1.0, 1.0),
        )

        assert (grid_pixels == np.array([[0, 0], [0, 1], [1, 0], [1, 1]])).all()

        # +1.0, -2.0, for origin of (1.0, -2.0)
        grid_scaled = np.array(
            [
                [4.0, -8.0],
                [4.0, -2.0],
                [4.0, 4.0],
                [1.0, -8.0],
                [1.0, -2.0],
                [1.0, 4.0],
                [-2.0, -8.0],
                [-2.0, -2.0],
                [-2.0, 4.0],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
            origin=(1.0, -2.0),
        )

        assert (
            grid_pixels
            == np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
            )
        ).all()

    def test__pixel_index_grid_2d_from_scaled_grid_2d__coordinates_in_origins_of_pixels(
        self,
    ):

        grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (grid_pixels == np.array([0, 1, 2, 3])).all()

        grid_scaled = np.array(
            [
                [3.0, -6.0],
                [3.0, 0.0],
                [3.0, 6.0],
                [0.0, -6.0],
                [0.0, 0.0],
                [0.0, 6.0],
                [-3.0, -6.0],
                [-3.0, 0.0],
                [-3.0, 6.0],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (grid_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above_2d_index__but_coordinates_are_top_left_of_each_pixel(self):

        grid_scaled = np.array(
            [[1.99, -3.99], [1.99, 0.01], [-0.01, -3.99], [-0.01, 0.01]]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (grid_pixels == np.array([0, 1, 2, 3])).all()

        grid_scaled = np.array(
            [
                [4.49, -8.99],
                [4.49, -2.99],
                [4.49, 3.01],
                [1.49, -8.99],
                [1.49, -2.99],
                [1.49, 3.01],
                [-1.51, -8.99],
                [-1.51, -2.99],
                [-1.51, 3.01],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (grid_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above_2d_index__but_coordinates_are_bottom_right_of_each_pixel(
        self,
    ):

        grid_scaled = np.array(
            [[0.01, -0.01], [0.01, 3.99], [-1.99, -0.01], [-1.99, 3.99]]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (grid_pixels == np.array([0, 1, 2, 3])).all()

        grid_scaled = np.array(
            [
                [1.51, -3.01],
                [1.51, 2.99],
                [1.51, 8.99],
                [-1.49, -3.01],
                [-1.49, 2.99],
                [-1.49, 8.99],
                [-4.49, -3.01],
                [-4.49, 2.99],
                [-4.49, 8.99],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (grid_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__same_as_above__2d_index__scaled_to_pixel_origin__but_nonzero_origin(self):

        # +1.0 for all entries for a origin of (1.0, 1.0)
        grid_scaled = np.array([[2.0, -1.0], [2.0, 3.0], [0.0, -1.0], [0.0, 3.0]])

        grid_pixels = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
            origin=(1.0, 1.0),
        )

        assert (grid_pixels == np.array([0, 1, 2, 3])).all()

        # +1.0, -2.0, for origin of (1.0, -2.0)
        grid_scaled = np.array(
            [
                [4.0, -8.0],
                [4.0, -2.0],
                [4.0, 4.0],
                [1.0, -8.0],
                [1.0, -2.0],
                [1.0, 4.0],
                [-2.0, -8.0],
                [-2.0, -2.0],
                [-2.0, 4.0],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
            origin=(1.0, -2.0),
        )

        assert (grid_pixels == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__scaled_grid_2d_from_pixel_origin_grid_2d__coordinates_in_origins_of_pixels(
        self,
    ):

        grid_pixels = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])

        grid_scaled = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (
            grid_scaled
            == np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])
        ).all()

        grid_pixels = np.array(
            [
                [0.5, 0.5],
                [0.5, 1.5],
                [0.5, 2.5],
                [1.5, 0.5],
                [1.5, 1.5],
                [1.5, 2.5],
                [2.5, 0.5],
                [2.5, 1.5],
                [2.5, 2.5],
            ]
        )

        grid_scaled = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (
            grid_scaled
            == np.array(
                [
                    [3.0, -6.0],
                    [3.0, 0.0],
                    [3.0, 6.0],
                    [0.0, -6.0],
                    [0.0, 0.0],
                    [0.0, 6.0],
                    [-3.0, -6.0],
                    [-3.0, 0.0],
                    [-3.0, 6.0],
                ]
            )
        ).all()

    def test__same_as_above__pixel_to_scaled__but_coordinates_are_top_left_of_each_pixel(
        self,
    ):

        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_scaled = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (
            grid_scaled == np.array([[2.0, -4], [2.0, 0.0], [0.0, -4], [0.0, 0.0]])
        ).all()

        grid_pixels = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        grid_scaled = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (
            grid_scaled
            == np.array(
                [
                    [4.5, -9.0],
                    [4.5, -3.0],
                    [4.5, 3.0],
                    [1.5, -9.0],
                    [1.5, -3.0],
                    [1.5, 3.0],
                    [-1.5, -9.0],
                    [-1.5, -3.0],
                    [-1.5, 3.0],
                ]
            )
        ).all()

    def test__same_as_above__pixel_to_scaled_but_coordinates_are_bottom_right_of_each_pixel(
        self,
    ):

        grid_pixels = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

        grid_scaled = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        assert (
            grid_scaled == np.array([[0.0, 0.0], [0.0, 4.0], [-2.0, 0.0], [-2.0, 4.0]])
        ).all()

        grid_pixels = np.array(
            [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]]
        )

        grid_scaled = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
        )

        assert (
            grid_scaled
            == np.array(
                [
                    [1.5, -3.0],
                    [1.5, 3.0],
                    [1.5, 9.0],
                    [-1.5, -3.0],
                    [-1.5, 3.0],
                    [-1.5, 9.0],
                    [-4.5, -3.0],
                    [-4.5, 3.0],
                    [-4.5, 9.0],
                ]
            )
        ).all()

    def test__same_as_above__pixel_to_scaled__nonzero_origin(self):

        grid_pixels = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])

        grid_scaled = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
            origin=(-1.0, -1.0),
        )

        # -1.0 from all entries for a origin of (-1.0, -1.0)
        assert (
            grid_scaled
            == np.array([[0.0, -3.0], [0.0, 1.0], [-2.0, -3.0], [-2.0, 1.0]])
        ).all()

        grid_pixels = np.array(
            [
                [0.5, 0.5],
                [0.5, 1.5],
                [0.5, 2.5],
                [1.5, 0.5],
                [1.5, 1.5],
                [1.5, 2.5],
                [2.5, 0.5],
                [2.5, 1.5],
                [2.5, 2.5],
            ]
        )

        grid_scaled = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
            origin=(-1.0, 2.0),
        )

        # -1.0, +2.0, for origin of (-1.0, 2.0)
        assert grid_scaled == pytest.approx(
            np.array(
                [
                    [2.0, -4.0],
                    [2.0, 2.0],
                    [2.0, 8.0],
                    [-1.0, -4.0],
                    [-1.0, 2.0],
                    [-1.0, 8.0],
                    [-4.0, -4.0],
                    [-4.0, 2.0],
                    [-4.0, 8.0],
                ]
            ),
            1e-4,
        )

    def test__pixel_centres_grid_2d_from_scaled_grid_2d__coordinates_in_origins_of_pixels(
        self,
    ):

        grid_scaled = np.array([[[1.0, -2.0], [1.0, 2.0]], [[-1.0, -2.0], [-1.0, 2.0]]])

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_from(
            grid_scaled_2d=grid_scaled, shape_native=(2, 2), pixel_scales=(2.0, 4.0)
        )

        assert (grid_pixels == np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])).all()

        grid_scaled = np.array(
            [
                [[3.0, -6.0], [3.0, 0.0], [3.0, 6.0]],
                [[0.0, -6.0], [0.0, 0.0], [0.0, 6.0]],
                [[-3.0, -6.0], [-3.0, 0.0], [-3.0, 6.0]],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_from(
            grid_scaled_2d=grid_scaled, shape_native=(3, 3), pixel_scales=(3.0, 6.0)
        )

        assert (
            grid_pixels
            == np.array(
                [
                    [[0, 0], [0, 1], [0, 2]],
                    [[1, 0], [1, 1], [1, 2]],
                    [[2, 0], [2, 1], [2, 2]],
                ]
            )
        ).all()

    def test__2d_same_as_above_but_coordinates_are_top_left_of_each_pixel(self):

        grid_scaled = np.array(
            [[[1.99, -3.99], [1.99, 0.01]], [[-0.01, -3.99], [-0.01, 0.01]]]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_from(
            grid_scaled_2d=grid_scaled, shape_native=(2, 2), pixel_scales=(2.0, 4.0)
        )

        assert (grid_pixels == np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])).all()

        grid_scaled = np.array(
            [
                [[4.49, -8.99], [4.49, -2.99], [4.49, 3.01]],
                [[1.49, -8.99], [1.49, -2.99], [1.49, 3.01]],
                [[-1.51, -8.99], [-1.51, -2.99], [-1.51, 3.01]],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_from(
            grid_scaled_2d=grid_scaled, shape_native=(3, 3), pixel_scales=(3.0, 6.0)
        )

        assert (
            grid_pixels
            == np.array(
                [
                    [[0, 0], [0, 1], [0, 2]],
                    [[1, 0], [1, 1], [1, 2]],
                    [[2, 0], [2, 1], [2, 2]],
                ]
            )
        ).all()

    def test__2d_same_as_above_but_coordinates_are_bottom_right_of_each_pixel(self):

        grid_scaled = np.array(
            [[[0.01, -0.01], [0.01, 3.99]], [[-1.99, -0.01], [-1.99, 3.99]]]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_from(
            grid_scaled_2d=grid_scaled, shape_native=(2, 2), pixel_scales=(2.0, 4.0)
        )

        assert (grid_pixels == np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])).all()

        grid_scaled = np.array(
            [
                [[1.51, -3.01], [1.51, 2.99], [1.51, 8.99]],
                [[-1.49, -3.01], [-1.49, 2.99], [-1.49, 8.99]],
                [[-4.49, -3.01], [-4.49, 2.99], [-4.49, 8.99]],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_from(
            grid_scaled_2d=grid_scaled, shape_native=(3, 3), pixel_scales=(3.0, 6.0)
        )

        assert (
            grid_pixels
            == np.array(
                [
                    [[0, 0], [0, 1], [0, 2]],
                    [[1, 0], [1, 1], [1, 2]],
                    [[2, 0], [2, 1], [2, 2]],
                ]
            )
        ).all()

    def test__2d_same_as_above__scaled_to_pixel_origin__but_nonzero_origin(self):

        # +1.0 for all entries for a origin of (1.0, 1.0)
        grid_scaled = np.array([[[2.0, -1.0], [2.0, 3.0]], [[0.0, -1.0], [0.0, 3.0]]])

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_from(
            grid_scaled_2d=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
            origin=(1.0, 1.0),
        )

        assert (grid_pixels == np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])).all()

        # +1.0, -2.0, for origin of (1.0, -2.0)
        grid_scaled = np.array(
            [
                [[4.0, -8.0], [4.0, -2.0], [4.0, 4.0]],
                [[1.0, -8.0], [1.0, -2.0], [1.0, 4.0]],
                [[-2.0, -8.0], [-2.0, -2.0], [-2.0, 4.0]],
            ]
        )

        grid_pixels = aa.util.grid_2d.grid_pixel_centres_2d_from(
            grid_scaled_2d=grid_scaled,
            shape_native=(3, 3),
            pixel_scales=(3.0, 6.0),
            origin=(1.0, -2.0),
        )

        assert (
            grid_pixels
            == np.array(
                [
                    [[0, 0], [0, 1], [0, 2]],
                    [[1, 0], [1, 1], [1, 2]],
                    [[2, 0], [2, 1], [2, 2]],
                ]
            )
        ).all()


class TestSubGrid2dFromSubGrid2D:
    def test__map_simple_grids__sub_grid_1(self):

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
                [[1, 1], [2, 2], [3, 3]],
                [[4, 4], [5, 5], [6, 6]],
                [[7, 7], [8, 8], [9, 9]],
            ]
        )

        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        grid_slim = aa.util.grid_2d.grid_2d_slim_from(
            grid_2d_native=grid_2d, mask=mask, sub_size=1
        )

        assert (grid_slim == np.array([[2, 2], [4, 4], [5, 5], [6, 6], [8, 8]])).all()

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

    def test__map_simple_grids__sub_grid_2(self):

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


class TestSubGrid2DFromSubGrid2d:
    def test__simple_2d_array__is_masked_and_mapped__sub_size_1(self):

        grid_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        mask = np.full(fill_value=False, shape=(2, 2))

        grid_2d = aa.util.grid_2d.grid_2d_native_from(
            grid_2d_slim=grid_slim, mask_2d=mask, sub_size=1
        )

        assert (
            grid_2d == np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
        ).all()

        grid_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        mask = np.array([[False, False], [False, True]])

        grid_2d = aa.util.grid_2d.grid_2d_native_from(
            grid_2d_slim=grid_slim, mask_2d=mask, sub_size=1
        )

        assert (
            grid_2d == np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [0.0, 0.0]]])
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

    def test__simple_2d_grid__is_masked_and_mapped__sub_size_2(self):

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


class TestGridUpscaled2d:
    def test__simple_grid_in_output_grid_is_upscaled(self):

        grid_slim = np.array([[1.0, 1.0]])

        grid_upscaled_2d = aa.util.grid_2d.grid_2d_slim_upscaled_from(
            grid_slim=grid_slim, upscale_factor=1, pixel_scales=(2.0, 2.0)
        )

        assert (grid_upscaled_2d == np.array([[1.0, 1.0]])).all()

        grid_upscaled_2d = aa.util.grid_2d.grid_2d_slim_upscaled_from(
            grid_slim=grid_slim, upscale_factor=2, pixel_scales=(2.0, 2.0)
        )

        assert (
            grid_upscaled_2d
            == np.array([[1.5, 0.5], [1.5, 1.5], [0.5, 0.5], [0.5, 1.5]])
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
