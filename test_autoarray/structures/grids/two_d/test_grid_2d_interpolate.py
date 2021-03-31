import numpy as np
import pytest

import autoarray as aa

from autoarray.mock.mock import ndarray_1d_from_grid, ndarray_2d_from_grid


class TestObj:
    def test__blurring_grid_from_mask__compare_to_array_util(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, False, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_mask_util = aa.util.mask_2d.blurring_mask_2d_from(
            mask_2d=mask, kernel_shape_native=(3, 5)
        )

        blurring_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

        blurring_grid = grid.blurring_grid_from_kernel_shape(kernel_shape_native=(3, 5))

        assert isinstance(blurring_grid, aa.Grid2DInterpolate)
        assert len(blurring_grid.shape) == 2
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)
        assert blurring_grid.pixel_scales_interp == (0.1, 0.1)

    def test__blurring_grid_from_kernel_shape__compare_to_array_util(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0))

        blurring_mask_util = aa.util.mask_2d.blurring_mask_2d_from(
            mask_2d=mask, kernel_shape_native=(3, 5)
        )

        blurring_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0))

        blurring_grid = aa.Grid2DInterpolate.blurring_grid_from_mask_and_kernel_shape(
            mask=mask, kernel_shape_native=(3, 5), pixel_scales_interp=0.1
        )

        assert isinstance(blurring_grid, aa.Grid2DInterpolate)
        assert len(blurring_grid.shape) == 2
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)
        assert blurring_grid.pixel_scales_interp == (0.1, 0.1)

    def test__padded_grid_from_kernel_shape(self):
        grid = aa.Grid2DInterpolate.uniform(
            shape_native=(4, 4), pixel_scales=3.0, pixel_scales_interp=0.1
        )

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_native=(3, 3))

        assert isinstance(padded_grid, aa.Grid2DInterpolate)
        assert padded_grid.pixel_scales_interp == (0.1, 0.1)

        mask = aa.Mask2D.unmasked(
            shape_native=(6, 6), pixel_scales=(3.0, 3.0), sub_size=1
        )

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

        assert isinstance(padded_grid, aa.Grid2DInterpolate)
        assert padded_grid.pixel_scales_interp == (0.1, 0.1)
        assert (padded_grid.vtx == grid.vtx).all()
        assert (padded_grid.wts == grid.wts).all()

        mask = aa.Mask2D.manual(
            mask=np.full((2, 5), False), pixel_scales=(8.0, 8.0), sub_size=4
        )

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.2)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_native=(5, 5))

        padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=4
        )

        assert isinstance(padded_grid, aa.Grid2DInterpolate)
        assert padded_grid.pixel_scales_interp == (0.2, 0.2)
        assert isinstance(padded_grid, aa.Grid2DInterpolate)
        assert padded_grid.shape == (864, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 9))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)


class TestInterpolatedResult:
    def test__function_returns_binary_ndarray_1d__returns_interpolated_array(self):

        # noinspection PyUnusedLocal

        class MockInterpolateClass:
            def func(self, profile, grid):
                result = np.zeros(grid.shape[0])
                result[0] = 1
                return result

        mask = aa.Mask2D.unmasked(
            shape_native=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.5)

        cls = MockInterpolateClass()

        interp_array = grid.result_from_func(func=cls.func, cls=MockInterpolateClass())

        assert isinstance(interp_array, aa.Array2D)
        assert interp_array.ndim == 1
        assert interp_array.shape == (9,)
        assert (interp_array != np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])).any()

    def test__function_is_false_in_config__does_not_use_interpolatin(self):

        # noinspection PyUnusedLocal

        class MockInterpolateClass:
            def func_off(self, profile, grid):
                result = np.zeros(grid.shape[0])
                result[0] = 1
                return result

        mask = aa.Mask2D.unmasked(
            shape_native=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.5)

        cls = MockInterpolateClass()

        arr = grid.result_from_func(func=cls.func_off, cls=MockInterpolateClass())

        assert isinstance(arr, aa.Array2D)
        assert arr.ndim == 1
        assert arr.shape == (9,)
        assert (arr == np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])).any()

    def test__function_returns_binary_ndarray_2d__returns_interpolated_grid(self):

        # noinspection PyUnusedLocal
        class MockInterpolateClass:
            def func(self, profile, grid):
                result = np.zeros((grid.shape[0], 2))
                result[0, :] = 1
                return result

        mask = aa.Mask2D.unmasked(
            shape_native=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.5)

        cls = MockInterpolateClass()

        interp_grid = grid.result_from_func(func=cls.func, cls=MockInterpolateClass())

        assert isinstance(interp_grid, aa.Grid2D)
        assert interp_grid.ndim == 2
        assert interp_grid.shape == (9, 2)
        assert (
            interp_grid
            != np.array(
                np.array(
                    [
                        [1, 1],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                    ]
                )
            )
        ).any()

    def test__function_returns_ndarray_1d__interpolation_used_and_accurate(self):

        # noinspection PyUnusedLocal
        class MockInterpolateObj:
            def ndarray_1d_from_grid(self, profile, grid):
                return ndarray_1d_from_grid(profile=profile, grid=grid)

        cls = MockInterpolateObj()

        mask = aa.Mask2D.circular_annular(
            shape_native=(20, 20),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
            inner_radius=3.0,
            outer_radius=8.0,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        true_array = ndarray_1d_from_grid(profile=None, grid=grid)

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=1.0)

        interpolated_array = grid.result_from_func(
            func=cls.ndarray_1d_from_grid, cls=MockInterpolateObj()
        )

        assert interpolated_array.shape[0] == mask.pixels_in_mask
        assert (true_array == interpolated_array).all()

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

        interpolated_array = grid.result_from_func(
            func=cls.ndarray_1d_from_grid, cls=MockInterpolateObj()
        )

        assert interpolated_array.shape[0] == mask.pixels_in_mask
        assert true_array[0] != interpolated_array[0]
        assert np.max(true_array - interpolated_array) < 0.001

        mask = aa.Mask2D.circular_annular(
            shape_native=(28, 28),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
            inner_radius=3.0,
            outer_radius=8.0,
            centre=(3.0, 3.0),
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        true_array = ndarray_1d_from_grid(profile=None, grid=grid)

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

        interpolated_array = grid.result_from_func(
            func=cls.ndarray_1d_from_grid, cls=MockInterpolateObj()
        )

        assert interpolated_array.shape[0] == mask.pixels_in_mask
        assert true_array[0] != interpolated_array[0]
        assert np.max(true_array - interpolated_array) < 0.001

    def test__function_returns_ndarray_2d__interpolation_used_and_accurate(self):

        # noinspection PyUnusedLocal
        class MockInterpolateObj:
            def ndarray_2d_from_grid(self, profile, grid):
                return ndarray_2d_from_grid(profile=profile, grid=grid)

        cls = MockInterpolateObj()

        mask = aa.Mask2D.circular_annular(
            shape_native=(20, 20),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
            inner_radius=3.0,
            outer_radius=8.0,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        true_grid = ndarray_2d_from_grid(profile=None, grid=grid)

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=1.0)

        interpolated_grid = grid.result_from_func(
            func=cls.ndarray_2d_from_grid, cls=MockInterpolateObj()
        )

        assert interpolated_grid.shape[0] == mask.pixels_in_mask
        assert interpolated_grid.shape[1] == 2
        assert (true_grid == interpolated_grid).all()

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

        interpolated_grid = grid.result_from_func(
            func=cls.ndarray_2d_from_grid, cls=MockInterpolateObj()
        )

        assert interpolated_grid.shape[0] == mask.pixels_in_mask
        assert interpolated_grid.shape[1] == 2
        assert true_grid[0, 0] != interpolated_grid[0, 0]
        assert np.max(true_grid[:, 0] - interpolated_grid[:, 0]) < 0.001
        assert np.max(true_grid[:, 1] - interpolated_grid[:, 1]) < 0.001

        mask = aa.Mask2D.circular_annular(
            shape_native=(28, 28),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
            inner_radius=3.0,
            outer_radius=8.0,
            centre=(3.0, 3.0),
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        true_grid = ndarray_2d_from_grid(profile=None, grid=grid)

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

        interpolated_grid = grid.result_from_func(
            func=cls.ndarray_2d_from_grid, cls=MockInterpolateObj()
        )

        assert interpolated_grid.shape[0] == mask.pixels_in_mask
        assert interpolated_grid.shape[1] == 2
        assert true_grid[0, 0] != interpolated_grid[0, 0]
        assert np.max(true_grid[:, 0] - interpolated_grid[:, 0]) < 0.01
        assert np.max(true_grid[:, 1] - interpolated_grid[:, 1]) < 0.01


class TestAPI:
    def test__manual_slim(self):

        grid = aa.Grid2DInterpolate.manual_slim(
            grid=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            shape_native=(2, 2),
            pixel_scales=1.0,
            pixel_scales_interp=0.1,
            origin=(0.0, 1.0),
        )

        assert type(grid) == aa.Grid2DInterpolate
        assert type(grid.slim) == aa.Grid2DInterpolate
        assert type(grid.native) == aa.Grid2DInterpolate
        assert (
            grid == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            grid.native
            == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.pixel_scales_interp == (0.1, 0.1)
        assert grid.origin == (0.0, 1.0)

        grid = aa.Grid2DInterpolate.manual_slim(
            grid=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            shape_native=(1, 1),
            pixel_scales=1.0,
            pixel_scales_interp=0.1,
            sub_size=2,
            origin=(0.0, 1.0),
        )

        assert type(grid) == aa.Grid2DInterpolate
        assert type(grid.slim) == aa.Grid2DInterpolate
        assert type(grid.native) == aa.Grid2DInterpolate
        assert (
            grid == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            grid.native
            == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (grid.binned.native == np.array([[[4.0, 5.0]]])).all()
        assert (grid.binned == np.array([[4.0, 5.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.pixel_scales_interp == (0.1, 0.1)
        assert grid.origin == (0.0, 1.0)
        assert grid.sub_size == 2

    def test__from_mask(self):

        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=1)

        grid_via_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, sub_size=1, pixel_scales=(2.0, 2.0)
        )

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

        assert type(grid) == aa.Grid2DInterpolate
        assert type(grid.slim) == aa.Grid2DInterpolate
        assert type(grid.native) == aa.Grid2DInterpolate
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.pixel_scales_interp == (0.1, 0.1)
        assert grid.sub_size == 1

        grid_via_util = aa.util.grid_2d.grid_2d_via_mask_from(
            mask_2d=mask, sub_size=1, pixel_scales=(2.0, 2.0)
        )

        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        grid_via_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=mask, sub_size=2, pixel_scales=(2.0, 2.0)
        )

        grid = aa.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

        assert type(grid) == aa.Grid2DInterpolate
        assert type(grid.slim) == aa.Grid2DInterpolate
        assert type(grid.native) == aa.Grid2DInterpolate
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.pixel_scales_interp == (0.1, 0.1)
        assert grid.sub_size == 2

    def test__uniform(self):

        grid = aa.Grid2DInterpolate.uniform(
            shape_native=(2, 2), pixel_scales=2.0, pixel_scales_interp=0.1
        )

        assert type(grid) == aa.Grid2DInterpolate
        assert type(grid.slim) == aa.Grid2DInterpolate
        assert type(grid.native) == aa.Grid2DInterpolate
        assert (
            grid == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert (
            grid.native
            == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
        ).all()
        assert (
            grid.slim == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.pixel_scales_interp == (0.1, 0.1)
        assert grid.origin == (0.0, 0.0)

        grid = aa.Grid2DInterpolate.uniform(
            shape_native=(2, 1), pixel_scales=1.0, pixel_scales_interp=0.2, sub_size=2
        )

        assert type(grid) == aa.Grid2DInterpolate
        assert type(grid.slim) == aa.Grid2DInterpolate
        assert type(grid.native) == aa.Grid2DInterpolate
        assert (
            grid.native
            == np.array(
                [
                    [[0.75, -0.25], [0.75, 0.25]],
                    [[0.25, -0.25], [0.25, 0.25]],
                    [[-0.25, -0.25], [-0.25, 0.25]],
                    [[-0.75, -0.25], [-0.75, 0.25]],
                ]
            )
        ).all()
        assert (
            grid.slim
            == np.array(
                [
                    [0.75, -0.25],
                    [0.75, 0.25],
                    [0.25, -0.25],
                    [0.25, 0.25],
                    [-0.25, -0.25],
                    [-0.25, 0.25],
                    [-0.75, -0.25],
                    [-0.75, 0.25],
                ]
            )
        ).all()
        assert (grid.binned.native == np.array([[[0.5, 0.0]], [[-0.5, 0.0]]])).all()
        assert (grid.binned == np.array([[0.5, 0.0], [-0.5, 0.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.pixel_scales_interp == (0.2, 0.2)
        assert grid.origin == (0.0, 0.0)
        assert grid.sub_size == 2
