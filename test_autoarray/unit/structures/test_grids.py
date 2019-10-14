import numpy as np
import pytest

import autoarray as aa
from autoarray import exc

@pytest.fixture(name="grid")
def make_grid():
    mask = aa.ScaledSubMask(
        np.array([[True, False, True], [False, False, False], [True, False, True]]),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    return aa.ScaledSubGrid.from_mask(mask=mask)


class TestAPIFactory:

    class TestGrid:

        def test__grid__makes_scaled_grid_with_pixel_scale(self):

            grid = aa.grid(grid=[[[1.0, 2.0], [3.0, 4.0]],
                                 [[5.0, 6.0], [7.0, 8.0]]], pixel_scales=1.0)

            assert type(grid) == aa.ScaledGrid
            assert (grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])).all()
            assert (grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
            assert grid.geometry.pixel_scales == (1.0, 1.0)
            assert grid.geometry.origin == (0.0, 0.0)

            grid = aa.grid(grid=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], shape_2d=(2,2), pixel_scales=1.0, origin=(0.0, 1.0))

            assert type(grid) == aa.ScaledGrid
            assert (grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])).all()
            assert (grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
            assert grid.geometry.pixel_scales == (1.0, 1.0)
            assert grid.geometry.origin == (0.0, 1.0)

            grid = aa.grid(grid=[[1.0, 2.0], [3.0, 4.0]], shape_2d=(2,1), pixel_scales=(2.0, 3.0))

            assert type(grid) == aa.ScaledGrid
            assert (grid.in_2d == np.array([[[1.0, 2.0]], [[3.0, 4.0]]])).all()
            assert (grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
            assert grid.geometry.pixel_scales == (2.0, 3.0)
            assert grid.geometry.origin == (0.0, 0.0)

        def test__grid__makes_scaled_sub_grid_with_pixel_scale_and_sub_size(self):

            grid = aa.grid(grid=[[[1.0, 2.0], [3.0, 4.0]],
                                 [[5.0, 6.0], [7.0, 8.0]]], pixel_scales=1.0, sub_size=1)

            assert type(grid) == aa.ScaledSubGrid
            assert (grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])).all()
            assert (grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
            assert (grid.in_2d_binned == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])).all()
            assert (grid.in_1d_binned == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
            assert grid.geometry.pixel_scales == (1.0, 1.0)
            assert grid.geometry.origin == (0.0, 0.0)
            assert grid.mask.sub_size == 1

            grid = aa.grid(grid=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                           shape_2d=(1,1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0))

            assert type(grid) == aa.ScaledSubGrid
            assert (grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])).all()
            assert (grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
            assert (grid.in_2d_binned == np.array([[[4.0, 5.0]]])).all()
            assert (grid.in_1d_binned == np.array([[4.0, 5.0]])).all()
            assert grid.geometry.pixel_scales == (1.0, 1.0)
            assert grid.geometry.origin == (0.0, 1.0)
            assert grid.mask.sub_size == 2

        def test__grid__input_is_1d_grid__no_shape_2d__raises_exception(self):

            with pytest.raises(exc.GridException):

                aa.grid(grid=[1.0, 2.0, 3.0], pixel_scales=1.0)

            with pytest.raises(exc.GridException):

                aa.grid(grid=[1.0, 2.0, 3.0], pixel_scales=1.0)

            with pytest.raises(exc.GridException):

                aa.grid(grid=[1.0, 2.0, 3.0], pixel_scales=1.0, sub_size=1)

            with pytest.raises(exc.GridException):

                aa.grid(grid=[1.0, 2.0, 3.0], pixel_scales=1.0, sub_size=1)

class TestGrid:

    def test__constructor_class_method_in_2d(self):

        grid = aa.ScaledGrid.from_grid_2d_and_pixel_scales(grid_2d=np.ones((3, 3, 2)), pixel_scales=(1.0, 1.0))

        assert type(grid) == aa.ScaledGrid
        assert type(grid.mask) == aa.ScaledMask
        assert (grid.in_1d == np.ones((9, 2))).all()
        assert (grid.in_2d == np.ones((3, 3, 2))).all()
        assert grid.mask.geometry.pixel_scale == 1.0
        assert grid.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert grid.geometry.shape_arcsec == pytest.approx((3.0, 3.0))
        assert grid.geometry.arc_second_maxima == (1.5, 1.5)
        assert grid.geometry.arc_second_minima == (-1.5, -1.5)

        grid = aa.ScaledGrid.from_grid_2d_and_pixel_scales(grid_2d=np.ones((3, 4, 2)), pixel_scales=(0.1, 0.1))

        assert type(grid) == aa.ScaledGrid
        assert type(grid.mask) == aa.ScaledMask
        assert (grid.in_1d == np.ones((12, 2))).all()
        assert (grid.in_2d == np.ones((3, 4, 2))).all()
        assert grid.mask.geometry.pixel_scale == 0.1
        assert grid.geometry.central_pixel_coordinates == (1.0, 1.5)
        assert grid.geometry.shape_arcsec == pytest.approx((0.3, 0.4))
        assert grid.geometry.arc_second_maxima == pytest.approx((0.15, 0.2), 1e-4)
        assert grid.geometry.arc_second_minima == pytest.approx((-0.15, -0.2), 1e-4)

        grid = aa.ScaledGrid.from_grid_2d_and_pixel_scales(
            grid_2d=np.ones((4, 3, 2)), pixel_scales=(0.1, 0.1), origin=(1.0, 1.0)
        )

        assert type(grid) == aa.ScaledGrid
        assert type(grid.mask) == aa.ScaledMask
        assert (grid.in_1d == np.ones((12, 2))).all()
        assert (grid.in_2d == np.ones((4, 3, 2))).all()
        assert grid.mask.geometry.pixel_scale == 0.1
        assert grid.geometry.central_pixel_coordinates == (1.5, 1.0)
        assert grid.geometry.shape_arcsec == pytest.approx((0.4, 0.3))
        assert grid.geometry.arc_second_maxima == pytest.approx((1.2, 1.15), 1e-4)
        assert grid.geometry.arc_second_minima == pytest.approx((0.8, 0.85), 1e-4)

        grid = aa.ScaledGrid.from_grid_2d_and_pixel_scales(
            grid_2d=np.ones((3, 3, 2)), pixel_scales=(2.0, 1.0), origin=(-1.0, -2.0)
        )

        assert type(grid) == aa.ScaledGrid
        assert type(grid.mask) == aa.ScaledMask
        assert grid.in_1d == pytest.approx(np.ones((9, 2)), 1e-4)
        assert grid.in_2d == pytest.approx(np.ones((3, 3, 2)), 1e-4)
        assert grid.mask.geometry.pixel_scales == (2.0, 1.0)
        assert grid.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert grid.geometry.shape_arcsec == pytest.approx((6.0, 3.0))
        assert grid.geometry.origin == (-1.0, -2.0)
        assert grid.geometry.arc_second_maxima == pytest.approx((2.0, -0.5), 1e-4)
        assert grid.geometry.arc_second_minima == pytest.approx((-4.0, -3.5), 1e-4)

    def test__constructor_class_method_in_1d(self):

        grid = aa.ScaledGrid.from_grid_1d_shape_2d_and_pixel_scales(
            grid_1d=np.ones((9, 2)), shape_2d=(3,3), pixel_scales=(2.0, 1.0), origin=(-1.0, -2.0)
        )

        assert type(grid) == aa.ScaledGrid
        assert type(grid.mask) == aa.ScaledMask
        assert grid.in_1d == pytest.approx(np.ones((9,2)), 1e-4)
        assert grid.in_2d == pytest.approx(np.ones((3, 3, 2)), 1e-4)
        assert grid.mask.geometry.pixel_scales == (2.0, 1.0)
        assert grid.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert grid.geometry.shape_arcsec == pytest.approx((6.0, 3.0))
        assert grid.geometry.origin == (-1.0, -2.0)
        assert grid.geometry.arc_second_maxima == pytest.approx((2.0, -0.5), 1e-4)
        assert grid.geometry.arc_second_minima == pytest.approx((-4.0, -3.5), 1e-4)

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

        mask = aa.ScaledSubMask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_mask_util = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask=mask, kernel_shape=(3, 5)
        )

        blurring_grid_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid = aa.ScaledGrid.from_mask(mask=mask)

        blurring_grid = grid.blurring_grid_from_kernel_shape(kernel_shape=(3, 5))

        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.geometry.pixel_scale == 2.0

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

        mask = aa.ScaledSubMask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_mask_util = aa.mask_util.blurring_mask_from_mask_and_kernel_shape(
            mask=mask, kernel_shape=(3, 5)
        )

        blurring_grid_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        mask = aa.ScaledSubMask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)
        blurring_grid = aa.ScaledGrid.blurring_grid_from_mask_and_kernel_shape(
            mask=mask, kernel_shape=(3, 5)
        )

        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.geometry.pixel_scale == 2.0

    def test__masked_shape_arcsec(self):
        mask = aa.ScaledSubMask.circular(
            shape=(3, 3), radius_arcsec=1.0, pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = aa.Grid(grid_1d=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=mask)
        assert grid.shape_arcsec == (3.0, 2.0)

        grid = aa.Grid(
            grid_1d=np.array([[1.5, 1.0], [-1.5, -1.0], [0.1, 0.1]]), mask=mask
        )
        assert grid.shape_arcsec == (3.0, 2.0)

        grid = aa.Grid(
            grid_1d=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0]]), mask=mask
        )
        assert grid.shape_arcsec == (4.5, 4.0)

        grid = aa.Grid(
            grid_1d=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0], [7.0, -5.0]]),
            mask=mask,
        )
        assert grid.shape_arcsec == (8.5, 8.0)

    def test__in_radians(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(2.0, 2.0))

        grid = aa.ScaledGrid.from_mask(mask=mask)

        assert grid.in_radians[0, 0] == pytest.approx(0.00000969627362, 1.0e-8)
        assert grid.in_radians[0, 1] == pytest.approx(0.00000484813681, 1.0e-8)

        assert grid.in_radians[0, 0] == pytest.approx(
            2.0 * np.pi / (180 * 3600), 1.0e-8
        )
        assert grid.in_radians[0, 1] == pytest.approx(
            1.0 * np.pi / (180 * 3600), 1.0e-8
        )

    def test__yticks(self):

        mask = aa.ScaledSubMask.circular(
            shape=(3, 3), radius_arcsec=1.0, pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = aa.Grid(grid_1d=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        grid = aa.Grid(grid_1d=np.array([[3.0, 1.0], [-3.0, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        grid = aa.Grid(grid_1d=np.array([[5.0, 3.5], [2.0, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__xticks(self):
        mask = aa.ScaledSubMask.circular(
            shape=(3, 3), radius_arcsec=1.0, pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = aa.Grid(grid_1d=np.array([[1.0, 1.5], [-1.0, -1.5]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        grid = aa.Grid(grid_1d=np.array([[1.0, 3.0], [-1.0, -3.0]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        grid = aa.Grid(grid_1d=np.array([[3.5, 2.0], [-1.0, 5.0]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__new_grid__with_interpolator__returns_grid_with_interpolator(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(2.0, 2.0))

        grid = aa.ScaledGrid.from_mask(mask=mask)

        grid_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=1.0
        )

        assert (grid[:, :] == grid_with_interp[:, :]).all()
        assert (grid.mask == grid_with_interp.mask).all()

        interpolator_manual = aa.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=1.0
        )

        assert (grid.interpolator.vtx == interpolator_manual.vtx).all()
        assert (grid.interpolator.wts == interpolator_manual.wts).all()

    def test__new_grid__with_binned__returns_grid_with_binned(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = aa.ScaledMask(array_2d=mask, pixel_scales=(2.0, 2.0))

        grid = aa.ScaledGrid.from_mask(mask=mask)

        grid.new_grid_with_binned_grid(binned_grid=1)

        assert grid.binned == 1


class TestSubGrid:

    def test__constructor_class_method_in_2d(self):

        grid = aa.ScaledSubGrid.from_sub_grid_2d_pixel_scales_and_sub_size(sub_grid_2d=np.ones((3, 3, 2)), sub_size=1, pixel_scales=(1.0, 1.0))

        assert (grid.in_1d == np.ones((9, 2))).all()
        assert (grid.in_2d == np.ones((3, 3, 2))).all()
        assert (grid.in_1d_binned == np.ones((9, 2))).all()
        assert (grid.in_2d_binned == np.ones((3, 3, 2))).all()
        assert grid.mask.geometry.pixel_scale == 1.0
        assert grid.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert grid.geometry.shape_arcsec == pytest.approx((3.0, 3.0))
        assert grid.geometry.arc_second_maxima == (1.5, 1.5)
        assert grid.geometry.arc_second_minima == (-1.5, -1.5)

        grid = aa.ScaledSubGrid.from_sub_grid_2d_pixel_scales_and_sub_size(sub_grid_2d=np.ones((4, 4, 2)), sub_size=2, pixel_scales=(0.1, 0.1))

        assert (grid.in_1d == np.ones((16, 2))).all()
        assert (grid.in_2d == np.ones((4, 4, 2))).all()
        assert (grid.in_1d_binned == np.ones((4, 2))).all()
        assert (grid.in_2d_binned == np.ones((2, 2, 2))).all()
        assert grid.mask.geometry.pixel_scale == 0.1
        assert grid.geometry.central_pixel_coordinates == (0.5, 0.5)
        assert grid.geometry.shape_arcsec == pytest.approx((0.2, 0.2))
        assert grid.geometry.arc_second_maxima == pytest.approx((0.1, 0.1), 1e-4)
        assert grid.geometry.arc_second_minima == pytest.approx((-0.1, -0.1), 1e-4)

        grid = aa.ScaledSubGrid.from_sub_grid_2d_pixel_scales_and_sub_size(
            sub_grid_2d=np.array([[[1.0, 2.0],
                                   [3.0, 4.0]],
                                   [[5.0, 6.0],
                                   [7.0, 8.0]]]), pixel_scales=(0.1, 0.1), sub_size=2, origin=(1.0, 1.0)
        )

        assert grid.in_2d.shape == (2, 2, 2)
        assert (grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
        assert (grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])).all()
        assert grid.in_2d_binned.shape == (1, 1, 2)
        assert (grid.in_1d_binned == np.array([4.0, 5.0])).all()
        assert (grid.in_2d_binned == np.array([[4.0, 5.0]])).all()
        assert grid.mask.geometry.pixel_scale == 0.1
        assert grid.geometry.central_pixel_coordinates == (0.0, 0.0)
        assert grid.geometry.shape_arcsec == pytest.approx((0.1, 0.1))
        assert grid.geometry.arc_second_maxima == pytest.approx((1.05, 1.05), 1e-4)
        assert grid.geometry.arc_second_minima == pytest.approx((0.95, 0.95), 1e-4)

        grid = aa.ScaledSubGrid.from_sub_grid_2d_pixel_scales_and_sub_size(
            sub_grid_2d=np.ones((3, 3, 2)), pixel_scales=(2.0, 1.0), sub_size=1, origin=(-1.0, -2.0)
        )

        assert grid.in_1d == pytest.approx(np.ones((9, 2)), 1e-4)
        assert grid.in_2d == pytest.approx(np.ones((3, 3, 2)), 1e-4)
        assert grid.mask.geometry.pixel_scales == (2.0, 1.0)
        assert grid.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert grid.geometry.shape_arcsec == pytest.approx((6.0, 3.0))
        assert grid.geometry.origin == (-1.0, -2.0)
        assert grid.geometry.arc_second_maxima == pytest.approx((2.0, -0.5), 1e-4)
        assert grid.geometry.arc_second_minima == pytest.approx((-4.0, -3.5), 1e-4)

    def test__constructor_class_method_in_1d(self):

        grid = aa.ScaledSubGrid.from_sub_grid_1d_shape_2d_pixel_scales_and_sub_size(
            sub_grid_1d=np.ones((9, 2)), shape_2d=(3,3), pixel_scales=(2.0, 1.0), sub_size=1, origin=(-1.0, -2.0)
        )

        assert grid.in_1d == pytest.approx(np.ones((9,2)), 1e-4)
        assert grid.in_2d == pytest.approx(np.ones((3, 3, 2)), 1e-4)
        assert grid.mask.geometry.pixel_scales == (2.0, 1.0)
        assert grid.geometry.central_pixel_coordinates == (1.0, 1.0)
        assert grid.geometry.shape_arcsec == pytest.approx((6.0, 3.0))
        assert grid.geometry.origin == (-1.0, -2.0)
        assert grid.geometry.arc_second_maxima == pytest.approx((2.0, -0.5), 1e-4)
        assert grid.geometry.arc_second_minima == pytest.approx((-4.0, -3.5), 1e-4)

    def test__from_mask__compare_to_array_util(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = aa.ScaledSubMask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=1)

        grid_via_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, sub_size=1, pixel_scales=(2.0, 2.0)
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        assert type(grid) == aa.ScaledSubGrid
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.geometry.pixel_scale == 2.0
        assert grid.interpolator == None

        grid_2d = mask.mapping.sub_grid_2d_from_sub_grid_1d(sub_grid_1d=grid)

        assert (grid.in_2d == grid_2d).all()

        mask = np.array([[True, True, True], [True, False, False], [True, True, False]])

        mask = aa.ScaledSubMask(mask, pixel_scales=(3.0, 3.0), sub_size=2)

        grid_via_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, pixel_scales=(3.0, 3.0), sub_size=2
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        assert grid == pytest.approx(grid_via_util, 1e-4)

    def test__from_shape_and_pixel_scale__compare_to_grid_util(self):
        mask = np.array(
            [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        )
        mask = aa.ScaledSubMask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=1)

        grid_via_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid = aa.ScaledSubGrid.from_shape_2d_pixel_scale_and_sub_size(
            shape_2d=(3, 4), pixel_scale=2.0, sub_size=1
        )

        assert type(grid) == aa.ScaledSubGrid
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.geometry.pixel_scale == 2.0

        mask = np.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )

        grid_via_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=mask, pixel_scales=(3.0, 3.0), sub_size=2
        )

        grid = aa.ScaledSubGrid.from_shape_2d_pixel_scale_and_sub_size(
            shape_2d=(3, 3), pixel_scale=3.0, sub_size=2
        )

        assert grid == pytest.approx(grid_via_util, 1e-4)

    def test__padded_grid_from_kernel_shape__matches_grid_2d_after_padding(self):

        grid = aa.ScaledSubGrid.from_shape_2d_pixel_scale_and_sub_size(
            shape_2d=(4, 4), pixel_scale=3.0, sub_size=1
        )

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape=(3, 3))

        padded_grid_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((6, 6), False), pixel_scales=(3.0, 3.0), sub_size=1
        )

        assert padded_grid.shape == (36, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 6))).all()
        assert (padded_grid == padded_grid_util).all()
        assert padded_grid.interpolator is None

        grid = aa.ScaledSubGrid.from_shape_2d_pixel_scale_and_sub_size(
            shape_2d=(4, 5), pixel_scale=2.0, sub_size=1
        )

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape=(3, 3))

        padded_grid_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((6, 7), False), pixel_scales=(2.0, 2.0), sub_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = aa.ScaledSubGrid.from_shape_2d_pixel_scale_and_sub_size(
            shape_2d=(5, 4), pixel_scale=1.0, sub_size=1
        )

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape=(3, 3))

        padded_grid_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((7, 6), False), pixel_scales=(1.0, 1.0), sub_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = aa.ScaledSubGrid.from_shape_2d_pixel_scale_and_sub_size(
            shape_2d=(5, 5), pixel_scale=8.0, sub_size=1
        )

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape=(2, 5))

        padded_grid_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=1
        )

        assert padded_grid.shape == (54, 2)
        assert (padded_grid == padded_grid_util).all()

        mask = aa.ScaledSubMask(
            array_2d=np.full((5, 4), False), pixel_scales=(2.0, 2.0), sub_size=2
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape=(3, 3))

        padded_grid_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((7, 6), False), pixel_scales=(2.0, 2.0), sub_size=2
        )

        assert padded_grid.shape == (168, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(7, 6))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)
        assert padded_grid.interpolator is None

        mask = aa.ScaledSubMask(
            array_2d=np.full((2, 5), False), pixel_scales=(8.0, 8.0), sub_size=4
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape=(5, 5))

        padded_grid_util = aa.grid_util.grid_1d_from_mask_pixel_scales_sub_size_and_origin(
            mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=4
        )

        assert padded_grid.shape == (864, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 9))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)

    def test__padded_grid_from_kernel_shape__has_interpolator_grid_if_had_one_before(
        self
    ):
        grid = aa.ScaledSubGrid.from_shape_2d_pixel_scale_and_sub_size(
            shape_2d=(4, 4), pixel_scale=3.0, sub_size=1
        )

        grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=0.1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape=(3, 3))

        assert padded_grid.interpolator is not None
        assert padded_grid.interpolator.pixel_scale_interpolation_grid == 0.1

        mask = aa.ScaledSubMask.unmasked_from_shape(
            shape=(6, 6), pixel_scales=(3.0, 3.0), sub_size=1
        )

        interpolator = aa.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=padded_grid, pixel_scale_interpolation_grid=0.1
        )

        assert (padded_grid.interpolator.vtx == interpolator.vtx).all()
        assert (padded_grid.interpolator.wts == interpolator.wts).all()

        mask = aa.ScaledSubMask(
            array_2d=np.full((5, 4), False), pixel_scales=(2.0, 2.0), sub_size=2
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=0.1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape=(3, 3))

        assert padded_grid.interpolator is not None
        assert padded_grid.interpolator.pixel_scale_interpolation_grid == 0.1

        mask = aa.ScaledSubMask.unmasked_from_shape(
            shape=(7, 6), pixel_scales=(2.0, 2.0), sub_size=2
        )

        interpolator = aa.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=padded_grid, pixel_scale_interpolation_grid=0.1
        )

        assert (padded_grid.interpolator.vtx == interpolator.vtx).all()
        assert (padded_grid.interpolator.wts == interpolator.wts).all()

    def test__sub_border_1d_indexes__compare_to_array_util(self):
        mask = np.array(
            [
                [False, False, False, False, False, False, False, True],
                [False, True, True, True, True, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, False, True, False, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, True, True, True, True, False, True],
                [False, False, False, False, False, False, False, True],
            ]
        )

        mask = aa.ScaledSubMask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        sub_border_1d_indexes_util = aa.mask_util.sub_border_pixel_1d_indexes_from_mask_and_sub_size(
            mask=mask, sub_size=2
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        assert grid.regions._sub_border_1d_indexes == pytest.approx(
            sub_border_1d_indexes_util, 1e-4
        )


class TestGridBorder(object):
    def test__sub_border_grid_for_simple_mask(self):
        mask = np.array(
            [
                [False, False, False, False, False, False, False, True],
                [False, True, True, True, True, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, False, True, False, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, True, True, True, True, False, True],
                [False, False, False, False, False, False, False, True],
            ]
        )

        mask = aa.ScaledSubMask(array_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        assert (
            grid.sub_border_grid
            == np.array(
                [
                    [6.5, -7.5],
                    [6.5, -5.5],
                    [6.5, -3.5],
                    [6.5, -0.5],
                    [6.5, 1.5],
                    [6.5, 3.5],
                    [6.5, 5.5],
                    [4.5, -7.5],
                    [4.5, 5.5],
                    [2.5, -7.5],
                ]
            )
        ).all()

    def test__inside_border_no_relocations(self):
        mask = aa.ScaledSubMask.circular(
            shape=(30, 30), radius_arcsec=1.0, pixel_scales=(0.1, 0.1), sub_size=1
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        grid_to_relocate = aa.Grid(
            grid_1d=np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert (
            relocated_grid == np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]])
        ).all()
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.mask.sub_size == 1

        mask = aa.ScaledSubMask.circular(
            shape=(30, 30), radius_arcsec=1.0, pixel_scales=(0.1, 0.1), sub_size=2
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        grid_to_relocate = aa.Grid(
            grid_1d=np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert (
            relocated_grid == np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]])
        ).all()
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.mask.sub_size == 2

    def test__outside_border_are_relocations(self):
        mask = aa.ScaledSubMask.circular(
            shape=(30, 30), radius_arcsec=1.0, pixel_scales=(0.1, 0.1), sub_size=1
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        grid_to_relocate = aa.Grid(
            grid_1d=np.array([[10.1, 0.0], [0.0, 10.1], [-10.1, -10.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array([[0.95, 0.0], [0.0, 0.95], [-0.7017, -0.7017]]), 0.1
        )
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.mask.sub_size == 1

        mask = aa.ScaledSubMask.circular(
            shape=(30, 30), radius_arcsec=1.0, pixel_scales=(0.1, 0.1), sub_size=2
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        grid_to_relocate = aa.Grid(
            grid_1d=np.array([[10.1, 0.0], [0.0, 10.1], [-10.1, -10.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array([[0.9778, 0.0], [0.0, 0.97788], [-0.7267, -0.7267]]), 0.1
        )
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.mask.sub_size == 2

    def test__outside_border_are_relocations__positive_origin_included_in_relocate(
        self
    ):
        mask = aa.ScaledSubMask.circular(
            shape=(60, 60),
            radius_arcsec=1.0,
            pixel_scales=(0.1, 0.1),
            centre=(1.0, 1.0),
            sub_size=1,
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        grid_to_relocate = aa.Grid(
            grid_1d=np.array([[11.1, 1.0], [1.0, 11.1], [-11.1, -11.1]]),
            sub_size=1,
            mask=mask,
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array(
                [[2.0, 1.0], [1.0, 2.0], [1.0 - np.sqrt(2) / 2, 1.0 - np.sqrt(2) / 2]]
            ),
            0.1,
        )
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.mask.sub_size == 1

        mask = aa.ScaledSubMask.circular(
            shape=(60, 60),
            radius_arcsec=1.0,
            pixel_scales=(0.1, 0.1),
            centre=(1.0, 1.0),
            sub_size=2,
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        grid_to_relocate = aa.Grid(
            grid_1d=np.array([[11.1, 1.0], [1.0, 11.1], [-11.1, -11.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array(
                [
                    [1.9263, 1.0 - 0.0226],
                    [1.0 - 0.0226, 1.9263],
                    [1.0 - 0.7267, 1.0 - 0.7267],
                ]
            ),
            0.1,
        )
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.mask.sub_size == 2


class TestBinnedGrid:
    def test__from_mask_and_pixel_scale_binned_grid__correct_binned_bin_up_calculated(
        self, mask_7x7, grid_7x7
    ):
        mask_7x7.pixel_scales = (1.0, 1.0)
        binned_grid = aa.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
            mask=mask_7x7, pixel_scale_binned_grid=1.0
        )

        assert (binned_grid == grid_7x7).all()
        assert (binned_grid.mask == mask_7x7).all()
        assert binned_grid.bin_up_factor == 1
        assert (
            binned_grid.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
        ).all()

        mask_7x7.pixel_scales = (1.0, 1.0)
        binned_grid = aa.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
            mask=mask_7x7, pixel_scale_binned_grid=1.9
        )

        assert binned_grid.bin_up_factor == 1
        assert (binned_grid.mask == mask_7x7).all()
        assert (
            binned_grid.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
        ).all()

        mask_7x7.pixel_scales = (1.0, 1.0)
        binned_grid = aa.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
            mask=mask_7x7, pixel_scale_binned_grid=2.0
        )

        assert binned_grid.bin_up_factor == 2
        assert (
            binned_grid.mask
            == np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            )
        ).all()

        assert (
            binned_grid
            == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert (
            binned_grid.binned_mask_1d_index_to_mask_1d_indexes
            == np.array([[0, -1, -1, -1], [1, 2, -1, -1], [3, 6, -1, -1], [4, 5, 7, 8]])
        ).all()

        mask_7x7.pixel_scales = (2.0, 2.0)
        binned_grid = aa.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
            mask=mask_7x7, pixel_scale_binned_grid=1.0
        )

        assert binned_grid.bin_up_factor == 1


class TestPixelizationGrid:
    def test__pixelization_grid__attributes(self):
        pix_grid = aa.PixelizationGrid(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0]]),
            nearest_pixelization_1d_index_for_mask_1d_index=np.array([0, 1]),
        )

        assert type(pix_grid) == aa.PixelizationGrid
        assert (pix_grid == np.array([[1.0, 1.0], [2.0, 2.0]])).all()
        assert (
            pix_grid.nearest_pixelization_1d_index_for_mask_1d_index == np.array([0, 1])
        ).all()

    def test__from_unmasked_sparse_shape_and_grid(self):
        mask = aa.ScaledSubMask(
            array_2d=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scales=(0.5, 0.5),
            sub_size=1,
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        sparse_to_grid = aa.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=(10, 10), grid=grid
        )

        pixelization_grid = aa.PixelizationGrid.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=(10, 10), grid=grid
        )

        assert (sparse_to_grid.sparse == pixelization_grid).all()
        assert (
            sparse_to_grid.sparse_1d_index_for_mask_1d_index
            == pixelization_grid.nearest_pixelization_1d_index_for_mask_1d_index
        ).all()


class TestSparseToGrid:
    class TestUnmaskedShape:
        def test__properties_consistent_with_util(self):
            mask = aa.ScaledSubMask(
                array_2d=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                pixel_scales=(0.5, 0.5),
                sub_size=1,
            )

            grid = aa.ScaledSubGrid.from_mask(mask=mask)

            sparse_to_grid = aa.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(10, 10), grid=grid
            )

            unmasked_sparse_grid_util = aa.grid_util.grid_1d_from_shape_pixel_scales_sub_size_and_origin(
                shape=(10, 10), pixel_scales=(0.15, 0.15), sub_size=1, origin=(0.0, 0.0)
            )

            unmasked_sparse_grid_pixel_centres = aa.grid_util.grid_pixel_centres_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
                grid_arcsec_1d=unmasked_sparse_grid_util,
                shape=grid.mask.shape,
                pixel_scales=grid.geometry.pixel_scales,
            ).astype(
                "int"
            )

            total_sparse_pixels = aa.mask_util.total_sparse_pixels_from_mask(
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            )

            regular_to_unmasked_sparse_util = aa.grid_util.grid_pixel_indexes_1d_from_grid_arcsec_1d_shape_and_pixel_scales(
                grid_arcsec_1d=grid,
                shape=(10, 10),
                pixel_scales=(0.15, 0.15),
                origin=(0.0, 0.0),
            ).astype(
                "int"
            )

            unmasked_sparse_for_sparse_util = aa.sparse_util.unmasked_sparse_for_sparse_from_mask_and_pixel_centres(
                total_sparse_pixels=total_sparse_pixels,
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            ).astype(
                "int"
            )

            sparse_for_unmasked_sparse_util = aa.sparse_util.sparse_for_unmasked_sparse_from_mask_and_pixel_centres(
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
                total_sparse_pixels=total_sparse_pixels,
            ).astype(
                "int"
            )

            sparse_1d_index_for_mask_1d_index_util = aa.sparse_util.sparse_1d_index_for_mask_1d_index_from_sparse_mappings(
                regular_to_unmasked_sparse=regular_to_unmasked_sparse_util,
                sparse_for_unmasked_sparse=sparse_for_unmasked_sparse_util,
            )

            sparse_grid_util = aa.sparse_util.sparse_grid_from_unmasked_sparse_grid(
                unmasked_sparse_grid=unmasked_sparse_grid_util,
                unmasked_sparse_for_sparse=unmasked_sparse_for_sparse_util,
            )

            assert (
                sparse_to_grid.sparse_1d_index_for_mask_1d_index
                == sparse_1d_index_for_mask_1d_index_util
            ).all()
            assert (sparse_to_grid.sparse == sparse_grid_util).all()

        def test__sparse_grid_overlaps_mask_perfectly__masked_pixels_in_masked_sparse_grid(
            self
        ):
            mask = aa.ScaledSubMask(
                array_2d=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.ScaledSubGrid.from_mask(mask=mask)

            sparse_to_grid = aa.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_to_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
                )
            ).all()

        def test__same_as_above_but_4x3_grid_and_mask(self):
            mask = aa.ScaledSubMask(
                array_2d=np.array(
                    [
                        [True, False, True],
                        [False, False, False],
                        [False, False, False],
                        [True, False, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.ScaledSubGrid.from_mask(mask=mask)

            sparse_to_grid = aa.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(4, 3), grid=grid
            )

            assert (
                sparse_to_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [
                        [1.5, 0.0],
                        [0.5, -1.0],
                        [0.5, 0.0],
                        [0.5, 1.0],
                        [-0.5, -1.0],
                        [-0.5, 0.0],
                        [-0.5, 1.0],
                        [-1.5, 0.0],
                    ]
                )
            ).all()

        def test__same_as_above_but_3x4_grid_and_mask(self):
            mask = aa.ScaledSubMask(
                array_2d=np.array(
                    [
                        [True, False, True, True],
                        [False, False, False, False],
                        [True, False, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.ScaledSubGrid.from_mask(mask=mask)

            sparse_to_grid = aa.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 4), grid=grid
            )

            assert (
                sparse_to_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [
                        [1.0, -0.5],
                        [0.0, -1.5],
                        [0.0, -0.5],
                        [0.0, 0.5],
                        [0.0, 1.5],
                        [-1.0, -0.5],
                    ]
                )
            ).all()

        def test__mask_with_offset_centre__origin_of_sparse_to_grid_moves_to_give_same_pairings(
            self
        ):
            mask = aa.ScaledSubMask(
                array_2d=np.array(
                    [
                        [True, True, True, False, True],
                        [True, True, False, False, False],
                        [True, True, True, False, True],
                        [True, True, True, True, True],
                        [True, True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.ScaledSubGrid.from_mask(mask=mask)

            # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
            # the central (3x3) pixels only.

            sparse_to_grid = aa.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_to_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [[2.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0]]
                )
            ).all()

        def test__same_as_above_but_different_offset(self):
            mask = aa.ScaledSubMask(
                array_2d=np.array(
                    [
                        [True, True, True, True, True],
                        [True, True, True, False, True],
                        [True, True, False, False, False],
                        [True, True, True, False, True],
                        [True, True, True, True, True],
                    ]
                ),
                pixel_scales=(2.0, 2.0),
                sub_size=1,
            )

            grid = aa.ScaledSubGrid.from_mask(mask=mask)

            # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
            # the central (3x3) pixels only.

            sparse_to_grid = aa.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_to_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [[2.0, 2.0], [0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [-2.0, 2.0]]
                )
            ).all()

        def test__from_grid_and_unmasked_shape__sets_up_with_correct_shape_and_pixel_scales(
            self, mask_7x7
        ):
            grid = aa.ScaledSubGrid.from_mask(mask=mask_7x7)

            sparse_to_grid = aa.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                grid=grid, unmasked_sparse_shape=(3, 3)
            )

            assert (
                sparse_to_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
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
                )
            ).all()

        def test__same_as_above__but_4x3_image(self):
            mask = aa.ScaledSubMask(
                array_2d=np.array(
                    [
                        [True, False, True],
                        [False, False, False],
                        [False, False, False],
                        [True, False, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.ScaledSubGrid.from_mask(mask=mask)

            sparse_to_grid = aa.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(4, 3), grid=grid
            )

            assert (
                sparse_to_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [
                        [1.5, 0.0],
                        [0.5, -1.0],
                        [0.5, 0.0],
                        [0.5, 1.0],
                        [-0.5, -1.0],
                        [-0.5, 0.0],
                        [-0.5, 1.0],
                        [-1.5, 0.0],
                    ]
                )
            ).all()

        def test__same_as_above__but_3x4_image(self):
            mask = aa.ScaledSubMask(
                array_2d=np.array(
                    [
                        [True, False, True, True],
                        [False, False, False, False],
                        [True, False, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.ScaledSubGrid.from_mask(mask=mask)

            sparse_to_grid = aa.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 4), grid=grid
            )

            assert (
                sparse_to_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [
                        [1.0, -0.5],
                        [0.0, -1.5],
                        [0.0, -0.5],
                        [0.0, 0.5],
                        [0.0, 1.5],
                        [-1.0, -0.5],
                    ]
                )
            ).all()

        def test__from_grid_and_shape__offset_mask__origin_shift_corrects(self):
            mask = aa.ScaledSubMask(
                array_2d=np.array(
                    [
                        [True, True, False, False, False],
                        [True, True, False, False, False],
                        [True, True, False, False, False],
                        [True, True, True, True, True],
                        [True, True, True, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.ScaledSubGrid.from_mask(mask=mask)

            sparse_to_grid = aa.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_to_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            ).all()
            assert (
                sparse_to_grid.sparse
                == np.array(
                    [
                        [2.0, 0.0],
                        [2.0, 1.0],
                        [2.0, 2.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [1.0, 2.0],
                        [0.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 2.0],
                    ]
                )
            ).all()

    class TestUnmaskeedShapeAndWeightImage:
        def test__binned_weight_map_all_ones__kmenas_grid_is_grid_overlapping_image(
            self
        ):
            mask = aa.ScaledSubMask(
                array_2d=np.array(
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ),
                pixel_scales=(0.5, 0.5),
                sub_size=1,
            )

            binned_grid = aa.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
                mask=mask, pixel_scale_binned_grid=mask.geometry.pixel_scale
            )

            binned_weight_map = np.ones(mask.pixels_in_mask)

            sparse_to_grid_weight = aa.SparseToGrid.from_total_pixels_binned_grid_and_weight_map(
                total_pixels=8,
                binned_grid=binned_grid,
                binned_weight_map=binned_weight_map,
                n_iter=10,
                max_iter=20,
                seed=1,
            )

            assert (
                sparse_to_grid_weight.sparse
                == np.array(
                    [
                        [-0.25, 0.25],
                        [0.5, -0.5],
                        [0.75, 0.5],
                        [0.25, 0.5],
                        [-0.5, -0.25],
                        [-0.5, -0.75],
                        [-0.75, 0.5],
                        [-0.25, 0.75],
                    ]
                )
            ).all()

            assert (
                sparse_to_grid_weight.sparse_1d_index_for_mask_1d_index
                == np.array([1, 1, 2, 2, 1, 1, 3, 3, 5, 4, 0, 7, 5, 4, 6, 6])
            ).all()

        def test__binned_weight_map_changes_grid_from_above(self):
            mask = aa.ScaledSubMask(
                array_2d=np.array(
                    [
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                        [False, False, False, False],
                    ]
                ),
                pixel_scales=(0.5, 0.5),
                sub_size=2,
            )

            binned_grid = aa.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
                mask=mask, pixel_scale_binned_grid=mask.geometry.pixel_scale
            )

            binned_weight_map = np.ones(mask.pixels_in_mask)
            binned_weight_map[0:15] = 0.00000001

            sparse_to_grid_weight = aa.SparseToGrid.from_total_pixels_binned_grid_and_weight_map(
                total_pixels=8,
                binned_grid=binned_grid,
                binned_weight_map=binned_weight_map,
                n_iter=10,
                max_iter=30,
                seed=1,
            )

            assert sparse_to_grid_weight.sparse[1] == pytest.approx(
                np.array([0.4166666, -0.0833333]), 1.0e-4
            )

            assert (
                sparse_to_grid_weight.sparse_1d_index_for_mask_1d_index
                == np.array([5, 1, 0, 0, 5, 1, 1, 4, 3, 6, 7, 4, 3, 6, 2, 2])
            ).all()

        def test__binned_weight_map_all_ones__pixel_scale_binned_grid_leads_to_binning_up_by_factor_2(
            self
        ):
            mask = aa.ScaledSubMask(
                array_2d=np.full(fill_value=False, shape=(8, 8)),
                pixel_scales=(0.5, 0.5),
                sub_size=2,
            )

            binned_grid = aa.BinnedGrid.from_mask_and_pixel_scale_binned_grid(
                mask=mask, pixel_scale_binned_grid=2.0 * mask.geometry.pixel_scale
            )

            binned_weight_map = np.ones(binned_grid.shape[0])

            sparse_to_grid_weight = aa.SparseToGrid.from_total_pixels_binned_grid_and_weight_map(
                total_pixels=8,
                binned_grid=binned_grid,
                binned_weight_map=binned_weight_map,
                n_iter=10,
                max_iter=30,
                seed=1,
            )

            assert (
                sparse_to_grid_weight.sparse
                == np.array(
                    [
                        [-0.5, 0.5],
                        [1.0, -1.0],
                        [1.5, 1.0],
                        [0.5, 1.0],
                        [-1.0, -0.5],
                        [-1.0, -1.5],
                        [-1.5, 1.0],
                        [-0.5, 1.5],
                    ]
                )
            ).all()

            assert (
                sparse_to_grid_weight.sparse_1d_index_for_mask_1d_index
                == np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        1,
                        1,
                        1,
                        1,
                        2,
                        2,
                        2,
                        2,
                        1,
                        1,
                        1,
                        1,
                        3,
                        3,
                        3,
                        3,
                        1,
                        1,
                        1,
                        1,
                        3,
                        3,
                        3,
                        3,
                        5,
                        5,
                        4,
                        4,
                        0,
                        0,
                        7,
                        7,
                        5,
                        5,
                        4,
                        4,
                        0,
                        0,
                        7,
                        7,
                        5,
                        5,
                        4,
                        4,
                        6,
                        6,
                        6,
                        6,
                        5,
                        5,
                        4,
                        4,
                        6,
                        6,
                        6,
                        6,
                    ]
                )
            ).all()


@aa.grid_interpolate
def grid_radii_from_grid(profile, grid, grid_radial_minimum=None):
    """
        The radius of each point of the grid from an origin of (0.0", 0.0")

        Parameters
        ----------
        grid : grids.Grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
    grid_radii = np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))
    return np.stack((grid_radii, grid_radii), axis=-1)


class TestInterpolator:
    def test_decorated_function__values_from_function_has_1_dimensions__returns_1d_result(
        self
    ):
        # noinspection PyUnusedLocal
        @aa.grid_interpolate
        def func(profile, grid, grid_radial_minimum=None):
            result = np.zeros(grid.shape[0])
            result[0] = 1
            return result

        grid = aa.ScaledSubGrid.from_mask(
            mask=aa.ScaledSubMask.unmasked_from_shape(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
        )

        values = func(None, grid)

        assert values.ndim == 1
        assert values.shape == (9,)
        assert (values == np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])).all()

        grid = aa.ScaledSubGrid.from_mask(
            mask=aa.ScaledSubMask.unmasked_from_shape(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
        )
        grid.interpolator = aa.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            grid.mask, grid, pixel_scale_interpolation_grid=0.5
        )
        interp_values = func(None, grid)
        assert interp_values.ndim == 1
        assert interp_values.shape == (9,)
        assert (interp_values != np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])).any()

    def test_decorated_function__values_from_function_has_2_dimensions__returns_2d_result(
        self
    ):
        # noinspection PyUnusedLocal
        @aa.grid_interpolate
        def func(profile, grid, grid_radial_minimum=None):
            result = np.zeros((grid.shape[0], 2))
            result[0, :] = 1
            return result

        grid = aa.ScaledSubGrid.from_mask(
            mask=aa.ScaledSubMask.unmasked_from_shape(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
        )

        values = func(None, grid)

        assert values.ndim == 2
        assert values.shape == (9, 2)
        assert (
            values
            == np.array(
                [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            )
        ).all()

        grid = aa.ScaledSubGrid.from_mask(
            mask=aa.ScaledSubMask.unmasked_from_shape(
                shape=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
            )
        )
        grid.interpolator = aa.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            grid.mask, grid, pixel_scale_interpolation_grid=0.5
        )

        interp_values = func(None, grid)
        assert interp_values.ndim == 2
        assert interp_values.shape == (9, 2)
        assert (
            interp_values
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

    def test__20x20_deflection_angles_no_central_pixels__interpolated_accurately(self):

        mask = aa.ScaledSubMask.circular_annular(
            shape=(20, 20),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
            inner_radius_arcsec=3.0,
            outer_radius_arcsec=8.0,
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        true_grid_radii = grid_radii_from_grid(profile=None, grid=grid)

        interpolator = aa.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=1.0
        )

        interp_grid_radii = grid_radii_from_grid(
            profile=None, grid=interpolator.interp_grid
        )

        interpolated_grid_radii_y = interpolator.interpolated_values_from_values(
            values=interp_grid_radii[:, 0]
        )
        interpolated_grid_radii_x = interpolator.interpolated_values_from_values(
            values=interp_grid_radii[:, 1]
        )

        assert np.max(true_grid_radii[:, 0] - interpolated_grid_radii_y) < 0.001
        assert np.max(true_grid_radii[:, 1] - interpolated_grid_radii_x) < 0.001

    def test__move_centre_of_galaxy__interpolated_accurately(self):

        mask = aa.ScaledSubMask.circular_annular(
            shape=(24, 24),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
            inner_radius_arcsec=3.0,
            outer_radius_arcsec=8.0,
            centre=(3.0, 3.0),
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        true_grid_radii = grid_radii_from_grid(profile=None, grid=grid)

        interpolator = aa.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=1.0
        )

        interp_grid_radii = grid_radii_from_grid(
            profile=None, grid=interpolator.interp_grid
        )

        interpolated_grid_radii_y = interpolator.interpolated_values_from_values(
            values=interp_grid_radii[:, 0]
        )
        interpolated_grid_radii_x = interpolator.interpolated_values_from_values(
            values=interp_grid_radii[:, 1]
        )

        assert np.max(true_grid_radii[:, 0] - interpolated_grid_radii_y) < 0.001
        assert np.max(true_grid_radii[:, 1] - interpolated_grid_radii_x) < 0.001

    def test__different_interpolation_pixel_scales_still_works(self):

        mask = aa.ScaledSubMask.circular_annular(
            shape=(28, 28),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
            inner_radius_arcsec=3.0,
            outer_radius_arcsec=8.0,
            centre=(3.0, 3.0),
        )

        grid = aa.ScaledSubGrid.from_mask(mask=mask)

        true_grid_radii = grid_radii_from_grid(profile=None, grid=grid)

        interpolator = aa.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.2
        )

        interp_grid_radii = grid_radii_from_grid(
            profile=None, grid=interpolator.interp_grid
        )

        interpolated_grid_radii_y = interpolator.interpolated_values_from_values(
            values=interp_grid_radii[:, 0]
        )
        interpolated_grid_radii_x = interpolator.interpolated_values_from_values(
            values=interp_grid_radii[:, 1]
        )

        assert np.max(true_grid_radii[:, 0] - interpolated_grid_radii_y) < 0.001
        assert np.max(true_grid_radii[:, 1] - interpolated_grid_radii_x) < 0.001

        interpolator = aa.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_grid_radii_values = grid_radii_from_grid(
            profile=None, grid=interpolator.interp_grid
        )

        interpolated_grid_radii_y = interpolator.interpolated_values_from_values(
            values=interp_grid_radii_values[:, 0]
        )
        interpolated_grid_radii_x = interpolator.interpolated_values_from_values(
            values=interp_grid_radii_values[:, 1]
        )

        assert np.max(true_grid_radii[:, 0] - interpolated_grid_radii_y) < 0.01
        assert np.max(true_grid_radii[:, 1] - interpolated_grid_radii_x) < 0.01

        interpolator = aa.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=1.1
        )

        interp_grid_radii_values = grid_radii_from_grid(
            profile=None, grid=interpolator.interp_grid
        )

        interpolated_grid_radii_y = interpolator.interpolated_values_from_values(
            values=interp_grid_radii_values[:, 0]
        )
        interpolated_grid_radii_x = interpolator.interpolated_values_from_values(
            values=interp_grid_radii_values[:, 1]
        )

        assert np.max(true_grid_radii[:, 0] - interpolated_grid_radii_y) < 0.1
        assert np.max(true_grid_radii[:, 1] - interpolated_grid_radii_x) < 0.1
