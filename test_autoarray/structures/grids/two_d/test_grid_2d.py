from os import path
import numpy as np
import pytest

from autoconf import conf
import autoarray as aa
from autoarray import exc

test_grid_dir = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


class TestAPI:
    def test__manual(self):

        grid = aa.Grid2D.manual_native(
            grid=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            pixel_scales=1.0,
            sub_size=1,
        )

        assert type(grid) == aa.Grid2D
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
        assert (
            grid.binned.native
            == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.binned == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 0.0)
        assert grid.sub_size == 1

        grid = aa.Grid2D.manual_slim(
            grid=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            shape_native=(1, 1),
            pixel_scales=1.0,
            sub_size=2,
            origin=(0.0, 1.0),
        )

        assert type(grid) == aa.Grid2D
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
        assert (grid.binned.slim == np.array([[4.0, 5.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)
        assert grid.sub_size == 2

        grid = aa.Grid2D.uniform(
            shape_native=(2, 2), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0)
        )

    def test__manual_mask(self):

        mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)
        grid = aa.Grid2D.manual_mask(
            grid=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], mask=mask
        )

        assert type(grid) == aa.Grid2D
        assert (
            grid.native
            == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 0.0)

        mask = aa.Mask2D.manual(
            [[True, False], [False, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        grid = aa.Grid2D.manual_mask(
            grid=[[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], mask=mask
        )

        assert type(grid) == aa.Grid2D
        assert (
            grid.native
            == np.array([[[0.0, 0.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (grid.slim == np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)

        mask = aa.Mask2D.manual(
            [[False], [True]], sub_size=2, pixel_scales=1.0, origin=(0.0, 1.0)
        )
        grid = aa.Grid2D.manual_mask(
            grid=[
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 7.0]],
            ],
            mask=mask,
        )

        assert type(grid) == aa.Grid2D
        assert (
            grid == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            grid.native
            == np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()
        assert (
            grid.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (grid.binned.native == np.array([[[4.0, 5.0]], [[0.0, 0.0]]])).all()
        assert (grid.binned.slim == np.array([[4.0, 5.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)
        assert grid.sub_size == 2

    def test__manual_mask__exception_raised_if_input_grid_is_2d_and_not_sub_shape_of_mask(
        self,
    ):

        with pytest.raises(exc.GridException):
            mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)
            aa.Grid2D.manual_mask(grid=[[[1.0, 1.0], [3.0, 3.0]]], mask=mask)

        with pytest.raises(exc.GridException):
            mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0, sub_size=2)
            aa.Grid2D.manual_mask(
                grid=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], mask=mask
            )

        with pytest.raises(exc.GridException):
            mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0, sub_size=2)
            aa.Grid2D.manual_mask(
                grid=[
                    [[1.0, 1.0], [2.0, 2.0]],
                    [[3.0, 3.0], [4.0, 4.0]],
                    [[5.0, 5.0], [6.0, 6.0]],
                ],
                mask=mask,
            )

    def test__manual_mask__exception_raised_if_input_grid_is_not_number_of_masked_sub_pixels(
        self,
    ):

        with pytest.raises(exc.GridException):
            mask = aa.Mask2D.manual(
                mask=[[False, False], [True, False]], pixel_scales=1.0, sub_size=1
            )
            aa.Grid2D.manual_mask(
                grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], mask=mask
            )

        with pytest.raises(exc.GridException):
            mask = aa.Mask2D.manual(
                mask=[[False, False], [True, False]], pixel_scales=1.0, sub_size=1
            )
            aa.Grid2D.manual_mask(grid=[[1.0, 1.0], [2.0, 2.0]], mask=mask)

        with pytest.raises(exc.GridException):
            mask = aa.Mask2D.manual(
                mask=[[False, True], [True, True]], pixel_scales=1.0, sub_size=2
            )
            aa.Grid2D.manual_mask(
                grid=[[[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]], mask=mask
            )

        with pytest.raises(exc.GridException):
            mask = aa.Mask2D.manual(
                mask=[[False, True], [True, True]], pixel_scales=1.0, sub_size=2
            )
            aa.Grid2D.manual_mask(
                grid=[[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]],
                mask=mask,
            )

    def test__manual_yx__makes_grid_with_pixel_scale(self):

        grid = aa.Grid2D.manual_yx_2d(
            y=[[1.0], [3.0]], x=[[2.0], [4.0]], pixel_scales=(2.0, 3.0)
        )

        assert type(grid) == aa.Grid2D
        assert (grid == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (grid.native == np.array([[[1.0, 2.0]], [[3.0, 4.0]]])).all()
        assert (grid.slim == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert grid.pixel_scales == (2.0, 3.0)
        assert grid.origin == (0.0, 0.0)

    def test__manual_yx__makes_sub_grid_with_pixel_scale_and_sub_size(self):

        grid = aa.Grid2D.manual_yx_1d(
            y=[1.0, 3.0, 5.0, 7.0],
            x=[2.0, 4.0, 6.0, 8.0],
            shape_native=(1, 1),
            pixel_scales=1.0,
            sub_size=2,
            origin=(0.0, 1.0),
        )

        assert type(grid) == aa.Grid2D
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
        assert (grid.binned.slim == np.array([[4.0, 5.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)
        assert grid.sub_size == 2

    def test__uniform__makes_grid_with_pixel_scale(self):

        grid = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=2.0)

        assert type(grid) == aa.Grid2D
        assert (
            grid.native
            == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
        ).all()
        assert (
            grid.slim == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (0.0, 0.0)

        grid = aa.Grid2D.uniform(
            shape_native=(2, 2), pixel_scales=2.0, origin=(1.0, 1.0)
        )

        assert type(grid) == aa.Grid2D
        assert (
            grid.native
            == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
        ).all()
        assert (
            grid.slim == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (1.0, 1.0)

        grid = aa.Grid2D.uniform(shape_native=(2, 1), pixel_scales=(2.0, 1.0))

        assert type(grid) == aa.Grid2D
        assert (grid.native == np.array([[[1.0, 0.0]], [[-1.0, 0.0]]])).all()
        assert (grid.slim == np.array([[1.0, 0.0], [-1.0, 0.0]])).all()
        assert grid.pixel_scales == (2.0, 1.0)
        assert grid.origin == (0.0, 0.0)

        grid = aa.Grid2D.uniform(
            shape_native=(2, 2), pixel_scales=2.0, origin=(1.0, 1.0)
        )

        assert type(grid) == aa.Grid2D
        assert (
            grid == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
        ).all()
        assert (
            grid.native
            == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
        ).all()
        assert (
            grid.slim == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (1.0, 1.0)

    def test__uniform__makes_sub_grid_with_pixel_scale_and_sub_size(self):

        grid = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=2.0, sub_size=1)

        assert type(grid) == aa.Grid2D
        assert (
            grid.native
            == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
        ).all()
        assert (
            grid.slim == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert (
            grid.binned.native
            == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
        ).all()
        assert (
            grid.binned
            == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (0.0, 0.0)
        assert grid.sub_size == 1

        grid = aa.Grid2D.uniform(
            shape_native=(2, 2), pixel_scales=2.0, sub_size=1, origin=(1.0, 1.0)
        )

        assert type(grid) == aa.Grid2D
        assert (
            grid.native
            == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
        ).all()
        assert (
            grid.slim == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
        ).all()
        assert (
            grid.binned.native
            == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
        ).all()
        assert (
            grid.binned == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (1.0, 1.0)
        assert grid.sub_size == 1

        grid = aa.Grid2D.uniform(shape_native=(2, 1), pixel_scales=1.0, sub_size=2)

        assert type(grid) == aa.Grid2D
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
        assert (grid.binned.slim == np.array([[0.5, 0.0], [-0.5, 0.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 0.0)
        assert grid.sub_size == 2

    def test__bounding_box__align_at_corners__grid_corner_is_at_bounding_box_corner(
        self,
    ):

        grid = aa.Grid2D.bounding_box(
            bounding_box=[-2.0, 2.0, -2.0, 2.0],
            shape_native=(3, 3),
            buffer_around_corners=False,
        )

        assert grid.slim == pytest.approx(
            np.array(
                [
                    [1.3333, -1.3333],
                    [1.3333, 0.0],
                    [1.3333, 1.3333],
                    [0.0, -1.3333],
                    [0.0, 0.0],
                    [0.0, 1.3333],
                    [-1.3333, -1.3333],
                    [-1.3333, 0.0],
                    [-1.3333, 1.3333],
                ]
            ),
            1.0e-4,
        )

        assert grid.pixel_scales == pytest.approx((1.33333, 1.3333), 1.0e-4)
        assert grid.origin == (0.0, 0.0)

        grid = aa.Grid2D.bounding_box(
            bounding_box=[-2.0, 2.0, -2.0, 2.0],
            shape_native=(2, 3),
            buffer_around_corners=False,
        )

        assert grid.slim == pytest.approx(
            np.array(
                [
                    [1.0, -1.3333],
                    [1.0, 0.0],
                    [1.0, 1.3333],
                    [-1.0, -1.3333],
                    [-1.0, 0.0],
                    [-1.0, 1.3333],
                ]
            ),
            1.0e-4,
        )
        assert grid.pixel_scales == pytest.approx((2.0, 1.33333), 1.0e4)
        assert grid.origin == (0.0, 0.0)

    def test__bounding_box__uniform_box__buffer_around_corners__makes_grid_with_correct_pixel_scales_and_origin(
        self,
    ):

        grid = aa.Grid2D.bounding_box(
            bounding_box=[-2.0, 2.0, -2.0, 2.0],
            shape_native=(3, 3),
            buffer_around_corners=True,
        )

        assert (
            grid.slim
            == np.array(
                [
                    [2.0, -2.0],
                    [2.0, 0.0],
                    [2.0, 2.0],
                    [0.0, -2.0],
                    [0.0, 0.0],
                    [0.0, 2.0],
                    [-2.0, -2.0],
                    [-2.0, 0.0],
                    [-2.0, 2.0],
                ]
            )
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (0.0, 0.0)

        grid = aa.Grid2D.bounding_box(
            bounding_box=[-2.0, 2.0, -2.0, 2.0],
            shape_native=(2, 3),
            buffer_around_corners=True,
        )

        assert (
            grid.slim
            == np.array(
                [
                    [2.0, -2.0],
                    [2.0, 0.0],
                    [2.0, 2.0],
                    [-2.0, -2.0],
                    [-2.0, 0.0],
                    [-2.0, 2.0],
                ]
            )
        ).all()
        assert grid.pixel_scales == (4.0, 2.0)
        assert grid.origin == (0.0, 0.0)

        grid = aa.Grid2D.bounding_box(
            bounding_box=[8.0, 10.0, -2.0, 3.0],
            shape_native=(3, 3),
            buffer_around_corners=True,
        )

        assert grid == pytest.approx(
            np.array(
                [
                    [10.0, -2.0],
                    [10.0, 0.5],
                    [10.0, 3.0],
                    [9.0, -2.0],
                    [9.0, 0.5],
                    [9.0, 3.0],
                    [8.0, -2.0],
                    [8.0, 0.5],
                    [8.0, 3.0],
                ]
            ),
            1.0e-4,
        )
        assert grid.slim == pytest.approx(
            np.array(
                [
                    [10.0, -2.0],
                    [10.0, 0.5],
                    [10.0, 3.0],
                    [9.0, -2.0],
                    [9.0, 0.5],
                    [9.0, 3.0],
                    [8.0, -2.0],
                    [8.0, 0.5],
                    [8.0, 3.0],
                ]
            ),
            1.0e-4,
        )
        assert grid.pixel_scales == (1.0, 2.5)
        assert grid.origin == (9.0, 0.5)


class TestGrid:
    def test__grid_via_deflection_grid_from(self):

        grid = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=2.0)

        grid_deflected = grid.grid_via_deflection_grid_from(deflection_grid=grid)

        assert type(grid_deflected) == aa.Grid2D
        assert (
            grid_deflected == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        ).all()
        assert (
            grid_deflected.native
            == np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        ).all()
        assert (
            grid_deflected.slim
            == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        ).all()
        assert (grid_deflected.mask == grid.mask).all()
        assert grid_deflected.pixel_scales == (2.0, 2.0)
        assert grid_deflected.origin == (0.0, 0.0)

    def test__blurring_grid_from__compare_to_array_util(self):
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

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_mask_util = aa.util.mask_2d.blurring_mask_2d_from(
            mask_2d=mask, kernel_shape_native=(3, 5)
        )

        blurring_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_grid = aa.Grid2D.blurring_grid_from(
            mask=mask, kernel_shape_native=(3, 5)
        )

        assert isinstance(blurring_grid, aa.Grid2D)
        assert len(blurring_grid.shape) == 2
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)

    def test__blurring_grid_via_kernel_shape_from__compare_to_array_util(self):
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

        grid = aa.Grid2D.from_mask(mask=mask)

        blurring_grid = grid.blurring_grid_via_kernel_shape_from(
            kernel_shape_native=(3, 5)
        )

        assert isinstance(blurring_grid, aa.Grid2D)
        assert len(blurring_grid.shape) == 2
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)

    def test__structure_2d_from__maps_numpy_array_to__auto_array_or_grid(self):

        mask = np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.Grid2D.from_mask(mask=mask)

        result = grid.structure_2d_from(result=np.array([1.0, 2.0, 3.0, 4.0]))

        assert isinstance(result, aa.Array2D)
        assert (
            result.native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0, 0.0],
                    [0.0, 3.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        result = grid.structure_2d_from(
            result=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        )

        assert isinstance(result, aa.Grid2D)
        assert (
            result.native
            == np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                    [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()

    def test__structure_2d_list_from__maps_list_to_auto_arrays_or_grids(self):

        mask = np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.Grid2D.from_mask(mask=mask)

        result = grid.structure_2d_list_from(
            result_list=[np.array([1.0, 2.0, 3.0, 4.0])]
        )

        assert isinstance(result[0], aa.Array2D)
        assert (
            result[0].native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0, 0.0],
                    [0.0, 3.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        result = grid.structure_2d_list_from(
            result_list=[np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])]
        )

        assert isinstance(result[0], aa.Grid2D)
        assert (
            result[0].native
            == np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                    [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()

    def test__from_mask_method_same_as_masked_grid(self):

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

        grid = aa.Grid2D.from_mask(mask=mask)

        assert type(grid) == aa.Grid2D
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.pixel_scales == (2.0, 2.0)

        grid_2d = aa.util.grid_2d.grid_2d_native_from(
            grid_2d_slim=grid, mask_2d=mask, sub_size=mask.sub_size
        )

        assert (grid.native == grid_2d).all()

    def test__to_and_from_fits_methods(self):

        grid = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=2.0)

        file_path = path.join(test_grid_dir, "grid.fits")

        grid.output_to_fits(file_path=file_path, overwrite=True)

        grid_from_fits = aa.Grid2D.from_fits(file_path=file_path, pixel_scales=2.0)

        assert type(grid) == aa.Grid2D
        assert (
            grid_from_fits.native
            == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
        ).all()
        assert (
            grid_from_fits.slim
            == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert grid_from_fits.pixel_scales == (2.0, 2.0)
        assert grid_from_fits.origin == (0.0, 0.0)
