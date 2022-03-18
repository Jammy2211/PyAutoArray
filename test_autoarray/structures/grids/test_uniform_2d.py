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

    def test__manual_extent(self):

        grid = aa.Grid2D.manual_extent(
            extent=(-1.0, 1.0, 2.0, 3.0), shape_native=(2, 3), sub_size=1
        )

        assert type(grid) == aa.Grid2D
        assert (
            grid
            == np.array(
                [
                    [3.0, -1.0],
                    [3.0, 0.0],
                    [3.0, 1.0],
                    [2.0, -1.0],
                    [2.0, 0.0],
                    [2.0, 1.0],
                ]
            )
        ).all()
        assert (
            grid.native
            == np.array(
                [
                    [[3.0, -1.0], [3.0, 0.0], [3.0, 1.0]],
                    [[2.0, -1.0], [2.0, 0.0], [2.0, 1.0]],
                ]
            )
        ).all()
        assert (
            grid.slim
            == np.array(
                [
                    [3.0, -1.0],
                    [3.0, 0.0],
                    [3.0, 1.0],
                    [2.0, -1.0],
                    [2.0, 0.0],
                    [2.0, 1.0],
                ]
            )
        ).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 0.0)

        grid = aa.Grid2D.manual_extent(
            extent=(-1.5, 1.5, -1.5, 1.5), shape_native=(2, 2), sub_size=2
        )

        assert type(grid) == aa.Grid2D
        assert (
            grid
            == np.array(
                [
                    [1.5, -1.5],
                    [1.5, -0.5],
                    [1.5, 0.5],
                    [1.5, 1.5],
                    [0.5, -1.5],
                    [0.5, -0.5],
                    [0.5, 0.5],
                    [0.5, 1.5],
                    [-0.5, -1.5],
                    [-0.5, -0.5],
                    [-0.5, 0.5],
                    [-0.5, 1.5],
                    [-1.5, -1.5],
                    [-1.5, -0.5],
                    [-1.5, 0.5],
                    [-1.5, 1.5],
                ]
            )
        ).all()
        assert (
            grid.native
            == np.array(
                [
                    [[1.5, -1.5], [1.5, -0.5], [1.5, 0.5], [1.5, 1.5]],
                    [[0.5, -1.5], [0.5, -0.5], [0.5, 0.5], [0.5, 1.5]],
                    [[-0.5, -1.5], [-0.5, -0.5], [-0.5, 0.5], [-0.5, 1.5]],
                    [[-1.5, -1.5], [-1.5, -0.5], [-1.5, 0.5], [-1.5, 1.5]],
                ]
            )
        ).all()
        assert (
            grid.slim
            == np.array(
                [
                    [1.5, -1.5],
                    [1.5, -0.5],
                    [1.5, 0.5],
                    [1.5, 1.5],
                    [0.5, -1.5],
                    [0.5, -0.5],
                    [0.5, 0.5],
                    [0.5, 1.5],
                    [-0.5, -1.5],
                    [-0.5, -0.5],
                    [-0.5, 0.5],
                    [-0.5, 1.5],
                    [-1.5, -1.5],
                    [-1.5, -0.5],
                    [-1.5, 0.5],
                    [-1.5, 1.5],
                ]
            )
        ).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 0.0)

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

    def test__recursive_shape_storage(self):

        grid = aa.Grid2D.manual_native(
            grid=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            pixel_scales=1.0,
            sub_size=1,
        )

        assert type(grid) == aa.Grid2D
        assert (
            grid.native.slim.native
            == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.slim.native.slim
            == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()

    def test__masked_shape_native_scaled(self):

        mask = aa.Mask2D.circular(
            shape_native=(3, 3), radius=1.0, pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = aa.Grid2D(grid=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=mask)
        assert grid.shape_native_scaled == (3.0, 2.0)

        grid = aa.Grid2D(
            grid=np.array([[1.5, 1.0], [-1.5, -1.0], [0.1, 0.1]]), mask=mask
        )
        assert grid.shape_native_scaled == (3.0, 2.0)

        grid = aa.Grid2D(
            grid=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0]]), mask=mask
        )
        assert grid.shape_native_scaled == (4.5, 4.0)

        grid = aa.Grid2D(
            grid=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0], [7.0, -5.0]]),
            mask=mask,
        )
        assert grid.shape_native_scaled == (8.5, 8.0)

    def test__flipped_property__returns_grid_as_x_then_y(self):

        grid = aa.Grid2D.manual_native(
            grid=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], pixel_scales=1.0
        )

        assert (
            grid.flipped.slim
            == np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])
        ).all()
        assert (
            grid.flipped.native
            == np.array([[[2.0, 1.0], [4.0, 3.0]], [[6.0, 5.0], [8.0, 7.0]]])
        ).all()

        grid = aa.Grid2D.manual_native(
            grid=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], pixel_scales=1.0
        )

        assert (
            grid.flipped.slim == np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]])
        ).all()
        assert (
            grid.flipped.native == np.array([[[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]])
        ).all()

    def test__grid_radial_projected_from__same_as_grid_util_but_also_includes_angle(
        self
    ):

        grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=(1.0, 2.0))

        grid_radii = grid.grid_2d_radial_projected_from(centre=(0.0, 0.0))
        grid_radial_shape_slim = grid.grid_2d_radial_projected_shape_slim_from(
            centre=(0.0, 0.0)
        )

        grid_radii_util = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
            extent=grid.extent,
            centre=(0.0, 0.0),
            pixel_scales=grid.pixel_scales,
            sub_size=grid.sub_size,
        )

        assert (grid_radii == grid_radii_util).all()
        assert grid_radial_shape_slim == grid_radii_util.shape[0]

        grid = aa.Grid2D.uniform(shape_native=(3, 4), pixel_scales=(3.0, 2.0))

        grid_radii = grid.grid_2d_radial_projected_from(centre=(0.3, 0.1))
        grid_radial_shape_slim = grid.grid_2d_radial_projected_shape_slim_from(
            centre=(0.3, 0.1)
        )

        grid_radii_util = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
            extent=grid.extent,
            centre=(0.3, 0.1),
            pixel_scales=grid.pixel_scales,
            sub_size=grid.sub_size,
        )

        assert (grid_radii == grid_radii_util).all()
        assert grid_radial_shape_slim == grid_radii_util.shape[0]

        grid_radii = grid.grid_2d_radial_projected_from(centre=(0.3, 0.1), angle=60.0)
        grid_radial_shape_slim = grid.grid_2d_radial_projected_shape_slim_from(
            centre=(0.3, 0.1)
        )

        grid_radii_util_angle = aa.util.geometry.transform_grid_2d_to_reference_frame(
            grid_2d=grid_radii_util, centre=(0.3, 0.1), angle=60.0
        )

        grid_radii_util_angle = aa.util.geometry.transform_grid_2d_from_reference_frame(
            grid_2d=grid_radii_util_angle, centre=(0.3, 0.1), angle=0.0
        )

        assert (grid_radii == grid_radii_util_angle).all()
        assert grid_radial_shape_slim == grid_radii_util.shape[0]

        conf.instance["general"]["grid"]["remove_projected_centre"] = True

        grid_radii = grid.grid_2d_radial_projected_from(centre=(0.0, 0.0))
        grid_radial_shape_slim = grid.grid_2d_radial_projected_shape_slim_from(
            centre=(0.0, 0.0)
        )

        grid_radii_util = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
            extent=grid.extent,
            centre=(0.0, 0.0),
            pixel_scales=grid.pixel_scales,
            sub_size=grid.sub_size,
        )

        assert (grid_radii == grid_radii_util[1:, :]).all()
        assert grid_radial_shape_slim == grid_radii_util.shape[0]

    def test__in_radians(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0))

        grid = aa.Grid2D.from_mask(mask=mask)

        assert grid.in_radians[0, 0] == pytest.approx(0.00000969627362, 1.0e-8)
        assert grid.in_radians[0, 1] == pytest.approx(0.00000484813681, 1.0e-8)

        assert grid.in_radians[0, 0] == pytest.approx(
            2.0 * np.pi / (180 * 3600), 1.0e-8
        )
        assert grid.in_radians[0, 1] == pytest.approx(
            1.0 * np.pi / (180 * 3600), 1.0e-8
        )

    def test__padded_grid_from__matches_grid_2d_after_padding(self):
        grid = aa.Grid2D.uniform(shape_native=(4, 4), pixel_scales=3.0, sub_size=1)

        padded_grid = grid.padded_grid_from(kernel_shape_native=(3, 3))

        padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=np.full((6, 6), False), pixel_scales=(3.0, 3.0), sub_size=1
        )

        assert isinstance(padded_grid, aa.Grid2D)
        assert padded_grid.shape == (36, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 6))).all()
        assert (padded_grid == padded_grid_util).all()

        grid = aa.Grid2D.uniform(shape_native=(4, 5), pixel_scales=2.0, sub_size=1)

        padded_grid = grid.padded_grid_from(kernel_shape_native=(3, 3))

        padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=np.full((6, 7), False), pixel_scales=(2.0, 2.0), sub_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = aa.Grid2D.uniform(shape_native=(5, 4), pixel_scales=1.0, sub_size=1)

        padded_grid = grid.padded_grid_from(kernel_shape_native=(3, 3))

        padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=np.full((7, 6), False), pixel_scales=(1.0, 1.0), sub_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = aa.Grid2D.uniform(shape_native=(5, 5), pixel_scales=8.0, sub_size=1)

        padded_grid = grid.padded_grid_from(kernel_shape_native=(2, 5))

        padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=1
        )

        assert padded_grid.shape == (54, 2)
        assert (padded_grid == padded_grid_util).all()

        mask = aa.Mask2D.manual(
            mask=np.full((5, 4), False), pixel_scales=(2.0, 2.0), sub_size=2
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        padded_grid = grid.padded_grid_from(kernel_shape_native=(3, 3))

        padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=np.full((7, 6), False), pixel_scales=(2.0, 2.0), sub_size=2
        )

        assert padded_grid.shape == (168, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(7, 6))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)

        mask = aa.Mask2D.manual(
            mask=np.full((2, 5), False), pixel_scales=(8.0, 8.0), sub_size=4
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        padded_grid = grid.padded_grid_from(kernel_shape_native=(5, 5))

        padded_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
            mask_2d=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=4
        )

        assert padded_grid.shape == (864, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 9))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)

    def test__sub_border_flat_indexes__compare_to_array_util(self):
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

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        sub_border_flat_indexes_util = aa.util.mask_2d.sub_border_pixel_slim_indexes_from(
            mask_2d=mask, sub_size=2
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        assert grid.mask.sub_border_flat_indexes == pytest.approx(
            sub_border_flat_indexes_util, 1e-4
        )

    def test__square_distance_from_coordinate_array(self):

        mask = aa.Mask2D.manual(
            [[True, False], [False, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        grid = aa.Grid2D.manual_mask(
            grid=[[1.0, 1.0], [2.0, 3.0], [1.0, 2.0]], mask=mask
        )

        square_distances = grid.squared_distances_to_coordinate_from(
            coordinate=(0.0, 0.0)
        )

        assert isinstance(square_distances, aa.Array2D)
        assert (square_distances.slim == np.array([2.0, 13.0, 5.0])).all()
        assert (square_distances.mask == mask).all()

        square_distances = grid.squared_distances_to_coordinate_from(
            coordinate=(0.0, 1.0)
        )

        assert isinstance(square_distances, aa.Array2D)
        assert (square_distances.slim == np.array([1.0, 8.0, 2.0])).all()
        assert (square_distances.mask == mask).all()

    def test__distance_from_coordinate_array(self):
        mask = aa.Mask2D.manual(
            [[True, False], [False, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        grid = aa.Grid2D.manual_mask(
            grid=[[1.0, 1.0], [2.0, 3.0], [1.0, 2.0]], mask=mask
        )

        square_distances = grid.distances_to_coordinate_from(coordinate=(0.0, 0.0))

        assert (
            square_distances.slim
            == np.array([np.sqrt(2.0), np.sqrt(13.0), np.sqrt(5.0)])
        ).all()
        assert (square_distances.mask == mask).all()

        square_distances = grid.distances_to_coordinate_from(coordinate=(0.0, 1.0))

        assert (
            square_distances.slim == np.array([1.0, np.sqrt(8.0), np.sqrt(2.0)])
        ).all()
        assert (square_distances.mask == mask).all()

    def test__grid_with_coordinates_within_distance_removed_from__single_coordinates_only(
        self,
    ):

        grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

        grid = grid.grid_with_coordinates_within_distance_removed_from(
            coordinates=(0.0, 0.0), distance=0.05
        )

        assert (
            grid
            == np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            )
        ).all()

        grid = grid.grid_with_coordinates_within_distance_removed_from(
            coordinates=(0.0, 0.0), distance=1.1
        )

        assert (
            grid == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()

        grid = aa.Grid2D.uniform(
            shape_native=(3, 3), pixel_scales=1.0, origin=(1.0, 1.0)
        )

        grid = grid.grid_with_coordinates_within_distance_removed_from(
            coordinates=(0.0, 0.0), distance=0.05
        )

        assert (
            grid
            == np.array(
                [
                    [2.0, 0.0],
                    [2.0, 1.0],
                    [2.0, 2.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 2.0],
                    [0.0, 1.0],
                    [0.0, 2.0],
                ]
            )
        ).all()

    def test__grid_with_coordinates_within_distance_removed_from__multiple_coordinates(
        self
    ):

        grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

        grid = grid.grid_with_coordinates_within_distance_removed_from(
            coordinates=[(0.0, 0.0), (1.0, -1.0), (-1.0, -1.0)], distance=0.05
        )

        assert (
            grid
            == np.array(
                [
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            )
        ).all()

        grid = aa.Grid2D.uniform(
            shape_native=(3, 3), pixel_scales=1.0, origin=(1.0, 1.0)
        )

        grid = grid.grid_with_coordinates_within_distance_removed_from(
            coordinates=[(0.0, 0.0), (1.0, -1.0), (-1.0, -1.0), (2.0, 2.0)],
            distance=0.05,
        )

        assert (
            grid
            == np.array(
                [
                    [2.0, 0.0],
                    [2.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 2.0],
                    [0.0, 1.0],
                    [0.0, 2.0],
                ]
            )
        ).all()


class TestGridRadialMinimum:
    def test__mock_profile__grid_radial_minimum_is_0_or_below_radial_coordinates__no_changes(
        self,
    ):

        grid = np.array([[2.5, 0.0], [4.0, 0.0], [6.0, 0.0]])
        mock_profile = aa.m.MockGridRadialMinimum()

        deflections = mock_profile.deflections_yx_2d_from(grid=grid)
        assert (deflections == grid).all()

    def test__mock_profile__grid_radial_minimum_is_above_some_radial_coordinates__moves_them_grid_radial_minimum(
        self,
    ):
        grid = np.array([[2.0, 0.0], [1.0, 0.0], [6.0, 0.0]])
        mock_profile = aa.m.MockGridRadialMinimum()

        deflections = mock_profile.deflections_yx_2d_from(grid=grid)

        assert (deflections == np.array([[2.5, 0.0], [2.5, 0.0], [6.0, 0.0]])).all()

    def test__mock_profile__same_as_above_but_diagonal_coordinates(self):
        grid = np.array(
            [
                [np.sqrt(2.0), np.sqrt(2.0)],
                [1.0, np.sqrt(8.0)],
                [np.sqrt(8.0), np.sqrt(8.0)],
            ]
        )

        mock_profile = aa.m.MockGridRadialMinimum()

        deflections = mock_profile.deflections_yx_2d_from(grid=grid)

        assert deflections == pytest.approx(
            np.array(
                [[1.7677, 1.7677], [1.0, np.sqrt(8.0)], [np.sqrt(8), np.sqrt(8.0)]]
            ),
            1.0e-4,
        )


class TestBorder:
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

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        grid = aa.Grid2D.from_mask(mask=mask)

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

        mask = aa.Mask2D.circular(
            shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=1
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        grid_to_relocate = aa.Grid2D(
            grid=np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from(grid=grid_to_relocate)

        assert (
            relocated_grid == np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]])
        ).all()
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.sub_size == 1

        mask = aa.Mask2D.circular(
            shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=2
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        grid_to_relocate = aa.Grid2D(
            grid=np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from(grid=grid_to_relocate)

        assert (
            relocated_grid == np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]])
        ).all()
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.sub_size == 2

    def test__outside_border_are_relocations(self):

        mask = aa.Mask2D.circular(
            shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=1
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        grid_to_relocate = aa.Grid2D(
            grid=np.array([[10.1, 0.0], [0.0, 10.1], [-10.1, -10.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array([[0.95, 0.0], [0.0, 0.95], [-0.7017, -0.7017]]), 0.1
        )
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.sub_size == 1

        mask = aa.Mask2D.circular(
            shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=2
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        grid_to_relocate = aa.Grid2D(
            grid=np.array([[10.1, 0.0], [0.0, 10.1], [-10.1, -10.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array([[0.9778, 0.0], [0.0, 0.97788], [-0.7267, -0.7267]]), 0.1
        )
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.sub_size == 2

    def test__outside_border_are_relocations__positive_origin_included_in_relocate(
        self,
    ):
        mask = aa.Mask2D.circular(
            shape_native=(60, 60),
            radius=1.0,
            pixel_scales=(0.1, 0.1),
            centre=(1.0, 1.0),
            sub_size=1,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        grid_to_relocate = aa.Grid2D(
            grid=np.array([[11.1, 1.0], [1.0, 11.1], [-11.1, -11.1]]),
            sub_size=1,
            mask=mask,
        )

        relocated_grid = grid.relocated_grid_from(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array(
                [[2.0, 1.0], [1.0, 2.0], [1.0 - np.sqrt(2) / 2, 1.0 - np.sqrt(2) / 2]]
            ),
            0.1,
        )
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.sub_size == 1

        mask = aa.Mask2D.circular(
            shape_native=(60, 60),
            radius=1.0,
            pixel_scales=(0.1, 0.1),
            centre=(1.0, 1.0),
            sub_size=2,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        grid_to_relocate = aa.Grid2D(
            grid=np.array([[11.1, 1.0], [1.0, 11.1], [-11.1, -11.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from(grid=grid_to_relocate)

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
        assert relocated_grid.sub_size == 2
