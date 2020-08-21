import os
import numpy as np
import pytest

import autoarray as aa
from autoarray import exc
from autoarray.structures import grids

from test_autoarray.mock import MockGridRadialMinimum

test_coordinates_dir = "{}/files/coordinates/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestAPI:
    def test__manual__makes_grid_with_pixel_scale(self):

        grid = aa.Grid.manual_2d(
            grid=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], pixel_scales=1.0
        )

        assert type(grid) == grids.Grid
        assert (
            grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 0.0)

        grid = aa.Grid.manual_1d(
            grid=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            shape_2d=(2, 2),
            pixel_scales=1.0,
            origin=(0.0, 1.0),
        )

        assert type(grid) == grids.Grid
        assert (
            grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)

        grid = aa.Grid.manual_1d(
            grid=[[1.0, 2.0], [3.0, 4.0]],
            shape_2d=(2, 1),
            pixel_scales=(2.0, 3.0),
            store_in_1d=True,
        )

        assert type(grid) == grids.Grid
        assert (grid == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (grid.in_2d == np.array([[[1.0, 2.0]], [[3.0, 4.0]]])).all()
        assert (grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert grid.pixel_scales == (2.0, 3.0)
        assert grid.origin == (0.0, 0.0)

        grid = aa.Grid.manual_1d(
            grid=[[1.0, 2.0], [3.0, 4.0]],
            shape_2d=(2, 1),
            pixel_scales=(2.0, 3.0),
            store_in_1d=False,
        )

        assert type(grid) == grids.Grid
        assert (grid == np.array([[[1.0, 2.0]], [[3.0, 4.0]]])).all()
        assert (grid.in_2d == np.array([[[1.0, 2.0]], [[3.0, 4.0]]])).all()
        assert (grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert grid.pixel_scales == (2.0, 3.0)
        assert grid.origin == (0.0, 0.0)

    def test__manual__makes_sub_grid_with_pixel_scale_and_sub_size(self):

        grid = aa.Grid.manual_2d(
            grid=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            pixel_scales=1.0,
            sub_size=1,
        )

        assert type(grid) == grids.Grid
        assert (
            grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            grid.in_2d_binned
            == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_1d_binned
            == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 0.0)
        assert grid.sub_size == 1

        grid = aa.Grid.manual_1d(
            grid=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            shape_2d=(1, 1),
            pixel_scales=1.0,
            sub_size=2,
            origin=(0.0, 1.0),
            store_in_1d=True,
        )

        assert type(grid) == grids.Grid
        assert (
            grid == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (grid.in_2d_binned == np.array([[[4.0, 5.0]]])).all()
        assert (grid.in_1d_binned == np.array([[4.0, 5.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)
        assert grid.sub_size == 2

        grid = aa.Grid.manual_1d(
            grid=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            shape_2d=(1, 1),
            pixel_scales=1.0,
            sub_size=2,
            origin=(0.0, 1.0),
            store_in_1d=False,
        )

        assert type(grid) == grids.Grid
        assert (
            grid == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (grid.in_2d_binned == np.array([[[4.0, 5.0]]])).all()
        assert (grid.in_1d_binned == np.array([[4.0, 5.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)
        assert grid.sub_size == 2

    def test__manual_yx__makes_grid_with_pixel_scale(self):

        grid = aa.Grid.manual_yx_1d(
            y=[1.0, 3.0, 5.0, 7.0],
            x=[2.0, 4.0, 6.0, 8.0],
            shape_2d=(2, 2),
            pixel_scales=1.0,
            origin=(0.0, 1.0),
            store_in_1d=False,
        )

        assert type(grid) == grids.Grid
        assert (
            grid == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)

        grid = aa.Grid.manual_yx_2d(
            y=[[1.0], [3.0]],
            x=[[2.0], [4.0]],
            pixel_scales=(2.0, 3.0),
            store_in_1d=True,
        )

        assert type(grid) == grids.Grid
        assert (grid == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (grid.in_2d == np.array([[[1.0, 2.0]], [[3.0, 4.0]]])).all()
        assert (grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert grid.pixel_scales == (2.0, 3.0)
        assert grid.origin == (0.0, 0.0)

    def test__manual_yx__makes_sub_grid_with_pixel_scale_and_sub_size(self):

        grid = aa.Grid.manual_yx_1d(
            y=[1.0, 3.0, 5.0, 7.0],
            x=[2.0, 4.0, 6.0, 8.0],
            shape_2d=(1, 1),
            pixel_scales=1.0,
            sub_size=2,
            origin=(0.0, 1.0),
            store_in_1d=True,
        )

        assert type(grid) == grids.Grid
        assert (
            grid == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (grid.in_2d_binned == np.array([[[4.0, 5.0]]])).all()
        assert (grid.in_1d_binned == np.array([[4.0, 5.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)
        assert grid.sub_size == 2

        grid = aa.Grid.manual_yx_2d(
            y=[[1.0, 3.0], [5.0, 7.0]],
            x=[[2.0, 4.0], [6.0, 8.0]],
            pixel_scales=1.0,
            sub_size=2,
            origin=(0.0, 1.0),
            store_in_1d=False,
        )

        assert type(grid) == grids.Grid
        assert (
            grid == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (grid.in_2d_binned == np.array([[[4.0, 5.0]]])).all()
        assert (grid.in_1d_binned == np.array([[4.0, 5.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)
        assert grid.sub_size == 2

    def test__uniform__makes_grid_with_pixel_scale(self):

        grid = aa.Grid.uniform(shape_2d=(2, 2), pixel_scales=2.0)

        assert type(grid) == grids.Grid
        assert (
            grid.in_2d
            == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (0.0, 0.0)

        grid = aa.Grid.uniform(shape_2d=(2, 2), pixel_scales=2.0, origin=(1.0, 1.0))

        assert type(grid) == grids.Grid
        assert (
            grid.in_2d == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (1.0, 1.0)

        grid = aa.Grid.uniform(shape_2d=(2, 1), pixel_scales=(2.0, 1.0))

        assert type(grid) == grids.Grid
        assert (grid.in_2d == np.array([[[1.0, 0.0]], [[-1.0, 0.0]]])).all()
        assert (grid.in_1d == np.array([[1.0, 0.0], [-1.0, 0.0]])).all()
        assert grid.pixel_scales == (2.0, 1.0)
        assert grid.origin == (0.0, 0.0)

        grid = aa.Grid.uniform(
            shape_2d=(2, 2), pixel_scales=2.0, origin=(1.0, 1.0), store_in_1d=True
        )

        assert type(grid) == grids.Grid
        assert (
            grid == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
        ).all()
        assert (
            grid.in_2d == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (1.0, 1.0)

        grid = aa.Grid.uniform(
            shape_2d=(2, 1), pixel_scales=(2.0, 1.0), store_in_1d=False
        )

        assert type(grid) == grids.Grid
        assert (grid == np.array([[[1.0, 0.0]], [[-1.0, 0.0]]])).all()
        assert (grid.in_2d == np.array([[[1.0, 0.0]], [[-1.0, 0.0]]])).all()
        assert (grid.in_1d == np.array([[1.0, 0.0], [-1.0, 0.0]])).all()
        assert grid.pixel_scales == (2.0, 1.0)
        assert grid.origin == (0.0, 0.0)

    def test__uniform__makes_sub_grid_with_pixel_scale_and_sub_size(self):

        grid = aa.Grid.uniform(shape_2d=(2, 2), pixel_scales=2.0, sub_size=1)

        assert type(grid) == grids.Grid
        assert (
            grid.in_2d
            == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert (
            grid.in_2d_binned
            == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
        ).all()
        assert (
            grid.in_1d_binned
            == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (0.0, 0.0)
        assert grid.sub_size == 1

        grid = aa.Grid.uniform(
            shape_2d=(2, 2), pixel_scales=2.0, sub_size=1, origin=(1.0, 1.0)
        )

        assert type(grid) == grids.Grid
        assert (
            grid.in_2d == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
        ).all()
        assert (
            grid.in_2d_binned
            == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
        ).all()
        assert (
            grid.in_1d_binned
            == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
        ).all()
        assert grid.pixel_scales == (2.0, 2.0)
        assert grid.origin == (1.0, 1.0)
        assert grid.sub_size == 1

        grid = aa.Grid.uniform(shape_2d=(2, 1), pixel_scales=1.0, sub_size=2)

        assert type(grid) == grids.Grid
        assert (
            grid.in_2d
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
            grid.in_1d
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
        assert (grid.in_2d_binned == np.array([[[0.5, 0.0]], [[-0.5, 0.0]]])).all()
        assert (grid.in_1d_binned == np.array([[0.5, 0.0], [-0.5, 0.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 0.0)
        assert grid.sub_size == 2

    def test__bounding_box__align_at_corners__grid_corner_is_at_bounding_box_corner(
        self
    ):

        grid = aa.Grid.bounding_box(
            bounding_box=[-2.0, 2.0, -2.0, 2.0],
            shape_2d=(3, 3),
            buffer_around_corners=False,
        )

        assert grid.in_1d == pytest.approx(
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

        grid = aa.Grid.bounding_box(
            bounding_box=[-2.0, 2.0, -2.0, 2.0],
            shape_2d=(2, 3),
            buffer_around_corners=False,
        )

        assert grid.in_1d == pytest.approx(
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
        self
    ):

        grid = aa.Grid.bounding_box(
            bounding_box=[-2.0, 2.0, -2.0, 2.0],
            shape_2d=(3, 3),
            buffer_around_corners=True,
        )

        assert (
            grid.in_1d
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

        grid = aa.Grid.bounding_box(
            bounding_box=[-2.0, 2.0, -2.0, 2.0],
            shape_2d=(2, 3),
            buffer_around_corners=True,
        )

        assert (
            grid.in_1d
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

        grid = aa.Grid.bounding_box(
            bounding_box=[8.0, 10.0, -2.0, 3.0],
            shape_2d=(3, 3),
            store_in_1d=True,
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
        assert grid.in_1d == pytest.approx(
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

        grid = aa.Grid.bounding_box(
            bounding_box=[8.0, 10.0, -2.0, 3.0],
            shape_2d=(3, 3),
            store_in_1d=False,
            buffer_around_corners=True,
        )

        assert grid.in_2d == pytest.approx(
            np.array(
                [
                    [[10.0, -2.0], [10.0, 0.5], [10.0, 3.0]],
                    [[9.0, -2.0], [9.0, 0.5], [9.0, 3.0]],
                    [[8.0, -2.0], [8.0, 0.5], [8.0, 3.0]],
                ]
            ),
            1.0e-4,
        )
        assert grid.in_1d == pytest.approx(
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
    def test__grid_from_deflection_grid(self):

        grid = aa.Grid.uniform(shape_2d=(2, 2), pixel_scales=2.0, store_in_1d=True)

        grid_deflected = grid.grid_from_deflection_grid(deflection_grid=grid)

        assert type(grid_deflected) == grids.Grid
        assert (
            grid_deflected == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        ).all()
        assert (
            grid_deflected.in_2d
            == np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        ).all()
        assert (
            grid_deflected.in_1d
            == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        ).all()
        assert (grid_deflected.mask == grid.mask).all()
        assert grid_deflected.pixel_scales == (2.0, 2.0)
        assert grid_deflected.origin == (0.0, 0.0)
        assert grid.store_in_1d == True

        grid = aa.Grid.uniform(shape_2d=(2, 2), pixel_scales=2.0, store_in_1d=False)

        grid_deflected = grid.grid_from_deflection_grid(deflection_grid=grid)

        assert type(grid_deflected) == grids.Grid
        assert (
            grid_deflected
            == np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        ).all()
        assert (
            grid_deflected.in_2d
            == np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        ).all()
        assert (
            grid_deflected.in_1d
            == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        ).all()
        assert (grid_deflected.mask == grid.mask).all()
        assert grid_deflected.pixel_scales == (2.0, 2.0)
        assert grid_deflected.origin == (0.0, 0.0)
        assert grid.store_in_1d == False

    def test__blurring_grid_from_mask_and_kernel_shape__compare_to_array_util(self):
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

        mask = aa.Mask.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_mask_util = aa.util.mask.blurring_mask_from(
            mask=mask, kernel_shape_2d=(3, 5)
        )

        blurring_grid_util = aa.util.grid.grid_1d_via_mask_from(
            mask=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        mask = aa.Mask.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_grid = grids.Grid.blurring_grid_from_mask_and_kernel_shape(
            mask=mask, kernel_shape_2d=(3, 5)
        )

        assert isinstance(blurring_grid, grids.Grid)
        assert len(blurring_grid.shape) == 2
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)

        blurring_grid = grids.Grid.blurring_grid_from_mask_and_kernel_shape(
            mask=mask, kernel_shape_2d=(3, 5), store_in_1d=False
        )

        assert isinstance(blurring_grid, grids.Grid)
        assert len(blurring_grid.shape) == 3
        assert blurring_grid.pixel_scales == (2.0, 2.0)

    def test__blurring_grid_from_kernel_shape__compare_to_array_util(self):
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

        mask = aa.Mask.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_mask_util = aa.util.mask.blurring_mask_from(
            mask=mask, kernel_shape_2d=(3, 5)
        )

        blurring_grid_util = aa.util.grid.grid_1d_via_mask_from(
            mask=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid = aa.Grid.from_mask(mask=mask)

        blurring_grid = grid.blurring_grid_from_kernel_shape(kernel_shape_2d=(3, 5))

        assert isinstance(blurring_grid, grids.Grid)
        assert len(blurring_grid.shape) == 2
        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)

    def test__structure_from_result__maps_numpy_array_to__auto_array_or_grid(self):

        mask = np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )

        mask = aa.Mask.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.Grid.from_mask(mask=mask)

        result = grid.structure_from_result(result=np.array([1.0, 2.0, 3.0, 4.0]))

        assert isinstance(result, aa.Array)
        assert (
            result.in_2d
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0, 0.0],
                    [0.0, 3.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        result = grid.structure_from_result(
            result=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        )

        assert isinstance(result, aa.Grid)
        assert (
            result.in_2d
            == np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                    [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()

    def test__structure_list_from_result_list__maps_list_to_auto_arrays_or_grids(self):

        mask = np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )

        mask = aa.Mask.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.Grid.from_mask(mask=mask)

        result = grid.structure_list_from_result_list(
            result_list=[np.array([1.0, 2.0, 3.0, 4.0])]
        )

        assert isinstance(result[0], aa.Array)
        assert (
            result[0].in_2d
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0, 0.0],
                    [0.0, 3.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        result = grid.structure_list_from_result_list(
            result_list=[np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])]
        )

        assert isinstance(result[0], aa.Grid)
        assert (
            result[0].in_2d
            == np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                    [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()

    def test__manual__makes_grid_with_pixel_scale(self):

        mask = aa.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
        grid = aa.Grid.manual_mask(
            grid=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], mask=mask
        )

        assert type(grid) == grids.Grid
        assert (
            grid.in_2d == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 0.0)

        mask = aa.Mask.manual(
            [[True, False], [False, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        grid = aa.Grid.manual_mask(grid=[[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], mask=mask)

        assert type(grid) == grids.Grid
        assert (
            grid.in_2d == np.array([[[0.0, 0.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (grid.in_1d == np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)

        mask = aa.Mask.manual(
            [[False], [True]], sub_size=2, pixel_scales=1.0, origin=(0.0, 1.0)
        )
        grid = aa.Grid.manual_mask(
            grid=[
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 7.0]],
            ],
            mask=mask,
            store_in_1d=True,
        )

        assert type(grid) == grids.Grid
        assert (
            grid == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            grid.in_2d
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
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (grid.in_2d_binned == np.array([[[4.0, 5.0]], [[0.0, 0.0]]])).all()
        assert (grid.in_1d_binned == np.array([[4.0, 5.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)
        assert grid.sub_size == 2

        grid = aa.Grid.manual_mask(
            grid=[
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 7.0]],
            ],
            mask=mask,
            store_in_1d=False,
        )

        assert type(grid) == grids.Grid
        assert (
            grid
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
            grid.in_2d
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
            grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (grid.in_2d_binned == np.array([[[4.0, 5.0]], [[0.0, 0.0]]])).all()
        assert (grid.in_1d_binned == np.array([[4.0, 5.0]])).all()
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.origin == (0.0, 1.0)
        assert grid.sub_size == 2

    def test__manual__exception_raised_if_input_grid_is_2d_and_not_sub_shape_of_mask(
        self
    ):

        with pytest.raises(exc.GridException):
            mask = aa.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)
            aa.Grid.manual_mask(grid=[[[1.0, 1.0], [3.0, 3.0]]], mask=mask)

        with pytest.raises(exc.GridException):
            mask = aa.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=2)
            aa.Grid.manual_mask(
                grid=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], mask=mask
            )

        with pytest.raises(exc.GridException):
            mask = aa.Mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=2)
            aa.Grid.manual_mask(
                grid=[
                    [[1.0, 1.0], [2.0, 2.0]],
                    [[3.0, 3.0], [4.0, 4.0]],
                    [[5.0, 5.0], [6.0, 6.0]],
                ],
                mask=mask,
            )

    def test__manual__exception_raised_if_input_grid_is_not_number_of_masked_sub_pixels(
        self
    ):

        with pytest.raises(exc.GridException):
            mask = aa.Mask.manual(mask=[[False, False], [True, False]], sub_size=1)
            aa.Grid.manual_mask(
                grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], mask=mask
            )

        with pytest.raises(exc.GridException):
            mask = aa.Mask.manual(mask=[[False, False], [True, False]], sub_size=1)
            aa.Grid.manual_mask(grid=[[1.0, 1.0], [2.0, 2.0]], mask=mask)

        with pytest.raises(exc.GridException):
            mask = aa.Mask.manual(mask=[[False, True], [True, True]], sub_size=2)
            aa.Grid.manual_mask(grid=[[[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]], mask=mask)

        with pytest.raises(exc.GridException):
            mask = aa.Mask.manual(mask=[[False, True], [True, True]], sub_size=2)
            aa.Grid.manual_mask(
                grid=[[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]],
                mask=mask,
            )

    def test__from_mask_method_same_as_masked_grid(self):

        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = aa.Mask.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=1)

        grid_via_util = aa.util.grid.grid_1d_via_mask_from(
            mask=mask, sub_size=1, pixel_scales=(2.0, 2.0)
        )

        grid = aa.Grid.from_mask(mask=mask)

        assert type(grid) == grids.Grid
        assert grid == pytest.approx(grid_via_util, 1e-4)
        assert grid.pixel_scales == (2.0, 2.0)

        grid_2d = aa.util.grid.sub_grid_2d_from(
            sub_grid_1d=grid, mask=mask, sub_size=mask.sub_size
        )

        assert (grid.in_2d == grid_2d).all()


class TestGridSparse:
    class TestUnmaskedShape:
        def test__properties_consistent_with_util(self):
            mask = aa.Mask.manual(
                mask=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                pixel_scales=(0.5, 0.5),
                sub_size=1,
            )

            grid = aa.Grid.from_mask(mask=mask)

            sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(10, 10), grid=grid
            )

            unmasked_sparse_grid_util = aa.util.grid.grid_1d_via_shape_2d_from(
                shape_2d=(10, 10),
                pixel_scales=(0.15, 0.15),
                sub_size=1,
                origin=(0.0, 0.0),
            )

            unmasked_sparse_grid_pixel_centres = aa.util.grid.grid_pixel_centres_1d_from(
                grid_scaled_1d=unmasked_sparse_grid_util,
                shape_2d=grid.mask.shape,
                pixel_scales=grid.pixel_scales,
            ).astype(
                "int"
            )

            total_sparse_pixels = aa.util.mask.total_sparse_pixels_from(
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            )

            regular_to_unmasked_sparse_util = aa.util.grid.grid_pixel_indexes_1d_from(
                grid_scaled_1d=grid,
                shape_2d=(10, 10),
                pixel_scales=(0.15, 0.15),
                origin=(0.0, 0.0),
            ).astype("int")

            unmasked_sparse_for_sparse_util = aa.util.sparse.unmasked_sparse_for_sparse_from(
                total_sparse_pixels=total_sparse_pixels,
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            ).astype(
                "int"
            )

            sparse_for_unmasked_sparse_util = aa.util.sparse.sparse_for_unmasked_sparse_from(
                mask=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
                total_sparse_pixels=total_sparse_pixels,
            ).astype(
                "int"
            )

            sparse_1d_index_for_mask_1d_index_util = aa.util.sparse.sparse_1d_index_for_mask_1d_index_from(
                regular_to_unmasked_sparse=regular_to_unmasked_sparse_util,
                sparse_for_unmasked_sparse=sparse_for_unmasked_sparse_util,
            )

            sparse_grid_util = aa.util.sparse.sparse_grid_via_unmasked_from(
                unmasked_sparse_grid=unmasked_sparse_grid_util,
                unmasked_sparse_for_sparse=unmasked_sparse_for_sparse_util,
            )

            assert (
                sparse_grid.sparse_1d_index_for_mask_1d_index
                == sparse_1d_index_for_mask_1d_index_util
            ).all()
            assert (sparse_grid.sparse == sparse_grid_util).all()

        def test__sparse_grid_overlaps_mask_perfectly__masked_pixels_in_masked_sparse_grid(
            self
        ):
            mask = aa.Mask.manual(
                mask=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.Grid.from_mask(mask=mask)

            sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4])
            ).all()
            assert (
                sparse_grid.sparse
                == np.array(
                    [[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
                )
            ).all()

        def test__same_as_above_but_4x3_grid_and_mask(self):
            mask = aa.Mask.manual(
                mask=np.array(
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

            grid = aa.Grid.from_mask(mask=mask)

            sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(4, 3), grid=grid
            )

            assert (
                sparse_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7])
            ).all()
            assert (
                sparse_grid.sparse
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
            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, False, True, True],
                        [False, False, False, False],
                        [True, False, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.Grid.from_mask(mask=mask)

            sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 4), grid=grid
            )

            assert (
                sparse_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5])
            ).all()
            assert (
                sparse_grid.sparse
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

        def test__mask_with_offset_centre__origin_of_sparse_grid_moves_to_give_same_pairings(
            self
        ):
            mask = aa.Mask.manual(
                mask=np.array(
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

            grid = aa.Grid.from_mask(mask=mask)

            # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
            # the central (3x3) pixels only.

            sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4])
            ).all()
            assert (
                sparse_grid.sparse
                == np.array(
                    [[2.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0]]
                )
            ).all()

        def test__same_as_above_but_different_offset(self):
            mask = aa.Mask.manual(
                mask=np.array(
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

            grid = aa.Grid.from_mask(mask=mask)

            # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
            # the central (3x3) pixels only.

            sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4])
            ).all()
            assert (
                sparse_grid.sparse
                == np.array(
                    [[2.0, 2.0], [0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [-2.0, 2.0]]
                )
            ).all()

        def test__from_grid_and_unmasked_shape__sets_up_with_correct_shape_and_pixel_scales(
            self, mask_7x7
        ):
            grid = aa.Grid.from_mask(mask=mask_7x7)

            sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
                grid=grid, unmasked_sparse_shape=(3, 3)
            )

            assert (
                sparse_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            ).all()
            assert (
                sparse_grid.sparse
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
            mask = aa.Mask.manual(
                mask=np.array(
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

            grid = aa.Grid.from_mask(mask=mask)

            sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(4, 3), grid=grid
            )

            assert (
                sparse_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7])
            ).all()
            assert (
                sparse_grid.sparse
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
            mask = aa.Mask.manual(
                mask=np.array(
                    [
                        [True, False, True, True],
                        [False, False, False, False],
                        [True, False, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.Grid.from_mask(mask=mask)

            sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 4), grid=grid
            )

            assert (
                sparse_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5])
            ).all()
            assert (
                sparse_grid.sparse
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
            mask = aa.Mask.manual(
                mask=np.array(
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

            grid = aa.Grid.from_mask(mask=mask)

            sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(3, 3), grid=grid
            )

            assert (
                sparse_grid.sparse_1d_index_for_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            ).all()
            assert (
                sparse_grid.sparse
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

    class TestUnmaskedShapeAndWeightImage:
        def test__weight_map_all_ones__kmeans_grid_is_grid_overlapping_image(self):
            mask = aa.Mask.manual(
                mask=np.array(
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

            grid = aa.Grid.from_mask(mask=mask)

            weight_map = np.ones(mask.pixels_in_mask)

            sparse_grid_weight = grids.GridSparse.from_total_pixels_grid_and_weight_map(
                total_pixels=8,
                grid=grid,
                weight_map=weight_map,
                n_iter=10,
                max_iter=20,
                seed=1,
            )

            assert (
                sparse_grid_weight.sparse
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
                sparse_grid_weight.sparse_1d_index_for_mask_1d_index
                == np.array([1, 1, 2, 2, 1, 1, 3, 3, 5, 4, 0, 7, 5, 4, 6, 6])
            ).all()

        def test__weight_map_changed_from_above(self):
            mask = aa.Mask.manual(
                mask=np.array(
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

            grid = aa.Grid.from_mask(mask=mask)

            weight_map = np.ones(mask.pixels_in_mask)
            weight_map[0:15] = 0.00000001

            sparse_grid_weight = grids.GridSparse.from_total_pixels_grid_and_weight_map(
                total_pixels=8,
                grid=grid,
                weight_map=weight_map,
                n_iter=10,
                max_iter=30,
                seed=1,
            )

            assert sparse_grid_weight.sparse[1] == pytest.approx(
                np.array([0.4166666, -0.0833333]), 1.0e-4
            )

            assert (
                sparse_grid_weight.sparse_1d_index_for_mask_1d_index
                == np.array([5, 1, 0, 0, 5, 1, 1, 4, 3, 6, 7, 4, 3, 6, 2, 2])
            ).all()

        def test__stochastic_true__every_grid_different(self):

            mask = aa.Mask.manual(
                mask=np.array(
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

            grid = aa.Grid.from_mask(mask=mask)

            weight_map = np.ones(mask.pixels_in_mask)

            sparse_grid_weight_0 = grids.GridSparse.from_total_pixels_grid_and_weight_map(
                total_pixels=8,
                grid=grid,
                weight_map=weight_map,
                n_iter=1,
                max_iter=2,
                seed=1,
                stochastic=False,
            )

            sparse_grid_weight_1 = grids.GridSparse.from_total_pixels_grid_and_weight_map(
                total_pixels=8,
                grid=grid,
                weight_map=weight_map,
                n_iter=1,
                max_iter=2,
                seed=1,
                stochastic=False,
            )

            assert (sparse_grid_weight_0.sparse == sparse_grid_weight_1.sparse).all()

            sparse_grid_weight_0 = grids.GridSparse.from_total_pixels_grid_and_weight_map(
                total_pixels=8,
                grid=grid,
                weight_map=weight_map,
                n_iter=1,
                max_iter=2,
                seed=1,
                stochastic=True,
            )

            sparse_grid_weight_1 = grids.GridSparse.from_total_pixels_grid_and_weight_map(
                total_pixels=8,
                grid=grid,
                weight_map=weight_map,
                n_iter=1,
                max_iter=2,
                seed=1,
                stochastic=True,
            )

            assert (sparse_grid_weight_0.sparse != sparse_grid_weight_1.sparse).any()


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

        mask = aa.Mask.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        grid = aa.Grid.from_mask(mask=mask)

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
        mask = aa.Mask.circular(
            shape_2d=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=1
        )

        grid = aa.Grid.from_mask(mask=mask)

        grid_to_relocate = grids.Grid(
            grid=np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert (
            relocated_grid == np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]])
        ).all()
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.sub_size == 1

        mask = aa.Mask.circular(
            shape_2d=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=2
        )

        grid = aa.Grid.from_mask(mask=mask)

        grid_to_relocate = grids.Grid(
            grid=np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert (
            relocated_grid == np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]])
        ).all()
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.sub_size == 2

    def test__outside_border_are_relocations(self):
        mask = aa.Mask.circular(
            shape_2d=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=1
        )

        grid = aa.Grid.from_mask(mask=mask)

        grid_to_relocate = grids.Grid(
            grid=np.array([[10.1, 0.0], [0.0, 10.1], [-10.1, -10.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array([[0.95, 0.0], [0.0, 0.95], [-0.7017, -0.7017]]), 0.1
        )
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.sub_size == 1

        mask = aa.Mask.circular(
            shape_2d=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=2
        )

        grid = aa.Grid.from_mask(mask=mask)

        grid_to_relocate = grids.Grid(
            grid=np.array([[10.1, 0.0], [0.0, 10.1], [-10.1, -10.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array([[0.9778, 0.0], [0.0, 0.97788], [-0.7267, -0.7267]]), 0.1
        )
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.sub_size == 2

    def test__outside_border_are_relocations__positive_origin_included_in_relocate(
        self
    ):
        mask = aa.Mask.circular(
            shape_2d=(60, 60),
            radius=1.0,
            pixel_scales=(0.1, 0.1),
            centre=(1.0, 1.0),
            sub_size=1,
        )

        grid = aa.Grid.from_mask(mask=mask)

        grid_to_relocate = grids.Grid(
            grid=np.array([[11.1, 1.0], [1.0, 11.1], [-11.1, -11.1]]),
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
        assert relocated_grid.sub_size == 1

        mask = aa.Mask.circular(
            shape_2d=(60, 60),
            radius=1.0,
            pixel_scales=(0.1, 0.1),
            centre=(1.0, 1.0),
            sub_size=2,
        )

        grid = aa.Grid.from_mask(mask=mask)

        grid_to_relocate = grids.Grid(
            grid=np.array([[11.1, 1.0], [1.0, 11.1], [-11.1, -11.1]]), mask=mask
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
        assert relocated_grid.sub_size == 2
