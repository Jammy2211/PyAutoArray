import os
import shutil
import numpy as np
import pytest
import scipy.spatial

import autoarray as aa
from autoarray import exc
from autoarray.structures import grids


@pytest.fixture(name="grid")
def make_grid():
    mask = aa.mask.manual(
        np.array([[True, False, True], [False, False, False], [True, False, True]]),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    return aa.masked.grid.from_mask(mask=mask)


class TestGridAPI:
    class TestManual:
        def test__grid__makes_scaled_grid_with_pixel_scale(self):

            grid = aa.grid.manual_2d(
                grid=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                pixel_scales=1.0,
            )

            assert type(grid) == grids.Grid
            assert (
                grid.in_2d
                == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            ).all()
            assert (
                grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert grid.pixel_scales == (1.0, 1.0)
            assert grid.origin == (0.0, 0.0)

            grid = aa.grid.manual_1d(
                grid=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                shape_2d=(2, 2),
                pixel_scales=1.0,
                origin=(0.0, 1.0),
            )

            assert type(grid) == grids.Grid
            assert (
                grid.in_2d
                == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            ).all()
            assert (
                grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert grid.pixel_scales == (1.0, 1.0)
            assert grid.origin == (0.0, 1.0)

            grid = aa.grid.manual_1d(
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

            grid = aa.grid.manual_1d(
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

        def test__grid__makes_scaled_sub_grid_with_pixel_scale_and_sub_size(self):

            grid = aa.grid.manual_2d(
                grid=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                pixel_scales=1.0,
                sub_size=1,
            )

            assert type(grid) == grids.Grid
            assert (
                grid.in_2d
                == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
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

            grid = aa.grid.manual_1d(
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
                grid.in_2d
                == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            ).all()
            assert (
                grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert (grid.in_2d_binned == np.array([[[4.0, 5.0]]])).all()
            assert (grid.in_1d_binned == np.array([[4.0, 5.0]])).all()
            assert grid.pixel_scales == (1.0, 1.0)
            assert grid.origin == (0.0, 1.0)
            assert grid.sub_size == 2

            grid = aa.grid.manual_1d(
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
                grid.in_2d
                == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            ).all()
            assert (
                grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert (grid.in_2d_binned == np.array([[[4.0, 5.0]]])).all()
            assert (grid.in_1d_binned == np.array([[4.0, 5.0]])).all()
            assert grid.pixel_scales == (1.0, 1.0)
            assert grid.origin == (0.0, 1.0)
            assert grid.sub_size == 2

    class TestManualYAndX:
        def test__grid__makes_scaled_grid_with_pixel_scale(self):

            grid = aa.grid.manual_yx_1d(
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
                grid.in_2d
                == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            ).all()
            assert (
                grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert grid.pixel_scales == (1.0, 1.0)
            assert grid.origin == (0.0, 1.0)

            grid = aa.grid.manual_yx_2d(
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

        def test__grid__makes_scaled_sub_grid_with_pixel_scale_and_sub_size(self):

            grid = aa.grid.manual_yx_1d(
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
                grid.in_2d
                == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            ).all()
            assert (
                grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert (grid.in_2d_binned == np.array([[[4.0, 5.0]]])).all()
            assert (grid.in_1d_binned == np.array([[4.0, 5.0]])).all()
            assert grid.pixel_scales == (1.0, 1.0)
            assert grid.origin == (0.0, 1.0)
            assert grid.sub_size == 2

            grid = aa.grid.manual_yx_2d(
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
                grid.in_2d
                == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            ).all()
            assert (
                grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert (grid.in_2d_binned == np.array([[[4.0, 5.0]]])).all()
            assert (grid.in_1d_binned == np.array([[4.0, 5.0]])).all()
            assert grid.pixel_scales == (1.0, 1.0)
            assert grid.origin == (0.0, 1.0)
            assert grid.sub_size == 2

    class TestGridUniform:
        def test__grid_uniform__makes_scaled_grid_with_pixel_scale(self):

            grid = aa.grid.uniform(shape_2d=(2, 2), pixel_scales=2.0)

            assert type(grid) == grids.Grid
            assert (
                grid.in_2d
                == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
            ).all()
            assert (
                grid.in_1d
                == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
            ).all()
            assert grid.pixel_scales == (2.0, 2.0)
            assert grid.origin == (0.0, 0.0)

            grid = aa.grid.uniform(shape_2d=(2, 2), pixel_scales=2.0, origin=(1.0, 1.0))

            assert type(grid) == grids.Grid
            assert (
                grid.in_2d
                == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
            ).all()
            assert (
                grid.in_1d == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
            ).all()
            assert grid.pixel_scales == (2.0, 2.0)
            assert grid.origin == (1.0, 1.0)

            grid = aa.grid.uniform(shape_2d=(2, 1), pixel_scales=(2.0, 1.0))

            assert type(grid) == grids.Grid
            assert (grid.in_2d == np.array([[[1.0, 0.0]], [[-1.0, 0.0]]])).all()
            assert (grid.in_1d == np.array([[1.0, 0.0], [-1.0, 0.0]])).all()
            assert grid.pixel_scales == (2.0, 1.0)
            assert grid.origin == (0.0, 0.0)

            grid = aa.grid.uniform(
                shape_2d=(2, 2), pixel_scales=2.0, origin=(1.0, 1.0), store_in_1d=True
            )

            assert type(grid) == grids.Grid
            assert (
                grid == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
            ).all()
            assert (
                grid.in_2d
                == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
            ).all()
            assert (
                grid.in_1d == np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 2.0]])
            ).all()
            assert grid.pixel_scales == (2.0, 2.0)
            assert grid.origin == (1.0, 1.0)

            grid = aa.grid.uniform(
                shape_2d=(2, 1), pixel_scales=(2.0, 1.0), store_in_1d=False
            )

            assert type(grid) == grids.Grid
            assert (grid == np.array([[[1.0, 0.0]], [[-1.0, 0.0]]])).all()
            assert (grid.in_2d == np.array([[[1.0, 0.0]], [[-1.0, 0.0]]])).all()
            assert (grid.in_1d == np.array([[1.0, 0.0], [-1.0, 0.0]])).all()
            assert grid.pixel_scales == (2.0, 1.0)
            assert grid.origin == (0.0, 0.0)

        def test__grid__makes_scaled_sub_grid_with_pixel_scale_and_sub_size(self):

            grid = aa.grid.uniform(shape_2d=(2, 2), pixel_scales=2.0, sub_size=1)

            assert type(grid) == grids.Grid
            assert (
                grid.in_2d
                == np.array([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
            ).all()
            assert (
                grid.in_1d
                == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
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

            grid = aa.grid.uniform(
                shape_2d=(2, 2), pixel_scales=2.0, sub_size=1, origin=(1.0, 1.0)
            )

            assert type(grid) == grids.Grid
            assert (
                grid.in_2d
                == np.array([[[2.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
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

            grid = aa.grid.uniform(shape_2d=(2, 1), pixel_scales=1.0, sub_size=2)

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

    class TestGridBoundingBox:
        def test__grid_bounding_box__align_at_corners__grid_corner_is_at_bounding_box_corner(
            self
        ):

            grid = aa.grid.bounding_box(
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

            grid = aa.grid.bounding_box(
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

        def test__grid_bounding_box__uniform_box__buffer_around_corners__makes_grid_with_correct_pixel_scales_and_origin(
            self
        ):

            grid = aa.grid.bounding_box(
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

            grid = aa.grid.bounding_box(
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

            grid = aa.grid.bounding_box(
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

            grid = aa.grid.bounding_box(
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


class TestGridMaskedAPI:
    class TestManual:
        def test__grid__makes_scaled_grid_with_pixel_scale(self):

            mask = aa.mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0)
            grid = aa.masked.grid.manual_2d(
                grid=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], mask=mask
            )

            assert type(grid) == grids.Grid
            assert (
                grid.in_2d
                == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            ).all()
            assert (
                grid.in_1d == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            ).all()
            assert grid.pixel_scales == (1.0, 1.0)
            assert grid.origin == (0.0, 0.0)

            mask = aa.mask.manual(
                [[True, False], [False, False]], pixel_scales=1.0, origin=(0.0, 1.0)
            )
            grid = aa.masked.grid.manual_1d(
                grid=[[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], mask=mask
            )

            assert type(grid) == grids.Grid
            assert (
                grid.in_2d
                == np.array([[[0.0, 0.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            ).all()
            assert (grid.in_1d == np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
            assert grid.pixel_scales == (1.0, 1.0)
            assert grid.origin == (0.0, 1.0)

            mask = aa.mask.manual(
                [[False], [True]], sub_size=2, pixel_scales=1.0, origin=(0.0, 1.0)
            )
            grid = aa.masked.grid.manual_2d(
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

            grid = aa.masked.grid.manual_2d(
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

        def test__exception_raised_if_input_grid_is_2d_and_not_sub_shape_of_mask(self):

            with pytest.raises(exc.GridException):
                mask = aa.mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)
                aa.masked.grid.manual_2d(grid=[[[1.0, 1.0], [3.0, 3.0]]], mask=mask)

            with pytest.raises(exc.GridException):
                mask = aa.mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=2)
                aa.masked.grid.manual_2d(
                    grid=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], mask=mask
                )

            with pytest.raises(exc.GridException):
                mask = aa.mask.unmasked(shape_2d=(2, 2), pixel_scales=1.0, sub_size=2)
                aa.masked.grid.manual_2d(
                    grid=[
                        [[1.0, 1.0], [2.0, 2.0]],
                        [[3.0, 3.0], [4.0, 4.0]],
                        [[5.0, 5.0], [6.0, 6.0]],
                    ],
                    mask=mask,
                )

        def test__exception_raised_if_input_grid_is_not_number_of_masked_sub_pixels(
            self
        ):

            with pytest.raises(exc.GridException):
                mask = aa.mask.manual(
                    mask_2d=[[False, False], [True, False]], sub_size=1
                )
                aa.masked.grid.manual_1d(
                    grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], mask=mask
                )

            with pytest.raises(exc.GridException):
                mask = aa.mask.manual(
                    mask_2d=[[False, False], [True, False]], sub_size=1
                )
                aa.masked.grid.manual_1d(grid=[[1.0, 1.0], [2.0, 2.0]], mask=mask)

            with pytest.raises(exc.GridException):
                mask = aa.mask.manual(mask_2d=[[False, True], [True, True]], sub_size=2)
                aa.masked.grid.manual_2d(
                    grid=[[[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]], mask=mask
                )

            with pytest.raises(exc.GridException):
                mask = aa.mask.manual(mask_2d=[[False, True], [True, True]], sub_size=2)
                aa.masked.grid.manual_2d(
                    grid=[[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]],
                    mask=mask,
                )

    class TestFromMask:
        def test__from_mask__compare_to_array_util(self):
            mask = np.array(
                [
                    [True, True, False, False],
                    [True, False, True, True],
                    [True, True, False, False],
                ]
            )
            mask = aa.mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0), sub_size=1)

            grid_via_util = aa.util.grid.grid_1d_via_mask_2d(
                mask_2d=mask, sub_size=1, pixel_scales=(2.0, 2.0)
            )

            grid = aa.masked.grid.from_mask(mask=mask)

            assert type(grid) == grids.Grid
            assert grid == pytest.approx(grid_via_util, 1e-4)
            assert grid.pixel_scales == (2.0, 2.0)
            assert grid.interpolator == None

            grid_2d = mask.mapping.grid_stored_2d_from_sub_grid_1d(sub_grid_1d=grid)

            assert (grid.in_2d == grid_2d).all()

            mask = np.array(
                [[True, True, True], [True, False, False], [True, True, False]]
            )

            mask = aa.mask.manual(mask, pixel_scales=(3.0, 3.0), sub_size=2)

            grid_via_util = aa.util.grid.grid_1d_via_mask_2d(
                mask_2d=mask, pixel_scales=(3.0, 3.0), sub_size=2
            )

            grid = aa.masked.grid.from_mask(mask=mask, store_in_1d=True)

            assert len(grid.shape) == 2
            assert grid == pytest.approx(grid_via_util, 1e-4)

            grid = aa.masked.grid.from_mask(mask=mask, store_in_1d=False)

            assert len(grid.shape) == 3

        def test__grid__from_mask_method_same_as_masked_grid(self):

            mask = np.array(
                [
                    [True, True, False, False],
                    [True, False, True, True],
                    [True, True, False, False],
                ]
            )
            mask = aa.mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0), sub_size=1)

            grid_via_util = aa.util.grid.grid_1d_via_mask_2d(
                mask_2d=mask, sub_size=1, pixel_scales=(2.0, 2.0)
            )

            grid = aa.grid.from_mask(mask=mask)

            assert type(grid) == grids.Grid
            assert grid == pytest.approx(grid_via_util, 1e-4)
            assert grid.pixel_scales == (2.0, 2.0)
            assert grid.interpolator == None

            grid_2d = mask.mapping.grid_stored_2d_from_sub_grid_1d(sub_grid_1d=grid)

            assert (grid.in_2d == grid_2d).all()


class TestGrid:
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

        mask = aa.mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_mask_util = aa.util.mask.blurring_mask_2d_from_mask_2d_and_kernel_shape_2d(
            mask_2d=mask, kernel_shape_2d=(3, 5)
        )

        blurring_grid_util = aa.util.grid.grid_1d_via_mask_2d(
            mask_2d=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        grid = aa.masked.grid.from_mask(mask=mask)

        blurring_grid = grid.blurring_grid_from_kernel_shape(kernel_shape_2d=(3, 5))

        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)

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

        mask = aa.mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        blurring_mask_util = aa.util.mask.blurring_mask_2d_from_mask_2d_and_kernel_shape_2d(
            mask_2d=mask, kernel_shape_2d=(3, 5)
        )

        blurring_grid_util = aa.util.grid.grid_1d_via_mask_2d(
            mask_2d=blurring_mask_util, pixel_scales=(2.0, 2.0), sub_size=1
        )

        mask = aa.mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)
        blurring_grid = grids.Grid.blurring_grid_from_mask_and_kernel_shape(
            mask=mask, kernel_shape_2d=(3, 5)
        )

        assert blurring_grid == pytest.approx(blurring_grid_util, 1e-4)
        assert blurring_grid.pixel_scales == (2.0, 2.0)

    def test__masked_shape_2d_arcsec(self):

        mask = aa.mask.circular(
            shape_2d=(3, 3), radius=1.0, pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = grids.Grid(grid=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=mask)
        assert grid.shape_2d_scaled == (3.0, 2.0)

        grid = grids.Grid(
            grid=np.array([[1.5, 1.0], [-1.5, -1.0], [0.1, 0.1]]), mask=mask
        )
        assert grid.shape_2d_scaled == (3.0, 2.0)

        grid = grids.Grid(
            grid=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0]]), mask=mask
        )
        assert grid.shape_2d_scaled == (4.5, 4.0)

        grid = grids.Grid(
            grid=np.array([[1.5, 1.0], [-1.5, -1.0], [3.0, 3.0], [7.0, -5.0]]),
            mask=mask,
        )
        assert grid.shape_2d_scaled == (8.5, 8.0)

    def test__flipped_property__returns_grid_as_x_then_y(self):

        grid = aa.grid.manual_2d(
            grid=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], pixel_scales=1.0
        )

        assert (
            grid.in_1d_flipped
            == np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])
        ).all()
        assert (
            grid.in_2d_flipped
            == np.array([[[2.0, 1.0], [4.0, 3.0]], [[6.0, 5.0], [8.0, 7.0]]])
        ).all()

        grid = aa.grid.manual_2d(
            grid=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], pixel_scales=1.0
        )

        assert (
            grid.in_1d_flipped == np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]])
        ).all()
        assert (
            grid.in_2d_flipped == np.array([[[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]])
        ).all()

    def test__in_radians(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = aa.mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0))

        grid = aa.masked.grid.from_mask(mask=mask)

        assert grid.in_radians[0, 0] == pytest.approx(0.00000969627362, 1.0e-8)
        assert grid.in_radians[0, 1] == pytest.approx(0.00000484813681, 1.0e-8)

        assert grid.in_radians[0, 0] == pytest.approx(
            2.0 * np.pi / (180 * 3600), 1.0e-8
        )
        assert grid.in_radians[0, 1] == pytest.approx(
            1.0 * np.pi / (180 * 3600), 1.0e-8
        )

    def test__yticks(self):

        mask = aa.mask.circular(
            shape_2d=(3, 3), radius=1.0, pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = grids.Grid(grid=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        grid = grids.Grid(grid=np.array([[3.0, 1.0], [-3.0, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        grid = grids.Grid(grid=np.array([[5.0, 3.5], [2.0, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__xticks(self):
        mask = aa.mask.circular(
            shape_2d=(3, 3), radius=1.0, pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = grids.Grid(grid=np.array([[1.0, 1.5], [-1.0, -1.5]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        grid = grids.Grid(grid=np.array([[1.0, 3.0], [-1.0, -3.0]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        grid = grids.Grid(grid=np.array([[3.5, 2.0], [-1.0, 5.0]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__new_grid__with_interpolator__returns_grid_with_interpolator(self):
        mask = np.array(
            [
                [True, True, False, False],
                [True, False, True, True],
                [True, True, False, False],
            ]
        )
        mask = aa.mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0))

        grid = aa.masked.grid.from_mask(mask=mask)

        grid_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=1.0
        )

        assert (grid[:, :] == grid_with_interp[:, :]).all()
        assert (grid.mask == grid_with_interp.mask).all()

        interpolator_manual = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
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
        mask = aa.mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0))

        grid = aa.masked.grid.from_mask(mask=mask)

        grid.new_grid_with_binned_grid(binned_grid=1)

        assert grid.binned == 1

    def test__padded_grid_from_kernel_shape__matches_grid_2d_after_padding(self):

        grid = grids.Grid.uniform(shape_2d=(4, 4), pixel_scales=3.0, sub_size=1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_2d(
            mask_2d=np.full((6, 6), False), pixel_scales=(3.0, 3.0), sub_size=1
        )

        assert padded_grid.shape == (36, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 6))).all()
        assert (padded_grid == padded_grid_util).all()
        assert padded_grid.interpolator is None

        grid = grids.Grid.uniform(shape_2d=(4, 5), pixel_scales=2.0, sub_size=1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_2d(
            mask_2d=np.full((6, 7), False), pixel_scales=(2.0, 2.0), sub_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = grids.Grid.uniform(shape_2d=(5, 4), pixel_scales=1.0, sub_size=1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_2d(
            mask_2d=np.full((7, 6), False), pixel_scales=(1.0, 1.0), sub_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = grids.Grid.uniform(shape_2d=(5, 5), pixel_scales=8.0, sub_size=1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(2, 5))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_2d(
            mask_2d=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=1
        )

        assert padded_grid.shape == (54, 2)
        assert (padded_grid == padded_grid_util).all()

        mask = aa.mask.manual(
            mask_2d=np.full((5, 4), False), pixel_scales=(2.0, 2.0), sub_size=2
        )

        grid = aa.masked.grid.from_mask(mask=mask)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_2d(
            mask_2d=np.full((7, 6), False), pixel_scales=(2.0, 2.0), sub_size=2
        )

        assert padded_grid.shape == (168, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(7, 6))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)
        assert padded_grid.interpolator is None

        mask = aa.mask.manual(
            mask_2d=np.full((2, 5), False), pixel_scales=(8.0, 8.0), sub_size=4
        )

        grid = aa.masked.grid.from_mask(mask=mask)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(5, 5))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_2d(
            mask_2d=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=4
        )

        assert padded_grid.shape == (864, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 9))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)

    def test__padded_grid_from_kernel_shape__has_interpolator_grid_if_had_one_before(
        self
    ):
        grid = grids.Grid.uniform(shape_2d=(4, 4), pixel_scales=3.0, sub_size=1)

        grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=0.1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        assert padded_grid.interpolator is not None
        assert padded_grid.interpolator.pixel_scale_interpolation_grid == 0.1

        mask = aa.mask.unmasked(shape_2d=(6, 6), pixel_scales=(3.0, 3.0), sub_size=1)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=padded_grid, pixel_scale_interpolation_grid=0.1
        )

        assert (padded_grid.interpolator.vtx == interpolator.vtx).all()
        assert (padded_grid.interpolator.wts == interpolator.wts).all()

        mask = aa.mask.manual(
            mask_2d=np.full((5, 4), False), pixel_scales=(2.0, 2.0), sub_size=2
        )

        grid = aa.masked.grid.from_mask(mask=mask)

        grid = grid.new_grid_with_interpolator(pixel_scale_interpolation_grid=0.1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        assert padded_grid.interpolator is not None
        assert padded_grid.interpolator.pixel_scale_interpolation_grid == 0.1

        mask = aa.mask.unmasked(shape_2d=(7, 6), pixel_scales=(2.0, 2.0), sub_size=2)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
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

        mask = aa.mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        sub_border_1d_indexes_util = aa.util.mask.sub_border_pixel_1d_indexes_from_mask_2d_and_sub_size(
            mask_2d=mask, sub_size=2
        )

        grid = aa.masked.grid.from_mask(mask=mask)

        assert grid.regions._sub_border_1d_indexes == pytest.approx(
            sub_border_1d_indexes_util, 1e-4
        )

    def test__square_distance_from_coordinate_array(self):

        mask = aa.mask.manual(
            [[True, False], [False, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        grid = aa.masked.grid.manual_1d(
            grid=[[1.0, 1.0], [2.0, 3.0], [1.0, 2.0]], mask=mask
        )

        square_distances = grid.squared_distances_from_coordinate(coordinate=(0.0, 0.0))

        assert (square_distances.in_1d == np.array([2.0, 13.0, 5.0])).all()
        assert (square_distances.mask == mask).all()

        square_distances = grid.squared_distances_from_coordinate(coordinate=(0.0, 1.0))

        assert (square_distances.in_1d == np.array([1.0, 8.0, 2.0])).all()
        assert (square_distances.mask == mask).all()

    def test__distance_from_coordinate_array(self):

        mask = aa.mask.manual(
            [[True, False], [False, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        grid = aa.masked.grid.manual_1d(
            grid=[[1.0, 1.0], [2.0, 3.0], [1.0, 2.0]], mask=mask
        )

        square_distances = grid.distances_from_coordinate(coordinate=(0.0, 0.0))

        assert (
            square_distances.in_1d
            == np.array([np.sqrt(2.0), np.sqrt(13.0), np.sqrt(5.0)])
        ).all()
        assert (square_distances.mask == mask).all()

        square_distances = grid.distances_from_coordinate(coordinate=(0.0, 1.0))

        assert (
            square_distances.in_1d == np.array([1.0, np.sqrt(8.0), np.sqrt(2.0)])
        ).all()
        assert (square_distances.mask == mask).all()


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

        mask = aa.mask.manual(mask_2d=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        grid = aa.masked.grid.from_mask(mask=mask)

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
        mask = aa.mask.circular(
            shape_2d=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=1
        )

        grid = aa.masked.grid.from_mask(mask=mask)

        grid_to_relocate = grids.Grid(
            grid=np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert (
            relocated_grid == np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]])
        ).all()
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.sub_size == 1

        mask = aa.mask.circular(
            shape_2d=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=2
        )

        grid = aa.masked.grid.from_mask(mask=mask)

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
        mask = aa.mask.circular(
            shape_2d=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=1
        )

        grid = aa.masked.grid.from_mask(mask=mask)

        grid_to_relocate = grids.Grid(
            grid=np.array([[10.1, 0.0], [0.0, 10.1], [-10.1, -10.1]]), mask=mask
        )

        relocated_grid = grid.relocated_grid_from_grid(grid=grid_to_relocate)

        assert relocated_grid == pytest.approx(
            np.array([[0.95, 0.0], [0.0, 0.95], [-0.7017, -0.7017]]), 0.1
        )
        assert (relocated_grid.mask == mask).all()
        assert relocated_grid.sub_size == 1

        mask = aa.mask.circular(
            shape_2d=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=2
        )

        grid = aa.masked.grid.from_mask(mask=mask)

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
        mask = aa.mask.circular(
            shape_2d=(60, 60),
            radius=1.0,
            pixel_scales=(0.1, 0.1),
            centre=(1.0, 1.0),
            sub_size=1,
        )

        grid = aa.masked.grid.from_mask(mask=mask)

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

        mask = aa.mask.circular(
            shape_2d=(60, 60),
            radius=1.0,
            pixel_scales=(0.1, 0.1),
            centre=(1.0, 1.0),
            sub_size=2,
        )

        grid = aa.masked.grid.from_mask(mask=mask)

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


class TestGridIrregular:
    def test__pixelization_grid__attributes(self):
        pix_grid = grids.GridIrregular(
            grid=np.array([[1.0, 1.0], [2.0, 2.0]]),
            nearest_irregular_1d_index_for_mask_1d_index=np.array([0, 1]),
        )

        assert type(pix_grid) == grids.GridIrregular
        assert (pix_grid == np.array([[1.0, 1.0], [2.0, 2.0]])).all()
        assert (
            pix_grid.nearest_irregular_1d_index_for_mask_1d_index == np.array([0, 1])
        ).all()

    def test__from_unmasked_sparse_shape_and_grid(self):
        mask = aa.mask.manual(
            mask_2d=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scales=(0.5, 0.5),
            sub_size=1,
        )

        grid = aa.masked.grid.from_mask(mask=mask)

        sparse_grid = grids.SparseGrid.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=(10, 10), grid=grid
        )

        pixelization_grid = grids.GridIrregular.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=(10, 10), grid=grid
        )

        assert (sparse_grid.sparse == pixelization_grid).all()
        assert (
            sparse_grid.sparse_1d_index_for_mask_1d_index
            == pixelization_grid.nearest_irregular_1d_index_for_mask_1d_index
        ).all()


class TestSparseGrid:
    class TestUnmaskedShape:
        def test__properties_consistent_with_util(self):
            mask = aa.mask.manual(
                mask_2d=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                pixel_scales=(0.5, 0.5),
                sub_size=1,
            )

            grid = aa.masked.grid.from_mask(mask=mask)

            sparse_grid = grids.SparseGrid.from_grid_and_unmasked_2d_grid_shape(
                unmasked_sparse_shape=(10, 10), grid=grid
            )

            unmasked_sparse_grid_util = aa.util.grid.grid_1d_via_shape_2d(
                shape_2d=(10, 10),
                pixel_scales=(0.15, 0.15),
                sub_size=1,
                origin=(0.0, 0.0),
            )

            unmasked_sparse_grid_pixel_centres = aa.util.grid.grid_pixel_centres_1d_from_grid_scaled_1d_shape_2d_and_pixel_scales(
                grid_scaled_1d=unmasked_sparse_grid_util,
                shape_2d=grid.mask.shape,
                pixel_scales=grid.pixel_scales,
            ).astype(
                "int"
            )

            total_sparse_pixels = aa.util.mask.total_sparse_pixels_from_mask_2d(
                mask_2d=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            )

            regular_to_unmasked_sparse_util = aa.util.grid.grid_pixel_indexes_1d_from_grid_scaled_1d_shape_2d_and_pixel_scales(
                grid_scaled_1d=grid,
                shape_2d=(10, 10),
                pixel_scales=(0.15, 0.15),
                origin=(0.0, 0.0),
            ).astype(
                "int"
            )

            unmasked_sparse_for_sparse_util = aa.util.sparse.unmasked_sparse_for_sparse_from_mask_2d_and_pixel_centres(
                total_sparse_pixels=total_sparse_pixels,
                mask_2d=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
            ).astype(
                "int"
            )

            sparse_for_unmasked_sparse_util = aa.util.sparse.sparse_for_unmasked_sparse_from_mask_2d_and_pixel_centres(
                mask_2d=mask,
                unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
                total_sparse_pixels=total_sparse_pixels,
            ).astype(
                "int"
            )

            sparse_1d_index_for_mask_1d_index_util = aa.util.sparse.sparse_1d_index_for_mask_1d_index_from_sparse_mappings(
                regular_to_unmasked_sparse=regular_to_unmasked_sparse_util,
                sparse_for_unmasked_sparse=sparse_for_unmasked_sparse_util,
            )

            sparse_grid_util = aa.util.sparse.sparse_grid_from_unmasked_sparse_grid(
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
            mask = aa.mask.manual(
                mask_2d=np.array(
                    [[True, False, True], [False, False, False], [True, False, True]]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.masked.grid.from_mask(mask=mask)

            sparse_grid = grids.SparseGrid.from_grid_and_unmasked_2d_grid_shape(
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
            mask = aa.mask.manual(
                mask_2d=np.array(
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

            grid = aa.masked.grid.from_mask(mask=mask)

            sparse_grid = grids.SparseGrid.from_grid_and_unmasked_2d_grid_shape(
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
            mask = aa.mask.manual(
                mask_2d=np.array(
                    [
                        [True, False, True, True],
                        [False, False, False, False],
                        [True, False, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.masked.grid.from_mask(mask=mask)

            sparse_grid = grids.SparseGrid.from_grid_and_unmasked_2d_grid_shape(
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
            mask = aa.mask.manual(
                mask_2d=np.array(
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

            grid = aa.masked.grid.from_mask(mask=mask)

            # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
            # the central (3x3) pixels only.

            sparse_grid = grids.SparseGrid.from_grid_and_unmasked_2d_grid_shape(
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
            mask = aa.mask.manual(
                mask_2d=np.array(
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

            grid = aa.masked.grid.from_mask(mask=mask)

            # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
            # the central (3x3) pixels only.

            sparse_grid = grids.SparseGrid.from_grid_and_unmasked_2d_grid_shape(
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
            grid = aa.masked.grid.from_mask(mask=mask_7x7)

            sparse_grid = grids.SparseGrid.from_grid_and_unmasked_2d_grid_shape(
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
            mask = aa.mask.manual(
                mask_2d=np.array(
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

            grid = aa.masked.grid.from_mask(mask=mask)

            sparse_grid = grids.SparseGrid.from_grid_and_unmasked_2d_grid_shape(
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
            mask = aa.mask.manual(
                mask_2d=np.array(
                    [
                        [True, False, True, True],
                        [False, False, False, False],
                        [True, False, True, True],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
                sub_size=1,
            )

            grid = aa.masked.grid.from_mask(mask=mask)

            sparse_grid = grids.SparseGrid.from_grid_and_unmasked_2d_grid_shape(
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
            mask = aa.mask.manual(
                mask_2d=np.array(
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

            grid = aa.masked.grid.from_mask(mask=mask)

            sparse_grid = grids.SparseGrid.from_grid_and_unmasked_2d_grid_shape(
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
            mask = aa.mask.manual(
                mask_2d=np.array(
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

            grid = aa.masked.grid.from_mask(mask=mask)

            weight_map = np.ones(mask.pixels_in_mask)

            sparse_grid_weight = grids.SparseGrid.from_total_pixels_grid_and_weight_map(
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
            mask = aa.mask.manual(
                mask_2d=np.array(
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

            grid = aa.masked.grid.from_mask(mask=mask)

            weight_map = np.ones(mask.pixels_in_mask)
            weight_map[0:15] = 0.00000001

            sparse_grid_weight = grids.SparseGrid.from_total_pixels_grid_and_weight_map(
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


class TestGridRectangular:
    class TestGridNeighbors:
        def test__3x3_grid__buffer_is_small__grid_give_min_minus_1_max_1__sets_up_geometry_correctly(
            self
        ):
            grid = np.array(
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

            pix_grid = aa.grid_rectangular.overlay_grid(
                shape_2d=(3, 3), grid=grid, buffer=1e-8
            )

            assert pix_grid.shape_2d == (3, 3)
            assert pix_grid.pixel_scales == pytest.approx((2.0 / 3.0, 2.0 / 3.0), 1e-2)
            assert (pix_grid.pixel_neighbors[0] == [1, 3, -1, -1]).all()
            assert (pix_grid.pixel_neighbors[1] == [0, 2, 4, -1]).all()
            assert (pix_grid.pixel_neighbors[2] == [1, 5, -1, -1]).all()
            assert (pix_grid.pixel_neighbors[3] == [0, 4, 6, -1]).all()
            assert (pix_grid.pixel_neighbors[4] == [1, 3, 5, 7]).all()
            assert (pix_grid.pixel_neighbors[5] == [2, 4, 8, -1]).all()
            assert (pix_grid.pixel_neighbors[6] == [3, 7, -1, -1]).all()
            assert (pix_grid.pixel_neighbors[7] == [4, 6, 8, -1]).all()
            assert (pix_grid.pixel_neighbors[8] == [5, 7, -1, -1]).all()

            assert (
                pix_grid.pixel_neighbors_size == np.array([2, 3, 2, 3, 4, 3, 2, 3, 2])
            ).all()

        def test__3x3_grid__same_as_above_change_buffer(self):
            grid = np.array(
                [
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ]
            )

            pix_grid = aa.grid_rectangular.overlay_grid(
                shape_2d=(3, 3), grid=grid, buffer=1e-8
            )

            assert pix_grid.shape_2d == (3, 3)
            assert pix_grid.pixel_scales == pytest.approx((2.0 / 3.0, 2.0 / 3.0), 1e-2)

        def test__5x4_grid__buffer_is_small(self):
            grid = np.array(
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

            pix_grid = aa.grid_rectangular.overlay_grid(
                shape_2d=(5, 4), grid=grid, buffer=1e-8
            )

            assert pix_grid.shape_2d == (5, 4)
            assert pix_grid.pixel_scales == pytest.approx((2.0 / 5.0, 2.0 / 4.0), 1e-2)

        def test__3x3_grid__larger_range_of_grid(self):
            grid = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])

            pix_grid = aa.grid_rectangular.overlay_grid(
                shape_2d=(3, 3), grid=grid, buffer=1e-8
            )

            assert pix_grid.shape_2d == (3, 3)
            assert pix_grid.pixel_scales == pytest.approx((6.0 / 3.0, 6.0 / 3.0), 1e-2)

    class TestPixelCentres:
        def test__3x3_grid__pixel_centres(self):
            grid = np.array(
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

            pix_grid = aa.grid_rectangular.overlay_grid(
                shape_2d=(3, 3), grid=grid, buffer=1e-8
            )

            assert pix_grid == pytest.approx(
                np.array(
                    [
                        [2.0 / 3.0, -2.0 / 3.0],
                        [2.0 / 3.0, 0.0],
                        [2.0 / 3.0, 2.0 / 3.0],
                        [0.0, -2.0 / 3.0],
                        [0.0, 0.0],
                        [0.0, 2.0 / 3.0],
                        [-2.0 / 3.0, -2.0 / 3.0],
                        [-2.0 / 3.0, 0.0],
                        [-2.0 / 3.0, 2.0 / 3.0],
                    ]
                )
            )

        def test__4x3_grid__pixel_centres(self):
            grid = np.array(
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

            pix_grid = aa.grid_rectangular.overlay_grid(
                shape_2d=(4, 3), grid=grid, buffer=1e-8
            )

            assert pix_grid == pytest.approx(
                np.array(
                    [
                        [0.75, -2.0 / 3.0],
                        [0.75, 0.0],
                        [0.75, 2.0 / 3.0],
                        [0.25, -2.0 / 3.0],
                        [0.25, 0.0],
                        [0.25, 2.0 / 3.0],
                        [-0.25, -2.0 / 3.0],
                        [-0.25, 0.0],
                        [-0.25, 2.0 / 3.0],
                        [-0.75, -2.0 / 3.0],
                        [-0.75, 0.0],
                        [-0.75, 2.0 / 3.0],
                    ]
                )
            )

    class TestPixelNeighbors:
        def test__compare_to_pixelization_util(self):
            # |0 | 1| 2| 3|
            # |4 | 5| 6| 7|
            # |8 | 9|10|11|
            # |12|13|14|15|

            pix_grid = aa.grid_rectangular.overlay_grid(
                shape_2d=(7, 5), grid=np.zeros((2, 2)), buffer=1e-8
            )

            pixel_neighbors_util, pixel_neighbors_size_util = aa.util.pixelization.rectangular_neighbors_from_shape(
                shape=(7, 5)
            )

            assert (pix_grid.pixel_neighbors == pixel_neighbors_util).all()
            assert (pix_grid.pixel_neighbors_size == pixel_neighbors_size_util).all()


class TestVoronoi:
    class TestVoronoiGrid:
        def test__9_points___check_voronoi_swaps_axis_from_y_x__to_x_y(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            grid = np.array(
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

            pix = aa.grid_voronoi(grid_1d=grid)

            assert (
                pix.voronoi.points
                == np.array(
                    [
                        [0.0, 2.0],
                        [1.0, 2.0],
                        [2.0, 2.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [2.0, 1.0],
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [2.0, 0.0],
                    ]
                )
            ).all()

        def test__points_in_x_cross_shape__sets_up_diamond_voronoi_vertices(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            grid = np.array(
                [[-1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]]
            )

            pix = aa.grid_voronoi(grid_1d=grid)

            pix.voronoi.vertices = list(map(lambda x: list(x), pix.voronoi.vertices))

            assert [0, 1.0] in pix.voronoi.vertices
            assert [-1.0, 0.0] in pix.voronoi.vertices
            assert [1.0, 0.0] in pix.voronoi.vertices
            assert [0.0, -1.0] in pix.voronoi.vertices

        def test__9_points_in_square___sets_up_square_of_voronoi_vertices(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            grid = np.array(
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

            pix = aa.grid_voronoi(grid_1d=grid)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            pix.voronoi.vertices = list(map(lambda x: list(x), pix.voronoi.vertices))

            assert [0.5, 1.5] in pix.voronoi.vertices
            assert [1.5, 0.5] in pix.voronoi.vertices
            assert [0.5, 0.5] in pix.voronoi.vertices
            assert [1.5, 1.5] in pix.voronoi.vertices

        def test__points_in_x_cross_shape__sets_up_pairs_of_voronoi_cells(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            grid = np.array(
                [[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]]
            )

            pix = aa.grid_voronoi(grid_1d=grid)
            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            pix.voronoi.ridge_grid = list(
                map(lambda x: list(x), pix.voronoi.ridge_points)
            )

            assert len(pix.voronoi.ridge_points) == 8

            assert [2, 0] in pix.voronoi.ridge_points or [
                0,
                2,
            ] in pix.voronoi.ridge_points
            assert [2, 1] in pix.voronoi.ridge_points or [
                1,
                2,
            ] in pix.voronoi.ridge_points
            assert [2, 3] in pix.voronoi.ridge_points or [
                3,
                2,
            ] in pix.voronoi.ridge_points
            assert [2, 4] in pix.voronoi.ridge_points or [
                4,
                2,
            ] in pix.voronoi.ridge_points
            assert [0, 1] in pix.voronoi.ridge_points or [
                1,
                0,
            ] in pix.voronoi.ridge_points
            assert [0.3] in pix.voronoi.ridge_points or [
                3,
                0,
            ] in pix.voronoi.ridge_points
            assert [3, 4] in pix.voronoi.ridge_points or [
                4,
                3,
            ] in pix.voronoi.ridge_points
            assert [4, 1] in pix.voronoi.ridge_points or [
                1,
                4,
            ] in pix.voronoi.ridge_points

        def test__9_points_in_square___sets_up_pairs_of_voronoi_cells(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            grid = np.array(
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

            pix = aa.grid_voronoi(grid_1d=grid)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            pix.voronoi.ridge_grid = list(
                map(lambda x: list(x), pix.voronoi.ridge_points)
            )

            assert len(pix.voronoi.ridge_points) == 12

            assert [0, 1] in pix.voronoi.ridge_points or [
                1,
                0,
            ] in pix.voronoi.ridge_points
            assert [1, 2] in pix.voronoi.ridge_points or [
                2,
                1,
            ] in pix.voronoi.ridge_points
            assert [3, 4] in pix.voronoi.ridge_points or [
                4,
                3,
            ] in pix.voronoi.ridge_points
            assert [4, 5] in pix.voronoi.ridge_points or [
                5,
                4,
            ] in pix.voronoi.ridge_points
            assert [6, 7] in pix.voronoi.ridge_points or [
                7,
                6,
            ] in pix.voronoi.ridge_points
            assert [7, 8] in pix.voronoi.ridge_points or [
                8,
                7,
            ] in pix.voronoi.ridge_points

            assert [0, 3] in pix.voronoi.ridge_points or [
                3,
                0,
            ] in pix.voronoi.ridge_points
            assert [1, 4] in pix.voronoi.ridge_points or [
                4,
                1,
            ] in pix.voronoi.ridge_points
            assert [4, 7] in pix.voronoi.ridge_points or [
                7,
                4,
            ] in pix.voronoi.ridge_points
            assert [2, 5] in pix.voronoi.ridge_points or [
                5,
                2,
            ] in pix.voronoi.ridge_points
            assert [5, 8] in pix.voronoi.ridge_points or [
                8,
                5,
            ] in pix.voronoi.ridge_points
            assert [3, 6] in pix.voronoi.ridge_points or [
                6,
                3,
            ] in pix.voronoi.ridge_points

    class TestNeighbors:
        def test__compare_to_pixelization_util(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            grid = np.array(
                [
                    [3.0, 0.0],
                    [2.0, 1.0],
                    [2.0, 2.0],
                    [8.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 9.0],
                    [6.0, 31.0],
                    [0.0, 2.0],
                    [3.0, 5.0],
                ]
            )

            pix = aa.grid_voronoi(grid_1d=grid)

            voronoi = scipy.spatial.Voronoi(
                np.asarray([grid[:, 1], grid[:, 0]]).T, qhull_options="Qbb Qc Qx Qm"
            )
            pixel_neighbors_util, pixel_neighbors_size_util = aa.util.pixelization.voronoi_neighbors_from_pixels_and_ridge_points(
                pixels=9, ridge_points=np.array(voronoi.ridge_points)
            )

            assert (pix.pixel_neighbors == pixel_neighbors_util).all()
            assert (pix.pixel_neighbors_size == pixel_neighbors_size_util).all()


@grids.grid_interpolate
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
        @grids.grid_interpolate
        def func(profile, grid, grid_radial_minimum=None):
            result = np.zeros(grid.shape[0])
            result[0] = 1
            return result

        grid = aa.masked.grid.from_mask(
            mask=aa.mask.unmasked(shape_2d=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1)
        )

        values = func(None, grid)

        assert values.ndim == 1
        assert values.shape == (9,)
        assert (values == np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])).all()

        grid = aa.masked.grid.from_mask(
            mask=aa.mask.unmasked(shape_2d=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1)
        )
        grid.interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
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
        @grids.grid_interpolate
        def func(profile, grid, grid_radial_minimum=None):
            result = np.zeros((grid.shape[0], 2))
            result[0, :] = 1
            return result

        grid = aa.masked.grid.from_mask(
            mask=aa.mask.unmasked(shape_2d=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1)
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

        grid = aa.masked.grid.from_mask(
            mask=aa.mask.unmasked(shape_2d=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1)
        )
        grid.interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
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

        mask = aa.mask.circular_annular(
            shape_2d=(20, 20),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
            inner_radius=3.0,
            outer_radius=8.0,
        )

        grid = aa.masked.grid.from_mask(mask=mask)

        true_grid_radii = grid_radii_from_grid(profile=None, grid=grid)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
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

        mask = aa.mask.circular_annular(
            shape_2d=(24, 24),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
            inner_radius=3.0,
            outer_radius=8.0,
            centre=(3.0, 3.0),
        )

        grid = aa.masked.grid.from_mask(mask=mask)

        true_grid_radii = grid_radii_from_grid(profile=None, grid=grid)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
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

        mask = aa.mask.circular_annular(
            shape_2d=(28, 28),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
            inner_radius=3.0,
            outer_radius=8.0,
            centre=(3.0, 3.0),
        )

        grid = aa.masked.grid.from_mask(mask=mask)

        true_grid_radii = grid_radii_from_grid(profile=None, grid=grid)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
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

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
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

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
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


test_positions_dir = "{}/../test_files/positions/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class MockPositionInput(object):
    def __init__(self):
        pass

    @grids.convert_positions_to_grid
    def float_values_from_grid(self, grid):
        return np.ones(shape=grid.shape[0])

    @grids.convert_positions_to_grid
    def tuple_values_from_grid(self, grid):
        return np.multiply(2.0, grid)

    @grids.convert_positions_to_grid
    def float_values_from_grid_returns_list(self, grid):
        return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]

    @grids.convert_positions_to_grid
    def tuple_values_from_grid_returns_list(self, grid):
        return [np.multiply(1.0, grid), np.multiply(2.0, grid)]


class TestPositions:
    def test__converts_to_and_from_pixels(self):

        mask = aa.mask.manual(
            mask_2d=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
        )

        positions = aa.positions(positions=[[(1.0, -1.0), (1.0, 1.0)]], mask=mask)

        assert type(positions.scaled) == grids.Positions
        assert positions.scaled == [[(1.0, -1.0), (1.0, 1.0)]]
        assert positions.pixels == [[(0, 0), (0, 1)]]

        positions = aa.positions.from_pixels_and_mask(
            pixels=[[(0, 0), (0, 1)]], mask=mask
        )

        assert positions.scaled == [[(1.0, -1.0), (1.0, 1.0)]]
        assert positions.pixels == [[(0, 0), (0, 1)]]

    def test__input_is_list_of_tuples__converts_to_irregular_1d_grid(self):

        positions = aa.positions(positions=[[(1.0, 1.0), (2.0, 2.0)]])

        assert positions == [[(1.0, 1.0), (2.0, 2.0)]]
        assert type(positions.in_1d) == grids.GridIrregular
        assert (positions.in_1d == np.array([[1.0, 1.0], [2.0, 2.0]])).all()
        assert (positions.list_in_1d[0] == np.array([[1.0, 1.0], [2.0, 2.0]])).all()

        positions = aa.positions(positions=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]])

        assert positions == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]
        assert type(positions.in_1d) == grids.GridIrregular
        assert (positions.in_1d == np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])).all()
        assert (positions.list_in_1d[0] == np.array([[1.0, 1.0], [2.0, 2.0]])).all()
        assert (positions.list_in_1d[1] == np.array([[3.0, 3.0]])).all()

    def test__retain_original_list_of_tuples_from_a_1d_grid(self):

        positions = aa.positions(positions=[[(1.0, 1.0), (2.0, 2.0)]])

        positions_from_1d = positions.from_1d_positions(
            positions_1d=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert type(positions_from_1d) == grids.Positions
        assert positions_from_1d == [[(1.0, 1.0), (2.0, 2.0)]]

        positions = aa.positions(positions=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]])

        positions_from_1d = positions.from_1d_positions(
            positions_1d=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(positions_from_1d) == grids.Positions
        assert positions_from_1d == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]

    def test__retain_origin_list_but_as_floats_for_single_values(self):

        positions = aa.positions(positions=[[(1.0, 1.0), (2.0, 2.0)]])

        values_from_1d = positions.from_1d_values(values_1d=np.array([1.0, 2.0]))

        assert values_from_1d == [[1.0, 2.0]]

        positions = aa.positions(positions=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]])

        values_from_1d = positions.from_1d_values(values_1d=np.array([1.0, 2.0, 3.0]))

        assert values_from_1d == [[1.0, 2.0], [3.0]]

    def test__load_positions__retains_list_structure(self):
        positions = aa.positions.from_file(
            positions_path=test_positions_dir + "positions_test.dat"
        )

        assert positions == [
            [(1.0, 1.0), (2.0, 2.0)],
            [(3.0, 3.0), (4.0, 4.0), (5.0, 6.0)],
        ]

    def test__output_positions(self):
        positions = aa.positions(
            [[(4.0, 4.0), (5.0, 5.0)], [(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)]]
        )

        output_data_dir = "{}/../test_files/positions/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        positions.output_to_file(positions_path=output_data_dir + "positions_test.dat")

        positions = aa.positions.from_file(
            positions_path=output_data_dir + "positions_test.dat"
        )

        assert positions == [
            [(4.0, 4.0), (5.0, 5.0)],
            [(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)],
        ]

    def test__convert_positions_decorator__positions_are_input__output_in_same_format(
        self
    ):

        positions_input = MockPositionInput()

        positions = aa.positions(positions=[[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0)]])

        positions_output = positions_input.float_values_from_grid(grid=positions)

        assert positions_output == [[1.0, 1.0], [1.0]]

        positions_output = positions_input.tuple_values_from_grid(grid=positions)

        assert positions_output == [[(2.0, 4.0), (6.0, 8.0)], [(10.0, 12.0)]]

    def test__convert_positions_decorator__same_as_above_but_output_is_a_list(self):

        positions_input = MockPositionInput()

        positions = aa.positions(positions=[[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0)]])

        positions_output = positions_input.float_values_from_grid_returns_list(
            grid=positions
        )

        assert positions_output == [[[1.0, 1.0], [1.0]], [[2.0, 2.0], [2.0]]]

        positions_output = positions_input.tuple_values_from_grid_returns_list(
            grid=positions
        )

        assert positions_output == [
            [[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0)]],
            [[(2.0, 4.0), (6.0, 8.0)], [(10.0, 12.0)]],
        ]
