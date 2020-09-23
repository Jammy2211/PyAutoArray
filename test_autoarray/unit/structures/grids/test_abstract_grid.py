import os
import numpy as np
import pytest

import autoarray as aa
from autoarray.structures import grids

from test_autoarray.mock import MockGridRadialMinimum

test_coordinates_dir = "{}/files/coordinates/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestGrid:
    def test__masked_shape_2d_arcsec(self):
        mask = aa.Mask.circular(
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
        grid = aa.Grid.manual_2d(
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

        grid = aa.Grid.manual_2d(
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
        mask = aa.Mask.manual(mask=mask, pixel_scales=(2.0, 2.0))

        grid = aa.Grid.from_mask(mask=mask)

        assert grid.in_radians[0, 0] == pytest.approx(0.00000969627362, 1.0e-8)
        assert grid.in_radians[0, 1] == pytest.approx(0.00000484813681, 1.0e-8)

        assert grid.in_radians[0, 0] == pytest.approx(
            2.0 * np.pi / (180 * 3600), 1.0e-8
        )
        assert grid.in_radians[0, 1] == pytest.approx(
            1.0 * np.pi / (180 * 3600), 1.0e-8
        )

    def test__yticks(self):
        mask = aa.Mask.circular(
            shape_2d=(3, 3), radius=1.0, pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = grids.Grid(grid=np.array([[1.5, 1.0], [-1.5, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        grid = grids.Grid(grid=np.array([[3.0, 1.0], [-3.0, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        grid = grids.Grid(grid=np.array([[5.0, 3.5], [2.0, -1.0]]), mask=mask)
        assert grid.yticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__xticks(self):
        mask = aa.Mask.circular(
            shape_2d=(3, 3), radius=1.0, pixel_scales=(1.0, 1.0), sub_size=1
        )

        grid = grids.Grid(grid=np.array([[1.0, 1.5], [-1.0, -1.5]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3)

        grid = grids.Grid(grid=np.array([[1.0, 3.0], [-1.0, -3.0]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([-3.0, -1, 1.0, 3.0]), 1e-3)

        grid = grids.Grid(grid=np.array([[3.5, 2.0], [-1.0, 5.0]]), mask=mask)
        assert grid.xticks == pytest.approx(np.array([2.0, 3.0, 4.0, 5.0]), 1e-3)

    def test__padded_grid_from_kernel_shape__matches_grid_2d_after_padding(self):
        grid = grids.Grid.uniform(shape_2d=(4, 4), pixel_scales=3.0, sub_size=1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_from(
            mask=np.full((6, 6), False), pixel_scales=(3.0, 3.0), sub_size=1
        )

        assert isinstance(padded_grid, grids.Grid)
        assert padded_grid.shape == (36, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 6))).all()
        assert (padded_grid == padded_grid_util).all()

        grid = grids.Grid.uniform(shape_2d=(4, 5), pixel_scales=2.0, sub_size=1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_from(
            mask=np.full((6, 7), False), pixel_scales=(2.0, 2.0), sub_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = grids.Grid.uniform(shape_2d=(5, 4), pixel_scales=1.0, sub_size=1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_from(
            mask=np.full((7, 6), False), pixel_scales=(1.0, 1.0), sub_size=1
        )

        assert padded_grid.shape == (42, 2)
        assert (padded_grid == padded_grid_util).all()

        grid = grids.Grid.uniform(shape_2d=(5, 5), pixel_scales=8.0, sub_size=1)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(2, 5))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_from(
            mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=1
        )

        assert padded_grid.shape == (54, 2)
        assert (padded_grid == padded_grid_util).all()

        mask = aa.Mask.manual(
            mask=np.full((5, 4), False), pixel_scales=(2.0, 2.0), sub_size=2
        )

        grid = aa.Grid.from_mask(mask=mask)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(3, 3))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_from(
            mask=np.full((7, 6), False), pixel_scales=(2.0, 2.0), sub_size=2
        )

        assert padded_grid.shape == (168, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(7, 6))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)

        mask = aa.Mask.manual(
            mask=np.full((2, 5), False), pixel_scales=(8.0, 8.0), sub_size=4
        )

        grid = aa.Grid.from_mask(mask=mask)

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=(5, 5))

        padded_grid_util = aa.util.grid.grid_1d_via_mask_from(
            mask=np.full((6, 9), False), pixel_scales=(8.0, 8.0), sub_size=4
        )

        assert padded_grid.shape == (864, 2)
        assert (padded_grid.mask == np.full(fill_value=False, shape=(6, 9))).all()
        assert padded_grid == pytest.approx(padded_grid_util, 1e-4)

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

        mask = aa.Mask.manual(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

        sub_border_1d_indexes_util = aa.util.mask.sub_border_pixel_1d_indexes_from(
            mask=mask, sub_size=2
        )

        grid = aa.Grid.from_mask(mask=mask)

        assert grid.regions._sub_border_1d_indexes == pytest.approx(
            sub_border_1d_indexes_util, 1e-4
        )

    def test__square_distance_from_coordinate_array(self):
        mask = aa.Mask.manual(
            [[True, False], [False, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        grid = aa.Grid.manual_mask(grid=[[1.0, 1.0], [2.0, 3.0], [1.0, 2.0]], mask=mask)

        square_distances = grid.squared_distances_from_coordinate(coordinate=(0.0, 0.0))

        assert (square_distances.in_1d == np.array([2.0, 13.0, 5.0])).all()
        assert (square_distances.mask == mask).all()

        square_distances = grid.squared_distances_from_coordinate(coordinate=(0.0, 1.0))

        assert (square_distances.in_1d == np.array([1.0, 8.0, 2.0])).all()
        assert (square_distances.mask == mask).all()

    def test__distance_from_coordinate_array(self):
        mask = aa.Mask.manual(
            [[True, False], [False, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        grid = aa.Grid.manual_mask(grid=[[1.0, 1.0], [2.0, 3.0], [1.0, 2.0]], mask=mask)

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

    def test__grid_with_coordinates_within_distance_removed__single_coordinates_only(
        self
    ):

        grid = aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0)

        grid = grid.grid_with_coordinates_within_distance_removed(
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

        grid = grid.grid_with_coordinates_within_distance_removed(
            coordinates=(0.0, 0.0), distance=1.1
        )

        assert (
            grid == np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
        ).all()

        grid = aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0, origin=(1.0, 1.0))

        grid = grid.grid_with_coordinates_within_distance_removed(
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

    def test__grid_with_coordinates_within_distance_removed__multiple_coordinates(self):

        grid = aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0)

        grid = grid.grid_with_coordinates_within_distance_removed(
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

        grid = aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0, origin=(1.0, 1.0))

        grid = grid.grid_with_coordinates_within_distance_removed(
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
        self
    ):

        grid = np.array([[2.5, 0.0], [4.0, 0.0], [6.0, 0.0]])
        mock_profile = MockGridRadialMinimum()

        deflections = mock_profile.deflections_from_grid(grid=grid)
        assert (deflections == grid).all()

    def test__mock_profile__grid_radial_minimum_is_above_some_radial_coordinates__moves_them_grid_radial_minimum(
        self
    ):
        grid = np.array([[2.0, 0.0], [1.0, 0.0], [6.0, 0.0]])
        mock_profile = MockGridRadialMinimum()

        deflections = mock_profile.deflections_from_grid(grid=grid)

        assert (deflections == np.array([[2.5, 0.0], [2.5, 0.0], [6.0, 0.0]])).all()

    def test__mock_profile__same_as_above_but_diagonal_coordinates(self):
        grid = np.array(
            [
                [np.sqrt(2.0), np.sqrt(2.0)],
                [1.0, np.sqrt(8.0)],
                [np.sqrt(8.0), np.sqrt(8.0)],
            ]
        )

        mock_profile = MockGridRadialMinimum()

        deflections = mock_profile.deflections_from_grid(grid=grid)

        assert deflections == pytest.approx(
            np.array(
                [[1.7677, 1.7677], [1.0, np.sqrt(8.0)], [np.sqrt(8), np.sqrt(8.0)]]
            ),
            1.0e-4,
        )
