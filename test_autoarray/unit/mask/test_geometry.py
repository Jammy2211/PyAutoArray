import numpy as np
import pytest

import autoarray as aa


class TestCoordinates:
    def test__central_pixel__gives_same_result_as_geometry_util(self):

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(0.1, 0.1)
        )

        central_pixel_coordinates_util = aa.util.geometry.central_pixel_coordinates_from(
            shape=(3, 3)
        )

        assert mask.geometry.central_pixel_coordinates == central_pixel_coordinates_util

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(5, 3)),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )

        central_pixel_coordinates_util = aa.util.geometry.central_pixel_coordinates_from(
            shape=(5, 3)
        )

        assert mask.geometry.central_pixel_coordinates == central_pixel_coordinates_util

    def test__centring__adapts_to_max_and_min_of_mask(self):
        mask = np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

        assert mask.geometry.mask_centre == (0.0, 0.0)

        mask = np.array(
            [
                [True, True, True, True],
                [True, False, False, False],
                [True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

        assert mask.geometry.mask_centre == (0.0, 0.5)

        mask = np.array(
            [
                [True, True, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

        assert mask.geometry.mask_centre == (0.5, 0.0)

        mask = np.array(
            [
                [True, True, True, True],
                [False, False, False, True],
                [True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

        assert mask.geometry.mask_centre == (0.0, -0.5)

        mask = np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

        assert mask.geometry.mask_centre == (-0.5, 0.0)

        mask = np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [False, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

        assert mask.geometry.mask_centre == (-0.5, -0.5)

    def test__pixel_grid__y_and_x_ticks(self):
        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry.yticks == pytest.approx(
            np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(0.5, 0.5)
        )
        assert mask.geometry.yticks == pytest.approx(
            np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(6, 3)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry.yticks == pytest.approx(
            np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 1)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry.yticks == pytest.approx(
            np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry.xticks == pytest.approx(
            np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(0.5, 0.5)
        )
        assert mask.geometry.xticks == pytest.approx(
            np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 6)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry.xticks == pytest.approx(
            np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(1, 3)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry.xticks == pytest.approx(
            np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(1.0, 5.0)
        )
        assert mask.geometry.yticks == pytest.approx(
            np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(0.5, 5.0)
        )
        assert mask.geometry.yticks == pytest.approx(
            np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(6, 3)), pixel_scales=(1.0, 5.0)
        )
        assert mask.geometry.yticks == pytest.approx(
            np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 6)), pixel_scales=(1.0, 5.0)
        )
        assert mask.geometry.yticks == pytest.approx(
            np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(5.0, 1.0)
        )
        assert mask.geometry.xticks == pytest.approx(
            np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(5.0, 0.5)
        )
        assert mask.geometry.xticks == pytest.approx(
            np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 6)), pixel_scales=(5.0, 1.0)
        )
        assert mask.geometry.xticks == pytest.approx(
            np.array([-3.0, -1.0, 1.0, 3.0]), 1e-3
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(6, 3)), pixel_scales=(5.0, 1.0)
        )
        assert mask.geometry.xticks == pytest.approx(
            np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3
        )


class TestGrids:
    def test__unmasked_grid__compare_to_array_util(self):

        grid_2d_util = aa.util.grid.grid_2d_via_shape_2d_from(
            shape_2d=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1
        )

        grid_1d_util = aa.util.grid.grid_1d_via_shape_2d_from(
            shape_2d=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(4, 7)), pixel_scales=(0.56, 0.56)
        )
        mask[0, 0] = True

        assert mask.geometry.unmasked_grid_sub_1.in_1d == pytest.approx(
            grid_1d_util, 1e-4
        )
        assert mask.geometry.unmasked_grid_sub_1.in_2d == pytest.approx(
            grid_2d_util, 1e-4
        )
        assert (
            mask.geometry.unmasked_grid_sub_1.mask
            == np.full(fill_value=False, shape=(4, 7))
        ).all()

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(1.0, 1.0)
        )

        assert (
            mask.geometry.unmasked_grid_sub_1.in_2d
            == np.array(
                [
                    [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]],
                    [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]],
                    [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]],
                ]
            )
        ).all()

        grid_2d_util = aa.util.grid.grid_2d_via_shape_2d_from(
            shape_2d=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1
        )

        grid_1d_util = aa.util.grid.grid_1d_via_shape_2d_from(
            shape_2d=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(4, 7)), pixel_scales=(0.8, 0.56)
        )

        assert mask.geometry.unmasked_grid_sub_1.in_1d == pytest.approx(
            grid_1d_util, 1e-4
        )
        assert mask.geometry.unmasked_grid_sub_1.in_2d == pytest.approx(
            grid_2d_util, 1e-4
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(1.0, 2.0)
        )

        assert (
            mask.geometry.unmasked_grid_sub_1.in_2d
            == np.array(
                [
                    [[1.0, -2.0], [1.0, 0.0], [1.0, 2.0]],
                    [[0.0, -2.0], [0.0, 0.0], [0.0, 2.0]],
                    [[-1.0, -2.0], [-1.0, 0.0], [-1.0, 2.0]],
                ]
            )
        ).all()

    def test__grid_with_nonzero_origins__compure_to_array_util(self):

        grid_2d_util = aa.util.grid.grid_2d_via_shape_2d_from(
            shape_2d=(4, 7), pixel_scales=(0.56, 0.56), origin=(1.0, 3.0), sub_size=1
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(4, 7)),
            pixel_scales=(0.56, 0.56),
            origin=(1.0, 3.0),
        )

        assert mask.geometry.unmasked_grid_sub_1.in_2d == pytest.approx(
            grid_2d_util, 1e-4
        )

        grid_1d_util = aa.util.grid.grid_1d_via_shape_2d_from(
            shape_2d=(4, 7), pixel_scales=(0.56, 0.56), origin=(-1.0, -4.0), sub_size=1
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(4, 7)),
            pixel_scales=(0.56, 0.56),
            origin=(-1.0, -4.0),
        )

        assert mask.geometry.unmasked_grid_sub_1.in_1d == pytest.approx(
            grid_1d_util, 1e-4
        )

        grid_2d_util = aa.util.grid.grid_2d_via_shape_2d_from(
            shape_2d=(4, 7), pixel_scales=(0.8, 0.56), origin=(1.0, 2.0), sub_size=1
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(4, 7)),
            pixel_scales=(0.8, 0.56),
            origin=(1.0, 2.0),
        )

        assert mask.geometry.unmasked_grid_sub_1.in_2d == pytest.approx(
            grid_2d_util, 1e-4
        )

        grid_1d_util = aa.util.grid.grid_1d_via_shape_2d_from(
            shape_2d=(4, 7), pixel_scales=(0.8, 0.56), origin=(-1.0, -4.0), sub_size=1
        )

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(4, 7)),
            pixel_scales=(0.8, 0.56),
            origin=(-1.0, -4.0),
        )

        assert mask.geometry.unmasked_grid_sub_1.in_1d == pytest.approx(
            grid_1d_util, 1e-4
        )

    def test__masked_grids_1d(self):

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(1.0, 1.0)
        )

        assert (
            mask.geometry.masked_grid_sub_1.in_1d
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

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(1.0, 1.0)
        )
        mask[1, 1] = True

        assert (
            mask.geometry.masked_grid_sub_1.in_1d
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

        mask = aa.Mask2D.manual(
            mask=np.array([[False, True], [True, False], [True, False]]),
            pixel_scales=(1.0, 1.0),
            origin=(3.0, -2.0),
        )

        assert (
            mask.geometry.masked_grid_sub_1.in_1d
            == np.array([[4.0, -2.5], [3.0, -1.5], [2.0, -1.5]])
        ).all()

    def test__edge_grid(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, False, True, False, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

        assert mask.geometry.edge_grid_sub_1.in_1d[0:11] == pytest.approx(
            np.array(
                [
                    [3.0, -3.0],
                    [3.0, -2.0],
                    [3.0, -1.0],
                    [3.0, -0.0],
                    [3.0, 1.0],
                    [3.0, 2.0],
                    [3.0, 3.0],
                    [2.0, -3.0],
                    [2.0, 3.0],
                    [1.0, -3.0],
                    [1.0, -1.0],
                ]
            ),
            1e-4,
        )

    def test__border_grid(self):
        mask = np.array(
            [
                [True, True, True, True, True, True, True, True, True],
                [True, False, False, False, False, False, False, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, False, True, False, True, False, True],
                [True, False, True, False, False, False, True, False, True],
                [True, False, True, True, True, True, True, False, True],
                [True, False, False, False, False, False, False, False, True],
                [True, True, True, True, True, True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

        assert mask.geometry.border_grid_sub_1.in_1d[0:11] == pytest.approx(
            np.array(
                [
                    [3.0, -3.0],
                    [3.0, -2.0],
                    [3.0, -1.0],
                    [3.0, -0.0],
                    [3.0, 1.0],
                    [3.0, 2.0],
                    [3.0, 3.0],
                    [2.0, -3.0],
                    [2.0, 3.0],
                    [1.0, -3.0],
                    [1.0, 3.0],
                ]
            ),
            1e-4,
        )

    def test__masked_sub_grid(self):
        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        assert (
            mask.geometry.masked_grid
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

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)),
            pixel_scales=(1.0, 1.0),
            sub_size=2,
        )

        assert (
            mask.geometry.masked_grid
            == np.array(
                [
                    [0.75, -0.75],
                    [0.75, -0.25],
                    [0.25, -0.75],
                    [0.25, -0.25],
                    [0.75, 0.25],
                    [0.75, 0.75],
                    [0.25, 0.25],
                    [0.25, 0.75],
                    [-0.25, -0.75],
                    [-0.25, -0.25],
                    [-0.75, -0.75],
                    [-0.75, -0.25],
                    [-0.25, 0.25],
                    [-0.25, 0.75],
                    [-0.75, 0.25],
                    [-0.75, 0.75],
                ]
            )
        ).all()

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )
        mask[1, 1] = True

        assert (
            mask.geometry.masked_grid
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

        mask = aa.Mask2D.manual(
            mask=np.array([[False, True], [True, False], [True, False]]),
            pixel_scales=(1.0, 1.0),
            sub_size=5,
            origin=(3.0, -2.0),
        )

        masked_grid_util = aa.util.grid.grid_1d_via_mask_from(
            mask=mask, pixel_scales=(1.0, 1.0), sub_size=5, origin=(3.0, -2.0)
        )

        assert (mask.geometry.masked_grid == masked_grid_util).all()

    def test__sub_border_1d_grid__compare_numerical_values(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, False, False, True, True, True, True],
                [True, True, True, True, False, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        assert (
            mask.geometry.border_grid_1d
            == np.array([[1.25, -2.25], [1.25, -1.25], [-0.25, 1.25]])
        ).all()

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=2)

        assert (
            mask.geometry.border_grid_1d
            == np.array(
                [
                    [1.25, -1.25],
                    [1.25, 0.25],
                    [1.25, 1.25],
                    [-0.25, -1.25],
                    [-0.25, 1.25],
                    [-1.25, -1.25],
                    [-1.25, 0.25],
                    [-1.25, 1.25],
                ]
            )
        ).all()


class TestScaledToPixel:
    def test__pixel_coordinates_2d_from__gives_same_result_as_geometry_util(self):

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(6, 7)),
            pixel_scales=(2.4, 1.8),
            origin=(1.0, 1.5),
        )

        pixel_coordinates_util = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.3, 1.2),
            shape_2d=(6, 7),
            pixel_scales=(2.4, 1.8),
            origins=(1.0, 1.5),
        )

        assert (
            mask.geometry.pixel_coordinates_2d_from(scaled_coordinates_2d=(2.3, 1.2))
            == pixel_coordinates_util
        )

    def test__scaled_coordinates_2d_from___gives_same_result_as_geometry_util(self,):

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(6, 7)),
            pixel_scales=(2.4, 1.8),
            origin=(1.0, 1.5),
        )

        pixel_coordinates_util = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(5, 4),
            shape_2d=(6, 7),
            pixel_scales=(2.4, 1.8),
            origins=(1.0, 1.5),
        )

        assert (
            mask.geometry.scaled_coordinates_2d_from(pixel_coordinates_2d=(5, 4))
            == pixel_coordinates_util
        )


class TestGridConversions:
    def test__grid_pixels_from_grid_scaled(self):
        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 4.0)
        )

        grid_scaled_1d = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels_util = aa.util.grid.grid_pixels_1d_from(
            grid_scaled_1d=grid_scaled_1d, shape_2d=(2, 2), pixel_scales=(2.0, 4.0)
        )
        grid_pixels = mask.geometry.grid_pixels_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled_1d
        )

        assert (grid_pixels == grid_pixels_util).all()
        assert (grid_pixels.in_1d == grid_pixels_util).all()

    def test__grid_pixel_centres_1d_from_grid_scaled_1d__same_as_grid_util(self):

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
        )

        grid_scaled_1d = np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5]])

        grid_pixels_util = aa.util.grid.grid_pixel_centres_1d_from(
            grid_scaled_1d=grid_scaled_1d, shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )

        grid_pixels = mask.geometry.grid_pixel_centres_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled_1d
        )

        assert (grid_pixels == grid_pixels_util).all()

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(7.0, 2.0)
        )

        grid_scaled_1d = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels_util = aa.util.grid.grid_pixel_centres_1d_from(
            grid_scaled_1d=grid_scaled_1d, shape_2d=(2, 2), pixel_scales=(7.0, 2.0)
        )

        grid_pixels = mask.geometry.grid_pixel_centres_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled_1d
        )

        assert (grid_pixels == grid_pixels_util).all()

    def test__grid_pixel_indexes_1d_from_grid_scaled_1d__same_as_grid_util(self):
        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
        )

        grid_scaled = np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])

        grid_pixel_indexes_util = aa.util.grid.grid_pixel_indexes_1d_from(
            grid_scaled_1d=grid_scaled, shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )

        grid_pixel_indexes = mask.geometry.grid_pixel_indexes_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )

        assert (grid_pixel_indexes == grid_pixel_indexes_util).all()

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 4.0)
        )

        grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels_util = aa.util.grid.grid_pixel_indexes_1d_from(
            grid_scaled_1d=grid_scaled, shape_2d=(2, 2), pixel_scales=(2.0, 4.0)
        )

        grid_pixels = mask.geometry.grid_pixel_indexes_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )

        assert (grid_pixels == grid_pixels_util).all()

    def test__grid_scaled_1d_from_grid_pixels_1d__same_as_grid_util(self):
        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
        )

        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_pixels_util = aa.util.grid.grid_scaled_1d_from(
            grid_pixels_1d=grid_pixels, shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )

        grid_pixels = mask.geometry.grid_scaled_from_grid_pixels_1d(
            grid_pixels_1d=grid_pixels
        )

        assert (grid_pixels == grid_pixels_util).all()

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
        )

        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_pixels_util = aa.util.grid.grid_scaled_1d_from(
            grid_pixels_1d=grid_pixels, shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )
        grid_pixels = mask.geometry.grid_scaled_from_grid_pixels_1d(
            grid_pixels_1d=grid_pixels
        )

        assert (grid_pixels == grid_pixels_util).all()

    def test__pixel_grid__grids_with_nonzero_centres__same_as_grid_util(self):
        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )

        grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels_util = aa.util.grid.grid_pixels_1d_from(
            grid_scaled_1d=grid_scaled,
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.geometry.grid_pixels_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = aa.util.grid.grid_pixel_indexes_1d_from(
            grid_scaled_1d=grid_scaled,
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.geometry.grid_pixel_indexes_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels_util = aa.util.grid.grid_pixel_centres_1d_from(
            grid_scaled_1d=grid_scaled,
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.geometry.grid_pixel_centres_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_scaled_util = aa.util.grid.grid_scaled_1d_from(
            grid_pixels_1d=grid_pixels,
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )

        grid_scaled = mask.geometry.grid_scaled_from_grid_pixels_1d(
            grid_pixels_1d=grid_pixels
        )

        assert (grid_scaled == grid_scaled_util).all()

        grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )

        grid_pixels_util = aa.util.grid.grid_pixels_1d_from(
            grid_scaled_1d=grid_scaled,
            shape_2d=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.geometry.grid_pixels_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = aa.util.grid.grid_pixel_indexes_1d_from(
            grid_scaled_1d=grid_scaled,
            shape_2d=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.geometry.grid_pixel_indexes_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = aa.util.grid.grid_pixel_centres_1d_from(
            grid_scaled_1d=grid_scaled,
            shape_2d=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.geometry.grid_pixel_centres_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_scaled_util = aa.util.grid.grid_scaled_1d_from(
            grid_pixels_1d=grid_pixels,
            shape_2d=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )

        grid_scaled = mask.geometry.grid_scaled_from_grid_pixels_1d(
            grid_pixels_1d=grid_pixels
        )

        assert (grid_scaled == grid_scaled_util).all()


class TestZoomCentreAndOffet:
    def test__odd_sized_false_mask__centre_is_0_0__pixels_from_centre_are_0_0(self):
        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry._zoom_centre == (1.0, 1.0)
        assert mask.geometry._zoom_offset_pixels == (0, 0)

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(5, 5)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry._zoom_centre == (2.0, 2.0)
        assert mask.geometry._zoom_offset_pixels == (0, 0)

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(3, 5)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry._zoom_centre == (1.0, 2.0)
        assert mask.geometry._zoom_offset_pixels == (0, 0)

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(5, 3)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry._zoom_centre == (2.0, 1.0)
        assert mask.geometry._zoom_offset_pixels == (0, 0)

    def test__even_sized_false_mask__centre_is_0_0__pixels_from_centre_are_0_0(self):
        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(4, 4)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry._zoom_centre == (1.5, 1.5)
        assert mask.geometry._zoom_offset_pixels == (0, 0)

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(6, 6)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry._zoom_centre == (2.5, 2.5)
        assert mask.geometry._zoom_offset_pixels == (0, 0)

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(4, 6)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry._zoom_centre == (1.5, 2.5)
        assert mask.geometry._zoom_offset_pixels == (0, 0)

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(6, 4)), pixel_scales=(1.0, 1.0)
        )
        assert mask.geometry._zoom_centre == (2.5, 1.5)
        assert mask.geometry._zoom_offset_pixels == (0, 0)

    def test__mask_is_single_false__extraction_centre_is_central_pixel(self):
        mask = aa.Mask2D.manual(
            mask=np.array(
                [[False, True, True], [True, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (0, 0)
        assert mask.geometry._zoom_offset_pixels == (-1, -1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, False], [True, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (0, 2)
        assert mask.geometry._zoom_offset_pixels == (-1, 1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [True, True, True], [False, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (2, 0)
        assert mask.geometry._zoom_offset_pixels == (1, -1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [True, True, True], [True, True, False]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (2, 2)
        assert mask.geometry._zoom_offset_pixels == (1, 1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, False, True], [True, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (0, 1)
        assert mask.geometry._zoom_offset_pixels == (-1, 0)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [False, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (1, 0)
        assert mask.geometry._zoom_offset_pixels == (0, -1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [True, True, False], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (1, 2)
        assert mask.geometry._zoom_offset_pixels == (0, 1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [True, True, True], [True, False, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (2, 1)
        assert mask.geometry._zoom_offset_pixels == (1, 0)

    def test__mask_is_x2_false__extraction_centre_is_central_pixel(self):
        mask = aa.Mask2D.manual(
            mask=np.array(
                [[False, True, True], [True, True, True], [True, True, False]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (1, 1)
        assert mask.geometry._zoom_offset_pixels == (0, 0)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[False, True, True], [True, True, True], [False, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (1, 0)
        assert mask.geometry._zoom_offset_pixels == (0, -1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[False, True, False], [True, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (0, 1)
        assert mask.geometry._zoom_offset_pixels == (-1, 0)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[False, False, True], [True, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.geometry._zoom_centre == (0, 0.5)
        assert mask.geometry._zoom_offset_pixels == (-1, -0.5)

    def test__rectangular_mask(self):
        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [False, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
        )

        assert mask.geometry._zoom_centre == (0, 0)
        assert mask.geometry._zoom_offset_pixels == (-1.0, -1.5)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, False],
                ]
            ),
            pixel_scales=(1.0, 1.0),
        )

        assert mask.geometry._zoom_centre == (2, 3)
        assert mask.geometry._zoom_offset_pixels == (1.0, 1.5)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                    [True, True, True, True, False],
                ]
            ),
            pixel_scales=(1.0, 1.0),
        )

        assert mask.geometry._zoom_centre == (2, 4)
        assert mask.geometry._zoom_offset_pixels == (1, 2)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, False],
                ]
            ),
            pixel_scales=(1.0, 1.0),
        )

        assert mask.geometry._zoom_centre == (2, 6)
        assert mask.geometry._zoom_offset_pixels == (1, 3)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, False],
                ]
            ),
            pixel_scales=(1.0, 1.0),
        )

        assert mask.geometry._zoom_centre == (4, 2)
        assert mask.geometry._zoom_offset_pixels == (2, 1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, False],
                ]
            ),
            pixel_scales=(1.0, 1.0),
        )

        assert mask.geometry._zoom_centre == (6, 2)
        assert mask.geometry._zoom_offset_pixels == (3, 1)
