import os
import numpy as np
import pytest

import autoarray as aa
from autoarray import exc
from autoarray.structures import grids

test_coordinates_dir = "{}/files/coordinates/".format(
    os.path.dirname(os.path.realpath(__file__))
)


def test_decorated_function__values_from_function_has_1_dimensions__returns_1d_result():
    # noinspection PyUnusedLocal
    @grids.interpolate
    def func(profile, grid, grid_radial_minimum=None):
        result = np.zeros(grid.shape[0])
        result[0] = 1
        return result

    grid = aa.Grid.from_mask(
        mask=aa.Mask.unmasked(shape_2d=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1)
    )

    values = func(None, grid)

    assert values.ndim == 1
    assert values.shape == (9,)
    assert (values == np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])).all()

    grid = aa.Grid.from_mask(
        mask=aa.Mask.unmasked(shape_2d=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1)
    )
    grid.interpolator = grids.GridInterpolate.from_mask_grid_and_interpolation_pixel_scales(
        grid.mask, grid, interpolation_pixel_scale=0.5
    )
    interp_values = func(None, grid)
    assert interp_values.ndim == 1
    assert interp_values.shape == (9,)
    assert (interp_values != np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])).any()


def test_decorated_function__values_from_function_has_2_dimensions__returns_2d_result():
    # noinspection PyUnusedLocal
    @grids.interpolate
    def func(profile, grid, grid_radial_minimum=None):
        result = np.zeros((grid.shape[0], 2))
        result[0, :] = 1
        return result

    grid = aa.Grid.from_mask(
        mask=aa.Mask.unmasked(shape_2d=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1)
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

    grid = aa.Grid.from_mask(
        mask=aa.Mask.unmasked(shape_2d=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1)
    )
    grid.interpolator = grids.GridInterpolate.from_mask_grid_and_interpolation_pixel_scales(
        grid.mask, grid, interpolation_pixel_scale=0.5
    )

    interp_values = func(None, grid)
    assert interp_values.ndim == 2
    assert interp_values.shape == (9, 2)
    assert (
        interp_values
        != np.array(
            np.array(
                [[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            )
        )
    ).any()


def test__20x20_deflection_angles_no_central_pixels__interpolated_accurately():
    @grids.interpolate
    def grid_radii_from_grid(profile, grid, grid_radial_minimum=None):
        grid_radii = np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))
        return np.stack((grid_radii, grid_radii), axis=-1)

    mask = aa.Mask.circular_annular(
        shape_2d=(20, 20),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
        inner_radius=3.0,
        outer_radius=8.0,
    )

    grid = aa.Grid.from_mask(mask=mask)

    true_grid_radii = grid_radii_from_grid(profile=None, grid=grid)

    interpolator = grids.GridInterpolate.from_mask_grid_and_interpolation_pixel_scales(
        mask=mask, grid=grid, interpolation_pixel_scale=1.0
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


def test__move_centre_of_galaxy__interpolated_accurately():
    @grids.interpolate
    def grid_radii_from_grid(profile, grid, grid_radial_minimum=None):
        grid_radii = np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))
        return np.stack((grid_radii, grid_radii), axis=-1)

    mask = aa.Mask.circular_annular(
        shape_2d=(24, 24),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
        inner_radius=3.0,
        outer_radius=8.0,
        centre=(3.0, 3.0),
    )

    grid = aa.Grid.from_mask(mask=mask)

    true_grid_radii = grid_radii_from_grid(profile=None, grid=grid)

    interpolator = grids.GridInterpolate.from_mask_grid_and_interpolation_pixel_scales(
        mask=mask, grid=grid, interpolation_pixel_scale=1.0
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


def test__different_interpolation_pixel_scales_still_works():
    @grids.interpolate
    def grid_radii_from_grid(profile, grid, grid_radial_minimum=None):
        grid_radii = np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))
        return np.stack((grid_radii, grid_radii), axis=-1)

    mask = aa.Mask.circular_annular(
        shape_2d=(28, 28),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
        inner_radius=3.0,
        outer_radius=8.0,
        centre=(3.0, 3.0),
    )

    grid = aa.Grid.from_mask(mask=mask)

    true_grid_radii = grid_radii_from_grid(profile=None, grid=grid)

    interpolator = grids.GridInterpolate.from_mask_grid_and_interpolation_pixel_scales(
        mask=mask, grid=grid, interpolation_pixel_scale=0.2
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

    interpolator = grids.GridInterpolate.from_mask_grid_and_interpolation_pixel_scales(
        mask=mask, grid=grid, interpolation_pixel_scale=0.5
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

    interpolator = grids.GridInterpolate.from_mask_grid_and_interpolation_pixel_scales(
        mask=mask, grid=grid, interpolation_pixel_scale=1.1
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
