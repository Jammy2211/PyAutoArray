from os import path
import numpy as np
import pytest

from autoconf import conf
import autoarray as aa
from autoarray import exc


def test__sub_border_flat_indexes():
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

    mask = aa.Mask2D(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

    sub_border_flat_indexes_util = aa.util.mask_2d.sub_border_pixel_slim_indexes_from(
        mask_2d=np.array(mask),
        sub_size=2,
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    assert grid_2d.mask.derive_indexes.sub_border_slim == pytest.approx(
        sub_border_flat_indexes_util, 1e-4
    )


def test__sub_border_grid():
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

    mask = aa.Mask2D(mask=mask, pixel_scales=(2.0, 2.0), sub_size=2)

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    assert (
        grid_2d.sub_border_grid
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


def test__relocated_grid_from__inside_border_no_relocations():
    mask = aa.Mask2D.circular(
        shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1)
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    grid_to_relocate = aa.Grid2D.no_mask(
        values=np.array([[[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]]]),
        pixel_scales=mask.pixel_scales,
    )

    relocated_grid = grid_2d.relocated_grid_from(grid=grid_to_relocate)

    assert (relocated_grid == np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]])).all()

    mask = aa.Mask2D.circular(
        shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=2
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    grid_to_relocate = aa.Grid2D.no_mask(
        values=np.array([[[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]]]),
        pixel_scales=mask.pixel_scales,
    )

    relocated_grid = grid_2d.relocated_grid_from(grid=grid_to_relocate)

    assert (relocated_grid == np.array([[0.1, 0.1], [0.3, 0.3], [-0.1, -0.2]])).all()


def test__relocated_grid_from__outside_border_includes_relocations():
    mask = aa.Mask2D.circular(
        shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1)
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    grid_to_relocate = aa.Grid2D.no_mask(
        values=np.array([[[10.1, 0.0], [0.0, 10.1], [-10.1, -10.1]]]),
        pixel_scales=mask.pixel_scales,
    )

    relocated_grid = grid_2d.relocated_grid_from(grid=grid_to_relocate)

    assert relocated_grid == pytest.approx(
        np.array([[0.95, 0.0], [0.0, 0.95], [-0.7017, -0.7017]]), 0.1
    )

    mask = aa.Mask2D.circular(
        shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1), sub_size=2
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    grid_to_relocate = aa.Grid2D.no_mask(
        values=np.array([[[10.1, 0.0], [0.0, 10.1], [-10.1, -10.1]]]),
        pixel_scales=mask.pixel_scales,
    )

    relocated_grid = grid_2d.relocated_grid_from(grid=grid_to_relocate)

    assert relocated_grid == pytest.approx(
        np.array([[0.9778, 0.0], [0.0, 0.97788], [-0.7267, -0.7267]]), 0.1
    )


def test__relocated_grid_from__positive_origin_included_in_relocate():
    mask = aa.Mask2D.circular(
        shape_native=(60, 60),
        radius=1.0,
        pixel_scales=(0.1, 0.1),
        centre=(1.0, 1.0),
        sub_size=1,
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    grid_to_relocate = aa.Grid2D.no_mask(
        values=np.array([[[11.1, 1.0], [1.0, 11.1], [-11.1, -11.1]]]),
        pixel_scales=mask.pixel_scales,
    )

    relocated_grid = grid_2d.relocated_grid_from(grid=grid_to_relocate)

    assert relocated_grid == pytest.approx(
        np.array(
            [[2.0, 1.0], [1.0, 2.0], [1.0 - np.sqrt(2) / 2, 1.0 - np.sqrt(2) / 2]]
        ),
        0.1,
    )

    mask = aa.Mask2D.circular(
        shape_native=(60, 60),
        radius=1.0,
        pixel_scales=(0.1, 0.1),
        centre=(1.0, 1.0),
        sub_size=2,
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    grid_to_relocate = aa.Grid2D.no_mask(
        values=np.array([[[11.1, 1.0], [1.0, 11.1], [-11.1, -11.1]]]),
        pixel_scales=mask.pixel_scales,
    )

    relocated_grid = grid_2d.relocated_grid_from(grid=grid_to_relocate)

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
