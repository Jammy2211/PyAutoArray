import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.pixelization.border_relocator import (
    sub_border_pixel_slim_indexes_from,
)


def test__sub_border_pixel_slim_indexes_from():
    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    sub_border_pixels = sub_border_pixel_slim_indexes_from(mask_2d=mask, sub_size=1)

    assert (sub_border_pixels == np.array([0])).all()

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

    sub_border_pixels = sub_border_pixel_slim_indexes_from(mask_2d=mask, sub_size=1)

    assert (sub_border_pixels == np.array([0, 1, 2])).all()

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

    sub_border_pixels = sub_border_pixel_slim_indexes_from(mask_2d=mask, sub_size=1)

    assert (sub_border_pixels == np.array([0, 1, 2, 3, 5, 6, 7, 8])).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    sub_border_pixels = sub_border_pixel_slim_indexes_from(mask_2d=mask, sub_size=2)

    assert (sub_border_pixels == np.array([3])).all()

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

    sub_border_pixels = sub_border_pixel_slim_indexes_from(mask_2d=mask, sub_size=2)

    assert (sub_border_pixels == np.array([0, 5, 9, 14, 23, 26, 31, 35])).all()

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

    sub_border_pixels = sub_border_pixel_slim_indexes_from(mask_2d=mask, sub_size=3)

    assert (sub_border_pixels == np.array([0, 11, 20, 33, 53, 60, 71, 80])).all()

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
            [True, True, True, True, True, True, True, True, True],
        ]
    )

    sub_border_pixels = sub_border_pixel_slim_indexes_from(mask_2d=mask, sub_size=2)

    assert (
        sub_border_pixels
        == np.array(
            [
                0,
                4,
                8,
                13,
                17,
                21,
                25,
                28,
                33,
                36,
                53,
                58,
                71,
                74,
                91,
                94,
                99,
                102,
                106,
                110,
                115,
                119,
                123,
                127,
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True, True],
            [True, False, True, True, True, True, True, False, True, True],
            [True, False, True, False, False, False, True, False, True, True],
            [True, False, True, False, True, False, True, False, True, True],
            [True, False, True, False, False, False, True, False, True, True],
            [True, False, True, True, True, True, True, False, True, True],
            [True, False, False, False, False, False, False, False, True, True],
            [True, True, True, True, True, True, True, True, True, True],
        ]
    )

    sub_border_pixels = sub_border_pixel_slim_indexes_from(mask_2d=mask, sub_size=2)

    assert (
        sub_border_pixels
        == np.array(
            [
                0,
                4,
                8,
                13,
                17,
                21,
                25,
                28,
                33,
                36,
                53,
                58,
                71,
                74,
                91,
                94,
                99,
                102,
                106,
                110,
                115,
                119,
                123,
                127,
            ]
        )
    ).all()


def test__sub_border_slim():
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

    mask = aa.Mask2D(mask=mask, pixel_scales=(2.0, 2.0))

    sub_border_flat_indexes_util = sub_border_pixel_slim_indexes_from(
        mask_2d=np.array(mask),
        sub_size=2,
    )

    border_relocator = aa.BorderRelocator(mask=mask, sub_size=2)

    assert border_relocator.sub_border_slim == pytest.approx(
        sub_border_flat_indexes_util, 1e-4
    )


def test__relocated_grid_from__inside_border_no_relocations():
    mask = aa.Mask2D.circular(
        shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1)
    )

    over_sampling = aa.OverSamplerUniform(mask=mask, sub_size=2)
    grid = over_sampling.over_sampled_grid
    grid[1, :] = [0.1, 0.1]

    border_relocator = aa.BorderRelocator(mask=mask, sub_size=2)

    relocated_grid = border_relocator.relocated_grid_from(grid=grid)

    assert (relocated_grid[1] == np.array([0.1, 0.1])).all()


def test__relocated_grid_from__outside_border_includes_relocations():
    mask = aa.Mask2D.circular(
        shape_native=(30, 30), radius=1.0, pixel_scales=(0.1, 0.1)
    )

    over_sampling = aa.OverSamplerUniform(mask=mask, sub_size=2)
    grid = over_sampling.over_sampled_grid
    grid[1, :] = [10.1, 0.1]

    border_relocator = aa.BorderRelocator(mask=mask, sub_size=2)

    relocated_grid = border_relocator.relocated_grid_from(grid=grid)

    assert relocated_grid[1] == pytest.approx([0.97783243, 0.00968151], 1e-4)


def test__relocated_grid_from__positive_origin_included_in_relocate():
    mask = aa.Mask2D.circular(
        shape_native=(60, 60),
        radius=1.0,
        pixel_scales=(0.1, 0.1),
        centre=(1.0, 1.0),
    )

    over_sampling = aa.OverSamplerUniform(mask=mask, sub_size=2)
    grid = over_sampling.over_sampled_grid
    grid[1, :] = [11.1, 1.0]

    border_relocator = aa.BorderRelocator(mask=mask, sub_size=2)

    relocated_grid = border_relocator.relocated_grid_from(grid=grid)

    assert relocated_grid[1] == pytest.approx([1.97783243, 1.0], 1e-4)
