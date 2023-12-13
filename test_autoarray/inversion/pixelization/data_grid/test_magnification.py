import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.pixelization.data_grid import magnification as data_grid


def test__unmasked_sparse_for_sparse_from():
    mask_2d = aa.Mask2D(
        mask=np.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    unmasked_sparse_grid_pixel_centres = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    total_masked_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
        mask_2d=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    )

    unmasked_sparse_for_sparse = data_grid.unmasked_sparse_for_sparse_from(
        total_sparse_pixels=total_masked_pixels,
        mask=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    )

    assert (unmasked_sparse_for_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    unmasked_sparse_grid_pixel_centres = np.array(
        [[0, 0], [0, 1], [2, 2], [1, 1], [0, 2], [2, 0], [0, 2]]
    )

    total_masked_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
        mask_2d=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    )

    unmasked_sparse_for_sparse = data_grid.unmasked_sparse_for_sparse_from(
        total_sparse_pixels=total_masked_pixels,
        mask=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    )

    assert (unmasked_sparse_for_sparse == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    mask_2d = aa.Mask2D(
        mask=np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    unmasked_sparse_grid_pixel_centres = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    total_masked_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
        mask_2d=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    )

    unmasked_sparse_for_sparse = data_grid.unmasked_sparse_for_sparse_from(
        total_sparse_pixels=total_masked_pixels,
        mask=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    )

    assert (unmasked_sparse_for_sparse == np.array([1, 3, 4, 5, 7])).all()

    mask_2d = aa.Mask2D(
        mask=np.array(
            [
                [True, True, False, True],
                [False, False, False, False],
                [True, True, False, True],
            ]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    unmasked_sparse_grid_pixel_centres = np.array(
        [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 3], [2, 2]]
    )

    total_masked_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
        mask_2d=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    )

    unmasked_sparse_for_sparse = data_grid.unmasked_sparse_for_sparse_from(
        total_sparse_pixels=total_masked_pixels,
        mask=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    )

    assert (unmasked_sparse_for_sparse == np.array([2, 3, 4, 5, 7])).all()


def test__sparse_for_unmasked_sparse_from():
    mask_2d = aa.Mask2D(
        mask=np.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    unmasked_sparse_grid_pixel_centres = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    sparse_for_unmasked_sparse = data_grid.sparse_for_unmasked_sparse_from(
        mask=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        total_sparse_pixels=9,
    )

    assert (sparse_for_unmasked_sparse == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    unmasked_sparse_grid_pixel_centres = np.array(
        [[0, 0], [0, 1], [2, 2], [1, 1], [0, 2], [2, 0], [0, 2]]
    )

    sparse_for_unmasked_sparse = data_grid.sparse_for_unmasked_sparse_from(
        mask=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        total_sparse_pixels=9,
    )

    assert (sparse_for_unmasked_sparse == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    mask_2d = aa.Mask2D(
        mask=np.array(
            [[False, False, True], [False, False, False], [True, False, False]]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    unmasked_sparse_grid_pixel_centres = np.array(
        [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1]]
    )

    sparse_for_unmasked_sparse = data_grid.sparse_for_unmasked_sparse_from(
        mask=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        total_sparse_pixels=4,
    )

    assert (sparse_for_unmasked_sparse == np.array([0, 1, 2, 2, 2, 2])).all()

    mask_2d = aa.Mask2D(
        mask=np.array(
            [
                [True, True, False, True],
                [False, False, False, False],
                [True, True, False, True],
            ]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    unmasked_sparse_grid_pixel_centres = np.array(
        [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 3], [0, 2]]
    )

    sparse_for_unmasked_sparse = data_grid.sparse_for_unmasked_sparse_from(
        mask=mask_2d,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        total_sparse_pixels=5,
    )

    assert (sparse_for_unmasked_sparse == np.array([0, 0, 0, 1, 2, 3, 4, 4])).all()


def test__sparse_slim_index_for_mask_slim_index_from():
    regular_to_unmasked_sparse = np.array([0, 1, 2, 3, 4])
    sparse_for_unmasked_sparse = np.array([0, 1, 2, 3, 4])
    sparse_index_for_slim_index = (
        data_grid.sparse_slim_index_for_mask_slim_index_from(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            sparse_for_unmasked_sparse=sparse_for_unmasked_sparse,
        )
    )

    assert (sparse_index_for_slim_index == np.array([0, 1, 2, 3, 4])).all()

    regular_to_unmasked_sparse = np.array([0, 1, 2, 3, 4])
    sparse_for_unmasked_sparse = np.array([0, 1, 5, 7, 18])
    sparse_index_for_slim_index = (
        data_grid.sparse_slim_index_for_mask_slim_index_from(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            sparse_for_unmasked_sparse=sparse_for_unmasked_sparse,
        )
    )

    assert (sparse_index_for_slim_index == np.array([0, 1, 5, 7, 18])).all()

    regular_to_unmasked_sparse = np.array([1, 1, 1, 1, 2])
    sparse_for_unmasked_sparse = np.array([0, 10, 15, 3, 4])
    sparse_index_for_slim_index = (
        data_grid.sparse_slim_index_for_mask_slim_index_from(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            sparse_for_unmasked_sparse=sparse_for_unmasked_sparse,
        )
    )

    assert (sparse_index_for_slim_index == np.array([10, 10, 10, 10, 15])).all()

    regular_to_unmasked_sparse = np.array([5, 6, 7, 8, 9])
    sparse_for_unmasked_sparse = np.array([0, 1, 2, 3, 4, 19, 18, 17, 16, 15])
    sparse_index_for_slim_index = (
        data_grid.sparse_slim_index_for_mask_slim_index_from(
            regular_to_unmasked_sparse=regular_to_unmasked_sparse,
            sparse_for_unmasked_sparse=sparse_for_unmasked_sparse,
        )
    )

    assert (sparse_index_for_slim_index == np.array([19, 18, 17, 16, 15])).all()


def test__sparse_grid_via_unmasked_from():
    unmasked_sparse_grid = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    unmasked_sparse_for_sparse = np.array([0, 1, 2, 3])
    pix_grid = data_grid.sparse_grid_via_unmasked_from(
        unmasked_sparse_grid=unmasked_sparse_grid,
        unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
    )

    assert (
        pix_grid == np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    ).all()

    unmasked_sparse_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
    unmasked_sparse_for_sparse = np.array([1, 0, 3, 2])
    pix_grid = data_grid.sparse_grid_via_unmasked_from(
        unmasked_sparse_grid=unmasked_sparse_grid,
        unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
    )

    assert (
        pix_grid == np.array([[4.0, 5.0], [0.0, 0.0], [8.0, 7.0], [2.0, 2.0]])
    ).all()

    unmasked_sparse_grid = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    unmasked_sparse_for_sparse = np.array([1, 2])
    pix_grid = data_grid.sparse_grid_via_unmasked_from(
        unmasked_sparse_grid=unmasked_sparse_grid,
        unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
    )

    assert (pix_grid == np.array([[1.0, 1.0], [2.0, 2.0]])).all()

    unmasked_sparse_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
    unmasked_sparse_for_sparse = np.array([2, 2, 3])
    pix_grid = data_grid.sparse_grid_via_unmasked_from(
        unmasked_sparse_grid=unmasked_sparse_grid,
        unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
    )

    assert (pix_grid == np.array([[2.0, 2.0], [2.0, 2.0], [8.0, 7.0]])).all()

    unmasked_sparse_grid = np.array(
        [
            [0.0, 0.0],
            [4.0, 5.0],
            [2.0, 2.0],
            [8.0, 7.0],
            [11.0, 11.0],
            [-20.0, -15.0],
        ]
    )
    unmasked_sparse_for_sparse = np.array([1, 0, 5, 2])
    pix_grid = data_grid.sparse_grid_via_unmasked_from(
        unmasked_sparse_grid=unmasked_sparse_grid,
        unmasked_sparse_for_sparse=unmasked_sparse_for_sparse,
    )

    assert (
        pix_grid == np.array([[4.0, 5.0], [0.0, 0.0], [-20.0, -15.0], [2.0, 2.0]])
    ).all()


def test__via_magnification_from__simple():
    
    mask = aa.Mask2D(
        mask=np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        ),
        pixel_scales=(0.5, 0.5),
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    sparse_grid = data_grid.via_magnification_from(
        unmasked_sparse_shape=(10, 10), grid=grid
    )

    unmasked_sparse_grid_util = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(10, 10), pixel_scales=(0.15, 0.15), sub_size=1, origin=(0.0, 0.0)
    )

    unmasked_sparse_grid_pixel_centres = (
        aa.util.geometry.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=unmasked_sparse_grid_util,
            shape_native=grid.mask.shape,
            pixel_scales=grid.pixel_scales,
        ).astype("int")
    )

    total_sparse_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
        mask_2d=mask,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    )

    unmasked_sparse_for_sparse_2d_util = data_grid.unmasked_sparse_for_sparse_from(
        total_sparse_pixels=total_sparse_pixels,
        mask=mask,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    ).astype("int")

    sparse_grid_util = data_grid.sparse_grid_via_unmasked_from(
        unmasked_sparse_grid=unmasked_sparse_grid_util,
        unmasked_sparse_for_sparse=unmasked_sparse_for_sparse_2d_util,
    )

    assert (sparse_grid == sparse_grid_util).all()


def test__via_magnification_from__sparse_grid_overlaps_mask_perfectly():
    mask = aa.Mask2D(
        mask=np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    sparse_grid = data_grid.via_magnification_from(
        unmasked_sparse_shape=(3, 3), grid=grid
    )

    assert (
        sparse_grid
        == np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    ).all()

    mask = aa.Mask2D(
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

    grid = aa.Grid2D.from_mask(mask=mask)

    sparse_grid = data_grid.via_magnification_from(
        unmasked_sparse_shape=(4, 3), grid=grid
    )
    assert (
        sparse_grid
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


def test__via_magnification_from__mask_with_offset_centre():
    mask = aa.Mask2D(
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

    grid = aa.Grid2D.from_mask(mask=mask)

    # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
    # the central (3x3) pixels only.

    sparse_grid = data_grid.via_magnification_from(
        unmasked_sparse_shape=(3, 3), grid=grid
    )

    assert (
        sparse_grid
        == np.array([[2.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
    ).all()

    mask = aa.Mask2D(
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

    grid = aa.Grid2D.from_mask(mask=mask)

    # Without a change in origin, only the central 3 pixels are paired as the unmasked sparse grid overlaps
    # the central (3x3) pixels only.

    sparse_grid = data_grid.via_magnification_from(
        unmasked_sparse_shape=(3, 3), grid=grid
    )
    assert (
        sparse_grid
        == np.array([[2.0, 2.0], [0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [-2.0, 2.0]])
    ).all()


def test__via_magnification_from__sets_up_with_correct_shape_and_pixel_scales(
    mask_2d_7x7,
):
    mask = aa.Mask2D(
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

    grid = aa.Grid2D.from_mask(mask=mask)

    sparse_grid = data_grid.via_magnification_from(
        unmasked_sparse_shape=(4, 3), grid=grid
    )
    assert (
        sparse_grid
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


def test__via_magnification_from__offset_mask__origin_shift_corrects():
    mask = aa.Mask2D(
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

    grid = aa.Grid2D.from_mask(mask=mask)

    sparse_grid = data_grid.via_magnification_from(
        unmasked_sparse_shape=(3, 3), grid=grid
    )
    assert (
        sparse_grid
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