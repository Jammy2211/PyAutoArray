import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.pixelization.image_mesh import overlay


def test__overlay_for_mask_from():
    mask_2d = aa.Mask2D(
        mask=np.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    overlaid_centres = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    total_masked_pixels = overlay.total_pixels_2d_from(
        mask_2d=mask_2d,
        overlaid_centres=overlaid_centres,
    )

    overlay_for_mask = overlay.overlay_for_mask_from(
        total_pixels=total_masked_pixels,
        mask=mask_2d,
        overlaid_centres=overlaid_centres,
    )

    assert (overlay_for_mask == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    overlaid_centres = np.array(
        [[0, 0], [0, 1], [2, 2], [1, 1], [0, 2], [2, 0], [0, 2]]
    )

    total_masked_pixels = overlay.total_pixels_2d_from(
        mask_2d=mask_2d,
        overlaid_centres=overlaid_centres,
    )

    overlay_for_mask = overlay.overlay_for_mask_from(
        total_pixels=total_masked_pixels,
        mask=mask_2d,
        overlaid_centres=overlaid_centres,
    )

    assert (overlay_for_mask == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    mask_2d = aa.Mask2D(
        mask=np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    overlaid_centres = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    total_masked_pixels = overlay.total_pixels_2d_from(
        mask_2d=mask_2d,
        overlaid_centres=overlaid_centres,
    )

    overlay_for_mask = overlay.overlay_for_mask_from(
        total_pixels=total_masked_pixels,
        mask=mask_2d,
        overlaid_centres=overlaid_centres,
    )

    assert (overlay_for_mask == np.array([1, 3, 4, 5, 7])).all()

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

    overlaid_centres = np.array(
        [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 3], [2, 2]]
    )

    total_masked_pixels = overlay.total_pixels_2d_from(
        mask_2d=mask_2d,
        overlaid_centres=overlaid_centres,
    )

    overlay_for_mask = overlay.overlay_for_mask_from(
        total_pixels=total_masked_pixels,
        mask=mask_2d,
        overlaid_centres=overlaid_centres,
    )

    assert (overlay_for_mask == np.array([2, 3, 4, 5, 7])).all()


def test__mask_for_overlay_from():
    mask_2d = aa.Mask2D(
        mask=np.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    overlaid_centres = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    mask_for_overlay = overlay.mask_for_overlay_from(
        mask=mask_2d,
        overlaid_centres=overlaid_centres,
        total_pixels=9,
    )

    assert (mask_for_overlay == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    overlaid_centres = np.array(
        [[0, 0], [0, 1], [2, 2], [1, 1], [0, 2], [2, 0], [0, 2]]
    )

    mask_for_overlay = overlay.mask_for_overlay_from(
        mask=mask_2d,
        overlaid_centres=overlaid_centres,
        total_pixels=9,
    )

    assert (mask_for_overlay == np.array([0, 1, 2, 3, 4, 5, 6])).all()

    mask_2d = aa.Mask2D(
        mask=np.array(
            [[False, False, True], [False, False, False], [True, False, False]]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    overlaid_centres = np.array([[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1]])

    mask_for_overlay = overlay.mask_for_overlay_from(
        mask=mask_2d,
        overlaid_centres=overlaid_centres,
        total_pixels=4,
    )

    assert (mask_for_overlay == np.array([0, 1, 2, 2, 2, 2])).all()

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

    overlaid_centres = np.array(
        [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2], [1, 1], [2, 3], [0, 2]]
    )

    mask_for_overlay = overlay.mask_for_overlay_from(
        mask=mask_2d,
        overlaid_centres=overlaid_centres,
        total_pixels=5,
    )

    assert (mask_for_overlay == np.array([0, 0, 0, 1, 2, 3, 4, 4])).all()


def test__overlay_via_unmasked_overlaid_from():
    unmasked_overlay_grid = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    overlay_for_mask = np.array([0, 1, 2, 3])
    pix_grid = overlay.overlay_via_unmasked_overlaid_from(
        unmasked_overlay_grid=unmasked_overlay_grid,
        overlay_for_mask=overlay_for_mask,
    )

    assert (
        pix_grid == np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    ).all()

    unmasked_overlay_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
    overlay_for_mask = np.array([1, 0, 3, 2])
    pix_grid = overlay.overlay_via_unmasked_overlaid_from(
        unmasked_overlay_grid=unmasked_overlay_grid,
        overlay_for_mask=overlay_for_mask,
    )

    assert (
        pix_grid == np.array([[4.0, 5.0], [0.0, 0.0], [8.0, 7.0], [2.0, 2.0]])
    ).all()

    unmasked_overlay_grid = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    overlay_for_mask = np.array([1, 2])
    pix_grid = overlay.overlay_via_unmasked_overlaid_from(
        unmasked_overlay_grid=unmasked_overlay_grid,
        overlay_for_mask=overlay_for_mask,
    )

    assert (pix_grid == np.array([[1.0, 1.0], [2.0, 2.0]])).all()

    unmasked_overlay_grid = np.array([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0], [8.0, 7.0]])
    overlay_for_mask = np.array([2, 2, 3])
    pix_grid = overlay.overlay_via_unmasked_overlaid_from(
        unmasked_overlay_grid=unmasked_overlay_grid,
        overlay_for_mask=overlay_for_mask,
    )

    assert (pix_grid == np.array([[2.0, 2.0], [2.0, 2.0], [8.0, 7.0]])).all()

    unmasked_overlay_grid = np.array(
        [
            [0.0, 0.0],
            [4.0, 5.0],
            [2.0, 2.0],
            [8.0, 7.0],
            [11.0, 11.0],
            [-20.0, -15.0],
        ]
    )
    overlay_for_mask = np.array([1, 0, 5, 2])
    pix_grid = overlay.overlay_via_unmasked_overlaid_from(
        unmasked_overlay_grid=unmasked_overlay_grid,
        overlay_for_mask=overlay_for_mask,
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

    grid_overlay = overlay.via_magnification_from(shape_overlay=(10, 10), grid=grid)

    unmasked_overlay_grid_util = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(10, 10), pixel_scales=(0.15, 0.15), sub_size=1, origin=(0.0, 0.0)
    )

    overlaid_centres = aa.util.geometry.grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=unmasked_overlay_grid_util,
        shape_native=grid.mask.shape,
        pixel_scales=grid.pixel_scales,
    ).astype("int")

    total_pixels = overlay.total_pixels_2d_from(
        mask_2d=mask,
        overlaid_centres=overlaid_centres,
    )

    overlay_for_mask_2d_util = overlay.overlay_for_mask_from(
        total_pixels=total_pixels,
        mask=mask,
        overlaid_centres=overlaid_centres,
    ).astype("int")

    grid_overlay_util = overlay.overlay_via_unmasked_overlaid_from(
        unmasked_overlay_grid=unmasked_overlay_grid_util,
        overlay_for_mask=overlay_for_mask_2d_util,
    )

    assert (grid_overlay == grid_overlay_util).all()


def test__via_magnification_from__grid_overlay_overlaps_mask_perfectly():
    mask = aa.Mask2D(
        mask=np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    grid_overlay = overlay.via_magnification_from(shape_overlay=(3, 3), grid=grid)

    assert (
        grid_overlay
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

    grid_overlay = overlay.via_magnification_from(shape_overlay=(4, 3), grid=grid)
    assert (
        grid_overlay
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

    grid_overlay = overlay.via_magnification_from(shape_overlay=(3, 3), grid=grid)

    assert (
        grid_overlay
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

    grid_overlay = overlay.via_magnification_from(shape_overlay=(3, 3), grid=grid)
    assert (
        grid_overlay
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

    grid_overlay = overlay.via_magnification_from(shape_overlay=(4, 3), grid=grid)
    assert (
        grid_overlay
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

    grid_overlay = overlay.via_magnification_from(shape_overlay=(3, 3), grid=grid)
    assert (
        grid_overlay
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
