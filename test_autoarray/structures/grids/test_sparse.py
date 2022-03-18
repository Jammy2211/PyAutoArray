from os import path
import numpy as np
import pytest

import autoarray as aa

test_grid_dir = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def test__unmasked_2d__properties_consistent_with_util():

    mask = aa.Mask2D.manual(
        mask=np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        ),
        pixel_scales=(0.5, 0.5),
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        unmasked_sparse_shape=(10, 10), grid=grid
    )

    unmasked_sparse_grid_util = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(10, 10), pixel_scales=(0.15, 0.15), sub_size=1, origin=(0.0, 0.0)
    )

    unmasked_sparse_grid_pixel_centres = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=unmasked_sparse_grid_util,
        shape_native=grid.mask.shape,
        pixel_scales=grid.pixel_scales,
    ).astype(
        "int"
    )

    total_sparse_pixels = aa.util.mask_2d.total_sparse_pixels_2d_from(
        mask_2d=mask,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    )

    regular_to_unmasked_sparse_2d_util = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
        grid_scaled_2d_slim=grid,
        shape_native=(10, 10),
        pixel_scales=(0.15, 0.15),
        origin=(0.0, 0.0),
    ).astype(
        "int"
    )

    unmasked_sparse_for_sparse_2d_util = aa.util.sparse.unmasked_sparse_for_sparse_from(
        total_sparse_pixels=total_sparse_pixels,
        mask=mask,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
    ).astype("int")

    sparse_for_unmasked_sparse_2d_util = aa.util.sparse.sparse_for_unmasked_sparse_from(
        mask=mask,
        unmasked_sparse_grid_pixel_centres=unmasked_sparse_grid_pixel_centres,
        total_sparse_pixels=total_sparse_pixels,
    ).astype("int")

    sparse_index_for_slim_index_util = aa.util.sparse.sparse_slim_index_for_mask_slim_index_from(
        regular_to_unmasked_sparse=regular_to_unmasked_sparse_2d_util,
        sparse_for_unmasked_sparse=sparse_for_unmasked_sparse_2d_util,
    )

    sparse_grid_util = aa.util.sparse.sparse_grid_via_unmasked_from(
        unmasked_sparse_grid=unmasked_sparse_grid_util,
        unmasked_sparse_for_sparse=unmasked_sparse_for_sparse_2d_util,
    )

    assert (
        sparse_grid.sparse_index_for_slim_index == sparse_index_for_slim_index_util
    ).all()
    assert (sparse_grid == sparse_grid_util).all()


def test__unmasked_2d__sparse_grid_overlaps_mask_perfectly__masked_pixels_in_masked_sparse_grid():
    mask = aa.Mask2D.manual(
        mask=np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        ),
        pixel_scales=(1.0, 1.0),
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        unmasked_sparse_shape=(3, 3), grid=grid
    )

    assert (sparse_grid.sparse_index_for_slim_index == np.array([0, 1, 2, 3, 4])).all()
    assert (
        sparse_grid
        == np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    ).all()

    mask = aa.Mask2D.manual(
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

    sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
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
    assert (
        sparse_grid.sparse_index_for_slim_index == np.array([0, 1, 2, 3, 4, 5, 6, 7])
    ).all()


def test__unmasked_2d__mask_with_offset_centre__origin_of_sparse_grid_moves_to_give_same_pairings():
    mask = aa.Mask2D.manual(
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

    sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        unmasked_sparse_shape=(3, 3), grid=grid
    )

    assert (
        sparse_grid
        == np.array([[2.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
    ).all()
    assert (sparse_grid.sparse_index_for_slim_index == np.array([0, 1, 2, 3, 4])).all()

    mask = aa.Mask2D.manual(
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

    sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
        unmasked_sparse_shape=(3, 3), grid=grid
    )
    assert (
        sparse_grid
        == np.array([[2.0, 2.0], [0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [-2.0, 2.0]])
    ).all()
    assert (sparse_grid.sparse_index_for_slim_index == np.array([0, 1, 2, 3, 4])).all()


def test__unmasked_2d__from_grid_and_unmasked_shape__sets_up_with_correct_shape_and_pixel_scales(
    mask_2d_7x7
):

    mask = aa.Mask2D.manual(
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

    sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
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
    assert (
        sparse_grid.sparse_index_for_slim_index == np.array([0, 1, 2, 3, 4, 5, 6, 7])
    ).all()


def test__unmasked_2d__from_grid_and_shape__offset_mask__origin_shift_corrects():

    mask = aa.Mask2D.manual(
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

    sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
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
    assert (
        sparse_grid.sparse_index_for_slim_index == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    ).all()


def test__via_kmeans__weight_map_all_ones__kmeans_grid_is_grid_overlapping_image():

    mask = aa.Mask2D.manual(
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

    grid = aa.Grid2D.from_mask(mask=mask)

    weight_map = np.ones(mask.pixels_in_mask)

    sparse_grid = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
        total_pixels=8, grid=grid, weight_map=weight_map, n_iter=10, max_iter=20, seed=1
    )

    assert (
        sparse_grid
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
        sparse_grid.sparse_index_for_slim_index
        == np.array([1, 1, 2, 2, 1, 1, 3, 3, 5, 4, 0, 7, 5, 4, 6, 6])
    ).all()


def test__weight_map_changed_from_above():

    mask = aa.Mask2D.manual(
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

    grid = aa.Grid2D.from_mask(mask=mask)

    weight_map = np.ones(mask.pixels_in_mask)
    weight_map[0:15] = 0.00000001

    sparse_grid = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
        total_pixels=8, grid=grid, weight_map=weight_map, n_iter=10, max_iter=30, seed=1
    )

    assert sparse_grid[1] == pytest.approx(np.array([0.4166666, -0.0833333]), 1.0e-4)

    assert (
        sparse_grid.sparse_index_for_slim_index
        == np.array([5, 1, 0, 0, 5, 1, 1, 4, 3, 6, 7, 4, 3, 6, 2, 2])
    ).all()


def test__stochastic_true__every_grid_different():

    mask = aa.Mask2D.manual(
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

    grid = aa.Grid2D.from_mask(mask=mask)

    weight_map = np.ones(mask.pixels_in_mask)

    sparse_grid_weight_0 = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=False,
    )

    sparse_grid_weight_1 = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=False,
    )

    assert (sparse_grid_weight_0 == sparse_grid_weight_1).all()

    sparse_grid_weight_0 = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=True,
    )

    sparse_grid_weight_1 = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
        total_pixels=8,
        grid=grid,
        weight_map=weight_map,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=True,
    )

    assert (sparse_grid_weight_0 != sparse_grid_weight_1).any()
