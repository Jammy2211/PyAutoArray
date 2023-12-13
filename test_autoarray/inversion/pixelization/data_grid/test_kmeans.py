import numpy as np
import pytest

import autoarray as aa


def test__from_pixels_grid_and_weight_map():
    mask = aa.Mask2D(
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

    kmeans = aa.image_mesh.KMeans(pixels=8, n_iter=10, max_iter=20, seed=1)
    image_mesh = kmeans.image_mesh_from(grid=grid, weight_map=weight_map)

    assert (
        image_mesh
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

    mask = aa.Mask2D(
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

    kmeans = aa.image_mesh.KMeans(pixels=8, n_iter=10, max_iter=30, seed=1)
    image_mesh = kmeans.image_mesh_from(grid=grid, weight_map=weight_map)

    assert image_mesh[1] == pytest.approx(np.array([0.4166666, -0.0833333]), 1.0e-4)


def test__from_pixels_grid_and_weight_map__stochastic_true():
    mask = aa.Mask2D(
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

    kmeans = aa.image_mesh.KMeans(
        pixels=8,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=False,
    )
    image_mesh_weight_0 = kmeans.image_mesh_from(grid=grid, weight_map=weight_map)

    kmeans = aa.image_mesh.KMeans(
        pixels=8,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=False,
    )
    image_mesh_weight_1 = kmeans.image_mesh_from(grid=grid, weight_map=weight_map)

    assert (image_mesh_weight_0 == image_mesh_weight_1).all()

    kmeans = aa.image_mesh.KMeans(
        pixels=8,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=True,
    )
    image_mesh_weight_0 = kmeans.image_mesh_from(grid=grid, weight_map=weight_map)

    kmeans = aa.image_mesh.KMeans(
        pixels=8,
        n_iter=1,
        max_iter=2,
        seed=1,
        stochastic=True,
    )
    image_mesh_weight_1 = kmeans.image_mesh_from(grid=grid, weight_map=weight_map)

    assert (image_mesh_weight_0 != image_mesh_weight_1).any()
