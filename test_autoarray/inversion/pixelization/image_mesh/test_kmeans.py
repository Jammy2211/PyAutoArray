import numpy as np
import pytest

import autoarray as aa


def test__brightness__weight_map_from():
    adapt_data = np.array([0.0, 1.0, 0.0])

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=0.0, weight_power=0.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert (weight_map == np.ones(3)).all()

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=0.0, weight_power=1.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert (weight_map == np.array([0.0, 1.0, 0.0])).all()

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=1.0, weight_power=1.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert (weight_map == np.array([1.0, 2.0, 1.0])).all()

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=1.0, weight_power=2.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert (weight_map == np.array([1.0, 4.0, 1.0])).all()

    adapt_data = np.array([-1.0, 1.0, 3.0])

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=0.0, weight_power=1.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert (weight_map == np.array([0.0, 0.5, 1.0])).all()

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=0.0, weight_power=2.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert (weight_map == np.array([0.0, 0.25, 1.0])).all()

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=1.0, weight_power=1.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert (weight_map == np.array([3.0, 3.5, 4.0])).all()


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
    image_mesh = kmeans.image_plane_mesh_grid_from(grid=grid, adapt_data=weight_map)

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
    )
    image_mesh_weight_0 = kmeans.image_plane_mesh_grid_from(
        grid=grid, adapt_data=weight_map
    )

    kmeans = aa.image_mesh.KMeans(
        pixels=8,
        n_iter=1,
        max_iter=2,
        seed=1,
    )
    image_mesh_weight_1 = kmeans.image_plane_mesh_grid_from(
        grid=grid, adapt_data=weight_map
    )

    assert (image_mesh_weight_0 == image_mesh_weight_1).all()

    kmeans = aa.image_mesh.KMeans(
        pixels=8,
        n_iter=1,
        max_iter=2,
        seed=1,
    )
    image_mesh_weight_0 = kmeans.image_plane_mesh_grid_from(
        grid=grid, adapt_data=weight_map
    )

    kmeans = aa.image_mesh.KMeans(
        pixels=8,
        n_iter=1,
        max_iter=2,
        seed=2,
    )
    image_mesh_weight_1 = kmeans.image_plane_mesh_grid_from(
        grid=grid, adapt_data=weight_map
    )

    assert (image_mesh_weight_0 != image_mesh_weight_1).any()
