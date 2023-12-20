import numpy as np
import pytest

import autoarray as aa


# def test__weight_map_from():
#     adapt_data = np.array([-1.0, 1.0, 3.0])
#
#     pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=0.0, weight_power=1.0)
#
#     weight_map = pixelization.weight_map_from(adapt_data=adapt_data)
#
#     assert (weight_map == np.array([0.0, 0.5, 1.0])).all()
#
#     pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=0.0, weight_power=2.0)
#
#     weight_map = pixelization.weight_map_from(adapt_data=adapt_data)
#
#     assert (weight_map == np.array([0.0, 0.25, 1.0])).all()
#
#     pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=1.0, weight_power=1.0)
#
#     weight_map = pixelization.weight_map_from(adapt_data=adapt_data)
#
#     assert (weight_map == np.array([3.0, 3.5, 4.0])).all()


def test__image_plane_mesh_grid_from():
    mask = aa.Mask2D.circular(
        shape_native=(4, 4),
        radius=2.0,
        pixel_scales=1.0,
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    adapt_data = aa.Array2D.ones(
        shape_native=mask.shape_native,
        pixel_scales=1.0,
    )

    kmeans = aa.image_mesh.Hilbert(pixels=8)
    image_mesh = kmeans.image_plane_mesh_grid_from(grid=grid, adapt_data=adapt_data)

    assert image_mesh[0, :] == pytest.approx(
        [-1.02590674, -1.70984456],
        1.0e-4,
    )
