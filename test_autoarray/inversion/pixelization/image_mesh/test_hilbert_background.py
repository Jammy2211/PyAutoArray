import pytest

import autoarray as aa


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

    kmeans = aa.image_mesh.HilbertBackground(pixels=10, pixels_background=5)
    image_mesh = kmeans.image_plane_mesh_grid_from(grid=grid, adapt_data=adapt_data)

    print(image_mesh)

    assert image_mesh[0, :] == pytest.approx(
        [-1.02590674, -1.70984456],
        1.0e-4,
    )
