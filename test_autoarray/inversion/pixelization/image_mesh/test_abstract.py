import numpy as np
import pytest

import autoarray as aa


def test__mesh_pixels_per_image_pixels_from():
    mask = aa.Mask2D.circular(
        shape_native=(3, 3),
        radius=2.0,
        pixel_scales=1.0,
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    mesh_grid = aa.Grid2DIrregular(
        values=[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 1.0), (0.0, 1.0), (-1.0, -1.0)]
    )

    image_mesh = aa.image_mesh.Hilbert(pixels=8)

    mesh_pixels_per_image_pixels = image_mesh.mesh_pixels_per_image_pixels_from(
        grid=grid, mesh_grid=mesh_grid
    )

    assert mesh_pixels_per_image_pixels.native == pytest.approx(
        np.array([[0, 0, 0], [0, 3, 2], [1, 0, 0]]), 1.0e-4
    )

def test__check_mesh_pixels_per_image_pixels():

    mask = aa.Mask2D.circular(
        shape_native=(3, 3),
        radius=2.0,
        pixel_scales=1.0,
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    mesh_grid = aa.Grid2DIrregular(
        values=[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 1.0), (0.0, 1.0), (-1.0, -1.0)]
    )

    image_mesh = aa.image_mesh.Hilbert(pixels=8)

    image_mesh.check_mesh_pixels_per_image_pixels(
        grid=grid,
        mesh_grid=mesh_grid,
        settings=None
    )

    image_mesh.check_mesh_pixels_per_image_pixels(
        grid=grid,
        mesh_grid=mesh_grid,
        settings=aa.SettingsInversion(image_mesh_min_mesh_pixels_per_pixel=3, image_mesh_min_mesh_number=1)
    )

    with pytest.raises(aa.exc.InversionException):
        image_mesh.check_mesh_pixels_per_image_pixels(
            grid=grid,
            mesh_grid=mesh_grid,
            settings=aa.SettingsInversion(image_mesh_min_mesh_pixels_per_pixel=5, image_mesh_min_mesh_number=1)
        )

    with pytest.raises(aa.exc.InversionException):
        image_mesh.check_mesh_pixels_per_image_pixels(
            grid=grid,
            mesh_grid=mesh_grid,
            settings=aa.SettingsInversion(image_mesh_min_mesh_pixels_per_pixel=3, image_mesh_min_mesh_number=2)
        )