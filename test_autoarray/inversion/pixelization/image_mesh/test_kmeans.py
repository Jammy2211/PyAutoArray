import numpy as np
import pytest

import autoarray as aa


def test__image_plane_mesh_grid_from():
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

    kmeans = aa.image_mesh.KMeans(pixels=8)
    image_mesh = kmeans.image_plane_mesh_grid_from(grid=grid, adapt_data=weight_map)

    assert image_mesh[0, :] == pytest.approx(
        [0.25, -0.5],
        1.0e-4,
    )
