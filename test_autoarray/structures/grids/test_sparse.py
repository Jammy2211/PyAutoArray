import numpy as np
import pytest

import autoarray as aa






def test__from_snr_split():
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

    snr_map = aa.Array2D(
        values=[
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 4.0, 4.0, 1.0],
            [1.0, 4.0, 4.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        mask=mask,
    )

    sparse_grid = aa.Grid2DSparse.from_snr_split(
        pixels=8,
        fraction_high_snr=0.5,
        snr_cut=3.0,
        grid=grid,
        snr_map=snr_map,
        n_iter=10,
        max_iter=20,
        seed=1,
    )

    assert sparse_grid == pytest.approx(
        np.array(
            [
                [0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, -0.25],
                [0.25, -0.25],
                [0.58333, 0.58333],
                [0.58333, -0.58333],
                [-0.58333, -0.58333],
                [-0.58333, 0.58333],
            ]
        ),
        1.0e-4,
    )
