import numpy as np
import pytest

import autoarray as aa


def test__interpolation_grid_from():

    mesh = aa.m.MockGrid2DMesh(extent=(-1.0, 1.0, -1.0, 1.0))

    interpolation_grid = mesh.interpolation_grid_from(shape_native=(3, 2))

    assert interpolation_grid.native == pytest.approx(
        np.array(
            [
                [[1.0, -1.0], [1.0, 1.0]],
                [[0.0, -1.0], [0.0, 1.0]],
                [[-1.0, -1.0], [-1.0, 1.0]],
            ]
        )
    )
    assert interpolation_grid.pixel_scales == (1.0, 2.0)

    interpolation_grid = mesh.interpolation_grid_from(shape_native=(2, 3))

    assert interpolation_grid.native == pytest.approx(
        np.array(
            [
                [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]],
                [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]],
            ]
        )
    )
    assert interpolation_grid.pixel_scales == (2.0, 1.0)

    mesh = aa.m.MockGrid2DMesh(extent=(-20.0, -5.0, -10.0, -5.0))

    interpolation_grid = mesh.interpolation_grid_from(shape_native=(3, 3))

    assert interpolation_grid.native == pytest.approx(
        np.array(
            [
                [[0.0, -20.0], [0.0, -12.5], [0.0, -5.0]],
                [[-7.5, -20.0], [-7.5, -12.5], [-7.5, -5.0]],
                [[-15.0, -20.0], [-15.0, -12.5], [-15.0, -5.0]],
            ]
        )
    )
    assert interpolation_grid.pixel_scales == (7.5, 7.5)
