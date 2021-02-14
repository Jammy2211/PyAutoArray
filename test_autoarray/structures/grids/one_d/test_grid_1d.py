from os import path
import numpy as np
import pytest

import autoarray as aa

test_grid_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "grids"
)


class TestAPI:
    def test__manual(self):

        grid = aa.Grid1D.manual_native(
            grid=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0, sub_size=2
        )

        assert type(grid) == aa.Grid1D
        assert (grid.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid.native_binned == np.array([1.5, 3.5])).all()
        assert (grid.slim_binned == np.array([1.5, 3.5])).all()
        assert grid.pixel_scales == (1.0,)
        assert grid.origin == (0.0,)

        grid = aa.Grid1D.manual_slim(
            grid=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0, sub_size=2
        )

        assert type(grid) == aa.Grid1D
        assert (grid.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid.native_binned == np.array([1.5, 3.5])).all()
        assert (grid.slim_binned == np.array([1.5, 3.5])).all()
        assert grid.pixel_scales == (1.0,)
        assert grid.origin == (0.0,)

    def test__manual_mask(self):

        mask = aa.Mask1D.unmasked(shape_slim=(2,), pixel_scales=1.0, sub_size=2)
        grid = aa.Grid1D.manual_mask(grid=[1.0, 2.0, 3.0, 4.0], mask=mask)

        assert type(grid) == aa.Grid1D
        assert (grid.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid.native_binned == np.array([1.5, 3.5])).all()
        assert (grid.slim_binned == np.array([1.5, 3.5])).all()
        assert grid.pixel_scales == (1.0,)
        assert grid.origin == (0.0,)

        mask = aa.Mask1D.manual(mask=[True, False, False], pixel_scales=1.0, sub_size=2)
        grid = aa.Grid1D.manual_mask(grid=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], mask=mask)

        assert type(grid) == aa.Grid1D
        assert (grid.native == np.array([0.0, 0.0, 3.0, 4.0, 5.0, 6.0])).all()
        assert (grid.slim == np.array([3.0, 4.0, 5.0, 6.0])).all()
        assert (grid.native_binned == np.array([0.0, 3.5, 5.5])).all()
        assert (grid.slim_binned == np.array([3.5, 5.5])).all()
        assert grid.pixel_scales == (1.0,)
        assert grid.origin == (0.0,)
