import autoarray as aa
import numpy as np
import pytest


class TestGrid1DActual1D:
    def test__sets_up_scaled_alone_grid(self):

        grid_slim = aa.util.grid_1d.grid_1d_via_shape_slim_from(
            shape_slim=(3,), pixel_scales=(1.0,), sub_size=1
        )

        assert (grid_slim == np.array([-1.0, 0.0, 1.0])).all()

        grid_slim = aa.util.grid_1d.grid_1d_via_shape_slim_from(
            shape_slim=(3,), pixel_scales=(1.0,), sub_size=2, origin=(1.0,)
        )

        assert (grid_slim == np.array([-0.25, 0.25, 0.75, 1.25, 1.75, 2.25])).all()

    def test__grid_1d_is_actual_via_via_mask_from(self):

        mask = np.array([False, True, False, False])

        grid_slim = aa.util.grid_1d.grid_1d_via_mask_from(
            mask_1d=mask, pixel_scales=(3.0,), sub_size=1
        )

        assert (grid_slim == np.array([-4.5, 1.5, 4.5])).all()

        mask = np.array([True, False, True, False])

        grid_slim = aa.util.grid_1d.grid_1d_via_mask_from(
            mask_1d=mask, pixel_scales=(2.0,), sub_size=2, origin=(1.0,)
        )

        assert (grid_slim == np.array([-0.5, 0.5, 3.5, 4.5])).all()
