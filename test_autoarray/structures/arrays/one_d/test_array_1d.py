from os import path
import numpy as np

import autoarray as aa

test_array_1d_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "array_1d"
)


class TestAPI:
    def test__manual__makes_array_1d_with_pixel_scale(self):

        array_1d = aa.Array1D.manual_slim(array=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

        assert type(array_1d) == aa.Array1D
        assert (array_1d.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (array_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (array_1d.grid_radial == np.array(([0.0, 1.0, 2.0, 3.0]))).all()
        assert array_1d.pixel_scale == 1.0
        assert array_1d.pixel_scales == (1.0,)
        assert array_1d.origin == (0.0,)

        array_1d = aa.Array1D.manual_native(
            array=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0
        )

        assert type(array_1d) == aa.Array1D
        assert (array_1d.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (array_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (array_1d.grid_radial == np.array(([0.0, 1.0, 2.0, 3.0]))).all()
        assert array_1d.pixel_scale == 1.0
        assert array_1d.pixel_scales == (1.0,)
        assert array_1d.origin == (0.0,)

    def test__manual_mask__makes_array_1d_using_input_mask(self):

        mask = aa.Mask1D.manual(
            mask=[True, False, False, True, False, False], pixel_scales=1.0, sub_size=1
        )

        array_1d = aa.Array1D.manual_mask(
            array=[100.0, 1.0, 2.0, 100.0, 3.0, 4.0], mask=mask
        )

        assert type(array_1d) == aa.Array1D
        assert (array_1d.native == np.array([0.0, 1.0, 2.0, 0.0, 3.0, 4.0])).all()
        assert (array_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert array_1d.pixel_scale == 1.0
        assert array_1d.pixel_scales == (1.0,)
        assert array_1d.origin == (0.0,)

    def test__recursive_shape_storage(self):

        array_1d = aa.Array1D.manual_slim(array=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

        assert (array_1d.native.slim.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (array_1d.slim.native.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()

        mask = aa.Mask1D.manual(
            mask=[True, False, False, True, False, False], pixel_scales=1.0, sub_size=1
        )

        array_1d = aa.Array1D.manual_mask(
            array=[100.0, 1.0, 2.0, 100.0, 3.0, 4.0], mask=mask
        )

        assert (
            array_1d.native.slim.native == np.array([0.0, 1.0, 2.0, 0.0, 3.0, 4.0])
        ).all()
        assert (array_1d.slim.native.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
