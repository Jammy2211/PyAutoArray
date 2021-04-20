import os

import numpy as np
import autoarray as aa


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestRegions:
    def test__parallel_overscan_array(self):

        array = aa.Array2D.manual_native(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 1, 0, 1))

        extracted_array = layout_2d.extract_parallel_overscan_array_2d_from(array=array)

        assert (extracted_array == np.array([[0.0]])).all()

        layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 3, 0, 2))

        extracted_array = layout_2d.extract_parallel_overscan_array_2d_from(array=array)

        assert (
            extracted_array.native == np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])
        ).all()

        layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 4, 2, 3))

        extracted_array = layout_2d.extract_parallel_overscan_array_2d_from(array=array)

        assert (extracted_array.native == np.array([[2.0], [5.0], [8.0], [11.0]])).all()

    def test__parallel_overscan_binned_line(self):

        array = aa.Array2D.manual_native(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 1, 0, 1))

        binned_line = layout_2d.parallel_overscan_binned_array_1d_from(array=array)

        assert (binned_line == np.array([0.0])).all()

        layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 3, 0, 2))

        binned_line = layout_2d.parallel_overscan_binned_array_1d_from(array=array)

        assert (binned_line == np.array([0.5, 3.5, 6.5])).all()

        layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 4, 2, 3))

        binned_line = layout_2d.parallel_overscan_binned_array_1d_from(array=array)

        assert (binned_line == np.array([2.0, 5.0, 8.0, 11.0])).all()

    def test__serial_overscan_array(self):

        array = aa.Array2D.manual_native(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 1, 0, 1))

        extracted_array = layout_2d.extract_serial_overscan_array_from(array=array)

        assert (extracted_array == np.array([[0.0]])).all()

        layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 3, 0, 2))

        extracted_array = layout_2d.extract_serial_overscan_array_from(array=array)

        assert (
            extracted_array.native == np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])
        ).all()

        layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 4, 2, 3))

        extracted_array = layout_2d.extract_serial_overscan_array_from(array=array)

        assert (extracted_array.native == np.array([[2.0], [5.0], [8.0], [11.0]])).all()

    def test__serial_overscan_binned_line(self):

        array = aa.Array2D.manual_native(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 1, 0, 1))

        binned_lined = layout_2d.serial_overscan_binned_array_1d_from(array=array)

        assert (binned_lined == np.array([0.0])).all()

        layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 3, 0, 2))

        binned_lined = layout_2d.serial_overscan_binned_array_1d_from(array=array)

        assert (binned_lined == np.array([3.0, 4.0])).all()

        layout_2d = aa.Layout2D(shape_2d=array.shape, serial_overscan=(0, 4, 2, 3))

        binned_lined = layout_2d.serial_overscan_binned_array_1d_from(array=array)

        assert (binned_lined == np.array([6.5])).all()
