import os

import numpy as np
import autoarray as aa


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestBinnedAcross:
    def test__parallel__different_arrays__gives_frame_binned(self):

        frame = aa.Frame.manual(array=np.ones((3, 3)))

        assert (frame.binned_across_parallel == np.array([1.0, 1.0, 1.0])).all()

        frame = aa.Frame.manual(array=np.ones((4, 3)))

        assert (frame.binned_across_parallel == np.array([1.0, 1.0, 1.0])).all()

        frame = aa.Frame.manual(array=np.ones((3, 4)))

        assert (frame.binned_across_parallel == np.array([1.0, 1.0, 1.0, 1.0])).all()

        frame = aa.Frame.manual(
            array=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]])
        )

        assert (frame.binned_across_parallel == np.array([2.0, 6.0, 9.0])).all()

    def test__parallel__same_as_above_but_with_mask(self):

        mask = np.ma.array(
            [[False, False, False], [False, False, False], [True, False, False]]
        )

        frame = aa.Frame.manual_mask(
            array=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]]),
            mask=mask,
        )

        assert (frame.binned_across_parallel == np.array([1.5, 6.0, 9.0])).all()

    def test__serial__different_arrays__gives_frame_binned(self):

        frame = aa.Frame.manual(array=np.ones((3, 3)))

        assert (frame.binned_across_serial == np.array([1.0, 1.0, 1.0])).all()

        frame = aa.Frame.manual(array=np.ones((4, 3)))

        assert (frame.binned_across_serial == np.array([1.0, 1.0, 1.0, 1.0])).all()

        frame = aa.Frame.manual(array=np.ones((3, 4)))

        assert (frame.binned_across_serial == np.array([1.0, 1.0, 1.0])).all()

        frame = aa.Frame.manual(
            array=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]])
        )

        assert (frame.binned_across_serial == np.array([2.0, 6.0, 9.0])).all()

    def test__serial__same_as_above_but_with_mask(self):

        mask = np.ma.array(
            [[False, False, True], [False, False, False], [False, False, False]]
        )

        frame = aa.Frame.manual_mask(
            array=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]]),
            mask=mask,
        )

        assert (frame.binned_across_serial == np.array([1.5, 6.0, 9.0])).all()


class TestFrameRegions:
    def test__parallel_overscan_frame(self):

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(parallel_overscan=(0, 1, 0, 1))
        )

        assert (frame.parallel_overscan_frame == np.array([[0.0]])).all()

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(parallel_overscan=(0, 3, 0, 2))
        )

        assert (
            frame.parallel_overscan_frame
            == np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])
        ).all()

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(parallel_overscan=(0, 4, 2, 3))
        )

        assert (
            frame.parallel_overscan_frame == np.array([[2.0], [5.0], [8.0], [11.0]])
        ).all()

    def test__parallel_overscan_binned_line(self):

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(parallel_overscan=(0, 1, 0, 1))
        )

        assert (frame.parallel_overscan_binned_line == np.array([0.0])).all()

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(parallel_overscan=(0, 3, 0, 2))
        )

        assert (frame.parallel_overscan_binned_line == np.array([0.5, 3.5, 6.5])).all()

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(parallel_overscan=(0, 4, 2, 3))
        )

        assert (
            frame.parallel_overscan_binned_line == np.array([2.0, 5.0, 8.0, 11.0])
        ).all()

    def test__parallel_front_edge_of_region__extracts_rows_within_bottom_of_region(
        self
    ):

        frame = aa.Frame.ones(shape_2d=(3, 3), roe_corner=(1, 0))

        region = aa.Region(region=(0, 3, 0, 3))

        # Front edge is row 0, so for 1 row we extract 0 -> 1

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(0, 1))

        assert front_edge == (0, 1, 0, 3)

        # Front edge is row 0, so for 2 rows we extract 0 -> 2

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(0, 2))

        assert front_edge == (0, 2, 0, 3)

        # Front edge is row 0, so for these 2 rows we extract 1 ->2

        front_edge = frame.parallel_front_edge_of_region(region=region, rows=(1, 3))

        assert front_edge == (1, 3, 0, 3)

    def test__parallel_trails_of_region__extracts_rows_above_region(self):

        frame = aa.Frame.ones(shape_2d=(3, 3), roe_corner=(1, 0))

        region = aa.Region(
            region=(0, 3, 0, 3)
        )  # The trails are row 3 and above, so extract 3 -> 4

        trails = frame.parallel_trails_of_region(region=region, rows=(0, 1))

        assert trails == (3, 4, 0, 3)

        # The trails are row 3 and above, so extract 3 -> 5

        trails = frame.parallel_trails_of_region(region=region, rows=(0, 2))

        assert trails == (3, 5, 0, 3)

        # The trails are row 3 and above, so extract 4 -> 6

        trails = frame.parallel_trails_of_region(region=region, rows=(1, 3))

        assert trails == (4, 6, 0, 3)

    def test__parallel_side_nearest_read_out_region(self):
        frame = aa.Frame.manual(array=np.ones((5, 5)), roe_corner=(1, 0))
        region = aa.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 5, 0, 1)

        frame = aa.Frame.manual(array=np.ones((4, 4)), roe_corner=(1, 0))
        region = aa.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(1, 3)
        )

        assert parallel_region == (0, 4, 1, 3)

        region = aa.Region(region=(1, 3, 2, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(1, 3)
        )

        assert parallel_region == (0, 4, 3, 5)

        frame = aa.Frame.manual(array=np.ones((2, 5)), roe_corner=(1, 0))
        region = aa.Region(region=(1, 3, 0, 5))

        parallel_region = frame.parallel_side_nearest_read_out_region(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 2, 0, 1)

    def test__serial_overscan_frame(self):

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(serial_overscan=(0, 1, 0, 1))
        )

        assert (frame.serial_overscan_frame == np.array([[0.0]])).all()

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(serial_overscan=(0, 3, 0, 2))
        )

        assert (
            frame.serial_overscan_frame
            == np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])
        ).all()

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(serial_overscan=(0, 4, 2, 3))
        )

        assert (
            frame.serial_overscan_frame == np.array([[2.0], [5.0], [8.0], [11.0]])
        ).all()

    def test__serial_overscan_binned_line(self):

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(serial_overscan=(0, 1, 0, 1))
        )

        assert (frame.serial_overscan_binned_line == np.array([0.0])).all()

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(serial_overscan=(0, 3, 0, 2))
        )

        assert (frame.serial_overscan_binned_line == np.array([3.0, 4.0])).all()

        frame = aa.Frame.manual(
            array=arr, roe_corner=(1, 0), scans=aa.Scans(serial_overscan=(0, 4, 2, 3))
        )

        assert (frame.serial_overscan_binned_line == np.array([6.5])).all()

    def test__serial_front_edge_of_region__extracts_region_within_left_of_region(self):
        frame = aa.Frame.ones(shape_2d=(3, 3), roe_corner=(1, 0))

        region = aa.Region(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for 1 column we extract 0 -> 1

        front_edge = frame.serial_front_edge_of_region(region=region, columns=(0, 1))

        assert front_edge == (0, 3, 0, 1)

        # Front edge is column 0, so for 2 columns we extract 0 -> 2

        front_edge = frame.serial_front_edge_of_region(region=region, columns=(0, 2))

        assert front_edge == (0, 3, 0, 2)

        # Front edge is column 0, so for these 2 columns we extract 1 ->2

        front_edge = frame.serial_front_edge_of_region(region=region, columns=(1, 3))

        assert front_edge == (0, 3, 1, 3)

    def test__serial_trails_of_regions__extracts_region_to_right_of_region(self):
        frame = aa.Frame.ones(shape_2d=(3, 3), roe_corner=(1, 0))

        region = aa.Region(
            region=(0, 3, 0, 3)
        )  # The trails are column 3 and above, so extract 3 -> 4

        trails = frame.serial_trails_of_region(region=region, columns=(0, 1))

        assert trails == (0, 3, 3, 4)

        # The trails are column 3 and above, so extract 3 -> 5

        trails = frame.serial_trails_of_region(region=region, columns=(0, 2))

        assert trails == (0, 3, 3, 5)

        # The trails are column 3 and above, so extract 4 -> 6

        trails = frame.serial_trails_of_region(region=region, columns=(1, 3))

        assert trails == (0, 3, 4, 6)

    def test__serial_entie_rows_of_regioons__full_region_from_left_most_prescan_to_right_most_end_of_trails(
        self
    ):

        frame = aa.Frame.manual(array=np.ones((5, 5)), roe_corner=(1, 0))
        region = aa.Region(region=(1, 3, 0, 5))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (1, 3, 0, 5)

        frame = aa.Frame.manual(array=np.ones((5, 25)), roe_corner=(1, 0))
        region = aa.Region(region=(1, 3, 0, 5))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (1, 3, 0, 25)

        frame = aa.Frame.manual(array=np.ones((8, 55)), roe_corner=(1, 0))
        region = aa.Region(region=(3, 5, 5, 30))

        serial_region = frame.serial_entire_rows_of_region(region=region)

        assert serial_region == (3, 5, 0, 55)
