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

        parallel_overscan_array = layout_2d.extract_parallel_overscan_array_from(
            array=array
        )

        assert (parallel_overscan_array == np.array([[0.0]])).all()

        layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 3, 0, 2))

        parallel_overscan_array = layout_2d.extract_parallel_overscan_array_from(
            array=array
        )

        assert (
            parallel_overscan_array.native
            == np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])
        ).all()

        layout_2d = aa.Layout2D(shape_2d=array.shape, parallel_overscan=(0, 4, 2, 3))

        parallel_overscan_array = layout_2d.extract_parallel_overscan_array_from(
            array=array
        )

        assert (
            parallel_overscan_array.native == np.array([[2.0], [5.0], [8.0], [11.0]])
        ).all()

    def test__parallel_overscan_binned_line(self):

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        array = aa.Array2D.manual(
            array=arr,
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=aa.Scans(parallel_overscan=(0, 1, 0, 1)),
        )

        assert (array.parallel_overscan_binned_line == np.array([0.0])).all()

        array = aa.Array2D.manual(
            array=arr,
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=aa.Scans(parallel_overscan=(0, 3, 0, 2)),
        )

        assert (array.parallel_overscan_binned_line == np.array([0.5, 3.5, 6.5])).all()

        array = aa.Array2D.manual(
            array=arr,
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=aa.Scans(parallel_overscan=(0, 4, 2, 3)),
        )

        assert (
            array.parallel_overscan_binned_line == np.array([2.0, 5.0, 8.0, 11.0])
        ).all()

    def test__parallel_front_edge_region_from__extracts_rows_within_bottom_of_region(
        self,
    ):

        array = aa.Array2D.ones(
            shape_native=(3, 3), pixel_scales=1.0, roe_corner=(1, 0)
        )

        region = aa.Region2D(region=(0, 3, 0, 3))

        # Front edge is row 0, so for 1 row we extract 0 -> 1

        front_edge = array.parallel_front_edge_region_from(region=region, rows=(0, 1))

        assert front_edge == (0, 1, 0, 3)

        # Front edge is row 0, so for 2 rows we extract 0 -> 2

        front_edge = array.parallel_front_edge_region_from(region=region, rows=(0, 2))

        assert front_edge == (0, 2, 0, 3)

        # Front edge is row 0, so for these 2 rows we extract 1 ->2

        front_edge = array.parallel_front_edge_region_from(region=region, rows=(1, 3))

        assert front_edge == (1, 3, 0, 3)

    def test__parallel_trails_of_region_from__extracts_rows_above_region(self):

        array = aa.Array2D.ones(
            shape_native=(3, 3), pixel_scales=1.0, roe_corner=(1, 0)
        )

        region = aa.Region2D(
            region=(0, 3, 0, 3)
        )  # The trails are row 3 and above, so extract 3 -> 4

        trails = array.parallel_trails_of_region_from(region=region, rows=(0, 1))

        assert trails == (3, 4, 0, 3)

        # The trails are row 3 and above, so extract 3 -> 5

        trails = array.parallel_trails_of_region_from(region=region, rows=(0, 2))

        assert trails == (3, 5, 0, 3)

        # The trails are row 3 and above, so extract 4 -> 6

        trails = array.parallel_trails_of_region_from(region=region, rows=(1, 3))

        assert trails == (4, 6, 0, 3)

    def test__parallel_side_nearest_read_out_region_from(self):
        array = aa.Array2D.manual(
            array=np.ones((5, 5)), pixel_scales=1.0, roe_corner=(1, 0)
        )
        region = aa.Region2D(region=(1, 3, 0, 5))

        parallel_region = array.parallel_side_nearest_read_out_region_from(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 5, 0, 1)

        array = aa.Array2D.manual(
            array=np.ones((4, 4)), pixel_scales=1.0, roe_corner=(1, 0)
        )
        region = aa.Region2D(region=(1, 3, 0, 5))

        parallel_region = array.parallel_side_nearest_read_out_region_from(
            region=region, columns=(1, 3)
        )

        assert parallel_region == (0, 4, 1, 3)

        region = aa.Region2D(region=(1, 3, 2, 5))

        parallel_region = array.parallel_side_nearest_read_out_region_from(
            region=region, columns=(1, 3)
        )

        assert parallel_region == (0, 4, 3, 5)

        array = aa.Array2D.manual(
            array=np.ones((2, 5)), pixel_scales=1.0, roe_corner=(1, 0)
        )
        region = aa.Region2D(region=(1, 3, 0, 5))

        parallel_region = array.parallel_side_nearest_read_out_region_from(
            region=region, columns=(0, 1)
        )

        assert parallel_region == (0, 2, 0, 1)

    def test__serial_overscan_array(self):

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        array = aa.Array2D.manual(
            array=arr,
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=aa.Scans(serial_overscan=(0, 1, 0, 1)),
        )

        assert (array.serial_overscan_array_from == np.array([[0.0]])).all()

        array = aa.Array2D.manual(
            array=arr,
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=aa.Scans(serial_overscan=(0, 3, 0, 2)),
        )

        assert (
            array.serial_overscan_array_from
            == np.array([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])
        ).all()

        array = aa.Array2D.manual(
            array=arr,
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=aa.Scans(serial_overscan=(0, 4, 2, 3)),
        )

        assert (
            array.serial_overscan_array_from == np.array([[2.0], [5.0], [8.0], [11.0]])
        ).all()

    def test__serial_overscan_binned_line(self):

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        array = aa.Array2D.manual(
            array=arr,
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=aa.Scans(serial_overscan=(0, 1, 0, 1)),
        )

        assert (array.serial_overscan_binned_array_1d_from == np.array([0.0])).all()

        array = aa.Array2D.manual(
            array=arr,
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=aa.Scans(serial_overscan=(0, 3, 0, 2)),
        )

        assert (
            array.serial_overscan_binned_array_1d_from == np.array([3.0, 4.0])
        ).all()

        array = aa.Array2D.manual(
            array=arr,
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=aa.Scans(serial_overscan=(0, 4, 2, 3)),
        )

        assert (array.serial_overscan_binned_array_1d_from == np.array([6.5])).all()

    def test__serial_front_edge_of_region__extracts_region_within_left_of_region(self):
        array = aa.Array2D.ones(
            shape_native=(3, 3), pixel_scales=1.0, roe_corner=(1, 0)
        )

        region = aa.Region2D(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for 1 column we extract 0 -> 1

        front_edge = array.serial_front_edge_of_region(region=region, columns=(0, 1))

        assert front_edge == (0, 3, 0, 1)

        # Front edge is column 0, so for 2 columns we extract 0 -> 2

        front_edge = array.serial_front_edge_of_region(region=region, columns=(0, 2))

        assert front_edge == (0, 3, 0, 2)

        # Front edge is column 0, so for these 2 columns we extract 1 ->2

        front_edge = array.serial_front_edge_of_region(region=region, columns=(1, 3))

        assert front_edge == (0, 3, 1, 3)

    def test__serial_trails_of_regions__extracts_region_to_right_of_region(self):
        array = aa.Array2D.ones(
            shape_native=(3, 3), pixel_scales=1.0, roe_corner=(1, 0)
        )

        region = aa.Region2D(
            region=(0, 3, 0, 3)
        )  # The trails are column 3 and above, so extract 3 -> 4

        trails = array.serial_trails_of_region_from(region=region, columns=(0, 1))

        assert trails == (0, 3, 3, 4)

        # The trails are column 3 and above, so extract 3 -> 5

        trails = array.serial_trails_of_region_from(region=region, columns=(0, 2))

        assert trails == (0, 3, 3, 5)

        # The trails are column 3 and above, so extract 4 -> 6

        trails = array.serial_trails_of_region_from(region=region, columns=(1, 3))

        assert trails == (0, 3, 4, 6)

    def test__serial_entie_rows_of_regioons__full_region_from_left_most_prescan_to_right_most_end_of_trails(
        self,
    ):

        array = aa.Array2D.manual(
            array=np.ones((5, 5)), pixel_scales=1.0, roe_corner=(1, 0)
        )
        region = aa.Region2D(region=(1, 3, 0, 5))

        serial_region = array.serial_entire_rows_of_region_from(region=region)

        assert serial_region == (1, 3, 0, 5)

        array = aa.Array2D.manual(
            array=np.ones((5, 25)), pixel_scales=1.0, roe_corner=(1, 0)
        )
        region = aa.Region2D(region=(1, 3, 0, 5))

        serial_region = array.serial_entire_rows_of_region_from(region=region)

        assert serial_region == (1, 3, 0, 25)

        array = aa.Array2D.manual(
            array=np.ones((8, 55)), pixel_scales=1.0, roe_corner=(1, 0)
        )
        region = aa.Region2D(region=(3, 5, 5, 30))

        serial_region = array.serial_entire_rows_of_region_from(region=region)

        assert serial_region == (3, 5, 0, 55)
