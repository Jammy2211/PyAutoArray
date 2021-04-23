import os

import numpy as np
import pytest
import autoarray as aa
from autoarray import exc


class TestRegion1D:
    def test__sanity_check__first_row_or_column_equal_too_or_bigger_than_second__raise_errors(
        self
    ):

        with pytest.raises(exc.RegionException):
            aa.Region1D(region=(2, 2))

        with pytest.raises(exc.RegionException):
            aa.Region1D(region=(2, 1))

    def test__sanity_check__negative_coordinates_raise_errors(self):

        with pytest.raises(exc.RegionException):

            aa.Region1D(region=(-1, 0))

        with pytest.raises(exc.RegionException):
            aa.Region1D(region=(0, -1))

    def test__extraction_via_slice(self):

        arr_1d = aa.Array1D.manual_native(
            array=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), pixel_scales=1.0
        )

        region = aa.Region1D(region=(0, 2))

        new_arr_1d = arr_1d[region.slice]

        assert (new_arr_1d == np.array([1.0, 2.0])).all()

        region = aa.Region1D(region=(1, 3))

        new_arr_1d = arr_1d[region.slice]

        assert (new_arr_1d == np.array([2.0, 3.0])).all()

    def test__add_region_to_arr_1d_via_slice(self):

        arr_1d = aa.Array1D.manual_native(
            array=np.array([1.0, 2.0, 3.0, 4.0]), pixel_scales=1.0
        )

        image = aa.Array1D.full(fill_value=1.0, shape_native=6, pixel_scales=1.0)

        region = aa.Region1D(region=(0, 1))

        arr_1d[region.slice] += image[region.slice]

        assert (arr_1d == np.array([2.0, 2.0, 3.0, 4.0])).all()

        arr_1d = aa.Array1D.manual_native(
            array=np.array([1.0, 2.0, 3.0, 4.0]), pixel_scales=1.0
        )

        region = aa.Region1D(region=(2, 4))

        arr_1d[region.slice] += image[region.slice]

        assert (arr_1d == np.array([1.0, 2.0, 4.0, 5.0])).all()

    def test__set_region_to_zero_via_slice(self):

        arr_1d = aa.Array1D.manual_native(
            array=np.array([1.0, 2.0, 3.0, 4.0]), pixel_scales=1.0
        )

        region = aa.Region1D(region=(0, 1))

        arr_1d[region.slice] = 0

        assert (arr_1d == np.array([0.0, 2.0, 3.0, 4.0])).all()

        arr_1d = aa.Array1D.manual_native(
            array=np.array([1.0, 2.0, 3.0, 4.0]), pixel_scales=1.0
        )

        region = aa.Region1D(region=(2, 4))

        arr_1d[region.slice] = 0

        assert (arr_1d == np.array([1.0, 2.0, 0.0, 0.0])).all()


class TestRegion2D:
    def test__sanity_check__first_row_or_column_equal_too_or_bigger_than_second__raise_errors(
        self
    ):
        with pytest.raises(exc.RegionException):
            aa.Region2D(region=(2, 2, 1, 2))

        with pytest.raises(exc.RegionException):
            aa.Region2D(region=(2, 1, 2, 2))

        with pytest.raises(exc.RegionException):
            aa.Region2D(region=(2, 1, 1, 2))

        with pytest.raises(exc.RegionException):
            aa.Region2D(region=(0, 1, 3, 2))

    def test__sanity_check__negative_coordinates_raise_errors(self):

        with pytest.raises(exc.RegionException):
            aa.Region2D(region=(-1, 0, 1, 2))

        with pytest.raises(exc.RegionException):
            aa.Region2D(region=(0, -1, 1, 2))

        with pytest.raises(exc.RegionException):
            aa.Region2D(region=(0, 0, -1, 2))

        with pytest.raises(exc.RegionException):
            aa.Region2D(region=(0, 1, 2, -1))

    def test__extraction_via_slice(self):

        array = aa.Array2D.manual(
            array=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            pixel_scales=1.0,
        )

        region = aa.Region2D(region=(0, 2, 0, 2))

        new_array = array.native[region.slice]

        assert (new_array == np.array([[1.0, 2.0], [4.0, 5.0]])).all()

        array = aa.Array2D.manual(
            array=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            pixel_scales=1.0,
        )

        region = aa.Region2D(region=(1, 3, 0, 3))

        new_array = array.native[region.slice]

        assert (new_array == np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])).all()

    def test__add_region_to_array_via_slice(self):

        array = aa.Array2D.manual(array=np.zeros((2, 2)), pixel_scales=1.0)
        array = array.native
        image = np.ones((2, 2))
        region = aa.Region2D(region=(0, 1, 0, 1))

        array[region.slice] += image[region.slice]

        assert (array == np.array([[1.0, 0.0], [0.0, 0.0]])).all()

        array = aa.Array2D.manual(array=np.ones((2, 2)), pixel_scales=1.0)
        array = array.native
        image = np.ones((2, 2))
        region = aa.Region2D(region=(0, 1, 0, 1))

        array[region.slice] += image[region.slice]

        assert (array == np.array([[2.0, 1.0], [1.0, 1.0]])).all()

        array = aa.Array2D.manual(array=np.ones((3, 3)), pixel_scales=1.0)
        array = array.native
        image = np.ones((3, 3))
        region = aa.Region2D(region=(1, 3, 2, 3))

        array[region.slice] += image[region.slice]

        assert (
            array == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
        ).all()

    def test__set_region_to_zero_via_slice(self):

        array = aa.Array2D.manual(array=np.ones((2, 2)), pixel_scales=1.0)
        array = array.native

        region = aa.Region2D(region=(0, 1, 0, 1))

        array[region.slice] = 0

        assert (array == np.array([[0.0, 1.0], [1.0, 1.0]])).all()

        array = aa.Array2D.manual(array=np.ones((3, 3)), pixel_scales=1.0)
        array = array.native

        region = aa.Region2D(region=(1, 3, 2, 3))

        array[region.slice] = 0

        assert (
            array == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        ).all()

    def test__parallel_front_edge_region_from__extracts_rows_within_bottom_of_region(
        self,
    ):

        region = aa.Region2D(region=(0, 3, 0, 3))

        # Front edge is row 0, so for 1 row we extract 0 -> 1

        front_edge = region.parallel_front_edge_region_from(rows=(0, 1))

        assert front_edge == (0, 1, 0, 3)

        # Front edge is row 0, so for 2 rows we extract 0 -> 2

        front_edge = region.parallel_front_edge_region_from(rows=(0, 2))

        assert front_edge == (0, 2, 0, 3)

        # Front edge is row 0, so for these 2 rows we extract 1 ->2

        front_edge = region.parallel_front_edge_region_from(rows=(1, 3))

        assert front_edge == (1, 3, 0, 3)

    def test__parallel_trails_of_region_from__extracts_rows_above_region(self):

        region = aa.Region2D(
            region=(0, 3, 0, 3)
        )  # The trails are row 3 and above, so extract 3 -> 4

        trails = region.parallel_trails_region_from(rows=(0, 1))

        assert trails == (3, 4, 0, 3)

        # The trails are row 3 and above, so extract 3 -> 5

        trails = region.parallel_trails_region_from(rows=(0, 2))

        assert trails == (3, 5, 0, 3)

        # The trails are row 3 and above, so extract 4 -> 6

        trails = region.parallel_trails_region_from(rows=(1, 3))

        assert trails == (4, 6, 0, 3)

    def test__parallel_side_nearest_read_out_region_from(self):

        region = aa.Region2D(region=(1, 3, 0, 5))

        parallel_region = region.parallel_side_nearest_read_out_region_from(
            shape_2d=(5, 5), columns=(0, 1)
        )

        assert parallel_region == (0, 5, 0, 1)

        parallel_region = region.parallel_side_nearest_read_out_region_from(
            shape_2d=(4, 4), columns=(1, 3)
        )

        assert parallel_region == (0, 4, 1, 3)

        region = aa.Region2D(region=(1, 3, 2, 5))

        parallel_region = region.parallel_side_nearest_read_out_region_from(
            shape_2d=(4, 4), columns=(1, 3)
        )

        assert parallel_region == (0, 4, 3, 5)

        region = aa.Region2D(region=(1, 3, 0, 5))

        parallel_region = region.parallel_side_nearest_read_out_region_from(
            shape_2d=(2, 5), columns=(0, 1)
        )

        assert parallel_region == (0, 2, 0, 1)

    def test__serial_front_edge_of_region__extracts_region_within_left_of_region(self):

        region = aa.Region2D(
            region=(0, 3, 0, 3)
        )  # Front edge is column 0, so for 1 column we extract 0 -> 1

        front_edge = region.serial_front_edge_region_from(columns=(0, 1))

        assert front_edge == (0, 3, 0, 1)

        # Front edge is column 0, so for 2 columns we extract 0 -> 2

        front_edge = region.serial_front_edge_region_from(columns=(0, 2))

        assert front_edge == (0, 3, 0, 2)

        # Front edge is column 0, so for these 2 columns we extract 1 ->2

        front_edge = region.serial_front_edge_region_from(columns=(1, 3))

        assert front_edge == (0, 3, 1, 3)

    def test__serial_trails_of_regions__extracts_region_to_right_of_region(self):

        region = aa.Region2D(
            region=(0, 3, 0, 3)
        )  # The trails are column 3 and above, so extract 3 -> 4

        trails = region.serial_trails_region_from(columns=(0, 1))

        assert trails == (0, 3, 3, 4)

        # The trails are column 3 and above, so extract 3 -> 5

        trails = region.serial_trails_region_from(columns=(0, 2))

        assert trails == (0, 3, 3, 5)

        # The trails are column 3 and above, so extract 4 -> 6

        trails = region.serial_trails_region_from(columns=(1, 3))

        assert trails == (0, 3, 4, 6)

    def test__serial_entire_rows_of_regions__full_region_from_left_most_prescan_to_right_most_end_of_trails(
        self,
    ):

        region = aa.Region2D(region=(1, 3, 0, 5))

        serial_region = region.serial_entire_rows_of_region_from(shape_2d=(5, 5))

        assert serial_region == (1, 3, 0, 5)

        region = aa.Region2D(region=(1, 3, 0, 5))

        serial_region = region.serial_entire_rows_of_region_from(shape_2d=(5, 25))

        assert serial_region == (1, 3, 0, 25)

        region = aa.Region2D(region=(3, 5, 5, 30))

        serial_region = region.serial_entire_rows_of_region_from(shape_2d=(8, 55))

        assert serial_region == (3, 5, 0, 55)
