import os

import numpy as np
import pytest
import autoarray as aa
from autoarray import exc


@pytest.fixture(scope="class")
def euclid_data():
    euclid_data = np.zeros((2086, 2119))
    return euclid_data


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestConstructor:
    def test__constructor__converts_region_to_cartesians(self):
        region = aa.Region(region=(0, 1, 2, 3))

        assert region == (0, 1, 2, 3)

        assert region.y0 == 0
        assert region.y1 == 1
        assert region.x0 == 2
        assert region.x1 == 3
        assert region.total_rows == 1
        assert region.total_columns == 1

    def test__first_row_or_column_equal_too_or_bigger_than_second__raise_errors(self):
        with pytest.raises(exc.RegionException):
            aa.Region(region=(2, 2, 1, 2))

        with pytest.raises(exc.RegionException):
            aa.Region(region=(2, 1, 2, 2))

        with pytest.raises(exc.RegionException):
            aa.Region(region=(2, 1, 1, 2))

        with pytest.raises(exc.RegionException):
            aa.Region(region=(0, 1, 3, 2))

    def test__negative_coordinates_raise_errors(self):
        with pytest.raises(exc.RegionException):
            aa.Region(region=(-1, 0, 1, 2))

        with pytest.raises(exc.RegionException):
            aa.Region(region=(0, -1, 1, 2))

        with pytest.raises(exc.RegionException):
            aa.Region(region=(0, 0, -1, 2))

        with pytest.raises(exc.RegionException):
            aa.Region(region=(0, 1, 2, -1))


class TestExtractRegionFromArray:
    def test__extracts_2x2_region_of_3x3_array(self):
        frame = aa.Frame.manual(
            array=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        )

        region = aa.Region(region=(0, 2, 0, 2))

        new_frame = frame[region.slice]

        assert (new_frame == np.array([[1.0, 2.0], [4.0, 5.0]])).all()

    def test__extracts_2x3_region_of_3x3_array(self):
        frame = aa.Frame.manual(
            array=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        )

        region = aa.Region(region=(1, 3, 0, 3))

        new_frame = frame[region.slice]

        assert (new_frame == np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])).all()


class TestAddRegionToArrayFromImage:
    def test__array_is_all_zeros__image_goes_into_correct_region_of_array(self):
        frame = aa.Frame.manual(array=np.zeros((2, 2)))
        image = np.ones((2, 2))
        region = aa.Region(region=(0, 1, 0, 1))

        frame[region.slice] += image[region.slice]

        assert (frame == np.array([[1.0, 0.0], [0.0, 0.0]])).all()

    def test__array_is_all_1s__image_goes_into_correct_region_of_array_and_adds_to_it(
        self
    ):
        frame = aa.Frame.manual(array=np.ones((2, 2)))
        image = np.ones((2, 2))
        region = aa.Region(region=(0, 1, 0, 1))

        frame[region.slice] += image[region.slice]

        assert (frame == np.array([[2.0, 1.0], [1.0, 1.0]])).all()

    def test__different_region(self):
        frame = aa.Frame.manual(array=np.ones((3, 3)))
        image = np.ones((3, 3))
        region = aa.Region(region=(1, 3, 2, 3))

        frame[region.slice] += image[region.slice]

        assert (
            frame == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
        ).all()


class TestSetRegionToZeros:
    def test__region_is_corner__sets_to_0(self):
        frame = aa.Frame.manual(array=np.ones((2, 2)))

        region = aa.Region(region=(0, 1, 0, 1))

        frame[region.slice] = 0

        assert (frame == np.array([[0.0, 1.0], [1.0, 1.0]])).all()

    def test__different_region___sets_to_0(self):
        frame = aa.Frame.manual(array=np.ones((3, 3)))

        region = aa.Region(region=(1, 3, 2, 3))

        frame[region.slice] = 0

        assert (
            frame == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        ).all()
