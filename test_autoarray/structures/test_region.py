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

        frame = aa.Frame2D.manual(
            array=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            pixel_scales=1.0,
        )

        region = aa.Region2D(region=(0, 2, 0, 2))

        new_frame = frame[region.slice]

        assert (new_frame == np.array([[1.0, 2.0], [4.0, 5.0]])).all()

        frame = aa.Frame2D.manual(
            array=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            pixel_scales=1.0,
        )

        region = aa.Region2D(region=(1, 3, 0, 3))

        new_frame = frame[region.slice]

        assert (new_frame == np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])).all()

    def test__add_region_to_frame_via_slice(self):

        frame = aa.Frame2D.manual(array=np.zeros((2, 2)), pixel_scales=1.0)
        image = np.ones((2, 2))
        region = aa.Region2D(region=(0, 1, 0, 1))

        frame[region.slice] += image[region.slice]

        assert (frame == np.array([[1.0, 0.0], [0.0, 0.0]])).all()

        frame = aa.Frame2D.manual(array=np.ones((2, 2)), pixel_scales=1.0)
        image = np.ones((2, 2))
        region = aa.Region2D(region=(0, 1, 0, 1))

        frame[region.slice] += image[region.slice]

        assert (frame == np.array([[2.0, 1.0], [1.0, 1.0]])).all()

        frame = aa.Frame2D.manual(array=np.ones((3, 3)), pixel_scales=1.0)
        image = np.ones((3, 3))
        region = aa.Region2D(region=(1, 3, 2, 3))

        frame[region.slice] += image[region.slice]

        assert (
            frame == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
        ).all()

    def test__set_region_to_zero_via_slice(self):

        frame = aa.Frame2D.manual(array=np.ones((2, 2)), pixel_scales=1.0)

        region = aa.Region2D(region=(0, 1, 0, 1))

        frame[region.slice] = 0

        assert (frame == np.array([[0.0, 1.0], [1.0, 1.0]])).all()

        frame = aa.Frame2D.manual(array=np.ones((3, 3)), pixel_scales=1.0)

        region = aa.Region2D(region=(1, 3, 2, 3))

        frame[region.slice] = 0

        assert (
            frame == np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        ).all()
