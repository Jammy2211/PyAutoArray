import os
from os import path

import numpy as np
import pytest
import shutil

import autoarray as aa

test_data_dir = path.join(
    "{}".format(os.path.dirname(os.path.realpath(__file__))), "files"
)


class TestAPI:
    def test__manual(self):

        arr = aa.Array2D.manual_native(
            array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, sub_size=1
        )

        assert type(arr) == aa.Array2D
        assert (arr == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (arr.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (arr.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 0.0)
        assert arr.mask.sub_size == 1

        arr = aa.Array2D.manual_slim(
            array=[1.0, 2.0, 3.0, 4.0],
            shape_native=(1, 1),
            pixel_scales=1.0,
            sub_size=2,
            origin=(0.0, 1.0),
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (arr.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)
        assert arr.mask.sub_size == 2

        arr = aa.Array2D.manual_slim(
            array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape_native=(2, 1),
            pixel_scales=2.0,
            sub_size=2,
        )

        assert type(arr) == aa.Array2D
        assert (
            arr.native == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (arr.slim == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
        assert (arr.binned.native == np.array([[2.5], [6.5]])).all()
        assert (arr.binned == np.array([2.5, 6.5])).all()
        assert arr.pixel_scales == (2.0, 2.0)
        assert arr.origin == (0.0, 0.0)
        assert arr.mask.sub_size == 2

    def test__manual_mask(self):

        mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)
        arr = aa.Array2D.manual_mask(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (arr.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 0.0)

        mask = aa.Mask2D.manual(
            mask=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        arr = aa.Array2D.manual_mask(array=[1.0, 2.0, 4.0], mask=mask)

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
        assert (arr.slim == np.array([1.0, 2.0, 4.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

        mask = aa.Mask2D.manual(
            mask=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )
        arr = aa.Array2D.manual_mask(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[1.0, 2.0], [0.0, 4.0]])).all()
        assert (arr.slim == np.array([1.0, 2.0, 4.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

        mask = aa.Mask2D.manual(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)
        arr = aa.Array2D.manual_mask(array=[1.0, 2.0, 3.0, 4.0], mask=mask)

        assert type(arr) == aa.Array2D
        assert (arr == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (
            arr.native == np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
        ).all()
        assert (arr.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (arr.binned.native == np.array([[2.5], [0.0]])).all()
        assert (arr.binned.slim == np.array([2.5])).all()
        assert arr.pixel_scales == (2.0, 2.0)
        assert arr.origin == (0.0, 0.0)
        assert arr.mask.sub_size == 2

        mask = aa.Mask2D.manual(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)
        arr = aa.Array2D.manual_slim(
            array=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape_native=(2, 1),
            pixel_scales=2.0,
            sub_size=2,
        )
        arr = arr.apply_mask(mask=mask)

        assert (
            arr.native == np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
        ).all()

    def test__manual_native__exception_raised_if_input_array_is_2d_and_not_sub_shape_of_mask(
        self,
    ):
        with pytest.raises(aa.exc.ArrayException):
            mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)
            aa.Array2D.manual_mask(array=[[1.0], [3.0]], mask=mask)

        with pytest.raises(aa.exc.ArrayException):
            mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0, sub_size=2)
            aa.Array2D.manual_mask(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

        with pytest.raises(aa.exc.ArrayException):
            mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0, sub_size=2)
            aa.Array2D.manual_mask(
                array=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], mask=mask
            )

    def test__exception_raised_if_input_array_is_1d_and_not_number_of_masked_sub_pixels(
        self,
    ):
        with pytest.raises(aa.exc.ArrayException):
            mask = aa.Mask2D.manual(
                mask=[[False, False], [True, False]], pixel_scales=1.0, sub_size=1
            )
            aa.Array2D.manual_mask(array=[1.0, 2.0, 3.0, 4.0], mask=mask)

        with pytest.raises(aa.exc.ArrayException):
            mask = aa.Mask2D.manual(
                mask=[[False, False], [True, False]], pixel_scales=1.0, sub_size=1
            )
            aa.Array2D.manual_mask(array=[1.0, 2.0], mask=mask)

        with pytest.raises(aa.exc.ArrayException):
            mask = aa.Mask2D.manual(
                mask=[[False, True], [True, True]], pixel_scales=1.0, sub_size=2
            )
            aa.Array2D.manual_mask(array=[1.0, 2.0, 4.0], mask=mask)

        with pytest.raises(aa.exc.ArrayException):
            mask = aa.Mask2D.manual(
                mask=[[False, True], [True, True]], pixel_scales=1.0, sub_size=2
            )
            aa.Array2D.manual_mask(array=[1.0, 2.0, 3.0, 4.0, 5.0], mask=mask)

    def test__full__makes_scaled_array_with_pixel_scale(self):

        arr = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
        assert (arr.slim == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 0.0)

        arr = aa.Array2D.full(
            fill_value=2.0, shape_native=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0)
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
        assert (arr.slim == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

    def test__full__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

        arr = aa.Array2D.full(
            fill_value=1.0, shape_native=(1, 4), pixel_scales=1.0, sub_size=1
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[1.0, 1.0, 1.0, 1.0]])).all()
        assert (arr.slim == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 0.0)
        assert arr.mask.sub_size == 1

        arr = aa.Array2D.full(
            fill_value=2.0,
            shape_native=(1, 1),
            pixel_scales=1.0,
            sub_size=2,
            origin=(0.0, 1.0),
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[2.0, 2.0], [2.0, 2.0]])).all()
        assert (arr.slim == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)
        assert arr.mask.sub_size == 2

    def test__ones_zeros__makes_array_without_other_inputs(self):

        arr = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
        assert (arr.slim == np.array([1.0, 1.0, 1.0, 1.0])).all()

        arr = aa.Array2D.zeros(shape_native=(2, 2), pixel_scales=1.0)

        assert type(arr) == aa.Array2D
        assert (arr == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert (arr.native == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert (arr.slim == np.array([0.0, 0.0, 0.0, 0.0])).all()

    def test__ones_zeros__makes_scaled_array_with_pixel_scale(self):

        arr = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
        assert (arr.slim == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 0.0)

        arr = aa.Array2D.zeros(shape_native=(2, 2), pixel_scales=1.0, origin=(0.0, 1.0))

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert (arr.slim == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

    def test__ones_zeros__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

        arr = aa.Array2D.ones(shape_native=(1, 4), pixel_scales=1.0, sub_size=1)

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[1.0, 1.0, 1.0, 1.0]])).all()
        assert (arr.slim == np.array([1.0, 1.0, 1.0, 1.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 0.0)
        assert arr.mask.sub_size == 1

        arr = aa.Array2D.zeros(
            shape_native=(1, 1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0)
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert (arr.slim == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)
        assert arr.mask.sub_size == 2

    def test__from_fits__makes_array_without_other_inputs(self):

        arr = aa.Array2D.from_fits(
            file_path=path.join(test_data_dir, "3x3_ones.fits"), hdu=0, pixel_scales=1.0
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == np.ones((3, 3))).all()
        assert (arr.slim == np.ones(9)).all()

        arr = aa.Array2D.from_fits(
            file_path=path.join(test_data_dir, "4x3_ones.fits"), hdu=0, pixel_scales=1.0
        )

        assert type(arr) == aa.Array2D
        assert (arr == np.ones((12,))).all()
        assert (arr.native == np.ones((4, 3))).all()
        assert (arr.slim == np.ones((12,))).all()

    def test__from_fits__loads_and_stores_header_info(self):

        arr = aa.Array2D.from_fits(
            file_path=path.join(test_data_dir, "3x3_ones.fits"), hdu=0, pixel_scales=1.0
        )

        assert arr.header.header_sci_obj["BITPIX"] == -64
        assert arr.header.header_hdu_obj["BITPIX"] == -64

        arr = aa.Array2D.from_fits(
            file_path=path.join(test_data_dir, "4x3_ones.fits"), hdu=0, pixel_scales=1.0
        )

        assert arr.header.header_sci_obj["BITPIX"] == -64
        assert arr.header.header_hdu_obj["BITPIX"] == -64

    def test__from_yx_values__use_manual_array_values__returns_input_array(self):

        arr = aa.Array2D.manual_native(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)

        y = arr.mask.unmasked_grid_sub_1[:, 0]
        x = arr.mask.unmasked_grid_sub_1[:, 1]
        arr_via_yx = aa.Array2D.manual_yx_and_values(
            y=y, x=x, values=arr, shape_native=arr.shape_native, pixel_scales=1.0
        )

        assert (arr == arr_via_yx).all()

        arr = aa.Array2D.manual_native(
            array=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], pixel_scales=1.0
        )

        y = arr.mask.unmasked_grid_sub_1[:, 0]
        x = arr.mask.unmasked_grid_sub_1[:, 1]

        arr_via_yx = aa.Array2D.manual_yx_and_values(
            y=y, x=x, values=arr, shape_native=arr.shape_native, pixel_scales=1.0
        )

        assert (arr == arr_via_yx).all()

        arr = aa.Array2D.manual_native(
            array=[[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]], pixel_scales=1.0
        )

        y = arr.mask.unmasked_grid_sub_1[:, 0]
        x = arr.mask.unmasked_grid_sub_1[:, 1]

        arr_via_yx = aa.Array2D.manual_yx_and_values(
            y=y, x=x, values=arr, shape_native=arr.shape_native, pixel_scales=1.0
        )

        assert (arr == arr_via_yx).all()

    def test__from_yx_values__use_input_values_which_swap_values_from_top_left_notation(
        self,
    ):

        arr = aa.Array2D.manual_yx_and_values(
            y=[0.5, 0.5, -0.5, -0.5],
            x=[-0.5, 0.5, -0.5, 0.5],
            values=[1.0, 2.0, 3.0, 4.0],
            shape_native=(2, 2),
            pixel_scales=1.0,
        )

        assert (arr.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        arr = aa.Array2D.manual_yx_and_values(
            y=[-0.5, 0.5, 0.5, -0.5],
            x=[-0.5, 0.5, -0.5, 0.5],
            values=[1.0, 2.0, 3.0, 4.0],
            shape_native=(2, 2),
            pixel_scales=1.0,
        )

        assert (arr.native == np.array([[3.0, 2.0], [1.0, 4.0]])).all()

        arr = aa.Array2D.manual_yx_and_values(
            y=[-0.5, 0.5, 0.5, -0.5],
            x=[0.5, 0.5, -0.5, -0.5],
            values=[1.0, 2.0, 3.0, 4.0],
            shape_native=(2, 2),
            pixel_scales=1.0,
        )

        assert (arr.native == np.array([[4.0, 2.0], [1.0, 3.0]])).all()

        arr = aa.Array2D.manual_yx_and_values(
            y=[1.0, 1.0, 0.0, 0.0, -1.0, -1.0],
            x=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
            values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape_native=(3, 2),
            pixel_scales=1.0,
        )

        assert (arr.native == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])).all()

        arr = aa.Array2D.manual_yx_and_values(
            y=[0.0, 1.0, -1.0, 0.0, -1.0, 1.0],
            x=[-0.5, 0.5, 0.5, 0.5, -0.5, -0.5],
            values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape_native=(3, 2),
            pixel_scales=1.0,
        )

        assert (arr.native == np.array([[3.0, 2.0], [6.0, 4.0], [5.0, 1.0]])).all()


class TestHeader:
    def test__header_has_date_and_time_of_observation__calcs_julian_date(self):

        header_sci_obj = {"DATE-OBS": "2000-01-01", "TIME-OBS": "00:00:00"}

        header = aa.Header(header_sci_obj=header_sci_obj, header_hdu_obj=None)

        assert header.modified_julian_date == 51544.0


class TestArray:
    def test__recursive_shape_storage(self):

        arr = aa.Array2D.manual_native(
            array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, sub_size=1
        )

        assert (arr.native.slim.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (arr.slim.native.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()


class TestConstructorMethods:
    def test__constructor_class_method_native(self):

        arr = aa.Array2D.manual_native(
            array=np.ones((3, 3)), sub_size=1, pixel_scales=(1.0, 1.0)
        )

        assert (arr == np.ones((9,))).all()
        assert (arr.slim == np.ones((9,))).all()
        assert (arr.native == np.ones((3, 3))).all()
        assert (arr.binned == np.ones((9,))).all()
        assert (arr.binned.native == np.ones((3, 3))).all()
        assert (arr.binned == np.ones((9,))).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.mask.central_pixel_coordinates == (1.0, 1.0)
        assert arr.mask.shape_native_scaled == pytest.approx((3.0, 3.0))
        assert arr.mask.scaled_maxima == (1.5, 1.5)
        assert arr.mask.scaled_minima == (-1.5, -1.5)

        arr = aa.Array2D.manual_native(
            array=np.ones((4, 4)), sub_size=2, pixel_scales=(0.1, 0.1)
        )

        assert (arr == np.ones((16,))).all()
        assert (arr.slim == np.ones((16,))).all()
        assert (arr.native == np.ones((4, 4))).all()
        assert (arr.binned == np.ones((4,))).all()
        assert (arr.binned.native == np.ones((2, 2))).all()
        assert (arr.binned == np.ones((4,))).all()
        assert arr.pixel_scales == (0.1, 0.1)
        assert arr.mask.central_pixel_coordinates == (0.5, 0.5)
        assert arr.mask.shape_native_scaled == pytest.approx((0.2, 0.2))
        assert arr.mask.scaled_maxima == pytest.approx((0.1, 0.1), 1e-4)
        assert arr.mask.scaled_minima == pytest.approx((-0.1, -0.1), 1e-4)

        arr = aa.Array2D.manual_native(
            array=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            pixel_scales=(0.1, 0.1),
            sub_size=2,
            origin=(1.0, 1.0),
        )

        assert (arr == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
        assert arr.shape_native == (2, 1)
        assert arr.sub_shape_native == (4, 2)
        assert (arr.slim == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
        assert (
            arr.native == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert arr.binned.native.shape == (2, 1)
        assert (arr.binned == np.array([2.5, 6.5])).all()
        assert (arr.binned.native == np.array([[2.5], [6.5]])).all()
        assert (arr.binned == np.array([2.5, 6.5])).all()
        assert arr.pixel_scales == (0.1, 0.1)
        assert arr.mask.central_pixel_coordinates == (0.5, 0.0)
        assert arr.mask.shape_native_scaled == pytest.approx((0.2, 0.1))
        assert arr.mask.scaled_maxima == pytest.approx((1.1, 1.05), 1e-4)
        assert arr.mask.scaled_minima == pytest.approx((0.9, 0.95), 1e-4)

        arr = aa.Array2D.manual_native(
            array=np.ones((3, 3)),
            pixel_scales=(2.0, 1.0),
            sub_size=1,
            origin=(-1.0, -2.0),
        )

        assert arr == pytest.approx(np.ones((9,)), 1e-4)
        assert arr.slim == pytest.approx(np.ones((9,)), 1e-4)
        assert arr.native == pytest.approx(np.ones((3, 3)), 1e-4)
        assert arr.pixel_scales == (2.0, 1.0)
        assert arr.mask.central_pixel_coordinates == (1.0, 1.0)
        assert arr.mask.shape_native_scaled == pytest.approx((6.0, 3.0))
        assert arr.origin == (-1.0, -2.0)
        assert arr.mask.scaled_maxima == pytest.approx((2.0, -0.5), 1e-4)
        assert arr.mask.scaled_minima == pytest.approx((-4.0, -3.5), 1e-4)

    def test__constructor_class_method_slim(self):
        arr = aa.Array2D.manual_slim(
            array=np.ones((9,)),
            shape_native=(3, 3),
            pixel_scales=(2.0, 1.0),
            sub_size=1,
            origin=(-1.0, -2.0),
        )

        assert arr == pytest.approx(np.ones((9,)), 1e-4)
        assert arr.slim == pytest.approx(np.ones((9,)), 1e-4)
        assert arr.native == pytest.approx(np.ones((3, 3)), 1e-4)
        assert arr.pixel_scales == (2.0, 1.0)
        assert arr.mask.central_pixel_coordinates == (1.0, 1.0)
        assert arr.mask.shape_native_scaled == pytest.approx((6.0, 3.0))
        assert arr.origin == (-1.0, -2.0)
        assert arr.mask.scaled_maxima == pytest.approx((2.0, -0.5), 1e-4)
        assert arr.mask.scaled_minima == pytest.approx((-4.0, -3.5), 1e-4)


class TestNewArrays:
    def test__pad__compare_to_array_util(self):

        array_2d = np.ones((5, 5))
        array_2d[2, 2] = 2.0

        arr = aa.Array2D.manual_native(
            array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
        )

        arr = arr.resized_from(new_shape=(7, 7))

        arr_resized_manual = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == arr_resized_manual).all()
        assert arr.mask.pixel_scales == (1.0, 1.0)

    def test__trim__compare_to_array_util(self):
        array_2d = np.ones((5, 5))
        array_2d[2, 2] = 2.0

        arr = aa.Array2D.manual_native(
            array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
        )

        arr = arr.resized_from(new_shape=(3, 3))

        arr_resized_manual = np.array(
            [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == arr_resized_manual).all()
        assert arr.mask.pixel_scales == (1.0, 1.0)

    def test__padded_before_convolution_from__padded_edge_of_zeros_where_extra_psf_blurring_is_performed(
        self,
    ):
        array_2d = np.ones((5, 5))
        array_2d[2, 2] = 2.0

        arr = aa.Array2D.manual_native(
            array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
        )

        new_arr = arr.padded_before_convolution_from(kernel_shape=(3, 3))

        assert type(new_arr) == aa.Array2D
        assert new_arr.native[0, 0] == 0.0
        assert new_arr.shape_native == (7, 7)
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

        new_arr = arr.padded_before_convolution_from(kernel_shape=(5, 5))

        assert type(new_arr) == aa.Array2D
        assert new_arr.native[0, 0] == 0.0
        assert new_arr.shape_native == (9, 9)
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

        array_2d = np.ones((9, 9))
        array_2d[4, 4] = 2.0

        arr = aa.Array2D.manual_native(
            array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
        )

        new_arr = arr.padded_before_convolution_from(kernel_shape=(7, 7))

        assert type(new_arr) == aa.Array2D
        assert new_arr.native[0, 0] == 0.0
        assert new_arr.shape_native == (15, 15)
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

    def test__trimmed_from_kernel_shape__trim_edges_where_extra_psf_blurring_is_performed(
        self,
    ):
        array_2d = np.ones((5, 5))
        array_2d[2, 2] = 2.0

        arr = aa.Array2D.manual_native(
            array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
        )

        new_arr = arr.trimmed_after_convolution_from(kernel_shape=(3, 3))

        assert type(new_arr) == aa.Array2D
        assert (
            new_arr.native
            == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

        new_arr = arr.trimmed_after_convolution_from(kernel_shape=(5, 5))

        assert type(new_arr) == aa.Array2D
        assert (new_arr.native == np.array([[2.0]])).all()
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

        array_2d = np.ones((9, 9))
        array_2d[4, 4] = 2.0

        arr = aa.Array2D.manual_native(
            array=array_2d, sub_size=1, pixel_scales=(1.0, 1.0)
        )

        new_arr = arr.trimmed_after_convolution_from(kernel_shape=(7, 7))

        assert type(new_arr) == aa.Array2D
        assert (
            new_arr.native
            == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()
        assert new_arr.mask.pixel_scales == (1.0, 1.0)

    def test__zoomed__2d_array_zoomed__uses_the_limits_of_the_mask(self):
        array_2d = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        arr_masked = aa.Array2D.manual_mask(array=array_2d, mask=mask)

        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

        assert (arr_zoomed.native == np.array([[6.0, 7.0], [10.0, 11.0]])).all()

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, False],
                    [True, False, False, False],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        arr_masked = aa.Array2D.manual_mask(array=array_2d, mask=mask)
        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
        assert (
            arr_zoomed.native == np.array([[6.0, 7.0, 8.0], [10.0, 11.0, 12.0]])
        ).all()

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, False, False, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        arr_masked = aa.Array2D.manual_mask(array=array_2d, mask=mask)
        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
        assert (
            arr_zoomed.native == np.array([[6.0, 7.0], [10.0, 11.0], [14.0, 15.0]])
        ).all()

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [False, False, False, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        arr_masked = aa.Array2D.manual_mask(array=array_2d, mask=mask)
        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

        assert (
            arr_zoomed.native == np.array([[0.0, 6.0, 7.0], [9.0, 10.0, 11.0]])
        ).all()

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, False, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        arr_masked = aa.Array2D.manual_mask(array=array_2d, mask=mask)
        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)
        assert (
            arr_zoomed.native == np.array([[2.0, 0.0], [6.0, 7.0], [10.0, 11.0]])
        ).all()

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        arr_masked = aa.Array2D.manual_mask(array=array_2d, mask=mask)
        arr_zoomed = arr_masked.zoomed_around_mask(buffer=1)

        assert (
            arr_zoomed.native
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 6.0, 7.0, 0.0],
                    [0.0, 10.0, 11.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__zoomed_2d_array_zoomed__centre_is_updated_using_original_mask(self):
        array_2d = np.ones(shape=(4, 4))

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        arr_masked = aa.Array2D.manual_mask(array=array_2d, mask=mask)

        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

        assert arr_zoomed.mask.origin == (0.0, 0.0)

        array_2d = np.ones(shape=(6, 6))

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True, True, True],
                    [True, True, True, False, False, True],
                    [True, True, True, False, False, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        arr_masked = aa.Array2D.manual_mask(array=array_2d, mask=mask)

        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

        assert arr_zoomed.mask.origin == (1.0, 1.0)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                    [True, True, True, False, False, True],
                    [True, True, True, False, False, True],
                    [True, True, True, True, True, True],
                    [True, True, True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 1.0),
            sub_size=1,
        )

        arr_masked = aa.Array2D.manual_mask(array=array_2d, mask=mask)

        arr_zoomed = arr_masked.zoomed_around_mask(buffer=0)

        assert arr_zoomed.mask.origin == (0.0, 1.0)

    def test__zoomed__array_extent__uses_the_limits_of_the_unzoomed_mask(self):
        array_2d = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]

        mask = aa.Mask2D.manual(
            mask=np.array(
                [
                    [True, True, True, False],
                    [True, False, False, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=(1.0, 2.0),
            sub_size=1,
        )

        arr_masked = aa.Array2D.manual_mask(array=array_2d, mask=mask)

        extent = arr_masked.extent_of_zoomed_array(buffer=1)

        assert extent == pytest.approx(np.array([-4.0, 6.0, -2.0, 3.0]), 1.0e-4)


class TestBinnedAcross:
    def test__columns__different_arrays__gives_array_binned(self):

        array = aa.Array2D.manual(array=np.ones((3, 3)), pixel_scales=1.0)

        assert (array.binned_across_columns == np.array([1.0, 1.0, 1.0])).all()

        array = aa.Array2D.manual(array=np.ones((4, 3)), pixel_scales=1.0)

        assert (array.binned_across_rows == np.array([1.0, 1.0, 1.0])).all()

        array = aa.Array2D.manual(array=np.ones((3, 4)), pixel_scales=1.0)

        assert (array.binned_across_rows == np.array([1.0, 1.0, 1.0, 1.0])).all()

        array = aa.Array2D.manual(
            array=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]]),
            pixel_scales=1.0,
        )

        assert (array.binned_across_rows == np.array([2.0, 6.0, 9.0])).all()

    def test__columns__same_as_above_but_with_mask(self):

        mask = aa.Mask2D.manual(
            mask=[[False, False, False], [False, False, False], [True, False, False]],
            pixel_scales=1.0,
        )

        array = aa.Array2D.manual_mask(
            array=np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]]),
            mask=mask,
        )

        assert (array.binned_across_rows == np.array([1.5, 6.0, 9.0])).all()

    def test__rows__different_arrays__gives_array_binned(self):

        array = aa.Array2D.manual(array=np.ones((3, 3)), pixel_scales=1.0)

        assert (array.binned_across_columns == np.array([1.0, 1.0, 1.0])).all()

        array = aa.Array2D.manual(array=np.ones((4, 3)), pixel_scales=1.0)

        assert (array.binned_across_columns == np.array([1.0, 1.0, 1.0, 1.0])).all()

        array = aa.Array2D.manual(array=np.ones((3, 4)), pixel_scales=1.0)

        assert (array.binned_across_columns == np.array([1.0, 1.0, 1.0])).all()

        array = aa.Array2D.manual(
            array=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]]),
            pixel_scales=1.0,
        )

        assert (array.binned_across_columns == np.array([2.0, 6.0, 9.0])).all()

    def test__rows__same_as_above_but_with_mask(self):

        mask = aa.Mask2D.manual(
            mask=[[False, False, True], [False, False, False], [False, False, False]],
            pixel_scales=1.0,
        )

        array = aa.Array2D.manual_mask(
            array=np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]]),
            mask=mask,
        )

        assert (array.binned_across_columns == np.array([1.5, 6.0, 9.0])).all()


class TestOutputToFits:
    def test__output_to_fits(self):

        arr = aa.Array2D.from_fits(
            file_path=path.join(test_data_dir, "3x3_ones.fits"), hdu=0, pixel_scales=1.0
        )

        output_data_dir = path.join(
            "{}".format(path.dirname(path.realpath(__file__))),
            "files",
            "array",
            "output_test",
        )
        if path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=path.join(output_data_dir, "array.fits"))

        array_from_out = aa.Array2D.from_fits(
            file_path=path.join(output_data_dir, "array.fits"), hdu=0, pixel_scales=1.0
        )

        assert (array_from_out.native == np.ones((3, 3))).all()

    def test__output_to_fits__shapes_of_arrays_are_2d(self):

        arr = aa.Array2D.from_fits(
            file_path=path.join(test_data_dir, "3x3_ones.fits"), hdu=0, pixel_scales=1.0
        )

        output_data_dir = path.join(
            "{}".format(path.dirname(path.realpath(__file__))),
            "files",
            "array",
            "output_test",
        )

        if path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        arr.output_to_fits(file_path=path.join(output_data_dir, "array.fits"))

        array_from_out = aa.util.array_2d.numpy_array_2d_via_fits_from(
            file_path=path.join(output_data_dir, "array.fits"), hdu=0
        )

        assert (array_from_out == np.ones((3, 3))).all()

        mask = aa.Mask2D.unmasked(shape_native=(3, 3), pixel_scales=0.1)

        masked_array = aa.Array2D.manual_mask(array=arr, mask=mask)

        masked_array.output_to_fits(
            file_path=path.join(output_data_dir, "masked_array.fits")
        )

        masked_array_from_out = aa.util.array_2d.numpy_array_2d_via_fits_from(
            file_path=path.join(output_data_dir, "masked_array.fits"), hdu=0
        )

        assert (masked_array_from_out == np.ones((3, 3))).all()
