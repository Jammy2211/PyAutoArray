import os
from os import path

import numpy as np
import pytest

import autoarray as aa
from autoconf import conf
from autoarray import exc

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

    def test__manual_native__exception_raised_if_input_array_is_2d_and_not_sub_shape_of_mask(
        self,
    ):
        with pytest.raises(exc.ArrayException):
            mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)
            aa.Array2D.manual_mask(array=[[1.0], [3.0]], mask=mask)

        with pytest.raises(exc.ArrayException):
            mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0, sub_size=2)
            aa.Array2D.manual_mask(array=[[1.0, 2.0], [3.0, 4.0]], mask=mask)

        with pytest.raises(exc.ArrayException):
            mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0, sub_size=2)
            aa.Array2D.manual_mask(
                array=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], mask=mask
            )

    def test__exception_raised_if_input_array_is_1d_and_not_number_of_masked_sub_pixels(
        self,
    ):
        with pytest.raises(exc.ArrayException):
            mask = aa.Mask2D.manual(
                mask=[[False, False], [True, False]], pixel_scales=1.0, sub_size=1
            )
            aa.Array2D.manual_mask(array=[1.0, 2.0, 3.0, 4.0], mask=mask)

        with pytest.raises(exc.ArrayException):
            mask = aa.Mask2D.manual(
                mask=[[False, False], [True, False]], pixel_scales=1.0, sub_size=1
            )
            aa.Array2D.manual_mask(array=[1.0, 2.0], mask=mask)

        with pytest.raises(exc.ArrayException):
            mask = aa.Mask2D.manual(
                mask=[[False, True], [True, True]], pixel_scales=1.0, sub_size=2
            )
            aa.Array2D.manual_mask(array=[1.0, 2.0, 4.0], mask=mask)

        with pytest.raises(exc.ArrayException):
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

    def test__from_fits__makes_scaled_array_with_pixel_scale(self):

        arr = aa.Array2D.from_fits(
            file_path=path.join(test_data_dir, "3x3_ones.fits"), hdu=0, pixel_scales=1.0
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == np.ones((3, 3))).all()
        assert (arr.slim == np.ones(9)).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 0.0)

        arr = aa.Array2D.from_fits(
            file_path=path.join(test_data_dir, "4x3_ones.fits"),
            hdu=0,
            pixel_scales=1.0,
            origin=(0.0, 1.0),
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == np.ones((4, 3))).all()
        assert (arr.slim == np.ones((12,))).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)

    def test__from_fits__makes_scaled_sub_array_with_pixel_scale_and_sub_size(self):

        arr = aa.Array2D.from_fits(
            file_path=path.join(test_data_dir, "3x3_ones.fits"),
            hdu=0,
            pixel_scales=1.0,
            sub_size=1,
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == np.ones((3, 3))).all()
        assert (arr.slim == np.ones(9)).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 0.0)
        assert arr.mask.sub_size == 1

        arr = aa.Array2D.from_fits(
            file_path=path.join(test_data_dir, "4x3_ones.fits"),
            hdu=0,
            pixel_scales=1.0,
            sub_size=1,
            origin=(0.0, 1.0),
        )

        assert type(arr) == aa.Array2D
        assert (arr.native == np.ones((4, 3))).all()
        assert (arr.slim == np.ones(12)).all()
        assert arr.pixel_scales == (1.0, 1.0)
        assert arr.origin == (0.0, 1.0)
        assert arr.mask.sub_size == 1

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

    def test__manual__store_slim_is_false_so_stores_in_native(self):

        conf.instance["general"]["structures"]["store_slim"] = False

        arr = aa.Array2D.manual_native(
            array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0, sub_size=1
        )

        assert type(arr) == aa.Array2D
        assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
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
        assert (arr == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
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
        assert (arr == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
        assert (
            arr.native == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (arr.slim == np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])).all()
        assert (arr.binned.native == np.array([[2.5], [6.5]])).all()
        assert (arr.binned == np.array([2.5, 6.5])).all()
        assert arr.pixel_scales == (2.0, 2.0)
        assert arr.origin == (0.0, 0.0)
        assert arr.mask.sub_size == 2

        conf.instance["general"]["structures"]["store_slim"] = True
