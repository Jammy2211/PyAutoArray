import os

import numpy as np
import autoarray as aa


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestArray2DEuclid:
    def test__euclid_array_for_four_quandrants__loads_data_and_dimensions(
        self, euclid_data
    ):

        euclid_array = aa.euclid.Array2DEuclid.top_left(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=51,
            serial_overscan_size=29,
            parallel_overscan_size=20,
        )

        assert euclid_array.layout.original_roe_corner == (0, 0)
        assert euclid_array.shape_native == (2086, 2128)
        assert (euclid_array.native == np.zeros((2086, 2128))).all()
        assert euclid_array.layout.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_array.layout.serial_prescan == (0, 2086, 0, 51)
        assert euclid_array.layout.serial_overscan == (20, 2086, 2099, 2128)

        euclid_array = aa.euclid.Array2DEuclid.top_left(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_array.layout.original_roe_corner == (0, 0)
        assert euclid_array.shape_native == (2086, 2128)
        assert (euclid_array.native == np.zeros((2086, 2128))).all()
        assert euclid_array.layout.parallel_overscan == (2071, 2086, 41, 2118)
        assert euclid_array.layout.serial_prescan == (0, 2086, 0, 41)
        assert euclid_array.layout.serial_overscan == (15, 2086, 2118, 2128)

        euclid_array = aa.euclid.Array2DEuclid.top_right(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=51,
            serial_overscan_size=29,
            parallel_overscan_size=20,
        )

        assert euclid_array.layout.original_roe_corner == (0, 1)
        assert euclid_array.shape_native == (2086, 2128)
        assert (euclid_array.native == np.zeros((2086, 2128))).all()
        assert euclid_array.layout.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_array.layout.serial_prescan == (0, 2086, 0, 51)
        assert euclid_array.layout.serial_overscan == (20, 2086, 2099, 2128)

        euclid_array = aa.euclid.Array2DEuclid.top_right(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_array.layout.original_roe_corner == (0, 1)
        assert euclid_array.shape_native == (2086, 2128)
        assert (euclid_array.native == np.zeros((2086, 2128))).all()
        assert euclid_array.layout.parallel_overscan == (2071, 2086, 41, 2118)
        assert euclid_array.layout.serial_prescan == (0, 2086, 0, 41)
        assert euclid_array.layout.serial_overscan == (15, 2086, 2118, 2128)

        euclid_array = aa.euclid.Array2DEuclid.bottom_left(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=51,
            serial_overscan_size=29,
            parallel_overscan_size=20,
        )

        assert euclid_array.layout.original_roe_corner == (1, 0)
        assert euclid_array.shape_native == (2086, 2128)
        assert (euclid_array.native == np.zeros((2086, 2128))).all()
        assert euclid_array.layout.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_array.layout.serial_prescan == (0, 2086, 0, 51)
        assert euclid_array.layout.serial_overscan == (0, 2066, 2099, 2128)

        euclid_array = aa.euclid.Array2DEuclid.bottom_left(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_array.layout.original_roe_corner == (1, 0)
        assert euclid_array.shape_native == (2086, 2128)
        assert (euclid_array.native == np.zeros((2086, 2128))).all()
        assert euclid_array.layout.parallel_overscan == (2071, 2086, 41, 2118)
        assert euclid_array.layout.serial_prescan == (0, 2086, 0, 41)
        assert euclid_array.layout.serial_overscan == (0, 2071, 2118, 2128)

        euclid_array = aa.euclid.Array2DEuclid.bottom_right(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=51,
            serial_overscan_size=29,
            parallel_overscan_size=20,
        )

        assert euclid_array.layout.original_roe_corner == (1, 1)
        assert euclid_array.shape_native == (2086, 2128)
        assert (euclid_array.native == np.zeros((2086, 2128))).all()
        assert euclid_array.layout.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_array.layout.serial_prescan == (0, 2086, 0, 51)
        assert euclid_array.layout.serial_overscan == (0, 2066, 2099, 2128)

        euclid_array = aa.euclid.Array2DEuclid.bottom_right(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_array.layout.original_roe_corner == (1, 1)
        assert euclid_array.shape_native == (2086, 2128)
        assert (euclid_array.native == np.zeros((2086, 2128))).all()
        assert euclid_array.layout.parallel_overscan == (2071, 2086, 41, 2118)
        assert euclid_array.layout.serial_prescan == (0, 2086, 0, 41)
        assert euclid_array.layout.serial_overscan == (0, 2071, 2118, 2128)

    def test__left_side__chooses_correct_array_given_input(self, euclid_data):

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="E"
        )

        assert array.layout.original_roe_corner == (1, 0)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="E"
        )

        assert array.layout.original_roe_corner == (1, 0)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="E"
        )

        assert array.layout.original_roe_corner == (1, 0)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="F"
        )

        assert array.layout.original_roe_corner == (1, 1)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="F"
        )

        assert array.layout.original_roe_corner == (1, 1)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="F"
        )

        assert array.layout.original_roe_corner == (1, 1)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="G"
        )

        assert array.layout.original_roe_corner == (0, 1)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="G"
        )

        assert array.layout.original_roe_corner == (0, 1)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="G"
        )

        assert array.layout.original_roe_corner == (0, 1)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="H"
        )

        assert array.layout.original_roe_corner == (0, 0)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="H"
        )

        assert array.layout.original_roe_corner == (0, 0)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="H"
        )

        assert array.layout.original_roe_corner == (0, 0)

    def test__right_side__chooses_correct_array_given_input(self, euclid_data):
        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="E"
        )

        assert array.layout.original_roe_corner == (0, 1)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="E"
        )

        assert array.layout.original_roe_corner == (0, 1)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="E"
        )

        assert array.layout.original_roe_corner == (0, 1)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="F"
        )

        assert array.layout.original_roe_corner == (0, 0)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="F"
        )

        assert array.layout.original_roe_corner == (0, 0)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="F"
        )

        assert array.layout.original_roe_corner == (0, 0)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="G"
        )

        assert array.layout.original_roe_corner == (1, 0)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="G"
        )

        assert array.layout.original_roe_corner == (1, 0)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="G"
        )

        assert array.layout.original_roe_corner == (1, 0)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="H"
        )

        assert array.layout.original_roe_corner == (1, 1)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="H"
        )

        assert array.layout.original_roe_corner == (1, 1)

        array = aa.euclid.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="H"
        )

        assert array.layout.original_roe_corner == (1, 1)
