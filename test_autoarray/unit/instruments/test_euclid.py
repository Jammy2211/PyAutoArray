import os

import numpy as np
import autoarray as aa


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestFrameEuclid:
    def test__euclid_frame_for_four_quandrants__loads_data_and_dimensions(
        self, euclid_data
    ):

        euclid_frame = aa.euclid.FrameEuclid.top_left(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=51,
            serial_overscan_size=29,
            parallel_overscan_size=20,
        )

        assert euclid_frame.original_roe_corner == (0, 0)
        assert euclid_frame.shape_2d == (2086, 2128)
        assert (euclid_frame == np.zeros((2086, 2128))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (20, 2086, 2099, 2128)

        euclid_frame = aa.euclid.FrameEuclid.top_left(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_frame.original_roe_corner == (0, 0)
        assert euclid_frame.shape_2d == (2086, 2128)
        assert (euclid_frame == np.zeros((2086, 2128))).all()
        assert euclid_frame.scans.parallel_overscan == (2071, 2086, 41, 2118)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 41)
        assert euclid_frame.scans.serial_overscan == (15, 2086, 2118, 2128)

        euclid_frame = aa.euclid.FrameEuclid.top_right(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=51,
            serial_overscan_size=29,
            parallel_overscan_size=20,
        )

        assert euclid_frame.original_roe_corner == (0, 1)
        assert euclid_frame.shape_2d == (2086, 2128)
        assert (euclid_frame == np.zeros((2086, 2128))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (20, 2086, 2099, 2128)

        euclid_frame = aa.euclid.FrameEuclid.top_right(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_frame.original_roe_corner == (0, 1)
        assert euclid_frame.shape_2d == (2086, 2128)
        assert (euclid_frame == np.zeros((2086, 2128))).all()
        assert euclid_frame.scans.parallel_overscan == (2071, 2086, 41, 2118)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 41)
        assert euclid_frame.scans.serial_overscan == (15, 2086, 2118, 2128)

        euclid_frame = aa.euclid.FrameEuclid.bottom_left(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=51,
            serial_overscan_size=29,
            parallel_overscan_size=20,
        )

        assert euclid_frame.original_roe_corner == (1, 0)
        assert euclid_frame.shape_2d == (2086, 2128)
        assert (euclid_frame == np.zeros((2086, 2128))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (0, 2066, 2099, 2128)

        euclid_frame = aa.euclid.FrameEuclid.bottom_left(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_frame.original_roe_corner == (1, 0)
        assert euclid_frame.shape_2d == (2086, 2128)
        assert (euclid_frame == np.zeros((2086, 2128))).all()
        assert euclid_frame.scans.parallel_overscan == (2071, 2086, 41, 2118)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 41)
        assert euclid_frame.scans.serial_overscan == (0, 2071, 2118, 2128)

        euclid_frame = aa.euclid.FrameEuclid.bottom_right(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=51,
            serial_overscan_size=29,
            parallel_overscan_size=20,
        )

        assert euclid_frame.original_roe_corner == (1, 1)
        assert euclid_frame.shape_2d == (2086, 2128)
        assert (euclid_frame == np.zeros((2086, 2128))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (0, 2066, 2099, 2128)

        euclid_frame = aa.euclid.FrameEuclid.bottom_right(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2128,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_frame.original_roe_corner == (1, 1)
        assert euclid_frame.shape_2d == (2086, 2128)
        assert (euclid_frame == np.zeros((2086, 2128))).all()
        assert euclid_frame.scans.parallel_overscan == (2071, 2086, 41, 2118)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 41)
        assert euclid_frame.scans.serial_overscan == (0, 2071, 2118, 2128)

    def test__left_side__chooses_correct_frame_given_input(self, euclid_data):
        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

    def test__right_side__chooses_correct_frame_given_input(self, euclid_data):
        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.euclid.FrameEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)
