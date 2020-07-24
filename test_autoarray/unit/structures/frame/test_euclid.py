import os

import numpy as np
import autoarray as aa


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestFrameAPI:
    def test__euclid_frame_for_four_quandrants__loads_data_and_dimensions(
        self, euclid_data
    ):

        euclid_frame = aa.EuclidFrame.top_left(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2119,
            serial_prescan_size=51,
            serial_overscan_size=20,
            parallel_overscan_size=20,
        )

        assert euclid_frame.original_roe_corner == (0, 0)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (20, 2086, 2099, 2119)

        euclid_frame = aa.EuclidFrame.top_left(
            array_electrons=euclid_data,
            parallel_size=2086,
            serial_size=2119,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_frame.original_roe_corner == (0, 0)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2071, 2086, 41, 2109)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 41)
        assert euclid_frame.scans.serial_overscan == (15, 2086, 2109, 2119)

        euclid_frame = aa.EuclidFrame.top_right(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2119,
            serial_prescan_size=51,
            serial_overscan_size=20,
            parallel_overscan_size=20,
        )

        assert euclid_frame.original_roe_corner == (0, 1)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (20, 2086, 2099, 2119)

        euclid_frame = aa.EuclidFrame.top_right(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2119,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_frame.original_roe_corner == (0, 1)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2071, 2086, 41, 2109)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 41)
        assert euclid_frame.scans.serial_overscan == (15, 2086, 2109, 2119)

        euclid_frame = aa.EuclidFrame.bottom_left(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2119,
            serial_prescan_size=51,
            serial_overscan_size=20,
            parallel_overscan_size=20,
        )

        assert euclid_frame.original_roe_corner == (1, 0)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (0, 2066, 2099, 2119)

        euclid_frame = aa.EuclidFrame.bottom_left(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2119,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_frame.original_roe_corner == (1, 0)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2071, 2086, 41, 2109)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 41)
        assert euclid_frame.scans.serial_overscan == (0, 2071, 2109, 2119)

        euclid_frame = aa.EuclidFrame.bottom_right(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2119,
            serial_prescan_size=51,
            serial_overscan_size=20,
            parallel_overscan_size=20,
        )

        assert euclid_frame.original_roe_corner == (1, 1)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (0, 2066, 2099, 2119)

        euclid_frame = aa.EuclidFrame.bottom_right(
            array=euclid_data,
            parallel_size=2086,
            serial_size=2119,
            serial_prescan_size=41,
            serial_overscan_size=10,
            parallel_overscan_size=15,
        )

        assert euclid_frame.original_roe_corner == (1, 1)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2071, 2086, 41, 2109)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 41)
        assert euclid_frame.scans.serial_overscan == (0, 2071, 2109, 2119)

    def test__left_side__chooses_correct_frame_given_input(self, euclid_data):
        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

    def test__right_side__chooses_correct_frame_given_input(self, euclid_data):
        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.EuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)


class TestMaskedFrameAPI:
    def test__euclid_frame_for_four_quandrants__loads_data_and_dimensions(
        self, euclid_data
    ):

        mask = np.full(shape=(2086, 2119), fill_value=False)
        mask[0, 0] = True

        euclid_frame = aa.MaskedEuclidFrame.top_left(array=euclid_data, mask=mask)

        assert euclid_frame.original_roe_corner == (0, 0)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (20, 2086, 2099, 2119)
        assert euclid_frame.mask[2085, 0] == True
        assert euclid_frame.mask[2085, 1] == False

        euclid_frame = aa.MaskedEuclidFrame.top_right(array=euclid_data, mask=mask)

        assert euclid_frame.original_roe_corner == (0, 1)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (20, 2086, 2099, 2119)
        assert euclid_frame.mask[2085, 2118] == True
        assert euclid_frame.mask[2085, 2117] == False

        euclid_frame = aa.MaskedEuclidFrame.bottom_left(array=euclid_data, mask=mask)

        assert euclid_frame.original_roe_corner == (1, 0)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (0, 2066, 2099, 2119)
        assert euclid_frame.mask[0, 0] == True
        assert euclid_frame.mask[0, 1] == False

        euclid_frame = aa.MaskedEuclidFrame.bottom_right(array=euclid_data, mask=mask)

        assert euclid_frame.original_roe_corner == (1, 1)
        assert euclid_frame.shape_2d == (2086, 2119)
        assert (euclid_frame == np.zeros((2086, 2119))).all()
        assert euclid_frame.scans.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_frame.scans.serial_prescan == (0, 2086, 0, 51)
        assert euclid_frame.scans.serial_overscan == (0, 2066, 2099, 2119)
        assert euclid_frame.mask[0, 2118] == True
        assert euclid_frame.mask[0, 2117] == False

    def test__left_side__chooses_correct_frame_given_input(self, euclid_data):

        mask = np.full(shape=(2086, 2119), fill_value=False)
        mask[0, 0] = True

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)
        assert frame.mask[0, 0] == True
        assert frame.mask[0, 1] == False

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quadrant_id="E"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quadrant_id="F"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quadrant_id="G"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text1", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text2", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text3", quadrant_id="H"
        )

        assert frame.original_roe_corner == (0, 0)

    def test__right_side__chooses_correct_frame_given_input(self, euclid_data):

        mask = np.full(shape=(2086, 2119), fill_value=False)
        mask[0, 0] = True

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)
        assert frame.mask[2085, 2118] == True
        assert frame.mask[2085, 2117] == False

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quadrant_id="E"
        )

        assert frame.original_roe_corner == (0, 1)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quadrant_id="F"
        )

        assert frame.original_roe_corner == (0, 0)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quadrant_id="G"
        )

        assert frame.original_roe_corner == (1, 0)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text4", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text5", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)

        frame = aa.MaskedEuclidFrame.from_ccd_and_quadrant_id(
            array=euclid_data, mask=mask, ccd_id="text6", quadrant_id="H"
        )

        assert frame.original_roe_corner == (1, 1)
