import numpy as np
import autoarray as aa

from astropy.io import fits
import copy
import shutil
import os

path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


def create_acs_fits(fits_path, acs_ccd, acs_ccd_0, acs_ccd_1, units):

    if os.path.exists(fits_path):
        shutil.rmtree(fits_path)

    os.makedirs(fits_path)

    new_hdul = fits.HDUList()

    new_hdul.append(fits.ImageHDU(acs_ccd))
    new_hdul.append(fits.ImageHDU(acs_ccd_0))
    new_hdul.append(fits.ImageHDU(acs_ccd))
    new_hdul.append(fits.ImageHDU(acs_ccd))
    new_hdul.append(fits.ImageHDU(acs_ccd_1))
    new_hdul.append(fits.ImageHDU(acs_ccd))

    if units in "COUNTS":
        new_hdul[1].header.set("BUNIT", "COUNTS", "brightness units")
        new_hdul[4].header.set("BUNIT", "COUNTS", "brightness units")
    elif units in "CPS":
        new_hdul[1].header.set("BUNIT", "CPS", "brightness units")
        new_hdul[4].header.set("BUNIT", "CPS", "brightness units")

    new_hdul[1].header.set(
        "BSCALE", 2.0, "scale factor for array value to physical value"
    )
    new_hdul[1].header.set("BZERO", 10.0, "physical value for an array value of zero")
    new_hdul[1].header.set("EXPTIME", 1000.0, "exposure duration (seconds)--calculated")
    new_hdul[4].header.set(
        "BSCALE", 2.0, "scale factor for array value to physical value"
    )
    new_hdul[4].header.set("BZERO", 10.0, "physical value for an array value of zero")
    new_hdul[4].header.set("EXPTIME", 2000.0, "exposure duration (seconds)--calculated")

    new_hdul.writeto(f"{fits_path}/acs_ccd.fits")


class TestFrameAPI:
    def test__acs_frame_for_left_and_right_quandrants__loads_data_and_dimensions(
        self, acs_quadrant
    ):

        acs_frame = aa.FrameACS.left(
            array_electrons=acs_quadrant,
            parallel_size=2068,
            serial_size=2072,
            serial_prescan_size=24,
            parallel_overscan_size=20,
        )

        assert acs_frame.original_roe_corner == (1, 0)
        assert acs_frame.shape_2d == (2068, 2072)
        assert (acs_frame == np.zeros((2068, 2072))).all()
        assert acs_frame.scans.parallel_overscan == (2048, 2068, 24, 2072)
        assert acs_frame.scans.serial_prescan == (0, 2068, 0, 24)

        acs_frame = aa.FrameACS.left(
            array_electrons=acs_quadrant,
            parallel_size=2070,
            serial_size=2072,
            serial_prescan_size=28,
            parallel_overscan_size=10,
        )

        assert acs_frame.original_roe_corner == (1, 0)
        assert acs_frame.shape_2d == (2068, 2072)
        assert (acs_frame == np.zeros((2068, 2072))).all()
        assert acs_frame.scans.parallel_overscan == (2060, 2070, 28, 2072)
        assert acs_frame.scans.serial_prescan == (0, 2070, 0, 28)

        acs_frame = aa.FrameACS.right(
            array=acs_quadrant,
            parallel_size=2068,
            serial_size=2072,
            serial_prescan_size=24,
            parallel_overscan_size=20,
        )

        assert acs_frame.original_roe_corner == (1, 1)
        assert acs_frame.shape_2d == (2068, 2072)
        assert (acs_frame == np.zeros((2068, 2072))).all()
        assert acs_frame.scans.parallel_overscan == (2048, 2068, 24, 2072)
        assert acs_frame.scans.serial_prescan == (0, 2068, 0, 24)

        acs_frame = aa.FrameACS.right(
            array=acs_quadrant,
            parallel_size=2070,
            serial_size=2072,
            serial_prescan_size=28,
            parallel_overscan_size=10,
        )

        assert acs_frame.original_roe_corner == (1, 1)
        assert acs_frame.shape_2d == (2068, 2072)
        assert (acs_frame == np.zeros((2068, 2072))).all()
        assert acs_frame.scans.parallel_overscan == (2060, 2070, 28, 2072)
        assert acs_frame.scans.serial_prescan == (0, 2070, 0, 28)

    def test__from_ccd__chooses_correct_frame_given_quadrant_letter(self, acs_ccd):

        frame = aa.FrameACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="B")

        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = aa.FrameACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="C")

        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = aa.FrameACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="A")

        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)

        frame = aa.FrameACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="D")

        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)

    def test__from_fits__in_counts__uses_fits_header_correctly_converts_and_picks_correct_quadrant(
        self, acs_ccd
    ):

        fits_path = "{}/files/acs/".format(os.path.dirname(os.path.realpath(__file__)))

        acs_ccd_0 = copy.copy(acs_ccd)
        acs_ccd_0[0, 0] = 10.0
        acs_ccd_0[0, 4143] = 20.0

        acs_ccd_1 = copy.copy(acs_ccd)
        acs_ccd_1[0, 0] = 30.0
        acs_ccd_1[0, 4143] = 40.0

        create_acs_fits(
            fits_path=fits_path,
            acs_ccd=acs_ccd,
            acs_ccd_0=acs_ccd_0,
            acs_ccd_1=acs_ccd_1,
            units="COUNTS",
        )

        frame = aa.FrameACS.from_fits(
            file_path=f"{fits_path}/acs_ccd.fits", quadrant_letter="B"
        )

        assert frame[0, 0] == (10.0 * 2.0) + 10.0
        assert frame.in_counts[0, 0] == 10.0
        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = aa.FrameACS.from_fits(
            file_path=f"{fits_path}/acs_ccd.fits", quadrant_letter="A"
        )

        assert frame[0, 0] == (20.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)

        frame = aa.FrameACS.from_fits(
            file_path=f"{fits_path}/acs_ccd.fits", quadrant_letter="C"
        )

        assert frame[0, 0] == (30.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = aa.FrameACS.from_fits(
            file_path=f"{fits_path}/acs_ccd.fits", quadrant_letter="D"
        )

        assert frame[0, 0] == (40.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)

    def test__from_fits__in_counts_per_second__uses_fits_header_correctly_converts_and_picks_correct_quadrant(
        self, acs_ccd
    ):

        fits_path = "{}/files/acs/".format(os.path.dirname(os.path.realpath(__file__)))

        acs_ccd_0 = copy.copy(acs_ccd)
        acs_ccd_0[0, 0] = 10.0
        acs_ccd_0[0, 4143] = 20.0

        acs_ccd_1 = copy.copy(acs_ccd)
        acs_ccd_1[0, 0] = 30.0
        acs_ccd_1[0, 4143] = 40.0

        create_acs_fits(
            fits_path=fits_path,
            acs_ccd=acs_ccd,
            acs_ccd_0=acs_ccd_0,
            acs_ccd_1=acs_ccd_1,
            units="CPS",
        )

        frame = aa.FrameACS.from_fits(
            file_path=f"{fits_path}/acs_ccd.fits", quadrant_letter="B"
        )

        assert frame[0, 0] == (10.0 * 1000.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = aa.FrameACS.from_fits(
            file_path=f"{fits_path}/acs_ccd.fits", quadrant_letter="A"
        )

        assert frame[0, 0] == (20.0 * 1000.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)

        frame = aa.FrameACS.from_fits(
            file_path=f"{fits_path}/acs_ccd.fits", quadrant_letter="C"
        )

        assert frame[0, 0] == (30.0 * 2000.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 0)
        assert frame.shape_2d == (2068, 2072)

        frame = aa.FrameACS.from_fits(
            file_path=f"{fits_path}/acs_ccd.fits", quadrant_letter="D"
        )

        assert frame[0, 0] == (40.0 * 2000.0 * 2.0) + 10.0
        assert frame.original_roe_corner == (1, 1)
        assert frame.shape_2d == (2068, 2072)
