import numpy as np
import autoarray as aa

from astropy.io import fits
import copy
import shutil
import os
from os import path
import pytest

acs_path = "{}".format(path.dirname(path.realpath(__file__)))


def create_acs_fits(fits_path, acs_ccd, acs_ccd_0, acs_ccd_1, units):

    if path.exists(fits_path):
        shutil.rmtree(fits_path)

    os.makedirs(fits_path)

    new_hdul = fits.HDUList()

    new_hdul.append(fits.ImageHDU(acs_ccd))
    new_hdul.append(fits.ImageHDU(acs_ccd_0))
    new_hdul.append(fits.ImageHDU(acs_ccd))
    new_hdul.append(fits.ImageHDU(acs_ccd))
    new_hdul.append(fits.ImageHDU(acs_ccd_1))
    new_hdul.append(fits.ImageHDU(acs_ccd))

    new_hdul[0].header.set("EXPTIME", 1000.0, "exposure duration (seconds)--calculated")
    new_hdul[0].header.set(
        "DATE-OBS", "2000-01-01", "UT date of start of observation (yyyy-mm-dd)"
    )
    new_hdul[0].header.set(
        "TIME-OBS", "00:00:00", "UT time of start of observation (hh:mm:ss)"
    )

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
    new_hdul[4].header.set(
        "BSCALE", 2.0, "scale factor for array value to physical value"
    )
    new_hdul[4].header.set("BZERO", 10.0, "physical value for an array value of zero")

    new_hdul.writeto(path.join(fits_path, "acs_ccd.fits"))


def create_acs_bias_fits(fits_path, bias_ccd, bias_ccd_0, bias_ccd_1):


    new_hdul = fits.HDUList()

    new_hdul.append(fits.ImageHDU(bias_ccd))
    new_hdul.append(fits.ImageHDU(bias_ccd_0))
    new_hdul.append(fits.ImageHDU(bias_ccd))
    new_hdul.append(fits.ImageHDU(bias_ccd))
    new_hdul.append(fits.ImageHDU(bias_ccd_1))
    new_hdul.append(fits.ImageHDU(bias_ccd))

    new_hdul.writeto(path.join(fits_path, "acs_bias_ccd.fits"))


class TestArray2DACS:
    def test__acs_array_for_left_and_right_quandrants__loads_data_and_dimensions(
        self, acs_quadrant
    ):

        acs_array = aa.acs.Array2DACS.quadrant_a(array_electrons=acs_quadrant)

        assert acs_array.shape_native == (2068, 2072)
        assert (acs_array.native == np.zeros((2068, 2072))).all()

        acs_array = aa.acs.Array2DACS.quadrant_d(array_electrons=acs_quadrant)

        assert acs_array.shape_native == (2068, 2072)
        assert (acs_array.native == np.zeros((2068, 2072))).all()

    def test__from_ccd__chooses_correct_array_given_quadrant_letter(self, acs_ccd):

        array = aa.acs.Array2DACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="B")

        assert array.shape_native == (2068, 2072)

        array = aa.acs.Array2DACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="C")

        assert array.shape_native == (2068, 2072)

        array = aa.acs.Array2DACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="A")

        assert array.shape_native == (2068, 2072)

        array = aa.acs.Array2DACS.from_ccd(array_electrons=acs_ccd, quadrant_letter="D")

        assert array.shape_native == (2068, 2072)

    def test__conversions_to_counts_and_counts_per_second_use_correct_values(self):

        array = aa.Array2D.ones(
            shape_native=(3, 3),
            pixel_scales=1.0,
            exposure_info=aa.acs.ExposureInfoACS(
                bscale=1.0, bzero=0.0, exposure_time=1.0
            ),
        )

        assert (array.in_counts.native == np.ones(shape=(3, 3))).all()
        assert (array.in_counts_per_second.native == np.ones(shape=(3, 3))).all()

        array = aa.Array2D.ones(
            shape_native=(3, 3),
            pixel_scales=1.0,
            exposure_info=aa.acs.ExposureInfoACS(
                bscale=2.0, bzero=0.0, exposure_time=1.0
            ),
        )

        assert (array.in_counts.native == 0.5 * np.ones(shape=(3, 3))).all()
        assert (array.in_counts_per_second.native == 0.5 * np.ones(shape=(3, 3))).all()

        array = aa.Array2D.ones(
            shape_native=(3, 3),
            pixel_scales=1.0,
            exposure_info=aa.acs.ExposureInfoACS(
                bscale=2.0, bzero=0.1, exposure_time=1.0
            ),
        )

        assert (array.in_counts.native == 0.45 * np.ones(shape=(3, 3))).all()
        assert (array.in_counts_per_second.native == 0.45 * np.ones(shape=(3, 3))).all()

        array = aa.Array2D.ones(
            shape_native=(3, 3),
            pixel_scales=1.0,
            exposure_info=aa.acs.ExposureInfoACS(
                bscale=2.0, bzero=0.1, exposure_time=2.0
            ),
        )

        assert (array.in_counts.native == 0.45 * np.ones(shape=(3, 3))).all()
        assert (
            array.in_counts_per_second.native == 0.225 * np.ones(shape=(3, 3))
        ).all()


class TestImageACS:
    def test__from_fits__reads_exposure_info_from_header_correctly(self, acs_ccd):

        fits_path = path.join(
            "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
        )

        file_path = path.join(fits_path, "acs_ccd.fits")

        create_acs_fits(
            fits_path=fits_path,
            acs_ccd=acs_ccd,
            acs_ccd_0=acs_ccd,
            acs_ccd_1=acs_ccd,
            units="COUNTS",
        )

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="B")

        assert array.exposure_info.exposure_time == 1000.0
        assert array.exposure_info.date_of_observation == "2000-01-01"
        assert array.exposure_info.time_of_observation == "00:00:00"
        assert array.exposure_info.modified_julian_date == 51544.0

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="C")

        assert array.exposure_info.exposure_time == 1000.0
        assert array.exposure_info.date_of_observation == "2000-01-01"
        assert array.exposure_info.time_of_observation == "00:00:00"
        assert array.exposure_info.modified_julian_date == 51544.0

    def test__from_fits__in_counts__uses_fits_header_correctly_converts_and_picks_correct_quadrant(
        self, acs_ccd
    ):

        fits_path = path.join(
            "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
        )

        file_path = path.join(fits_path, "acs_ccd.fits")

        acs_ccd_0 = copy.copy(acs_ccd)
        acs_ccd_0[0, 0] = 10.0
        acs_ccd_0[0, -1] = 20.0

        acs_ccd_1 = copy.copy(acs_ccd)
        acs_ccd_1[-1, 0] = 30.0
        acs_ccd_1[-1, -1] = 40.0

        create_acs_fits(
            fits_path=fits_path,
            acs_ccd=acs_ccd,
            acs_ccd_0=acs_ccd_0,
            acs_ccd_1=acs_ccd_1,
            units="COUNTS",
        )

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="A")

        assert array.native[0, 0] == (30.0 * 2.0) + 10.0
        assert array.in_counts.native[0, 0] == 30.0
        assert array.shape_native == (2068, 2072)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="B")

        assert array.native[0, 0] == (40.0 * 2.0) + 10.0
        assert array.shape_native == (2068, 2072)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="C")

        assert array.native[0, 0] == (10.0 * 2.0) + 10.0
        assert array.shape_native == (2068, 2072)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="D")

        assert array.native[0, 0] == (20.0 * 2.0) + 10.0
        assert array.shape_native == (2068, 2072)

    def test__from_fits__in_counts_per_second__uses_fits_header_correctly_converts_and_picks_correct_quadrant(
        self, acs_ccd
    ):

        fits_path = path.join(
            "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
        )

        file_path = path.join(fits_path, "acs_ccd.fits")

        acs_ccd_0 = copy.copy(acs_ccd)
        acs_ccd_0[0, 0] = 10.0
        acs_ccd_0[0, -1] = 20.0

        acs_ccd_1 = copy.copy(acs_ccd)
        acs_ccd_1[-1, 0] = 30.0
        acs_ccd_1[-1, -1] = 40.0

        create_acs_fits(
            fits_path=fits_path,
            acs_ccd=acs_ccd,
            acs_ccd_0=acs_ccd_0,
            acs_ccd_1=acs_ccd_1,
            units="CPS",
        )

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="A")

        assert array.native[0, 0] == (30.0 * 1000.0 * 2.0) + 10.0
        assert array.shape_native == (2068, 2072)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="B")

        assert array.native[0, 0] == (40.0 * 1000.0 * 2.0) + 10.0
        assert array.shape_native == (2068, 2072)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="C")

        assert array.native[0, 0] == (10.0 * 1000.0 * 2.0) + 10.0
        assert array.shape_native == (2068, 2072)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="D")

        assert array.native[0, 0] == (20.0 * 1000.0 * 2.0) + 10.0
        assert array.shape_native == (2068, 2072)

    def test__from_fits__in_counts__uses_bias_prescan_correctly(
        self, acs_ccd
    ):

        fits_path = path.join(
            "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
        )

        file_path = path.join(fits_path, "acs_ccd.fits")

        acs_ccd_0 = copy.copy(acs_ccd)
        acs_ccd_0[0, 0] = 10.0
        acs_ccd_0[0, -1] = 20.0

        acs_ccd_1 = copy.copy(acs_ccd)
        acs_ccd_1[-1, 0] = 30.0
        acs_ccd_1[-1, -1] = 40.0

        create_acs_fits(
            fits_path=fits_path,
            acs_ccd=acs_ccd,
            acs_ccd_0=acs_ccd_0,
            acs_ccd_1=acs_ccd_1,
            units="COUNTS",
        )

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="A", bias_subtract_via_prescan=True)

        assert array.native[0, 0] == pytest.approx(10.0, (30.0 * 2.0) + 10.0 - 10.0, 1.0e-4)
        assert array.exposure_info.bias_serial_prescan_column[0][0] == pytest.approx(10.0, 1.0e-4)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="B", bias_subtract_via_prescan=True)

        assert array.native[0, 0] == pytest.approx(10.0, (40.0 * 2.0) + 10.0 - 10.0, 1.0e-4)
        assert array.exposure_info.bias_serial_prescan_column[0][0] == pytest.approx(10.0, 1.0e-4)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="C", bias_subtract_via_prescan=True)

        assert array.native[0, 0] == pytest.approx(10.0, (10.0 * 2.0) + 10.0 - 10.0, 1.0e-4)
        assert array.exposure_info.bias_serial_prescan_column[0][0] == pytest.approx(10.0, 1.0e-4)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="D", bias_subtract_via_prescan=True)

        assert array.native[0, 0] == pytest.approx(10.0, (20.0 * 2.0) + 10.0 - 10.0, 1.0e-4)
        assert array.exposure_info.bias_serial_prescan_column[0][0] == pytest.approx(10.0, 1.0e-4)

    def test__from_fits__in_counts__uses_bias_file_subtraction_correctly(
        self, acs_ccd
    ):

        fits_path = path.join(
            "{}".format(path.dirname(path.realpath(__file__))), "files", "acs"
        )

        acs_ccd_0 = copy.copy(acs_ccd)
        acs_ccd_0[0, 0] = 10.0
        acs_ccd_0[0, -1] = 20.0

        acs_ccd_1 = copy.copy(acs_ccd)
        acs_ccd_1[-1, 0] = 30.0
        acs_ccd_1[-1, -1] = 40.0

        create_acs_fits(
            fits_path=fits_path,
            acs_ccd=acs_ccd,
            acs_ccd_0=acs_ccd_0,
            acs_ccd_1=acs_ccd_1,
            units="COUNTS",
        )

        create_acs_bias_fits(
            fits_path=fits_path,
            bias_ccd=np.zeros((2068, 4144)),
            bias_ccd_0=np.ones((2068, 4144)),
            bias_ccd_1=2.0*np.ones((2068, 4144)),
        )

        file_path = path.join(fits_path, "acs_ccd.fits")
        bias_path = path.join(fits_path, "acs_bias_ccd.fits")

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="A", bias_path=bias_path)


        assert array.native[0, 0] == pytest.approx(10.0, (30.0 * 2.0) + 10.0 - 2.0, 1.0e-4)
        assert array.exposure_info.bias[0][0] == pytest.approx(2.0, 1.0e-4)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="B", bias_path=bias_path)

        assert array.native[0, 0] == pytest.approx(10.0, (40.0 * 2.0) + 10.0 - 2.0, 1.0e-4)
        assert array.exposure_info.bias[0][0] == pytest.approx(2.0, 1.0e-4)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="C", bias_path=bias_path)

        assert array.native[0, 0] == pytest.approx(10.0, (10.0 * 2.0) + 10.0 - 1.0, 1.0e-4)
        assert array.exposure_info.bias[0][0] == pytest.approx(1.0, 1.0e-4)

        array = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="D", bias_path=bias_path)

        assert array.native[0, 0] == pytest.approx(10.0, (20.0 * 2.0) + 10.0 - 1.0, 1.0e-4)
        assert array.exposure_info.bias[0][0] == pytest.approx(1.0, 1.0e-4)

    # def test__update_fits__if_new_file_is_not_presnet_copies_original_file_and_updates(
    #     self, acs_ccd
    # ):
    #
    #     fits_path = "{}/files/acs".format(path.dirname(path.realpath(__file__)))
    #
    #     create_acs_fits(
    #         fits_path=fits_path,
    #         acs_ccd=acs_ccd,
    #         acs_ccd_0=acs_ccd,
    #         acs_ccd_1=acs_ccd,
    #         units="COUNTS",
    #     )
    #
    #     hdulist = fits.open(f"{fits_path}/acs_ccd.fits")
    #     print(hdulist[4].header)
    #     ext_header = hdulist[4].header
    #     bscale = ext_header["BSCALE"]
    #     print(bscale)
    #
    #     array = aa.acs.Array2DACS.from_fits(
    #         file_path=f"{fits_path}/acs_ccd.fits", quadrant_letter="B"
    #     )
    #
    #     array[0, 0] = 101.0
    #
    #     array.update_fits(
    #         original_file_path=f"{fits_path}/acs_ccd.fits",
    #         new_file_path=f"{fits_path}/acs_ccd_new.fits",
    #     )
    #
    #     hdulist = fits.open(f"{fits_path}/acs_ccd_new.fits")
    #     print(hdulist[4].header)
    #     ext_header = hdulist[4].header
    #     bscale = ext_header["BSCALE"]
    #     print(bscale)
    #     stop
    #
    #     array = aa.acs.Array2DACS.from_fits(
    #         file_path=f"{fits_path}/acs_ccd_new.fits", quadrant_letter="B"
    #     )
    #
    #     print(array)


class TestLayout2DACS:
    def test__acs_layout_for_left_and_right_quandrants__loads_data_and_dimensions(
        self, acs_quadrant
    ):

        layout = aa.acs.Layout2DACS.from_sizes(
            roe_corner=(1, 0),
            parallel_size=2068,
            serial_size=2072,
            serial_prescan_size=24,
            parallel_overscan_size=20,
        )

        assert layout.original_roe_corner == (1, 0)
        assert layout.shape_2d == (2068, 2072)
        assert layout.parallel_overscan == (2048, 2068, 24, 2072)
        assert layout.serial_prescan == (0, 2068, 0, 24)