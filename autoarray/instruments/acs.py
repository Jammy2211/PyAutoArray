from astropy.io import fits
import copy
import logging
import numpy as np
import os
from os import path
import shutil

from autoarray.structures.header import Header
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.layout.layout import Layout2D
from autoarray.layout.region import Region2D

from autoarray import exc
from autoarray.structures.arrays import array_2d_util
from autoarray.layout import layout_util

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")


def fits_hdu_via_quadrant_letter_from(quadrant_letter):

    if quadrant_letter == "D" or quadrant_letter == "C":
        return 1
    elif quadrant_letter == "B" or quadrant_letter == "A":
        return 4
    else:
        raise exc.ArrayException("Quadrant letter for FrameACS must be A, B, C or D.")


def array_eps_to_counts(array_eps, bscale, bzero):

    if bscale is None:
        raise exc.ArrayException(
            "Cannot convert a Frame2D to units COUNTS without a bscale attribute (bscale = None)."
        )

    return (array_eps - bzero) / bscale


class Array2DACS(Array2D):
    """
    An ACS array consists of four quadrants ('A', 'B', 'C', 'D') which have the following layout (which are described
    at the following STScI 
    link https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418).

       <--------S-----------   ---------S----------->
    [] [========= 2 =========] [========= 3 =========] []          /\
    /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /        |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
    P   [xxxxxxxxx B/C xxxxxxx] [xxxxxxxxx A/D xxxxxxx]  P         | clocks an image
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
    \/  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  \/        | (e.g. towards row 0
                                                                   | of the NumPy arrays)

    For a ACS .fits file:

    - The images contained in hdu 1 correspond to quadrants B (left) and A (right).
    - The images contained in hdu 4 correspond to quadrants C (left) and D (right).
    """

    @classmethod
    def from_fits(cls, file_path, quadrant_letter):
        """
        Use the input .fits file and quadrant letter to extract the quadrant from the full CCD, perform
        the rotations required to give correct arctic clocking and convert the image from units of COUNTS / CPS to
        ELECTRONS.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        hdu = fits_hdu_via_quadrant_letter_from(quadrant_letter=quadrant_letter)

        array = array_2d_util.numpy_array_2d_via_fits_from(file_path=file_path, hdu=hdu)

        return cls.from_ccd(array_electrons=array, quadrant_letter=quadrant_letter)

    @classmethod
    def from_ccd(
        cls,
        array_electrons,
        quadrant_letter,
        header=None,
        bias_subtract_via_prescan=False,
        bias=None,
    ):
        """
        Using an input array of both quadrants in electrons, use the quadrant letter to extract the quadrant from the
        full CCD and perform the rotations required to give correct arctic.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """
        if quadrant_letter == "A":

            array_electrons = array_electrons[0:2068, 0:2072]
            roe_corner = (1, 0)
            use_flipud = True

            if bias is not None:
                bias = bias[0:2068, 0:2072]

        elif quadrant_letter == "B":

            array_electrons = array_electrons[0:2068, 2072:4144]
            roe_corner = (1, 1)
            use_flipud = True

            if bias is not None:
                bias = bias[0:2068, 2072:4144]

        elif quadrant_letter == "C":

            array_electrons = array_electrons[0:2068, 0:2072]

            roe_corner = (1, 0)
            use_flipud = False

            if bias is not None:
                bias = bias[0:2068, 0:2072]

        elif quadrant_letter == "D":

            array_electrons = array_electrons[0:2068, 2072:4144]

            roe_corner = (1, 1)
            use_flipud = False

            if bias is not None:
                bias = bias[0:2068, 2072:4144]

        else:
            raise exc.ArrayException(
                "Quadrant letter for FrameACS must be A, B, C or D."
            )

        return cls.quadrant_a(
            array_electrons=array_electrons,
            header=header,
            roe_corner=roe_corner,
            use_flipud=use_flipud,
            bias_subtract_via_prescan=bias_subtract_via_prescan,
            bias=bias,
        )

    @classmethod
    def quadrant_a(
        cls,
        array_electrons,
        roe_corner,
        use_flipud,
        header=None,
        bias_subtract_via_prescan=False,
        bias=None,
    ):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=roe_corner
        )

        if use_flipud:
            array_electrons = np.flipud(array_electrons)

        if bias_subtract_via_prescan:

            bias_serial_prescan_value = prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:

            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=roe_corner
            )

            if use_flipud:
                bias = np.flipud(bias)

            array_electrons -= bias

            header.bias = Array2DACS.manual_native(array=bias, pixel_scales=0.05)

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_b(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the right quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 1)
        )

        array_electrons = np.flipud(array_electrons)

        if bias_subtract_via_prescan:

            bias_serial_prescan_value = prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:

            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=(1, 1)
            )

            bias = np.flipud(bias)

            array_electrons -= bias

            header.bias = Array2DACS.manual_native(array=bias, pixel_scales=0.05)

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_c(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 0)
        )

        if bias_subtract_via_prescan:

            bias_serial_prescan_value = prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:

            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=(1, 0)
            )

            array_electrons -= bias

            header.bias = Array2DACS.manual_native(array=bias, pixel_scales=0.05)

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_d(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the right quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418
        """

        array_electrons = layout_util.rotate_array_via_roe_corner_from(
            array=array_electrons, roe_corner=(1, 1)
        )

        if bias_subtract_via_prescan:
            bias_serial_prescan_value = prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:

            bias = layout_util.rotate_array_via_roe_corner_from(
                array=bias, roe_corner=(1, 1)
            )

            array_electrons -= bias

            header.bias = Array2DACS.manual_native(array=bias, pixel_scales=0.05)

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    def update_fits(self, original_file_path, new_file_path):
        """
        Output the array to a .fits file.

        Parameters
        ----------
        file_path
            The path the file is output to, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        """

        new_file_dir = os.path.split(new_file_path)[0]

        if not os.path.exists(new_file_dir):

            os.makedirs(new_file_dir)

        if not os.path.exists(new_file_path):

            shutil.copy(original_file_path, new_file_path)

        hdulist = fits.open(new_file_path)

        hdulist[self.header.hdu].data = self.layout_2d.original_orientation_from(
            array=self
        )

        ext_header = hdulist[4].header
        bscale = ext_header["BSCALE"]

        os.remove(new_file_path)

        hdulist.writeto(new_file_path)


class ImageACS(Array2DACS):
    """
    The layout of an ACS array and image is given in `FrameACS`.

    This class handles specifically the image of an ACS observation, assuming that it contains specific
    header info.
    """

    @classmethod
    def from_fits(
        cls,
        file_path,
        quadrant_letter,
        bias_subtract_via_bias_file=False,
        bias_subtract_via_prescan=False,
        bias_file_path=None,
        use_calibrated_gain=True,
    ):
        """
        Use the input .fits file and quadrant letter to extract the quadrant from the full CCD, perform
        the rotations required to give correct arctic clocking and convert the image from units of COUNTS / CPS to
        ELECTRONS.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.

        Also see https://github.com/spacetelescope/hstcal/blob/master/pkg/acs/calacs/acscte/dopcte-gen2.c#L418

        Parameters
        ----------
        file_path
            The full path of the file that the image is loaded from, including the file name and ``.fits`` extension.
        quadrant_letter
            The letter of the ACS quadrant the image is extracted from and loaded.
        bias_subtract_via_bias_file
            If True, the corresponding bias file of the image is loaded (via the name of the file in the fits header).
        bias_subtract_via_prescan
            If True, the prescan on the image is used to estimate a component of bias that is subtracted from the image.
        bias_file_path
            If `bias_subtract_via_bias_file=True`, this overwrites the path to the bias file instead of the default
            behaviour of using the .fits header.
        use_calibrated_gain
            If True, the calibrated gain values are used to convert from COUNTS to ELECTRONS.
        """

        hdu = fits_hdu_via_quadrant_letter_from(quadrant_letter=quadrant_letter)

        header_sci_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=0)
        header_hdu_obj = array_2d_util.header_obj_from(file_path=file_path, hdu=hdu)

        header = HeaderACS(
            header_sci_obj=header_sci_obj,
            header_hdu_obj=header_hdu_obj,
            hdu=hdu,
            quadrant_letter=quadrant_letter,
        )

        if header.header_sci_obj["TELESCOP"] != "HST":
            raise exc.ArrayException(
                f"The file {file_path} does not point to a valid HST ACS dataset."
            )

        if header.header_sci_obj["INSTRUME"] != "ACS":
            raise exc.ArrayException(
                f"The file {file_path} does not point to a valid HST ACS dataset."
            )

        array = array_2d_util.numpy_array_2d_via_fits_from(
            file_path=file_path, hdu=hdu, do_not_scale_image_data=True
        )

        array = header.array_original_to_electrons(
            array=array, use_calibrated_gain=use_calibrated_gain
        )

        if bias_subtract_via_bias_file:

            if bias_file_path is None:

                file_dir = os.path.split(file_path)[0]
                bias_file_path = path.join(file_dir, header.bias_file)

            bias = array_2d_util.numpy_array_2d_via_fits_from(
                file_path=bias_file_path, hdu=hdu, do_not_scale_image_data=True
            )

            header_sci_obj = array_2d_util.header_obj_from(
                file_path=bias_file_path, hdu=0
            )
            header_hdu_obj = array_2d_util.header_obj_from(
                file_path=bias_file_path, hdu=hdu
            )

            bias_header = HeaderACS(
                header_sci_obj=header_sci_obj,
                header_hdu_obj=header_hdu_obj,
                hdu=hdu,
                quadrant_letter=quadrant_letter,
            )

            if bias_header.original_units != "COUNTS":

                raise exc.ArrayException("Cannot use bias frame not in counts.")

            bias = bias * bias_header.calibrated_gain

        else:

            bias = None

        return cls.from_ccd(
            array_electrons=array,
            quadrant_letter=quadrant_letter,
            header=header,
            bias_subtract_via_prescan=bias_subtract_via_prescan,
            bias=bias,
        )


class Layout2DACS(Layout2D):
    @classmethod
    def from_sizes(cls, roe_corner, serial_prescan_size=24, parallel_overscan_size=20):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the HST FPA, quadrants and
        rotations.
        """

        parallel_overscan = Region2D(
            (2068 - parallel_overscan_size, 2068, serial_prescan_size, 2072)
        )

        serial_prescan = Region2D((0, 2068, 0, serial_prescan_size))

        return Layout2D.rotated_from_roe_corner(
            roe_corner=roe_corner,
            shape_native=(2068, 2072),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
        )


class HeaderACS(Header):
    def __init__(
        self,
        header_sci_obj,
        header_hdu_obj,
        quadrant_letter=None,
        hdu=None,
        bias=None,
        bias_serial_prescan_column=None,
    ):

        super().__init__(header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj)

        self.bias = bias
        self.bias_serial_prescan_column = bias_serial_prescan_column
        self.quadrant_letter = quadrant_letter
        self.hdu = hdu

    @property
    def bscale(self):
        return self.header_hdu_obj["BSCALE"]

    @property
    def bzero(self):
        return self.header_hdu_obj["BZERO"]

    @property
    def gain(self):
        return self.header_sci_obj["CCDGAIN"]

    @property
    def calibrated_gain(self):

        if round(self.gain) == 1:
            calibrated_gain = [0.99989998, 0.97210002, 1.01070000, 1.01800000]
        elif round(self.gain) == 2:
            calibrated_gain = [2.002, 1.945, 2.028, 1.994]
        elif round(self.gain) == 4:
            calibrated_gain = [4.011, 3.902, 4.074, 3.996]
        else:
            raise exc.ArrayException(
                "Calibrated gain of ACS file does not round to 1, 2 or 4."
            )

        if self.quadrant_letter == "A":
            return calibrated_gain[0]
        elif self.quadrant_letter == "B":
            return calibrated_gain[1]
        elif self.quadrant_letter == "C":
            return calibrated_gain[2]
        elif self.quadrant_letter == "D":
            return calibrated_gain[3]

    @property
    def original_units(self):
        return self.header_hdu_obj["BUNIT"]

    @property
    def bias_file(self):
        return self.header_sci_obj["BIASFILE"].replace("jref$", "")

    def array_eps_to_counts(self, array_eps):
        return array_eps_to_counts(
            array_eps=array_eps, bscale=self.bscale, bzero=self.bzero
        )

    def array_original_to_electrons(self, array, use_calibrated_gain):

        if self.original_units in "COUNTS":
            array = (array * self.bscale) + self.bzero
        elif self.original_units in "CPS":
            array = (array * self.exposure_time * self.bscale) + self.bzero

        if use_calibrated_gain:
            return array * self.calibrated_gain
        else:
            return array * self.gain

    def array_electrons_to_original(self, array, use_calibrated_gain):

        if use_calibrated_gain:
            array /= self.calibrated_gain
        else:
            array /= self.gain

        if self.original_units in "COUNTS":
            return (array - self.bzero) / self.bscale
        elif self.original_units in "CPS":
            return (array - self.bzero) / (self.exposure_time * self.bscale)


def prescan_fitted_bias_column(prescan, n_rows=2048, n_rows_ov=20):
    """
    Generate a bias column to be subtracted from the main image by doing a
    least squares fit to the serial prescan region.

    e.g. image -= prescan_fitted_bias_column(image[18:24])

    See Anton & Rorres (2013), S9.3, p460.

    Parameters
    ----------
    prescan : [[float]]
        The serial prescan part of the image. Should usually cover the full
        number of rows but may skip the first few columns of the prescan to
        avoid trails.

    n_rows
        The number of rows in the image, exculding overscan.

    n_rows_ov, int
        The number of overscan rows in the image.

    Returns
    -------
    bias_column : [float]
        The fitted bias to be subtracted from all image columns.
    """
    n_columns_fit = prescan.shape[1]

    # Flatten the multiple fitting columns to a long 1D array
    # y = [y_1_1, y_2_1, ..., y_nrow_1, y_1_2, y_2_2, ..., y_nrow_ncolfit]
    y = prescan[:-n_rows_ov].T.flatten()
    # x = [1, 2, ..., nrow, 1, ..., nrow, 1, ..., nrow, ...]
    x = np.tile(np.arange(n_rows), n_columns_fit)

    # M = [[1, 1, ..., 1], [x_1, x_2, ..., x_n]].T
    M = np.array([np.ones(n_rows * n_columns_fit), x]).T

    # Best-fit values for y = M v
    v = np.dot(np.linalg.inv(np.dot(M.T, M)), np.dot(M.T, y))

    # Map to full image size for easy subtraction
    bias_column = v[0] + v[1] * np.arange(n_rows + n_rows_ov)

    return np.transpose([bias_column])


def output_quadrants_to_fits(
    file_path: str,
    quadrant_a,
    quadrant_b,
    quadrant_c,
    quadrant_d,
    header_a=None,
    header_b=None,
    header_c=None,
    header_d=None,
    overwrite: bool = False,
):

    file_dir = os.path.split(file_path)[0]

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    array_hdu_1 = np.zeros((2068, 4144))
    array_hdu_4 = np.zeros((2068, 4144))

    def get_header(quadrant):
        try:
            return quadrant.header
        except AttributeError:
            raise (
                "You must pass in the header of the quadrants to output them to an ACS fits file."
            )

    header_a = get_header(quadrant_a) if header_a is None else header_a
    try:
        quadrant_a = copy.copy(np.asarray(quadrant_a.native))
    except AttributeError:
        quadrant_a = copy.copy(np.asarray(quadrant_a))

    quadrant_a = quadrant_convert_to_original(
        quadrant=quadrant_a, roe_corner=(1, 0), header=header_a, use_flipud=True
    )
    array_hdu_4[0:2068, 0:2072] = quadrant_a

    header_b = get_header(quadrant_b) if header_b is None else header_b

    try:
        quadrant_b = copy.copy(np.asarray(quadrant_b.native))
    except AttributeError:
        quadrant_b = copy.copy(np.asarray(quadrant_b))
    quadrant_b = quadrant_convert_to_original(
        quadrant=quadrant_b, roe_corner=(1, 1), header=header_b, use_flipud=True
    )
    array_hdu_4[0:2068, 2072:4144] = quadrant_b

    header_c = get_header(quadrant_c) if header_c is None else header_c
    try:
        quadrant_c = copy.copy(np.asarray(quadrant_c.native))
    except AttributeError:
        quadrant_c = copy.copy(np.asarray(quadrant_c))
    quadrant_c = quadrant_convert_to_original(
        quadrant=quadrant_c, roe_corner=(1, 0), header=header_c, use_flipud=False
    )
    array_hdu_1[0:2068, 0:2072] = quadrant_c

    header_d = get_header(quadrant_d) if header_d is None else header_d
    try:
        quadrant_d = copy.copy(np.asarray(quadrant_d.native))
    except AttributeError:
        quadrant_d = copy.copy(np.asarray(quadrant_d))
    quadrant_d = quadrant_convert_to_original(
        quadrant=quadrant_d, roe_corner=(1, 1), header=header_d, use_flipud=False
    )
    array_hdu_1[0:2068, 2072:4144] = quadrant_d

    hdu_list = fits.HDUList()

    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU(array_hdu_1))
    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU())
    hdu_list.append(fits.ImageHDU(array_hdu_4))
    hdu_list.append(fits.ImageHDU())

    def set_header(header):
        header.set("cticor", "ARCTIC", "CTI CORRECTION PERFORMED USING ARCTIC")
        return header

    hdu_list[0].header = set_header(header_a.header_sci_obj)
    hdu_list[1].header = set_header(header_c.header_hdu_obj)
    hdu_list[4].header = set_header(header_a.header_hdu_obj)
    hdu_list.writeto(file_path)


def quadrant_convert_to_original(
    quadrant, roe_corner, header, use_flipud=False, use_calibrated_gain=True
):

    if header.bias is not None:
        quadrant += header.bias.native

    if header.bias_serial_prescan_column is not None:
        quadrant += header.bias_serial_prescan_column

    quadrant = header.array_electrons_to_original(
        array=quadrant, use_calibrated_gain=use_calibrated_gain
    )

    if use_flipud:
        quadrant = np.flipud(quadrant)

    return layout_util.rotate_array_via_roe_corner_from(
        array=quadrant, roe_corner=roe_corner
    )
