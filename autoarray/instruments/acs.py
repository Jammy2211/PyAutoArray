from autoarray.structures.arrays import abstract_array
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray.structures.arrays.two_d import array_2d
from autoarray.layout import layout as lo, layout_util
from autoarray.layout import region as reg
from autoarray import exc

from astropy.io import fits

import numpy as np
import shutil
import os
from os import path

import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")


def fits_hdu_from_quadrant_letter(quadrant_letter):

    if quadrant_letter == "D" or quadrant_letter == "C":
        return 1
    elif quadrant_letter == "B" or quadrant_letter == "A":
        return 4
    else:
        raise exc.FrameException("Quadrant letter for FrameACS must be A, B, C or D.")


def array_eps_to_counts(array_eps, bscale, bzero):

    if bscale is None:
        raise exc.FrameException(
            "Cannot convert a Frame2D to units COUNTS without a bscale attribute (bscale = None)."
        )

    return (array_eps - bzero) / bscale


class Array2DACS(array_2d.Array2D):
    """
    An ACS array consists of four quadrants ('A', 'B', 'C', 'D') which have the following layout:

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

        See the docstring of the `FrameACS` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        hdu = fits_hdu_from_quadrant_letter(quadrant_letter=quadrant_letter)

        array = array_2d_util.numpy_array_2d_from_fits(file_path=file_path, hdu=hdu)

        return cls.from_ccd(array_electrons=array, quadrant_letter=quadrant_letter)

    @classmethod
    def from_ccd(
        cls,
        array_electrons,
        quadrant_letter,
        parallel_size=2068,
        serial_size=2072,
        header=None,
        bias_subtract_via_prescan=False,
        bias=None,
    ):
        """
        Using an input array of both quadrants in electrons, use the quadrant letter to extract the quadrant from the
        full CCD and perform the rotations required to give correct arctic.

        See the docstring of the `FrameACS` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """
        if quadrant_letter == "A":

            if bias is not None:
                bias = bias[0:parallel_size, 0:serial_size]

            return cls.quadrant_a(
                array_electrons=array_electrons[0:parallel_size, 0:serial_size],
                header=header,
                bias_subtract_via_prescan=bias_subtract_via_prescan,
                bias=bias,
            )
        elif quadrant_letter == "B":

            if bias is not None:
                bias = bias[0:parallel_size, serial_size : serial_size * 2]

            return cls.quadrant_b(
                array_electrons=array_electrons[
                    0:parallel_size, serial_size : serial_size * 2
                ],
                header=header,
                bias_subtract_via_prescan=bias_subtract_via_prescan,
                bias=bias,
            )
        elif quadrant_letter == "C":

            if bias is not None:
                bias = bias[0:parallel_size, 0:serial_size]

            return cls.quadrant_c(
                array_electrons=array_electrons[0:parallel_size, 0:serial_size],
                header=header,
                bias_subtract_via_prescan=bias_subtract_via_prescan,
                bias=bias,
            )
        elif quadrant_letter == "D":

            if bias is not None:
                bias = bias[0:parallel_size, serial_size : serial_size * 2]

            return cls.quadrant_d(
                array_electrons=array_electrons[
                    0:parallel_size, serial_size : serial_size * 2
                ],
                header=header,
                bias_subtract_via_prescan=bias_subtract_via_prescan,
                bias=bias,
            )
        else:
            raise exc.FrameException(
                "Quadrant letter for FrameACS must be A, B, C or D."
            )

    @classmethod
    def quadrant_a(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        array_electrons = layout_util.rotate_array_from_roe_corner(
            array=array_electrons, roe_corner=(1, 0)
        )

        array_electrons = np.flipud(array_electrons)

        if bias_subtract_via_prescan:

            bias_serial_prescan_value = prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:

            bias = layout_util.rotate_array_from_roe_corner(
                array=bias, roe_corner=(1, 0)
            )

            bias = np.flipud(bias)

            array_electrons -= bias

            header.bias = bias

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_b(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the right quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        array_electrons = layout_util.rotate_array_from_roe_corner(
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

            bias = layout_util.rotate_array_from_roe_corner(
                array=bias, roe_corner=(1, 1)
            )

            bias = np.flipud(bias)

            array_electrons -= bias

            header.bias = bias

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_c(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        array_electrons = layout_util.rotate_array_from_roe_corner(
            array=array_electrons, roe_corner=(1, 0)
        )

        if bias_subtract_via_prescan:

            bias_serial_prescan_value = prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:

            bias = layout_util.rotate_array_from_roe_corner(
                array=bias, roe_corner=(1, 0)
            )

            array_electrons -= bias

            header.bias = bias

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    @classmethod
    def quadrant_d(
        cls, array_electrons, header=None, bias_subtract_via_prescan=False, bias=None
    ):
        """
        Use an input array of the right quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        array_electrons = layout_util.rotate_array_from_roe_corner(
            array=array_electrons, roe_corner=(1, 1)
        )

        if bias_subtract_via_prescan:
            bias_serial_prescan_value = prescan_fitted_bias_column(
                array_electrons[:, 18:24]
            )

            array_electrons -= bias_serial_prescan_value

            header.bias_serial_prescan_column = bias_serial_prescan_value

        if bias is not None:

            bias = layout_util.rotate_array_from_roe_corner(
                array=bias, roe_corner=(1, 1)
            )

            array_electrons -= bias

            header.bias = bias

        return cls.manual(array=array_electrons, header=header, pixel_scales=0.05)

    def update_fits(self, original_file_path, new_file_path):
        """
        Output the array to a .fits file.

        Parameters
        ----------
        file_path : str
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
        cls, file_path, quadrant_letter, bias_subtract_via_prescan=False, bias_path=None
    ):
        """
        Use the input .fits file and quadrant letter to extract the quadrant from the full CCD, perform
        the rotations required to give correct arctic clocking and convert the image from units of COUNTS / CPS to
        ELECTRONS.

        See the docstring of the `FrameACS` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        hdu = fits_hdu_from_quadrant_letter(quadrant_letter=quadrant_letter)

        header_sci_obj = array_2d_util.header_obj_from_fits(file_path=file_path, hdu=0)
        header_hdu_obj = array_2d_util.header_obj_from_fits(
            file_path=file_path, hdu=hdu
        )

        header = HeaderACS(
            header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj, hdu=hdu
        )

        array = array_2d_util.numpy_array_2d_from_fits(
            file_path=file_path, hdu=hdu, do_not_scale_image_data=True
        )

        array = header.array_from_original_to_electrons(array=array)

        if bias_path is not None:

            bias = array_2d_util.numpy_array_2d_from_fits(
                file_path=bias_path, hdu=hdu, do_not_scale_image_data=True
            )

        else:

            bias = None

        return cls.from_ccd(
            array_electrons=array,
            quadrant_letter=quadrant_letter,
            header=header,
            bias_subtract_via_prescan=bias_subtract_via_prescan,
            bias=bias,
        )


class Layout2DACS(lo.Layout2D):
    @classmethod
    def from_sizes(
        cls,
        roe_corner,
        parallel_size=2068,
        serial_size=2072,
        serial_prescan_size=24,
        parallel_overscan_size=20,
    ):
        """
        Use an input array of the left quadrant in electrons and perform the rotations required to give correct
        arctic clocking.

        See the docstring of the `FrameACS` class for a complete description of the Euclid FPA, quadrants and
        rotations.
        """

        parallel_overscan = reg.Region2D(
            (
                parallel_size - parallel_overscan_size,
                parallel_size,
                serial_prescan_size,
                serial_size,
            )
        )

        serial_prescan = reg.Region2D((0, parallel_size, 0, serial_prescan_size))

        return lo.Layout2D.rotated_from_roe_corner(
            roe_corner=roe_corner,
            shape_native=(parallel_size, serial_size),
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
        )


class HeaderACS(abstract_array.Header):
    def __init__(
        self,
        header_sci_obj,
        header_hdu_obj,
        hdu=None,
        bias=None,
        bias_serial_prescan_column=None,
    ):

        super().__init__(header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj)

        self.bias = bias
        self.bias_serial_prescan_column = bias_serial_prescan_column
        self.hdu = hdu

    @property
    def bscale(self):
        return self.header_hdu_obj["BSCALE"]

    @property
    def bzero(self):
        return self.header_hdu_obj["BZERO"]

    @property
    def original_units(self):
        return self.header_hdu_obj["BUNIT"]

    def array_eps_to_counts(self, array_eps):
        return array_eps_to_counts(
            array_eps=array_eps, bscale=self.bscale, bzero=self.bzero
        )

    def array_from_original_to_electrons(self, array):

        if self.original_units in "COUNTS":
            return (array * self.bscale) + self.bzero
        elif self.original_units in "CPS":
            return (array * self.exposure_time * self.bscale) + self.bzero

    def array_from_electrons_to_original(self, array):

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

    n_rows : int
        The number of rows in the image, exculding overscan.

    n_rows_ov : int, int
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

    # plt.figure()
    # pixels = np.arange(n_rows + n_rows_ov)
    # for i in range(n_columns_fit):
    #     plt.scatter(pixels, prescan[:, i])
    # plt.plot(pixels, bias_column)
    # plt.show()

    return np.transpose([bias_column])


def output_quadrants_to_fits(
    quadrant_a,
    quadrant_b,
    quadrant_c,
    quadrant_d,
    file_path: str,
    overwrite: bool = False,
    parallel_size=2068,
    serial_size=2072,
):

    if path.exists(file_path):
        shutil.rmtree(file_path)

    os.makedirs(file_path)

    array_hdu_1 = np.zeros((2068, 4144))
    array_hdu_1[0:parallel_size, 0:serial_size] = quadrant_c
    array_hdu_1[0:parallel_size, serial_size : serial_size * 2] = quadrant_d

    array_hdu_4 = np.zeros((2068, 4144))
    array_hdu_4[0:parallel_size, 0:serial_size] = np.flipud(quadrant_a)
    array_hdu_4[0:parallel_size, serial_size : serial_size * 2] = np.flipud(quadrant_b)

    hdul = fits.HDUList()

    hdul.append(fits.ImageHDU(np.zeros(array_hdu_1.shape)))
    hdul.append(fits.ImageHDU(array_hdu_1))
    hdul.append(fits.ImageHDU(np.zeros(array_hdu_1.shape)))
    hdul.append(fits.ImageHDU(np.zeros(array_hdu_1.shape)))
    hdul.append(fits.ImageHDU(array_hdu_4))
    hdul.append(fits.ImageHDU(np.zeros(array_hdu_1.shape)))

    hdul[0].header = quadrant_a.header.header_sci_obj
    hdul[1].header = quadrant_c.header.header_hdu_obj
    hdul[4].header = quadrant_a.header.header_hdu_obj

    hdul.writeto(file_path)
