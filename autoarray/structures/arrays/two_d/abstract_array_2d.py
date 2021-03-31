import logging
from astropy import time
import numpy as np

from autoarray import exc
from autoarray.mask import mask_2d as msk
from autoarray.structures import abstract_structure
from autoarray.structures.arrays import abstract_array
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray.dataset import preprocess

logging.basicConfig()
logger = logging.getLogger(__name__)


def check_array_2d(array_2d):
    if len(array_2d.shape) != 1:
        raise exc.ArrayException(
            "An array input into the array_2d.Array2D.__new__ method is not of shape 1."
        )


def convert_array_2d(array_2d, mask_2d):
    """
    Manual array functions take as input a list or ndarray which is to be returned as an Array2D. This function
    performs the following and checks and conversions on the input:

    1) If the input is a list, convert it to an ndarray.
    2) Check that the number of sub-pixels in the array is identical to that of the mask.
    3) Map the input ndarray to its `slim` representation.

    For an Array2D, `slim` refers to a 1D NumPy array of shape [total_values] and `native` a 2D NumPy array of shape
    [total_y_values, total_values].

    Parameters
    ----------
    array_2d : np.ndarray or list
        The input structure which is converted to an ndarray if it is a list.
    mask_2d : Mask2D
        The mask of the output Array2D.
    """

    array_2d = abstract_array.convert_array(array=array_2d)

    if len(array_2d.shape) == 1:

        array_2d_slim = abstract_array.convert_array(array=array_2d)

        if array_2d_slim.shape[0] != mask_2d.sub_pixels_in_mask:
            raise exc.ArrayException(
                "The input 1D array does not have the same number of entries as sub-pixels in"
                "the mask."
            )

        return array_2d_slim

    if array_2d.shape != mask_2d.sub_shape_native:
        raise exc.ArrayException(
            "The input array is 2D but not the same dimensions as the sub-mask "
            "(e.g. the mask 2D shape multipled by its sub size."
        )

    sub_array_1d = array_2d_util.array_2d_slim_from(
        array_2d_native=array_2d, mask_2d=mask_2d, sub_size=mask_2d.sub_size
    )

    return sub_array_1d


class AbstractArray2D(abstract_structure.AbstractStructure2D):

    exposure_info = None

    def _new_structure(self, array, mask):
        return self.__class__(array=array, mask=mask)

    @property
    def readout_offsets(self):
        if self.exposure_info is not None:
            return self.exposure_info.readout_offsets
        return (0, 0)

    @property
    def slim(self):
        """
        Return an `Array2D` where the data is stored its `slim` representation, which is an ndarray of shape
        [total_unmasked_pixels * sub_size**2].

        If it is already stored in its `slim` representation it is returned as it is. If not, it is  mapped from
        `native` to `slim` and returned as a new `Array2D`.
        """

        if len(self.shape) == 1:
            return self

        sub_array_1d = array_2d_util.array_2d_slim_from(
            array_2d_native=self, mask_2d=self.mask, sub_size=self.mask.sub_size
        )

        return self._new_structure(array=sub_array_1d, mask=self.mask)

    @property
    def native(self):
        """
        Return a `Array2D` where the data is stored in its `native` representation, which is an ndarray of shape
        [sub_size*total_y_pixels, sub_size*total_x_pixels].

        If it is already stored in its `native` representation it is return as it is. If not, it is mapped from
        `slim` to `native` and returned as a new `Array2D`.
        """

        if len(self.shape) != 1:
            return self

        sub_array_2d = array_2d_util.array_2d_native_from(
            array_2d_slim=self, mask_2d=self.mask, sub_size=self.mask.sub_size
        )
        return self._new_structure(array=sub_array_2d, mask=self.mask)

    @property
    def binned(self):
        """
        Convenience method to access the binned-up array in its 1D representation, which is a Grid2D stored as an
        ndarray of shape [total_unmasked_pixels, 2].

        The binning up process converts a array from (y,x) values where each value is a coordinate on the sub-array to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a array with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the array is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.
        """

        array_2d_slim = self.slim

        binned_array_1d = np.multiply(
            self.mask.sub_fraction,
            array_2d_slim.reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        return self._new_structure(array=binned_array_1d, mask=self.mask.mask_sub_1)

    @property
    def extent(self):
        return self.mask.extent

    @property
    def in_counts(self):
        return self.exposure_info.array_eps_to_counts(array_eps=self)

    @property
    def in_counts_per_second(self):
        return self.exposure_info.array_counts_to_counts_per_second(
            array_counts=self.in_counts
        )

    def new_with_array(self, array):
        """
        Parameters
        ----------
        array: np.ndarray
            An ndarray

        Returns
        -------
        new_array: Array2D
            A new instance of this class that shares all of this instances attributes with a new ndarray.
        """
        arguments = vars(self)
        arguments.update({"array": array})
        return self.__class__(**arguments)

    def map(self, func):
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                func(y, x)

    def zoomed_around_mask(self, buffer=1):
        """Extract the 2D region of an array corresponding to the rectangle encompassing all unmasked values.

        This is used to extract and visualize only the region of an image that is used in an analysis.

        Parameters
        ----------
        buffer : int
            The number pixels around the extracted array used as a buffer.
        """

        extracted_array_2d = array_2d_util.extracted_array_2d_from(
            array_2d=self.native,
            y0=self.mask.zoom_region[0] - buffer,
            y1=self.mask.zoom_region[1] + buffer,
            x0=self.mask.zoom_region[2] - buffer,
            x1=self.mask.zoom_region[3] + buffer,
        )

        mask = msk.Mask2D.unmasked(
            shape_native=extracted_array_2d.shape,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.mask.mask_centre,
        )

        array = convert_array_2d(array_2d=extracted_array_2d, mask_2d=mask)

        return self._new_structure(array=array, mask=mask)

    def extent_of_zoomed_array(self, buffer=1):
        """For an extracted zoomed array computed from the method *zoomed_around_mask* compute its extent in scaled
        coordinates.

        The extent of the grid in scaled units returned as an ndarray of the form [x_min, x_max, y_min, y_max].

        This is used visualize zoomed and extracted arrays via the imshow() method.

        Parameters
        ----------
        buffer : int
            The number pixels around the extracted array used as a buffer.
        """
        extracted_array_2d = array_2d_util.extracted_array_2d_from(
            array_2d=self.native,
            y0=self.mask.zoom_region[0] - buffer,
            y1=self.mask.zoom_region[1] + buffer,
            x0=self.mask.zoom_region[2] - buffer,
            x1=self.mask.zoom_region[3] + buffer,
        )

        mask = msk.Mask2D.unmasked(
            shape_native=extracted_array_2d.shape,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.mask.mask_centre,
        )

        return mask.extent

    def resized_from(self, new_shape):
        """Resize the array around its centre to a new input shape.

        If a new_shape dimension is smaller than the current dimension, the data at the edges is trimmed and removed.
        If it is larger, the data is padded with zeros.

        If the array has even sized dimensions, the central pixel around which data is trimmed / padded is chosen as
        the top-left pixel of the central quadrant of pixels.

        Parameters
        -----------
        new_shape : (int, int)
            The new 2D shape of the array.
        """

        resized_array_2d = array_2d_util.resized_array_2d_from_array_2d(
            array_2d=self.native, resized_shape=new_shape
        )

        resized_mask = self.mask.resized_mask_from_new_shape(new_shape=new_shape)

        array = convert_array_2d(array_2d=resized_array_2d, mask_2d=resized_mask)

        return self._new_structure(array=array, mask=resized_mask)

    def padded_before_convolution_from(self, kernel_shape):
        """When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the array can be padded, where it is 'buffed' such that it includes all
        pixels whose signal will be convolved into the unmasked pixels given the 2D kernel shape. The values of
        these pixels are zeros.

        Parameters
        ----------
        kernel_shape : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        new_shape = (
            self.shape_native[0] + (kernel_shape[0] - 1),
            self.shape_native[1] + (kernel_shape[1] - 1),
        )
        return self.resized_from(new_shape=new_shape)

    def trimmed_after_convolution_from(self, kernel_shape):
        """When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the array can be padded, a padded array can be computed via the method
        *padded_from_kernel_shape*. This function trims the array back to its original shape, after the padded array
        has been used for computationl.

        Parameters
        ----------
        kernel_shape : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        psf_cut_y = int(np.ceil(kernel_shape[0] / 2)) - 1
        psf_cut_x = int(np.ceil(kernel_shape[1] / 2)) - 1
        array_y = int(self.mask.shape[0])
        array_x = int(self.mask.shape[1])
        trimmed_array_2d = self.native[
            psf_cut_y : array_y - psf_cut_y, psf_cut_x : array_x - psf_cut_x
        ]

        resized_mask = self.mask.resized_mask_from_new_shape(
            new_shape=trimmed_array_2d.shape
        )

        array = convert_array_2d(array_2d=trimmed_array_2d, mask_2d=resized_mask)

        return self.__class__(array=array, mask=resized_mask)

    def output_to_fits(self, file_path, overwrite=False):
        """
        Output the array to a .fits file.

        Parameters
        ----------
        file_path : str
            The path the file is output to, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        overwrite : bool
            If a file already exists at the path, if overwrite=True it is overwritten else an error is raised.
        """
        array_2d_util.numpy_array_2d_to_fits(
            array_2d=self.native, file_path=file_path, overwrite=overwrite
        )
