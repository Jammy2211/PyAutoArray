import logging
from astropy import time
import numpy as np

from autoarray import exc
from autoarray.dataset import preprocess
from autoarray.structures import abstract_structure
from autoarray.mask import mask as msk
from autoarray.util import binning_util, array_util

logging.basicConfig()
logger = logging.getLogger(__name__)


def check_array(array):

    if array.store_in_1d and len(array.shape) != 1:
        raise exc.ArrayException(
            "An array input into the arrays.Array.__new__ method has store_in_1d = True but"
            "the input shape of the array is not 1."
        )


def convert_array(array):
    """If the input array input a convert is of type list, convert it to type NumPy array.

    Parameters
    ----------
    array : list or ndarray
        The array which may be converted to an ndarray
    """

    if type(array) is list:
        array = np.asarray(array)

    return array


def convert_manual_1d_array(array_1d, mask, store_in_1d):
    """
    Manual 1D Array functions take as input a list or ndarray which is to be returned as an Array. This function
    performs the following and checks and conversions on the input:

    1) If the input is a list, convert it to a 1D ndarray of shape [total_values].
    2) Check that the number of sub-pixels in the array is identical to that of the mask.
    3) Return the array in 1D if it is to be stored in 1D, else return it in 2D.

    For Arrays, `1d' refers to a 1D NumPy array of shape [total_values] and '2d' a 2D NumPy array of shape
    [total_y_values, total_values].

    Parameters
    ----------
    array_1d : ndarray or list
        The input structure which is converted to a 1D ndarray if it is a list.
    mask : Mask
        The mask of the output Array.
    store_in_1d : bool
        Whether the memory-representation of the array is in 1D or 2D.
    """

    array_1d = convert_array(array=array_1d)

    if array_1d.shape[0] != mask.sub_pixels_in_mask:
        raise exc.ArrayException(
            "The input 1D array does not have the same number of entries as sub-pixels in"
            "the mask."
        )

    if store_in_1d:
        return array_1d

    return array_util.sub_array_2d_from(
        sub_array_1d=array_1d, mask=mask, sub_size=mask.sub_size
    )


def convert_manual_2d_array(array_2d, mask, store_in_1d):
    """
    Manual 2D Array functions take as input a list or ndarray which is to be returned as an Array. This function
    performs the following and checks and conversions on the input:

    1) If the input is a list, convert it to a 2D ndarray.
    2) Check that the number of sub-pixels in the array is identical to that of the mask.
    3) Return the array in 1D if it is to be stored in 1D, else return it in 2D.

    For Arrays, `1d' refers to a 1D NumPy array of shape [total_values] and '2d' a 2D NumPy array of shape
    [total_y_values, total_values].

    Parameters
    ----------
    array_2d : ndarray or list
        The input structure which is converted to a 2D ndarray if it is a list.
    mask : Mask
        The mask of the output Array.
    store_in_1d : bool
        Whether the memory-representation of the array is in 1D or 2D.
    """
    array_2d = convert_array(array=array_2d)

    if array_2d.shape != mask.sub_shape_2d:
        raise exc.ArrayException(
            "The input array is 2D but not the same dimensions as the sub-mask "
            "(e.g. the mask 2D shape multipled by its sub size."
        )

    sub_array_1d = array_util.sub_array_1d_from(
        sub_array_2d=array_2d, mask=mask, sub_size=mask.sub_size
    )

    if store_in_1d:
        return sub_array_1d

    return array_util.sub_array_2d_from(
        sub_array_1d=sub_array_1d, mask=mask, sub_size=mask.sub_size
    )


def convert_manual_array(array, mask, store_in_1d):
    """
    Manual array functions take as input a list or ndarray which is to be returned as an Array. This function
    performs the following and checks and conversions on the input:

    1) If the input is a list, convert it to an ndarray.
    2) Check that the number of sub-pixels in the array is identical to that of the mask.
    3) Return the array in 1D if it is to be stored in 1D, else return it in 2D.

    Parameters
    ----------
    array : ndarray or list
        The input structure which is converted to an ndarray if it is a list.
    mask : Mask
        The mask of the output Array.
    store_in_1d : bool
        Whether the memory-representation of the array is in 1D or 2D.
    """

    array = convert_array(array=array)

    if len(array.shape) == 1:
        return convert_manual_1d_array(
            array_1d=array, mask=mask, store_in_1d=store_in_1d
        )
    return convert_manual_2d_array(array_2d=array, mask=mask, store_in_1d=store_in_1d)


class AbstractArray(abstract_structure.AbstractStructure):

    exposure_info = None

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(AbstractArray, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        class_dict = {}
        for key, value in self.__dict__.items():
            class_dict[key] = value
        new_state = pickled_state[2] + (class_dict,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return pickled_state[0], pickled_state[1], new_state

    # noinspection PyMethodOverriding
    def __setstate__(self, state):

        for key, value in state[-1].items():
            setattr(self, key, value)
        super(AbstractArray, self).__setstate__(state[0:-1])

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __eq__(self, other):
        super_result = super(AbstractArray, self).__eq__(other)
        try:
            return super_result.all()
        except AttributeError:
            return super_result

    @property
    def in_1d(self):
        """Convenience method to access the array's 1D representation, which is an ndarray of shape
        [total_unmasked_pixels*(sub_size**2)].

        If the grid is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D."""
        if self.store_in_1d:
            return self

        sub_array_1d = array_util.sub_array_1d_from(
            sub_array_2d=self, mask=self.mask, sub_size=self.mask.sub_size
        )

        return self.__class__(array=sub_array_1d, mask=self.mask, store_in_1d=True)

    @property
    def in_2d(self):
        """Convenience method to access the array's 2D representation, which is an ndarray of shape
        [sub_size*total_y_pixels, sub_size*total_x_pixels, 2] where all masked values are given values (0.0, 0.0).

        If the array is stored in 2D it is return as is. If it is stored in 1D, it must first be mapped from 1D to 2D."""
        if self.store_in_1d:
            sub_array_2d = array_util.sub_array_2d_from(
                sub_array_1d=self, mask=self.mask, sub_size=self.mask.sub_size
            )
            return self.__class__(array=sub_array_2d, mask=self.mask, store_in_1d=False)

        return self

    @property
    def in_1d_binned(self):
        """Convenience method to access the binned-up array in its 1D representation, which is a Grid stored as an
        ndarray of shape [total_unmasked_pixels, 2].

        The binning up process converts a array from (y,x) values where each value is a coordinate on the sub-array to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a array with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the array is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D."""

        if not self.store_in_1d:

            sub_array_1d = array_util.sub_array_1d_from(
                sub_array_2d=self, mask=self.mask, sub_size=self.mask.sub_size
            )

        else:

            sub_array_1d = self

        binned_array_1d = np.multiply(
            self.mask.sub_fraction,
            sub_array_1d.reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        return self.__class__(
            array=binned_array_1d, mask=self.mask.mask_sub_1, store_in_1d=True
        )

    @property
    def in_2d_binned(self):
        """Convenience method to access the binned-up array in its 2D representation, which is a Grid stored as an
        ndarray of shape [total_y_pixels, total_x_pixels, 2].

        The binning up process conerts a array from (y,x) values where each value is a coordinate on the sub-array to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a array with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the array is stored in 2D it is return as is. If it is stored in 1D, it must first be mapped from 1D to 2D."""
        if not self.store_in_1d:

            sub_array_1d = array_util.sub_array_1d_from(
                sub_array_2d=self, mask=self.mask, sub_size=self.mask.sub_size
            )

        else:

            sub_array_1d = self

        binned_array_1d = np.multiply(
            self.mask.sub_fraction,
            sub_array_1d[:].reshape(-1, self.mask.sub_length).sum(axis=1),
        )

        binned_array_2d = array_util.sub_array_2d_from(
            sub_array_1d=binned_array_1d, mask=self.mask, sub_size=1
        )

        return self.__class__(
            array=binned_array_2d, mask=self.mask.mask_sub_1, store_in_1d=False
        )

    @property
    def extent(self):
        return self.mask.geometry.extent

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
        array: ndarray
            An ndarray

        Returns
        -------
        new_array: Array
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

        extracted_array_2d = array_util.extracted_array_2d_from(
            array_2d=self.in_2d,
            y0=self.geometry._zoom_region[0] - buffer,
            y1=self.geometry._zoom_region[1] + buffer,
            x0=self.geometry._zoom_region[2] - buffer,
            x1=self.geometry._zoom_region[3] + buffer,
        )

        mask = msk.Mask.unmasked(
            shape_2d=extracted_array_2d.shape,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.mask.geometry.mask_centre,
        )

        array = convert_manual_2d_array(
            array_2d=extracted_array_2d, mask=mask, store_in_1d=self.store_in_1d
        )

        return self.__class__(array=array, mask=mask, store_in_1d=self.store_in_1d)

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
        extracted_array_2d = array_util.extracted_array_2d_from(
            array_2d=self.in_2d,
            y0=self.geometry._zoom_region[0] - buffer,
            y1=self.geometry._zoom_region[1] + buffer,
            x0=self.geometry._zoom_region[2] - buffer,
            x1=self.geometry._zoom_region[3] + buffer,
        )

        mask = msk.Mask.unmasked(
            shape_2d=extracted_array_2d.shape,
            pixel_scales=self.pixel_scales,
            sub_size=self.sub_size,
            origin=self.mask.geometry.mask_centre,
        )

        return mask.geometry.extent

    def resized_from_new_shape(self, new_shape):
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

        resized_array_2d = array_util.resized_array_2d_from_array_2d(
            array_2d=self.in_2d, resized_shape=new_shape
        )

        resized_mask = self.mask.resized_mask_from_new_shape(new_shape=new_shape)

        array = convert_manual_2d_array(
            array_2d=resized_array_2d, mask=resized_mask, store_in_1d=self.store_in_1d
        )

        return self.__class__(
            array=array, mask=resized_mask, store_in_1d=self.store_in_1d
        )

    def padded_from_kernel_shape(self, kernel_shape_2d):
        """When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the array can be padded, where it is 'buffed' such that it includes all
        pixels whose signal will be convolved into the unmasked pixels given the 2D kernel shape. The values of
        these pixels are zeros.

        Parameters
        ----------
        kernel_shape_2d : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        new_shape = (
            self.shape_2d[0] + (kernel_shape_2d[0] - 1),
            self.shape_2d[1] + (kernel_shape_2d[1] - 1),
        )
        return self.resized_from_new_shape(new_shape=new_shape)

    def trimmed_from_kernel_shape(self, kernel_shape_2d):
        """When the edge pixels of a mask are unmasked and a convolution is to occur, the signal of edge pixels will be
        'missing' if the grid is used to evaluate the signal via an analytic function.

        To ensure this signal is included the array can be padded, a padded array can be computed via the method
        *padded_from_kernel_shape*. This function trims the array back to its original shape, after the padded array
        has been used for computationl.

        Parameters
        ----------
        kernel_shape_2d : (float, float)
            The 2D shape of the kernel which convolves signal from masked pixels to unmasked pixels.
        """
        psf_cut_y = np.int(np.ceil(kernel_shape_2d[0] / 2)) - 1
        psf_cut_x = np.int(np.ceil(kernel_shape_2d[1] / 2)) - 1
        array_y = np.int(self.mask.shape[0])
        array_x = np.int(self.mask.shape[1])
        trimmed_array_2d = self.in_2d[
            psf_cut_y : array_y - psf_cut_y, psf_cut_x : array_x - psf_cut_x
        ]

        resized_mask = self.mask.resized_mask_from_new_shape(
            new_shape=trimmed_array_2d.shape
        )

        array = convert_manual_2d_array(
            array_2d=trimmed_array_2d, mask=resized_mask, store_in_1d=self.store_in_1d
        )

        return self.__class__(
            array=array, mask=resized_mask, store_in_1d=self.store_in_1d
        )

    def binned_from_bin_up_factor(self, bin_up_factor, method):
        """Compute a binned version of the Array, where binning up occurs by coming all pixel values in a set of
        (bin_up_factor x bin_up_factor) pixels.

        The pixels can be combined:

        - By taking the mean of their values, which one may use for binning up an image.
        - By adding them in quadranture, which one may use for binning up a noise-map.
        - By summing them, which one may use for binning up an exposure time map.

        Parameters
        ----------
        bin_up_factor : int
            The factor which the array is binned up by (e.g. a value of 2 bins every 2 x 2 pixels into one pixel).
        method : str
            The method used to combine the set of values that are binned up.
        """

        binned_mask = self.mask.binned_mask_from_bin_up_factor(
            bin_up_factor=bin_up_factor
        )

        if method is "mean":

            binned_array_2d = binning_util.bin_array_2d_via_mean(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

        elif method is "quadrature":

            binned_array_2d = binning_util.bin_array_2d_via_quadrature(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

        elif method is "sum":

            binned_array_2d = binning_util.bin_array_2d_via_sum(
                array_2d=self.in_2d, bin_up_factor=bin_up_factor
            )

        else:

            raise exc.ArrayException(
                "The method used in binned_up_array_from_array is not a valid method "
                "[mean I quadrature I sum]"
            )

        binned_array_1d = array_util.sub_array_1d_from(
            mask=binned_mask, sub_array_2d=binned_array_2d, sub_size=1
        )

        array = convert_manual_1d_array(
            array_1d=binned_array_1d, mask=binned_mask, store_in_1d=self.store_in_1d
        )

        return self.__class__(
            array=array, mask=binned_mask, store_in_1d=self.store_in_1d
        )

    def output_to_fits(self, file_path, overwrite=False):
        """Output the array to a .fits file.

        Parameters
        ----------
        file_path : str
            The path the file is output to, including the filename and the '.fits' extension,
            e.g. '/path/to/filename.fits'
        overwrite : bool
            If a file already exists at the path, if overwrite=True it is overwritten else an error is raised."""
        array_util.numpy_array_2d_to_fits(
            array_2d=self.in_2d, file_path=file_path, overwrite=overwrite
        )


class ExposureInfo:
    def __init__(
        self, date_of_observation=None, time_of_observation=None, exposure_time=None
    ):

        self.date_of_observation = date_of_observation
        self.time_of_observation = time_of_observation
        self.exposure_time = exposure_time

    @property
    def modified_julian_date(self):
        if (
            self.date_of_observation is not None
            and self.time_of_observation is not None
        ):
            t = time.Time(self.date_of_observation + "T" + self.time_of_observation)
            return t.mjd
        return None

    def array_eps_to_counts(self, array_eps):
        raise NotImplementedError()

    def array_counts_to_counts_per_second(self, array_counts):
        return preprocess.array_counts_to_counts_per_second(
            array_counts=array_counts, exposure_time=self.exposure_time
        )
