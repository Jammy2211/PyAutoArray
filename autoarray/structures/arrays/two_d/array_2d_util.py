import inspect
import os

from autoconf import conf
from autoarray import decorator_util
from autoarray.mask import mask_2d_util
import numpy as np
from astropy.io import fits
from functools import wraps


class Memoizer:
    def __init__(self):
        """
        Class to store the results of a function given a set of inputs.
        """
        self.results = {}
        self.calls = 0
        self.arg_names = None

    def __call__(self, func):
        """
        Memoize decorator. Any time a function is called that a memoizer has been attached to its results are stored in
        the results dictionary or retrieved from the dictionary if the function has aaready been called with those
        arguments.

        Note that the same memoizer persists over all instances of a class. Any state for a given instance that is not
        given in the representation of that instance will be ignored. That is, it is possible that the memoizer will
        give incorrect results if instance state does not affect __str__ but does affect the value returned by the
        memoized method.

        Parameters
        ----------
        func: function
            A function for which results should be memoized

        Returns
        -------
        decorated : function
            A function that memoizes results
        """
        if self.arg_names is not None:
            raise AssertionError("Instantiate a new Memoizer for each function")
        self.arg_names = inspect.getfullargspec(func).args

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = ", ".join(
                [
                    "('{}', {})".format(arg_name, arg)
                    for arg_name, arg in list(zip(self.arg_names, args))
                    + [(k, v) for k, v in kwargs.items()]
                ]
            )
            if key not in self.results:
                self.calls += 1
            self.results[key] = func(*args, **kwargs)
            return self.results[key]

        return wrapper


@decorator_util.jit()
def extracted_array_2d_from(
    array_2d: np.ndarray, y0: int, y1: int, x0: int, x1: int
) -> np.ndarray:
    """
    Resize an array to a new size by extracting a sub-set of the array.

    The extracted input coordinates use NumPy convention, such that the upper values should be specified as +1 the
    dimensions of the extracted array.

    In the example below, an array of size (5,5) is extracted using the coordinates y0=1, y1=4, x0=1, x1=4. This
    extracts an array of dimensions (3,3) and is equivalent to array_2d[1:4, 1:4].

    This function is necessary work with numba jit tags and is why a standard Numpy array extraction is not used.

    Parameters
    ----------
    array_2d : np.ndarray
        The 2D array that an array is extracted from.
    y0 : int
        The top row number (e.g. the higher y-coodinate) of the array that is extracted for the resize.
    y1 : int
        The bottom row number (e.g. the lower y-coodinate) of the array that is extracted for the resize.
    x0 : int
        The left column number (e.g. the lower x-coodinate) of the array that is extracted for the resize.
    x1 : int
        The right column number (e.g. the higher x-coodinate) of the array that is extracted for the resize.

    Returns
    -------
    ndarray
        The extracted 2D array from the input 2D array.

    Examples
    --------
    array_2d = np.ones((5,5))
    extracted_array = extract_array_2d(array_2d=array_2d, y0=1, y1=4, x0=1, x1=4)
    """

    new_shape = (y1 - y0, x1 - x0)

    resized_array = np.zeros(shape=new_shape)

    for y_resized, y in enumerate(range(y0, y1)):
        for x_resized, x in enumerate(range(x0, x1)):
            if (
                y >= 0
                and x >= 0
                and y <= array_2d.shape[0] - 1
                and x <= array_2d.shape[1] - 1
            ):
                resized_array[y_resized, x_resized] = array_2d[y, x]

    return resized_array


@decorator_util.jit()
def resized_array_2d_from_array_2d(
    array_2d: np.ndarray,
    resized_shape: (int, int),
    origin: (int, int) = (-1, -1),
    pad_value: int = 0.0,
) -> np.ndarray:
    """
    Resize an array to a new size around a central pixel.

    If the origin (e.g. the central pixel) of the resized array is not specified, the central pixel of the array is
    calculated automatically. For example, a (5,5) array's central pixel is (2,2). For even dimensions the central
    pixel is assumed to be the lower indexed value, e.g. a (6,4) array's central pixel is calculated as (2,1).

    The default origin is (-1, -1) because numba requires that the function input is the same type throughout the
    function, thus a default 'None' value cannot be used.

    Parameters
    ----------
    array_2d : np.ndarray
        The 2D array that is resized.
    resized_shape : (int, int)
        The (y,x) new pixel dimension of the trimmed array.
    origin : (int, int)
        The oigin of the resized array, e.g. the central pixel around which the array is extracted.
    pad_value : float
        If the reszied array is bigger in size than the input array, the value the padded edge values are filled in
        using.

    Returns
    -------
    ndarray
        The resized 2D array from the input 2D array.

    Examples
    --------
    array_2d = np.ones((5,5))
    resize_array = resize_array_2d(array_2d=array_2d, new_shape=(2,2), origin=(2, 2))
    """

    y_is_even = int(array_2d.shape[0]) % 2 == 0
    x_is_even = int(array_2d.shape[1]) % 2 == 0

    if origin == (-1, -1):

        if y_is_even:
            y_centre = int(array_2d.shape[0] / 2)
        elif not y_is_even:
            y_centre = int(array_2d.shape[0] / 2)

        if x_is_even:
            x_centre = int(array_2d.shape[1] / 2)
        elif not x_is_even:
            x_centre = int(array_2d.shape[1] / 2)

        origin = (y_centre, x_centre)

    resized_array = np.zeros(shape=resized_shape)

    if y_is_even:
        y_min = origin[0] - int(resized_shape[0] / 2)
        y_max = origin[0] + int((resized_shape[0] / 2)) + 1
    elif not y_is_even:
        y_min = origin[0] - int(resized_shape[0] / 2)
        y_max = origin[0] + int((resized_shape[0] / 2)) + 1

    if x_is_even:
        x_min = origin[1] - int(resized_shape[1] / 2)
        x_max = origin[1] + int((resized_shape[1] / 2)) + 1
    elif not x_is_even:
        x_min = origin[1] - int(resized_shape[1] / 2)
        x_max = origin[1] + int((resized_shape[1] / 2)) + 1

    for y_resized, y in enumerate(range(y_min, y_max)):
        for x_resized, x in enumerate(range(x_min, x_max)):
            if y >= 0 and y < array_2d.shape[0] and x >= 0 and x < array_2d.shape[1]:
                if (
                    y_resized >= 0
                    and y_resized < resized_shape[0]
                    and x_resized >= 0
                    and x_resized < resized_shape[1]
                ):
                    resized_array[y_resized, x_resized] = array_2d[y, x]
            else:
                if (
                    y_resized >= 0
                    and y_resized < resized_shape[0]
                    and x_resized >= 0
                    and x_resized < resized_shape[1]
                ):
                    resized_array[y_resized, x_resized] = pad_value

    return resized_array


@decorator_util.jit()
def replace_noise_map_2d_values_where_image_2d_values_are_negative(
    image_2d: np.ndarray, noise_map_2d: np.ndarray, target_signal_to_noise: float = 2.0
) -> np.ndarray:
    """
    If the values of a 2D image array are negative, this function replaces the corresponding 2D noise-map array
    values to meet a specified target to noise value.

    This routine is necessary because of anomolous values in images which come from our HST ACS data_type-reduction
    pipeline, where image-pixels with negative values (e.g. due to the background sky subtraction) have extremely
    small noise values, which inflate their signal-to-noise values and chi-squared contributions in the modeling.

    Parameters
    ----------
    image_2d : np.ndarray
        The 2D image array used to locate the pixel indexes in the noise-map which are replaced.
    noise_map_2d : np.ndarray
        The 2D noise-map array whose values are replaced.
    target_signal_to_noise : float
        The target signal-to-noise the noise-map valueus are changed to.

    Returns
    -------
    ndarray
        The 2D noise-map with values changed.

    Examples
    --------
    image_2d = np.ones((5,5))
    image_2d[2,2] = -1.0
    noise_map_2d = np.ones((5,5))

    noise_map_2d_replaced = replace_noise_map_2d_values_where_image_2d_values_are_negative(
        image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=2.0):
    """
    for y in range(image_2d.shape[0]):
        for x in range(image_2d.shape[1]):
            if image_2d[y, x] < 0.0:
                absolute_signal_to_noise = np.abs(image_2d[y, x]) / noise_map_2d[y, x]
                if absolute_signal_to_noise >= target_signal_to_noise:
                    noise_map_2d[y, x] = np.abs(image_2d[y, x]) / target_signal_to_noise

    return noise_map_2d


@decorator_util.jit()
def index_2d_for_index_slim_from(indexes_slim: np.ndarray, shape_native) -> np.ndarray:
    """
    For pixels on a native 2D array of shape (total_y_pixels, total_x_pixels), this array maps the slimmed 1D pixel
    indexes to their corresponding 2D pixel indexes.

    Indexing is defined from the top-left corner rightwards and downwards, whereby the top-left pixel on the 2D array
    corresponds to index 0, the pixel to its right pixel 1, and so on.

    For a 2D array of shape (3,3), flattened 1D pixel indexes are converted as follows:

    - 1D Pixel index 0 maps -> 2D pixel index [0,0].
    - 1D Pixel index 1 maps -> 2D pixel index [0,1].
    - 1D Pixel index 4 maps -> 2D pixel index [1,0].
    - 1D Pixel index 8 maps -> 2D pixel index [2,2].

    Parameters
    ----------
    indexes_slim : np.ndarray
        The slim 1D pixel indexes which are mapped to 2D indexes.
    shape_native : (int, int)
        The shape of the 2D array which the pixels are natively defined on.

    Returns
    -------
    ndarray
        An array of native 2d pixel indexes with dimensions (total_indexes, 2).

    Examples
    --------
    indexes_slim = np.array([0, 1, 2, 5])
    indexes_2d = index_2d_for_index_slim_from(indexes_slim=indexes_slim, shape=(3,3))
    """
    index_2d_for_index_slim = np.zeros((indexes_slim.shape[0], 2))

    for i, index_slim in enumerate(indexes_slim):

        index_2d_for_index_slim[i, 0] = int(index_slim / shape_native[1])
        index_2d_for_index_slim[i, 1] = int(index_slim % shape_native[1])

    return index_2d_for_index_slim


@decorator_util.jit()
def index_slim_for_index_2d_from(indexes_2d: np.ndarray, shape_native) -> np.ndarray:
    """
    For pixels on a native 2D array of shape (total_y_pixels, total_x_pixels), this array maps the 2D pixel indexes to
    their corresponding slimmed 1D pixel indexes.

    Indexing is defined from the top-left corner rightwards and downwards, whereby the top-left pixel on the 2D array
    corresponds to index 0, the pixel to its right pixel 1, and so on.

    For a 2D array of shape (3,3), 2D pixel indexes are converted as follows:

    - 2D Pixel index [0,0] maps -> 1D pixel index 0.
    - 2D Pixel index [0,1] maps -> 2D pixel index 1.
    - 2D Pixel index [1,0] maps -> 2D pixel index 4.
    - 2D Pixel index [2,2] maps -> 2D pixel index 8.

    Parameters
    ----------
    indexes_2d : np.ndarray
        The native 2D pixel indexes which are mapped to slimmed 1D indexes.
    shape_native : (int, int)
        The shape of the 2D array which the pixels are defined on.

    Returns
    -------
    ndarray
        An array of 1d pixel indexes with dimensions (total_indexes).

    Examples
    --------
    indexes_2d = np.array([[0,0], [1,0], [2,0], [2,2]])
    indexes_flat = index_flat_for_index_2d_from(indexes_2d=indexes_2d, shape=(3,3))
    """
    index_slim_for_index_native_2d = np.zeros(indexes_2d.shape[0])

    for i in range(indexes_2d.shape[0]):
        index_slim_for_index_native_2d[i] = int(
            (indexes_2d[i, 0]) * shape_native[1] + indexes_2d[i, 1]
        )

    return index_slim_for_index_native_2d


@decorator_util.jit()
def array_2d_slim_from(
    array_2d_native: np.ndarray, mask_2d: np.ndarray, sub_size: int
) -> np.ndarray:
    """
    For a 2D sub array and mask, map the values of all unmasked pixels to its slimmed 1D sub-array.

    A sub-array is an array whose native dimensions correspond to the normal array multiplied by the sub_size. For
    example if an array is shape [total_y_pixels, total_x_pixels] and the ``sub_size=2``, the sub_array is shape
    [total_y_pixels*sub_size, total_x_pixels*sub_size].

    The pixel coordinate origin is at the top left corner of the 2D array and goes right-wards and downwards,
    with sub-pixels then going right and downwards in each pixel. For example, for an array of shape (3,3) and a
    sub-grid size of 2 where all pixels are unmasked:

    - pixel [0,0] of the 2D array will correspond to index 0 of the 1D array.
    - pixel [0,1] of the 2D array will correspond to index 1 of the 1D array.
    - pixel [1,0] of the 2D array will correspond to index 2 of the 1D array.
    - pixel [2,0] of the 2D array will correspond to index 4 of the 1D array.
    - pixel [1,0] of the 2D array will correspond to index 12 of the 1D array.

    Parameters
    ----------
    array_2d_native : np.ndarray
        A 2D array of values on the dimensions of the sub-grid.
    mask_2d : np.ndarray
        A 2D array of bools, where `False` values mean unmasked and are included in the mapping.
    array_2d_native : np.ndarray
        The 2D array of values which are mapped to a 1D array.

    Returns
    -------
    ndarray
        The slimmed 1D array of values mapped from the native 2D array with dimensions (total_unmasked_pixels).

    Examples
    --------

    sub_array_2d = np.array([[ 1.0,  2.0,  5.0,  6.0],
                             [ 3.0,  4.0,  7.0,  8.0],
                             [ 9.0, 10.0, 13.0, 14.0],
                             [11.0, 12.0, 15.0, 16.0])

    mask = np.array([[True, False],
                     [False, False]])

    sub_array_2d_slim = array_2d_slim_from(mask=mask, array_2d=array_2d, sub_size=2)
    """

    total_sub_pixels = mask_2d_util.total_sub_pixels_2d_from(
        mask_2d=mask_2d, sub_size=sub_size
    )

    array_2d_slim = np.zeros(shape=total_sub_pixels)
    index = 0

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            if not mask_2d[y, x]:
                for y1 in range(sub_size):
                    for x1 in range(sub_size):
                        array_2d_slim[index] = array_2d_native[
                            y * sub_size + y1, x * sub_size + x1
                        ]
                        index += 1

    return array_2d_slim


def array_2d_native_from(
    array_2d_slim: np.ndarray, mask_2d: np.ndarray, sub_size: int
) -> np.ndarray:
    """
    For a slimmed 2D array that was computed by mapping unmasked values from a native 2D array of shape
    [total_y_pixels, total_x_pixels], map its values back to the original 2D array where masked values are set to zero.

    This uses the array ``slim_to_native`` where each index gives the 2D sub pixel indexes of the 1D array's
    unmasked pixels, for example:

    - If ``slim_to_native[0] = [0,0]``, the first value of the 1D array maps to the pixel [0,0] of the 2D array.
    - If ``slim_to_native[1] = [0,1]``, the second value of the 1D array maps to the pixel [0,1] of the 2D array.
    - If ``slim_to_native[4] = [1,1]``, the fifth value of the 1D array maps to the pixel [1,1] of the 2D array.

    This example uses an array `sub_slim_to_native` which includes sub-gridding in the mapping.

    Parameters
    ----------
    array_2d_slim : np.ndarray
        The slimmed array of values which are mapped to a 2D array.
    mask_2d : np.ndarray
        A 2D array of bools, where `False` values mean unmasked and are included in the mapping.
    sub_size : int
        The size (sub_size x sub_size) of each unmasked pixels sub-array.

    Returns
    -------
    ndarray
        A 2D array of values mapped from the 1D array with dimensions shape.

    Examples
    --------
    slim_to_native = np.array([[0,1], [1,0], [1,1], [1,2], [2,1]])

    array_1d = np.array([[2.0, 4.0, 5.0, 6.0, 8.0])

    array_2d = map_masked_1d_array_to_2d_array_from_array_1d_shape_and_slim_to_native(
        array_1d=array_1d, shape=(3,3), slim_to_native=slim_to_native)
    """

    sub_shape = (mask_2d.shape[0] * sub_size, mask_2d.shape[1] * sub_size)

    native_index_for_slim_index_2d = mask_2d_util.native_index_for_slim_index_2d_from(
        mask_2d=mask_2d, sub_size=sub_size
    ).astype("int")

    return array_2d_via_indexes_from(
        array_2d_slim=array_2d_slim,
        sub_shape=sub_shape,
        native_index_for_slim_index_2d=native_index_for_slim_index_2d,
    )


@decorator_util.jit()
def array_2d_via_indexes_from(
    array_2d_slim: np.ndarray,
    sub_shape: (int, int),
    native_index_for_slim_index_2d: np.ndarray,
) -> np.ndarray:
    """
    For a slimmed sub array with sub-indexes mapping the slimmed sub-array values to their native sub-array,
    return the native 2D sub-array.

    A sub-array is an array whose dimensions correspond to the normal array multiplied by the sub_size. For example
    if an array is shape [total_y_pixels, total_x_pixels] and the ``sub_size=2``, the sub_array is shape
    [total_y_pixels*sub_size, total_x_pixels*sub_size].

    The pixel coordinate origin is at the top left corner of the 2D array and goes right-wards and downwards,
    with sub-pixels then going right and downwards in each pixel. For example, for an array of shape (3,3) and a
    sub-grid size of 2 where all pixels are unmasked:

    - pixel [0,0] of the native 2D array will correspond to index 0 of the slim array.
    - pixel [0,1] of the native 2D array will correspond to index 1 of the slim array.
    - pixel [1,0] of the native 2D array will correspond to index 2 of the slim array.
    - pixel [2,0] of the native 2D array will correspond to index 4 of the slim array.
    - pixel [1,0] of the native 2D array will correspond to index 12 of the slim array.

    Parameters
    ----------
    array_2d_slim : np.ndarray
        The slimmed array of shape [total_values] which are mapped to the native array..
    sub_shape : (int, int)
        The 2D dimensions of the native 2D sub-array.
    native_index_for_slim_index_2d : np.narray
        An array of shape [total_values] that maps from the slimmed array to the native array.

    Returns
    -------
    ndarray
        The native 2D array of values mapped from the slimmed array with dimensions (total_values, total_values).
    """
    sub_array_native_2d = np.zeros(sub_shape)

    for slim_index in range(len(native_index_for_slim_index_2d)):

        sub_array_native_2d[
            native_index_for_slim_index_2d[slim_index, 0],
            native_index_for_slim_index_2d[slim_index, 1],
        ] = array_2d_slim[slim_index]

    return sub_array_native_2d


@decorator_util.jit()
def array_2d_slim_complex_from(
    array_2d_native: np.ndarray, mask: np.ndarray, sub_size: int
) -> np.ndarray:
    """
    For a 2D sub array and mask, map the values of all unmasked pixels to a 1D sub-array.

    A sub-array is an array whose dimensions correspond to the normal array (e.g. used to make the grid)
    multiplied by the sub_size. E.g., it is an array that would be generated using the sub-grid and not binning
    up values in sub-pixels back to the grid.

    The pixel coordinate origin is at the top left corner of the 2D array and goes right-wards and downwards,
    with sub-pixels then going right and downwards in each pixel. For example, for an array of shape (3,3) and a
    sub-grid size of 2 where all pixels are unmasked:

    - pixel [0,0] of the 2D array will correspond to index 0 of the 1D array.
    - pixel [0,1] of the 2D array will correspond to index 1 of the 1D array.
    - pixel [1,0] of the 2D array will correspond to index 2 of the 1D array.
    - pixel [2,0] of the 2D array will correspond to index 4 of the 1D array.
    - pixel [1,0] of the 2D array will correspond to index 12 of the 1D array.

    Parameters
    ----------
    array_2d_native : np.ndarray
        A 2D array of values on the dimensions of the sub-grid.
    mask : np.ndarray
        A 2D array of bools, where `False` values mean unmasked and are included in the mapping.
    array_2d : np.ndarray
        The 2D array of values which are mapped to a 1D array.

    Returns
    -------
    ndarray
        A 1D array of values mapped from the 2D array with dimensions (total_unmasked_pixels).

    Examples
    --------

    sub_array_2d = np.array([[ 1.0,  2.0,  5.0,  6.0],
                             [ 3.0,  4.0,  7.0,  8.0],
                             [ 9.0, 10.0, 13.0, 14.0],
                             [11.0, 12.0, 15.0, 16.0])

    mask = np.array([[True, False],
                     [False, False]])

    sub_array_1d = map_sub_array_2d_to_masked_sub_array_1d_from_sub_array_2d_mask_and_sub_size(
        mask=mask, array_2d=array_2d)
    """

    total_sub_pixels = mask_2d_util.total_sub_pixels_2d_from(
        mask_2d=mask, sub_size=sub_size
    )

    sub_array_1d = 0 + 0j * np.zeros(shape=total_sub_pixels)
    index = 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                for y1 in range(sub_size):
                    for x1 in range(sub_size):
                        sub_array_1d[index] = array_2d_native[
                            y * sub_size + y1, x * sub_size + x1
                        ]
                        index += 1

    return sub_array_1d


@decorator_util.jit()
def array_2d_native_complex_via_indexes_from(
    array_2d_slim: np.ndarray,
    sub_shape_native: (int, int),
    native_index_for_slim_index_2d: np.ndarray,
) -> np.ndarray:

    sub_array_2d = 0 + 0j * np.zeros(sub_shape_native)

    for slim_index in range(len(native_index_for_slim_index_2d)):
        sub_array_2d[
            native_index_for_slim_index_2d[slim_index, 0],
            native_index_for_slim_index_2d[slim_index, 1],
        ] = array_2d_slim[slim_index]

    return sub_array_2d


def numpy_array_2d_to_fits(
    array_2d: np.ndarray, file_path: str, overwrite: bool = False
):
    """
    Write a 2D NumPy array to a .fits file.

    Before outputting a NumPy array, the array may be flipped upside-down using np.flipud depending on the project
    config files. This is for Astronomy projects so that structures appear the same orientation as ``.fits`` files
    loaded in DS9.

    Parameters
    ----------
    array_2d : np.ndarray
        The 2D array that is written to fits.
    file_path : str
        The full path of the file that is output, including the file name and ``.fits`` extension.
    overwrite : bool
        If `True` and a file already exists with the input file_path the .fits file is overwritten. If `False`, an
        error is raised.

    Returns
    -------
    None

    Examples
    --------
    array_2d = np.ones((5,5))
    numpy_array_to_fits(array_2d=array_2d, file_path='/path/to/file/filename.fits', overwrite=True)
    """

    file_dir = os.path.split(file_path)[0]

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    new_hdr = fits.Header()

    flip_for_ds9 = conf.instance["general"]["fits"]["flip_for_ds9"]

    if flip_for_ds9:
        hdu = fits.PrimaryHDU(np.flipud(array_2d), new_hdr)
    else:
        hdu = fits.PrimaryHDU(array_2d, new_hdr)
    hdu.writeto(file_path)


def numpy_array_2d_from_fits(
    file_path: str, hdu: int, do_not_scale_image_data: bool = False
):
    """
    Read a 2D NumPy array from a .fits file.

    After loading the NumPy array, the array is flipped upside-down using np.flipud. This is so that the structures
    appear the same orientation as .fits files loaded in DS9.

    Parameters
    ----------
    file_path : str
        The full path of the file that is loaded, including the file name and ``.fits`` extension.
    hdu : int
        The HDU extension of the array that is loaded from the .fits file.
    do_not_scale_image_data : bool
        If True, the .fits file is not rescaled automatically based on the .fits header info.

    Returns
    -------
    ndarray
        The NumPy array that is loaded from the .fits file.

    Examples
    --------
    array_2d = numpy_array_from_fits(file_path='/path/to/file/filename.fits', hdu=0)
    """
    hdu_list = fits.open(file_path, do_not_scale_image_data=do_not_scale_image_data)

    flip_for_ds9 = conf.instance["general"]["fits"]["flip_for_ds9"]

    if flip_for_ds9:
        return np.flipud(np.array(hdu_list[hdu].data)).astype("float64")
    return np.array(hdu_list[hdu].data).astype("float64")
