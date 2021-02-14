from autoarray.mask import mask_1d_util

import os
from autoarray import decorator_util
import numpy as np
from astropy.io import fits


@decorator_util.jit()
def array_1d_slim_from(
    array_1d_native: np.ndarray, mask_1d: np.ndarray, sub_size: int
) -> np.ndarray:
    """
    For a 1D array and mask, map the values of all unmasked pixels to its slimmed 1D array.

    The 1D array has native dimensions corresponding to the array pixels (without masking) multiplied by the
    sub_size. For example if a native array is shape [total_unmasked_pixels] and the ``sub_size=2``, the array is
    shape [total_unmasked_x_pixels*sub_size].

    The pixel coordinate origin is at the left of the 1D array and goes right, with pixels then going right in
    each pixel. For example, for an array of shape (3,) and a sub-grid size of 2 where all pixels are unmasked:

    - pixel[0] of the native 1D array will correspond to index 0 of the slim array (which is the first sub-pixel in
    the array).
    - pixel[1] of the native 1D array will correspond to index 1 of the slim array (which is the second sub-pixel in
    the array).

    If the native array is masked and the third pixel is masked (e.g. its mask_1d entry is `True`) then:

    - pixels [0], [1], [2] and [3] of the native 1D array will correspond to indexes 0, 1, 2, 3 of the slim array.
    - pixels [4] and [5] of the native 1D array do not map to the slim array (these sub-pixels are masked).
    - pixel [6], [7], etc. of the native 1D array will correspond to indexes 4, 5, etc. of the slim array.

    Parameters
    ----------
    array_1d_native : np.ndarray
        A 1D array of values on the dimensions of the native array.
    mask_1d : np.ndarray
        A 1D array of bools, where `False` values mean unmasked and are included in the mapping.
    sub_size : int
        The sub-grid size of the array.

    Returns
    -------
    ndarray
        The slimmed 1D array of values mapped from the native 1d array with
        dimensions [total_unmasked_pixels*sub_size].

    Examples
    --------

    array_1d_native = np.array([ 1.0,  2.0,  5.0,  6.0])

    mask = np.array([True, False, False, False]])

    array_1d_slim = array_1d_slim_from(array_1d_native, array_2d=array_2d, sub_size=2)
    """

    total_sub_pixels = mask_1d_util.total_sub_pixels_1d_from(
        mask_1d=mask_1d, sub_size=sub_size
    )

    line_1d_slim = np.zeros(shape=total_sub_pixels)
    index = 0

    for x in range(mask_1d.shape[0]):
        if not mask_1d[x]:
            for x1 in range(sub_size):
                line_1d_slim[index] = array_1d_native[x * sub_size + x1]
                index += 1

    return line_1d_slim


def array_1d_native_from(
    array_1d_slim: np.ndarray, mask_1d: np.ndarray, sub_size: int
) -> np.ndarray:

    sub_shape = mask_1d.shape[0] * sub_size

    native_index_for_slim_index_1d = mask_1d_util.native_index_for_slim_index_1d_from(
        mask_1d=mask_1d, sub_size=sub_size
    ).astype("int")

    return array_1d_via_indexes_1d_from(
        array_1d_slim=array_1d_slim,
        sub_shape=sub_shape,
        native_index_for_slim_index_1d=native_index_for_slim_index_1d,
    )


@decorator_util.jit()
def array_1d_via_indexes_1d_from(
    array_1d_slim: np.ndarray,
    sub_shape: int,
    native_index_for_slim_index_1d: np.ndarray,
) -> np.ndarray:
    """
    For a slimmed 1D array with sub-indexes mapping the slimmed array values to their native array indexes,
    return the native 1D array.

    The 1D array has dimensions correspond to the size of the 1D array multiplied by the sub_size. For example
    if an array is shape [total_x_pixels] and the `sub_size=2`, the array is shape [total_x_pixels*sub_size].

    The pixel coordinate origin is at the left of the 1D array and goes right, with sub-pixels then going right in
    each pixel. For example, for an array of shape (3,3) and a sub-grid size of 2 where all pixels are unmasked:

    - pixel[0] of the native 1D array will correspond to index 0 of the slim array (which is the first sub-pixel in
    the line).
    - pixel[1] of the native 1D array will correspond to index 1 of the slim array (which is the second sub-pixel in
    the line).

    If the native line is masked and the third pixel is masked (e.g. its mask_1d entry is `True`) then:

    - pixels [0], [1], [2] and [3] of the native 1D array will correspond to indexes 0, 1, 2, 3 of the slim array.
    - pixels [4] and [5] of the native 1D array do not map to the slim array (these sub-pixels are masked).
    - pixel [6], [7], etc. of the native 1D array will correspond to indexes 4, 5, etc. of the slim array.

    Parameters
    ----------
    array_1d_slim : np.ndarray
        The slimmed array of shape [total_x_pixels*sub_size] which are mapped to the native array.
    sub_shape : int
        The 1D dimensions of the native 1D sub line.
    native_index_for_slim_index_1d : np.narray
        An array of shape [total_x_pixels*sub_size] that maps from the slimmed array to the native array.

    Returns
    -------
    ndarray
        The native 1D array of values mapped from the slimmed array with dimensions (total_x_pixels).
    """
    array_1d_native = np.zeros(sub_shape)

    for slim_index in range(len(native_index_for_slim_index_1d)):

        array_1d_native[native_index_for_slim_index_1d[slim_index]] = array_1d_slim[
            slim_index
        ]

    return array_1d_native


def numpy_array_1d_to_fits(
    array_1d: np.ndarray, file_path: str, overwrite: bool = False
):
    """
    Write a 1D NumPy array to a .fits file.

    Parameters
    ----------
    array_2d : np.ndarray
        The 2D array that is written to fits.
    file_path : str
        The full path of the file that is output, including the file name and ``.fits`` extension.
    overwrite : bool
        If `True` and a file already exists with the input file_path the .fits file is overwritten. If False, an error
        will be raised.

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
    hdu = fits.PrimaryHDU(array_1d, new_hdr)
    hdu.writeto(file_path)


def numpy_array_1d_from_fits(file_path: str, hdu: int):
    """
    Read a 1D NumPy array from a .fits file.

    After loading the NumPy array, the array is flipped upside-down using np.flipud. This is so that the structures
    appear the same orientation as .fits files loaded in DS9.

    Parameters
    ----------
    file_path : str
        The full path of the file that is loaded, including the file name and ``.fits`` extension.
    hdu : int
        The HDU extension of the array that is loaded from the .fits file.

    Returns
    -------
    ndarray
        The NumPy array that is loaded from the .fits file.

    Examples
    --------
    array_2d = numpy_array_from_fits(file_path='/path/to/file/filename.fits', hdu=0)
    """
    hdu_list = fits.open(file_path)
    return np.array(hdu_list[hdu].data)
