import numpy as np

from autoarray import numba_util


@numba_util.jit()
def total_pixels_1d_from(mask_1d: np.ndarray) -> int:
    """
    Returns the total number of unmasked pixels in a mask.

    Parameters
    ----------
    mask_1d
        A 2D array of bools, where `False` values are unmasked and included when counting pixels.

    Returns
    -------
    int
        The total number of pixels that are unmasked.

    Examples
    --------

    mask = np.array([[True, False, True],
                 [False, False, False]
                 [True, False, True]])

    total_regular_pixels = total_regular_pixels_from(mask=mask)
    """

    total_regular_pixels = 0

    for x in range(mask_1d.shape[0]):
        if not mask_1d[x]:
            total_regular_pixels += 1

    return total_regular_pixels


@numba_util.jit()
def total_sub_pixels_1d_from(mask_1d: np.ndarray, sub_size: int) -> int:
    """
    Returns the total number of sub-pixels in unmasked pixels in a mask.

    Parameters
    ----------
    mask_1d
        A 2D array of bools, where `False` values are unmasked and included when counting sub pixels.
    sub_size
        The size of the sub-grid that each pixel of the 2D mask array is divided into.

    Returns
    -------
    int
        The total number of sub pixels that are unmasked.

    Examples
    --------

    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])

    total_sub_pixels = total_sub_pixels_from(mask=mask, sub_size=2)
    """
    return total_pixels_1d_from(mask_1d) * sub_size


@numba_util.jit()
def native_index_for_slim_index_1d_from(
    mask_1d: np.ndarray, sub_size: int
) -> np.ndarray:
    """
    Returns an array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its
    corresponding native 2D pixel using its (y,x) pixel indexes.

    For example, for a sub-grid size of 2x2, if pixel [2,5] corresponds to the first pixel in the masked slim array:

    - The first sub-pixel in this pixel on the 1D array is sub_native_index_for_sub_slim_index_2d[4] = [2,5]
    - The second sub-pixel in this pixel on the 1D array is sub_native_index_for_sub_slim_index_2d[5] = [2,6]
    - The third sub-pixel in this pixel on the 1D array is sub_native_index_for_sub_slim_index_2d[5] = [3,5]

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked.
    sub_size
        The size of the sub-grid in each mask pixel.

    Returns
    -------
    ndarray
        An array that maps pixels from a slimmed array of shape [total_unmasked_pixels*sub_size] to its native array
        of shape [total_pixels*sub_size, total_pixels*sub_size].

    Examples
    --------
    mask_2d = np.array([[True, True, True],
                     [True, False, True]
                     [True, True, True]])

    sub_native_index_for_sub_slim_index_2d = sub_native_index_for_sub_slim_index_via_mask_2d_from(mask_2d=mask_2d, sub_size=1)

    """

    total_sub_pixels = total_sub_pixels_1d_from(mask_1d=mask_1d, sub_size=sub_size)
    sub_native_index_for_sub_slim_index_1d = np.zeros(shape=total_sub_pixels)

    sub_slim_index = 0

    for x in range(mask_1d.shape[0]):
        if not mask_1d[x]:
            for x1 in range(sub_size):
                sub_native_index_for_sub_slim_index_1d[sub_slim_index] = (
                    x * sub_size
                ) + x1
                sub_slim_index += 1

    return sub_native_index_for_sub_slim_index_1d
