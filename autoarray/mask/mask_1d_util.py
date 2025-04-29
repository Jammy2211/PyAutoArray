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
def native_index_for_slim_index_1d_from(
    mask_1d: np.ndarray,
) -> np.ndarray:
    """
    Returns an array of shape [total_unmasked_pixels] that maps every unmasked pixel to its
    corresponding native 2D pixel using its (y,x) pixel indexes.

    For example, for a grid size of 3x3, if pixel [0,1] corresponds to the first pixel in the masked slim array:

    - The first pixel in this pixel on the 1D array is native_index_for_slim_index_2d[0] = [0,1]

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked.

    Returns
    -------
    ndarray
        An array that maps pixels from a slimmed array of shape [total_unmasked_pixels] to its native array
        of shape [total_pixels, total_pixels].

    Examples
    --------
    mask_2d = np.array([[True, True, True],
                     [True, False, True]
                     [True, True, True]])

    native_index_for_slim_index_1d =  native_index_for_slim_index_1d_from(mask_2d=mask_2d)

    """

    total_pixels = total_pixels_1d_from(mask_1d=mask_1d)
    native_index_for_slim_index_1d = np.zeros(shape=total_pixels)

    slim_index = 0

    for x in range(mask_1d.shape[0]):
        if not mask_1d[x]:
            native_index_for_slim_index_1d[slim_index] = x
            slim_index += 1

    return native_index_for_slim_index_1d
