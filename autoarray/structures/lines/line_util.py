import numpy as np

from autoarray import decorator_util
from autoarray.mask import mask_1d_util


# @decorator_util.jit()
def line_1d_slim_from(
    line_1d_native: np.ndarray, mask_1d: np.ndarray, sub_size: int
) -> np.ndarray:
    """
    For a 1D sub line and mask, map the values of all unmasked pixels to its slimmed 1D sub-line.

    A sub-line is an array whose native dimensions correspond to the normal line (without masking) multiplied by the
    sub_size. For example if a native line is shape [total_unmasked_x_pixels] and the ``sub_size=2``, the sub_line is
    shape [total_unmasked_x_pixels*sub_size].

    The pixel coordinate origin is at the left of the 1D array and goes right, with sub-pixels then going right in
    each pixel. For example, for an array of shape (3,3) and a sub-grid size of 2 where all pixels are unmasked:

    - pixel[0] of the native 1D array will correspond to index 0 of the slim array (which is the first sub-pixel in
    the line).
    - pixel[1] of the native 1D array will correspond to index 1 of the slim array (which is the second sub-pixel in
    the line).

    If the native lined is masked and the third pixel is masked (e.g. its mask_1d entry is `True`) then:

    - pixels [0], [1], [2] and [3] of the native 1D array will correspond to indexes 0, 1, 2, 3 of the slim array.
    - pixels [4] and [5] of the native 1D array do not map to the slim array (these sub-pixels are masked).
    - pixel [6], [7], etc. of the native 1D array will correspond to indexes 4, 5, etc. of the slim array.

    Parameters
    ----------
    line_1d_native : np.ndarray
        A 1D array of values on the dimensions of the native sub-line.
    mask_1d : np.ndarray
        A 1D array of bools, where `False` values mean unmasked and are included in the mapping.
    sub_size : int
        The sub-grid size of the sub-line.

    Returns
    -------
    ndarray
        The slimmed 1D sub-line of values mapped from the native 1d sub-line with
        dimensions [total_unmasked_x_pixels*sub_size].

    Examples
    --------

    sub_line_1d_native = np.array([ 1.0,  2.0,  5.0,  6.0])

    mask = np.array([True, False, False, False]])

    sub_line_1d_slim = sub_line_1d_slim_from(sub_line_1d_native, array_2d=array_2d, sub_size=2)
    """

    total_sub_pixels = mask_1d_util.total_sub_pixels_1d_from(
        mask_1d=mask_1d, sub_size=sub_size
    )

    line_1d_slim = np.zeros(shape=total_sub_pixels)
    index = 0

    for x in range(mask_1d.shape[0]):
        if not mask_1d[x]:
            for x1 in range(sub_size):
                line_1d_slim[index] = line_1d_native[x * sub_size + x1]
                index += 1

    return line_1d_slim


def line_1d_native_from(
    line_1d_slim: np.ndarray, mask_1d: np.ndarray, sub_size: int
) -> np.ndarray:

    sub_shape = mask_1d.shape[0] * sub_size

    native_index_for_slim_index_1d = mask_1d_util.native_index_for_slim_index_1d_from(
        mask_1d=mask_1d, sub_size=sub_size
    ).astype("int")

    return line_1d_via_indexes_1d_from(
        line_1d_slim=line_1d_slim,
        sub_shape=sub_shape,
        native_index_for_slim_index_1d=native_index_for_slim_index_1d,
    )


@decorator_util.jit()
def line_1d_via_indexes_1d_from(
    line_1d_slim: np.ndarray, sub_shape: int, native_index_for_slim_index_1d: np.ndarray
) -> np.ndarray:
    """
    For a slimmed sub line with sub-indexes mapping the slimmed sub line values to their native sub line,
    return the native 1D sub line.

    A sub line is an array whose dimensions correspond to the normal line multiplied by the sub_size. For example
    if an array is shape [total_x_pixels] and the `sub_size=2`, the sub-line is shape [total_x_pixels*sub_size].

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
    line_1d_slim : np.ndarray
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
    sub_line_1d_native = np.zeros(sub_shape)

    for slim_index in range(len(native_index_for_slim_index_1d)):

        sub_line_1d_native[native_index_for_slim_index_1d[slim_index]] = line_1d_slim[
            slim_index
        ]

    return sub_line_1d_native
