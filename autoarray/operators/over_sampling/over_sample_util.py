from autoarray.numpy_wrapper import register_pytree_node_class


import numpy as np
from typing import Tuple

from autoarray.mask.mask_2d import Mask2D

from autoarray.geometry import geometry_util
from autoarray import numba_util
from autoarray.mask import mask_2d_util
from autoarray import type as ty


@numba_util.jit()
def total_sub_pixels_2d_from(sub_size: np.ndarray) -> int:
    """
    Returns the total number of sub-pixels in unmasked pixels in a mask.

    Parameters
    ----------
    mask_2d
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
    return int(np.sum(sub_size**2))


@numba_util.jit()
def native_sub_index_for_slim_sub_index_2d_from(
    mask_2d: np.ndarray, sub_size: np.ndarray
) -> np.ndarray:
    """
    Returns an array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its
    corresponding native 2D pixel using its (y,x) pixel indexes.

    For example, for the following ``Mask2D`` for ``sub_size=1``:

    ::
        [[True,  True,  True, True]
         [True, False, False, True],
         [True, False,  True, True],
         [True,  True,  True, True]]

    This has three unmasked (``False`` values) which have the ``slim`` indexes:

    ::
        [0, 1, 2]

    The array ``native_index_for_slim_index_2d`` is therefore:

    ::
        [[1,1], [1,2], [2,1]]

    For a ``Mask2D`` with ``sub_size=2`` each unmasked ``False`` entry is split into a sub-pixel of size 2x2 and
    there are therefore 12 ``slim`` indexes:

    ::
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    The array ``native_index_for_slim_index_2d`` is therefore:

    ::
        [[2,2], [2,3], [2,4], [2,5], [3,2], [3,3], [3,4], [3,5], [4,2], [4,3], [5,2], [5,3]]

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

    total_sub_pixels = total_sub_pixels_2d_from(sub_size=sub_size)
    sub_native_index_for_sub_slim_index_2d = np.zeros(shape=(total_sub_pixels, 2))

    slim_index = 0
    sub_slim_index = 0

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            if not mask_2d[y, x]:
                sub = sub_size[slim_index]

                for y1 in range(sub):
                    for x1 in range(sub):
                        sub_native_index_for_sub_slim_index_2d[sub_slim_index, :] = (
                            (y * sub) + y1,
                            (x * sub) + x1,
                        )
                        sub_slim_index += 1

                slim_index += 1

    return sub_native_index_for_sub_slim_index_2d


@numba_util.jit()
def slim_index_for_sub_slim_index_via_mask_2d_from(
    mask_2d: np.ndarray, sub_size: np.ndarray
) -> np.ndarray:
    """ "
    For pixels on a native 2D array of shape (total_y_pixels, total_x_pixels), compute a slimmed array which, for
    every unmasked pixel on the native 2D array, maps the slimmed sub-pixel indexes to their slimmed pixel indexes.

    For example, for a sub-grid size of 2, the following mappings from sub-pixels to 2D array pixels are:

    - slim_index_for_sub_slim_index[0] = 0 -> The first sub-pixel maps to the first unmasked pixel on the native 2D array.
    - slim_index_for_sub_slim_index[3] = 0 -> The fourth sub-pixel maps to the first unmasked pixel on the native 2D array.
    - slim_index_for_sub_slim_index[7] = 1 -> The eighth sub-pixel maps to the second unmasked pixel on the native 2D array.

    Parameters
    ----------
    mask_2d
        The mask whose indexes are mapped.
    sub_size
        The sub-size of the grid on the mask, so that the sub-mask indexes can be computed correctly.

    Returns
    -------
    np.ndarray
        The array of shape [total_unmasked_pixels] mapping every unmasked pixel on the native 2D mask array to its
        slimmed index on the sub-mask array.

    Examples
    --------
    mask = np.array([[True, False, True]])
    slim_index_for_sub_slim_index = slim_index_for_sub_slim_index_via_mask_2d_from(mask_2d=mask_2d, sub_size=2)
    """

    total_sub_pixels = total_sub_pixels_2d_from(sub_size=sub_size)

    slim_index_for_sub_slim_index = np.zeros(shape=total_sub_pixels)
    slim_index = 0
    sub_slim_index = 0

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            if not mask_2d[y, x]:
                sub = sub_size[slim_index]

                for y1 in range(sub):
                    for x1 in range(sub):
                        slim_index_for_sub_slim_index[sub_slim_index] = slim_index
                        sub_slim_index += 1

                slim_index += 1

    return slim_index_for_sub_slim_index


@numba_util.jit()
def sub_slim_index_for_sub_native_index_from(sub_mask_2d: np.ndarray):
    """
    Returns a 2D array which maps every `False` entry of a 2D mask to its sub slim mask array. Every
    True entry is given a value -1.

    This is used as a convenience tool for creating structures util between different grids and structures.

    For example, if we had a 3x4 mask:

    [[False, True, False, False],
     [False, True, False, False],
     [False, False, False, True]]]

    The sub_slim_index_for_sub_native_index array would be:

    [[0, -1, 2, 3],
     [4, -1, 5, 6],
     [7, 8, 9, -1]]

    Parameters
    ----------
    sub_mask_2d
        The 2D mask that the util array is created for.

    Returns
    -------
    ndarray
        The 2D array mapping 2D mask entries to their 1D masked array indexes.

    Examples
    --------
    mask = np.full(fill_value=False, shape=(9,9))
    sub_two_to_one = mask_to_mask_1d_index_from(mask=mask)
    """

    sub_slim_index_for_sub_native_index = -1 * np.ones(shape=sub_mask_2d.shape)

    sub_mask_1d_index = 0

    for sub_mask_y in range(sub_mask_2d.shape[0]):
        for sub_mask_x in range(sub_mask_2d.shape[1]):
            if sub_mask_2d[sub_mask_y, sub_mask_x] == False:
                sub_slim_index_for_sub_native_index[
                    sub_mask_y, sub_mask_x
                ] = sub_mask_1d_index
                sub_mask_1d_index += 1

    return sub_slim_index_for_sub_native_index


@numba_util.jit()
def oversample_mask_2d_from(mask: np.ndarray, sub_size: int) -> np.ndarray:
    """
    Returns a new mask of shape (mask.shape[0] * sub_size, mask.shape[1] * sub_size) where all boolean values are
    expanded according to the `sub_size`.

    For example, if the input mask is:

    mask = np.array([
        [True, True, True],
        [True, False, True],
        [True, True, True]
    ])

    and the sub_size is 2, the output mask would be:

    expanded_mask = np.array([
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [True, True, False, False, True, True],
        [True, True, False, False, True, True],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True]
    ])

    This is used throughout the code to handle uniform oversampling calculations.

    Parameters
    ----------
    mask
        The mask from which the over sample mask is computed.
    sub_size
        The factor by which the mask is oversampled.

    Returns
    -------
    The mask oversampled by the input sub_size.
    """
    oversample_mask = np.full(
        (mask.shape[0] * sub_size, mask.shape[1] * sub_size), True
    )

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x]:
                oversample_mask[
                    y * sub_size : (y + 1) * sub_size, x * sub_size : (x + 1) * sub_size
                ] = False

    return oversample_mask


@numba_util.jit()
def grid_2d_slim_over_sampled_via_mask_from(
    mask_2d: np.ndarray,
    pixel_scales: ty.PixelScales,
    sub_size: np.ndarray,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into
    a finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
    scaled coordinates a the centre of every sub-pixel defined by this 2D mask array.

    The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_size**2, 2). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index. Masked coordinates are therefore
    removed and not included in the slimmed grid.

    Grid2D are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0.
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked and therefore included as part of the calculated
        sub-grid.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    sub_size
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    -------
    ndarray
        A slimmed sub grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The sub grid array has dimensions (total_unmasked_pixels*sub_size**2, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_slim = grid_2d_slim_over_sampled_via_mask_from(mask=mask, pixel_scales=(0.5, 0.5), sub_size=1, origin=(0.0, 0.0))
    """

    total_sub_pixels = np.sum(sub_size**2)

    grid_slim = np.zeros(shape=(total_sub_pixels, 2))

    centres_scaled = geometry_util.central_scaled_coordinate_2d_from(
        shape_native=mask_2d.shape, pixel_scales=pixel_scales, origin=origin
    )

    index = 0
    sub_index = 0

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            if not mask_2d[y, x]:
                sub = sub_size[index]

                y_sub_half = pixel_scales[0] / 2
                y_sub_step = pixel_scales[0] / (sub)

                x_sub_half = pixel_scales[1] / 2
                x_sub_step = pixel_scales[1] / (sub)

                y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
                x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

                for y1 in range(sub):
                    for x1 in range(sub):
                        grid_slim[sub_index, 0] = -(
                            y_scaled - y_sub_half + y1 * y_sub_step + (y_sub_step / 2.0)
                        )
                        grid_slim[sub_index, 1] = (
                            x_scaled - x_sub_half + x1 * x_sub_step + (x_sub_step / 2.0)
                        )
                        sub_index += 1

                index += 1

    return grid_slim


@numba_util.jit()
def binned_array_2d_from(
    array_2d: np.ndarray,
    mask_2d: np.ndarray,
    sub_size: np.ndarray,
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into
    a finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
    scaled coordinates a the centre of every sub-pixel defined by this 2D mask array.

    The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_size**2, 2). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index. Masked coordinates are therefore
    removed and not included in the slimmed grid.

    Grid2D are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0.
    Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked and therefore included as part of the calculated
        sub-grid.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    sub_size
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin
        The (y,x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    -------
    ndarray
        A slimmed sub grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The sub grid array has dimensions (total_unmasked_pixels*sub_size**2, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_slim = grid_2d_slim_over_sampled_via_mask_from(mask=mask, pixel_scales=(0.5, 0.5), sub_size=1, origin=(0.0, 0.0))
    """

    total_pixels = mask_2d_util.total_pixels_2d_from(
        mask_2d=mask_2d,
    )

    sub_fraction = 1.0 / sub_size**2

    binned_array_2d_slim = np.zeros(shape=total_pixels)

    index = 0
    sub_index = 0

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            if not mask_2d[y, x]:
                sub = sub_size[index]

                for y1 in range(sub):
                    for x1 in range(sub):
                        binned_array_2d_slim[index] += (
                            array_2d[sub_index] * sub_fraction[index]
                        )
                        sub_index += 1

                index += 1

    return binned_array_2d_slim
