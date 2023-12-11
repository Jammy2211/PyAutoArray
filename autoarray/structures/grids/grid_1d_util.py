from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, List, Union, Tuple

if TYPE_CHECKING:
    from autoarray.mask.mask_1d import Mask1D

from autoarray.structures.arrays import array_1d_util
from autoarray import numba_util
from autoarray.geometry import geometry_util
from autoarray.structures.grids import grid_2d_util
from autoarray.mask import mask_1d_util
from autoarray import type as ty


def convert_grid_1d(
    grid_1d: Union[np.ndarray, List], mask_1d: Mask1D, store_native: bool = False
) -> np.ndarray:
    """
    The `manual` classmethods in the Grid2D object take as input a list or ndarray which is returned as a Grid2D.

    This function performs the following and checks and conversions on the input:

    1: If the input is a list, convert it to an ndarray.
    2: Check that the number of sub-pixels in the array is identical to that of the mask.
    3) Map the input ndarray to its `slim` representation.

    For a Grid2D, `slim` refers to a 2D NumPy array of shape [total_coordinates, 2] and `native` a 3D NumPy array of
    shape [total_y_coordinates, total_x_coordinates, 2]

    Parameters
    ----------
    grid_2d
        The input (y,x) grid of coordinates which is converted to an ndarray if it is a list.
    mask_2d
        The mask of the output Array2D.
    store_native
        If True, the ndarray is stored in its native format [total_y_pixels, total_x_pixels, 2]. This avoids
        mapping large data arrays to and from the slim / native formats, which can be a computational bottleneck.
    """

    grid_1d = grid_2d_util.convert_grid(grid=grid_1d)

    is_native = grid_1d.shape[0] == mask_1d.sub_shape_native[0]

    if is_native == store_native:
        return grid_1d
    elif not store_native:
        return grid_1d_slim_from(
            grid_1d_native=grid_1d, mask_1d=mask_1d, sub_size=mask_1d.sub_size
        )
    return grid_1d_native_from(
        grid_1d_slim=grid_1d, mask_1d=mask_1d, sub_size=mask_1d.sub_size
    )


def grid_1d_slim_via_shape_slim_from(
    shape_slim: Tuple[int],
    pixel_scales: ty.PixelScales,
    sub_size: int,
    origin: Tuple[float] = (0.0,),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 1D mask with shape (total_pixels,) is divided into a finer uniform
    grid of shape (total_pixels*sub_size, ). This routine computes the (x) scaled coordinates at the centre of every
    sub-pixel defined by a 1D shape of the overall grid.

    Grid2D are defined from left to right, where the first unmasked sub-pixel corresponds to index 0. Sub-pixels that
    are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_size, ).

    Parameters
    ----------
    shape_slim
        The (x) shape of the 1D array the sub-grid of coordinates is computed for.
    pixel_scales
        The (x) scaled units to pixel units conversion factor of the 1D mask array.
    sub_size
        The size of the sub-grid that each pixel of the 1D mask array is divided into.
    origin
        The (x) origin of the 1D array, which the sub-grid is shifted around.

    Returns
    -------
    ndarray
        A sub grid of (x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The sub grid array has dimensions (total_unmasked_pixels*sub_size, ).

    Examples
    --------
    sub_grid_1d = grid_1d_via_shape_slim_from(mask=mask, pixel_scales=(0.5, 0.5), sub_size=2, origin=(0.0, 0.0))
    """
    return grid_1d_slim_via_mask_from(
        mask_1d=np.full(fill_value=False, shape=shape_slim),
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        origin=origin,
    )


@numba_util.jit()
def grid_1d_slim_via_mask_from(
    mask_1d: np.ndarray,
    pixel_scales: ty.PixelScales,
    sub_size: int,
    origin: Tuple[float] = (0.0,),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 1D mask with shape (total_pixels,) is divided into a finer uniform
    grid of shape (total_pixels*sub_size, ). This routine computes the (x) scaled coordinates at the centre of every
    sub-pixel defined by this 1D mask array.

    Grid2D are defined from left to right, where the first unmasked sub-pixel corresponds to index 0. Sub-pixels that
    are part of the same mask array pixel are indexed next to one another, such that the second
    sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.

    The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_size, ).

    Parameters
    ----------
    mask_1d
        A 1D array of bools, where `False` values are unmasked and therefore included as part of the calculated
        sub-grid.
    pixel_scales
        The (x) scaled units to pixel units conversion factor of the 2D mask array.
    sub_size
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin
        The (x) origin of the 2D array, which the sub-grid is shifted around.

    Returns
    -------
    ndarray
        A sub grid of (x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The sub grid array has dimensions (total_unmasked_pixels*sub_size, ).

    Examples
    --------
    mask = np.array([True, False, True, False, False, False])
    grid_slim = grid_1d_via_mask_from(mask_1d=mask_1d, pixel_scales=(0.5, 0.5), sub_size=1, origin=(0.0, 0.0))
    """

    total_sub_pixels = mask_1d_util.total_sub_pixels_1d_from(mask_1d, sub_size)

    grid_1d = np.zeros(shape=(total_sub_pixels,))

    centres_scaled = geometry_util.central_scaled_coordinate_1d_from(
        shape_slim=mask_1d.shape, pixel_scales=pixel_scales, origin=origin
    )

    sub_index = 0

    x_sub_half = pixel_scales[0] / 2
    x_sub_step = pixel_scales[0] / (sub_size)

    for x in range(mask_1d.shape[0]):
        if not mask_1d[x]:
            x_scaled = (x - centres_scaled[0]) * pixel_scales[0]

            for x1 in range(sub_size):
                grid_1d[sub_index] = (
                    x_scaled - x_sub_half + x1 * x_sub_step + (x_sub_step / 2.0)
                )
                sub_index += 1

    return grid_1d


def grid_1d_slim_from(
    grid_1d_native: np.ndarray, mask_1d: np.ndarray, sub_size: int
) -> np.ndarray:
    """
    For a native 1D grid and mask of shape [total_pixels], map the values of all unmasked pixels to a slimmed grid of
    shape [total_unmasked_pixels].

    The pixel coordinate origin is at the left of the native grid and goes right-wards, such
    that for a grid of shape (4,) where pixels 0, 1 and 3 are unmasked:

    - pixel [0] of the 1D native grid will correspond to index 0 of the 1D grid.
    - pixel [1] of the 1D native grid will correspond to index 1 of the 1D grid.
    - pixel [3] of the 1D native grid will correspond to index 2 of the 1D grid.

    Parameters
    ----------
    grid_1d_native : ndarray
        The native grid of (x) values which are mapped to the slimmed grid.
    mask_1d
        A 1D array of bools, where `False` values mean unmasked and are included in the mapping.
    sub_size
        The size (sub_size x sub_size) of each unmasked pixels sub-array.

    Returns
    -------
    ndarray
        A 1D slim grid of values mapped from the 1D native grid with dimensions (total_unmasked_pixels).
    """

    return array_1d_util.array_1d_slim_from(
        array_1d_native=grid_1d_native, mask_1d=mask_1d, sub_size=sub_size
    )


def grid_1d_native_from(
    grid_1d_slim: np.ndarray, mask_1d: np.ndarray, sub_size: int
) -> np.ndarray:
    """
    For a slimmed 1D grid of shape [total_unmasked_pixels*sub_size], that was computed by extracting the unmasked values
    from a native 1D grid of shape [total_pixels*sub_size], map the slimmed grid's coordinates back to the native 1D
    grid where masked values are set to zero.

    This uses a 1D array 'slim_to_native' where each index gives the 1D pixel indexes of the grid's native unmasked
    pixels, for example:

    - If slim_to_native[0] = [0], the first value of the 1D slimmed grid maps to the pixel [0] of the native 1D grid.
    - If slim_to_native[1] = [2], the second value of the 1D slimmed grid maps to the pixel [2] of the native 1D grid.
    - If slim_to_native[4] = [9], the fifth value of the 1D slimmed grid maps to the pixels [9] of the native 1D grid.

    Parameters
    ----------
    grid_1d_slim
        The (x) values of the slimmed 1D grid which are mapped to the native 1D grid.
    mask_1d
        A 1D array of bools, where `False` values means unmasked and are included in the mapping.
    sub_size
        The size of each unmasked pixels sub-grid.

    Returns
    -------
    ndarray
        A NumPy array of shape [total_pixels*sub_size] corresponding to the (x) values of the native 1D
        mapped from the slimmed grid.
    """
    return array_1d_util.array_1d_native_from(
        array_1d_slim=grid_1d_slim, mask_1d=mask_1d, sub_size=sub_size
    )
