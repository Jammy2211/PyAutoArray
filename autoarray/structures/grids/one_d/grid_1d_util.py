from autoarray import decorator_util
import numpy as np

from autoarray.util import mask_util
from autoarray.geometry import geometry_util

from typing import Tuple


def grid_1d_via_shape_slim_from(
    shape_slim: (int,),
    pixel_scales: Tuple[float],
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
    shape_slim : (int, int)
        The (x) shape of the 1D array the sub-grid of coordinates is computed for.
    pixel_scales : (float, float)
        The (x) scaled units to pixel units conversion factor of the 1D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 1D mask array is divided into.
    origin : (float, flloat)
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
    return grid_1d_via_mask_from(
        mask_1d=np.full(fill_value=False, shape=shape_slim),
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        origin=origin,
    )


@decorator_util.jit()
def grid_1d_via_mask_from(
    mask_1d: np.ndarray,
    pixel_scales: Tuple[float],
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
    mask_1d : np.ndarray
        A 1D array of bools, where `False` values are unmasked and therefore included as part of the calculated
        sub-grid.
    pixel_scales : (float, float)
        The (x) scaled units to pixel units conversion factor of the 2D mask array.
    sub_size : int
        The size of the sub-grid that each pixel of the 2D mask array is divided into.
    origin : (float, float)
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

    total_sub_pixels = mask_util.total_sub_pixels_1d_from(mask_1d, sub_size)

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
