from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Union
from typing import List, Tuple

from autoarray.structures.arrays.uniform_2d import Array2D

if TYPE_CHECKING:
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.mask.mask_2d import Mask2D


from autoarray import type as ty


def over_sample_size_convert_to_array_2d_from(
    over_sample_size: Union[int, np.ndarray], mask: Union[np.ndarray, Mask2D]
):
    """
    Returns the over sample size as an `Array2D` object, for example converting it from a single integer.

    The interface allows a user to specify the `over_sample_size` as either:

    - A single integer, whereby over sampling is performed to this degree for every pixel.
    - An ndarray with the same number of entries as the mask, to enable adaptive over sampling.

    This function converts these input structures to an `Array2D` which is used internally in the source code
    to perform computations.

    Parameters
    ----------
    over_sample_size
        The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
        values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
        into each pixel.

    Returns
    -------

    """
    if isinstance(over_sample_size, int):
        over_sample_size = np.full(
            fill_value=over_sample_size, shape=mask.pixels_in_mask
        )

    return Array2D(values=np.array(over_sample_size).astype("int"), mask=mask)


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

    # Step 1: Identify unmasked (False) pixels
    unmasked_indices = np.argwhere(~mask_2d)
    n_unmasked = unmasked_indices.shape[0]

    # Step 2: Compute total number of sub-pixels
    sub_pixels_per_pixel = sub_size**2

    # Step 3: Repeat slim indices for each sub-pixel
    slim_indices = np.arange(n_unmasked)
    slim_index_for_sub_slim_index = np.repeat(slim_indices, sub_pixels_per_pixel)

    return slim_index_for_sub_slim_index


def sub_size_radial_bins_from(
    radial_grid: np.ndarray,
    sub_size_list: np.ndarray,
    radial_list: np.ndarray,
) -> np.ndarray:
    """
    Returns an adaptive sub-grid size based on the radial distance of every pixel from the centre of the mask.

    The adaptive sub-grid size is computed as follows:

    1) Compute the radial distance of every pixel in the mask from the centre of the mask.
    2) For every pixel, determine the sub-grid size based on the radial distance of that pixel. For example, if
    the first entry in `radial_list` is 0.5 and the first entry in `sub_size_list` 8, all pixels with a radial
    distance less than 0.5 will have a sub-grid size of 8x8.

    This scheme can produce high sub-size values towards the centre of the mask, where the galaxy is brightest and
    has the most rapidly changing light profile which requires a high sub-grid size to resolve accurately.

    Parameters
    ----------
    mask
        The mask defining the 2D region where the over-sampled grid is computed.
    radial_grid
        The radial distance of every pixel from the centre of the mask.
    sub_size_list
        The sub-grid size for every radial bin.
    radial_list
        The radial distance defining each bin, which are refeneced based on the previous entry. For example, if
        the first entry is 0.5, the second 1.0 and the third 1.5, the adaptive sub-grid size will be between 0.5
        and 1.0 for the first sub-grid size, between 1.0 and 1.5 for the second sub-grid size, etc.

    Returns
    -------
    A uniform over-sampling object with an adaptive sub-grid size based on the radial distance of every pixel from
    the centre of the mask.
    """

    # Use np.searchsorted to find the first index where radial_grid[i] < radial_list[j]
    bin_indices = np.searchsorted(radial_list, radial_grid, side="left")

    # Clip indices to stay within bounds of sub_size_list
    bin_indices = np.clip(bin_indices, 0, len(sub_size_list) - 1)

    return sub_size_list[bin_indices]

def grid_2d_slim_over_sampled_via_mask_from(
    mask_2d: np.ndarray,
    pixel_scales: ty.PixelScales,
    sub_size: np.ndarray,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into
    a finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
    scaled coordinates at the centre of every sub-pixel defined by this 2D mask array.

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

    H, W = mask_2d.shape
    sy, sx = pixel_scales
    oy, ox = origin

    # 1) Find unmasked pixel indices in row-major order
    rows, cols = np.nonzero(~mask_2d)
    Npix = rows.size

    # 2) Broadcast or validate sub_size array
    sub_arr = np.asarray(sub_size)
    if sub_arr.ndim == 0:
        sub_arr = np.full(Npix, int(sub_arr), int)
    elif sub_arr.ndim == 1 and sub_arr.size == Npix:
        sub_arr = sub_arr.astype(int)
    else:
        raise ValueError(f"sub_size must be scalar or length-{Npix} array, got shape {sub_arr.shape}")

    # 3) Compute pixel centers (y ↑ up, x → right)
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    y_pix = (cy - rows) * sy + oy
    x_pix = (cols - cx) * sx + ox

    # 4) For each pixel, generate its sub-pixel coords and collect
    coords_list = []
    for i in range(Npix):
        s = sub_arr[i]
        dy = sy / s
        dx = sx / s

        # y offsets: from top (+sy/2 - dy/2) down to bottom (-sy/2 + dy/2)
        y_off = np.linspace(+sy/2 - dy/2, -sy/2 + dy/2, s)
        # x offsets: left to right
        x_off = np.linspace(-sx/2 + dx/2, +sx/2 - dx/2, s)

        # build subgrid
        y_sub, x_sub = np.meshgrid(y_off, x_off, indexing="ij")
        y_sub = y_sub.ravel()
        x_sub = x_sub.ravel()

        # center + offsets
        y_center = y_pix[i]
        x_center = x_pix[i]
        coords = np.stack([y_center + y_sub, x_center + x_sub], axis=1)

        coords_list.append(coords)

    # 5) Concatenate all sub-pixel blocks in row-major pixel order
    return np.vstack(coords_list)


def over_sample_size_via_radial_bins_from(
    grid: Grid2D,
    sub_size_list: List[int],
    radial_list: List[float],
    centre_list: List[Tuple] = None,
) -> Array2D:
    """
    Returns an adaptive sub-grid size based on the radial distance of every pixel from the centre of the mask.

    The adaptive sub-grid size is computed as follows:

    1) Compute the radial distance of every pixel in the mask from the centre of the mask (or input centres).
    2) For every pixel, determine the sub-grid size based on the radial distance of that pixel. For example, if
    the first entry in `radial_list` is 0.5 and the first entry in `sub_size_list` 8, all pixels with a radial
    distance less than 0.5 will have a sub-grid size of 8x8.

    This scheme can produce high sub-size values towards the centre of the mask, where the galaxy is brightest and
    has the most rapidly changing light profile which requires a high sub-grid size to resolve accurately.

    If the data has multiple galaxies, the `centre_list` can be used to define the centre of each galaxy
    and therefore increase the sub-grid size based on the light profile of each individual galaxy.

    Parameters
    ----------
    mask
        The mask defining the 2D region where the over-sampled grid is computed.
    sub_size_list
        The sub-grid size for every radial bin.
    radial_list
        The radial distance defining each bin, which are refeneced based on the previous entry. For example, if
        the first entry is 0.5, the second 1.0 and the third 1.5, the adaptive sub-grid size will be between 0.5
        and 1.0 for the first sub-grid size, between 1.0 and 1.5 for the second sub-grid size, etc.
    centre_list
        A list of centres for each galaxy whose centres require higher sub-grid sizes.

    Returns
    -------
    A uniform over-sampling object with an adaptive sub-grid size based on the radial distance of every pixel from
    the centre of the mask.
    """

    if centre_list is None:
        centre_list = [grid.mask.mask_centre]

    sub_size = np.zeros(grid.shape_slim)

    for centre in centre_list:
        radial_grid = grid.distances_to_coordinate_from(coordinate=centre)

        sub_size_of_centre = sub_size_radial_bins_from(
            radial_grid=np.array(radial_grid.array),
            sub_size_list=np.array(sub_size_list),
            radial_list=np.array(radial_list),
        )

        sub_size = np.where(sub_size_of_centre > sub_size, sub_size_of_centre, sub_size)

    return Array2D(values=sub_size, mask=grid.mask)


def over_sample_size_via_adapt_from(
    data: Array2D,
    noise_map: Array2D,
    signal_to_noise_cut: float = 5.0,
    sub_size_lower: int = 2,
    sub_size_upper: int = 4,
) -> Array2D:
    """
    Returns an adaptive sub-grid size based on the signal-to-noise of the data.

    The adaptive sub-grid size is computed as follows:

    1) The signal-to-noise of every pixel is computed as the data divided by the noise-map.
    2) For all pixels with signal-to-noise above the signal-to-noise cut, the sub-grid size is set to the upper
      value. For all other pixels, the sub-grid size is set to the lower value.

    This scheme can produce low sub-size values over entire datasets if the data has a low signal-to-noise. However,
    just because the data has a low signal-to-noise does not mean that the sub-grid size should be low.

    To mitigate this, the signal-to-noise cut is set to the maximum signal-to-noise of the data divided by 2.0 if
    it this value is below the signal-to-noise cut.

    Parameters
    ----------
    data
        The data which is to be fitted via a calculation using this over-sampling sub-grid.
    noise_map
        The noise-map of the data.
    signal_to_noise_cut
        The signal-to-noise cut which defines whether the sub-grid size is the upper or lower value.
    sub_size_lower
        The sub-grid size for pixels with signal-to-noise below the signal-to-noise cut.
    sub_size_upper
        The sub-grid size for pixels with signal-to-noise above the signal-to-noise cut.

    Returns
    -------
    The adaptive sub-grid sizes.
    """
    signal_to_noise = data / noise_map

    if np.max(signal_to_noise) < (2.0 * signal_to_noise_cut):
        signal_to_noise_cut = np.max(signal_to_noise) / 2.0

    sub_size = np.where(
        signal_to_noise > signal_to_noise_cut, sub_size_upper, sub_size_lower
    )

    return Array2D(values=sub_size, mask=data.mask)
