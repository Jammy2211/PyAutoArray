from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Union
from typing import List, Tuple

from autoarray.structures.arrays.uniform_2d import Array2D

if TYPE_CHECKING:
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.geometry import geometry_util
from autoarray.mask.mask_2d import Mask2D

from autoarray import numba_util

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
        ).astype("int")

    return Array2D(values=over_sample_size, mask=mask)


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
                sub_slim_index_for_sub_native_index[sub_mask_y, sub_mask_x] = (
                    sub_mask_1d_index
                )
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

    sub_size = sub_size_list[-1] * np.ones(radial_grid.shape)

    for i in range(radial_grid.shape[0]):
        for j in range(len(radial_list)):
            if radial_grid[i] < radial_list[j]:
                # if use_jax:
                #     # while this makes it run, it is very, very slow
                #     sub_size = sub_size.at[i].set(sub_size_list[j])
                # else:
                sub_size[i] = sub_size_list[j]
                break

    return sub_size


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

    total_sub_pixels = np.sum(sub_size**2)

    grid_slim = np.zeros(shape=(total_sub_pixels, 2))

    centres_scaled = geometry_util.central_scaled_coordinate_2d_numba_from(
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
                        # if use_jax:
                        #     # while this makes it run, it is very, very slow
                        #     grid_slim = grid_slim.at[sub_index, 0].set(
                        #         -(
                        #             y_scaled
                        #             - y_sub_half
                        #             + y1 * y_sub_step
                        #             + (y_sub_step / 2.0)
                        #         )
                        #     )
                        #     grid_slim = grid_slim.at[sub_index, 1].set(
                        #         x_scaled
                        #         - x_sub_half
                        #         + x1 * x_sub_step
                        #         + (x_sub_step / 2.0)
                        #     )
                        # else:
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

    total_pixels = np.sum(~mask_2d)

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
                        # if use_jax:
                        #     binned_array_2d_slim = binned_array_2d_slim.at[index].add(
                        #         array_2d[sub_index] * sub_fraction[index]
                        #     )
                        # else:
                        binned_array_2d_slim[index] += (
                            array_2d[sub_index] * sub_fraction[index]
                        )
                        sub_index += 1

                index += 1

    return binned_array_2d_slim


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
