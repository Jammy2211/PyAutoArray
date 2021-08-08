import numpy as np
from typing import Tuple
from autoarray import decorator_util

from autoarray import exc


@decorator_util.jit()
def data_slim_to_pixelization_unique_from(
    data_pixels, pixelization_index_for_sub_slim_index: np.ndarray, sub_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) by computing it using `w_tilde_preload`
    (see `w_tilde_preload_interferometer_from`) for an imaging inversion.

    To compute the curvature matrix via w_tilde the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * w_tilde * mapping matrix

    This function speeds this calculation up in two ways:

    1) Instead of using `w_tilde` (dimensions [image_pixels, image_pixels] it uses `w_tilde_preload` (dimensions
    [image_pixels, kernel_overlap]). The massive reduction in the size of this matrix in memory allows for much fast
    computation.

    2) It omits the `mapping_matrix` and instead uses directly the 1D vector that maps every image pixel to a source
    pixel `native_index_for_slim_index`. This exploits the sparsity in the `mapping_matrix` to directly
    compute the `curvature_matrix` (e.g. it condenses the triple matrix multiplication into a double for loop!).

    Parameters
    ----------
    w_tilde_preload
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    pixelization_index_for_sub_slim_index
        The mappings between the pixelization grid's pixels and the data's slimmed pixels.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pixelization_pixels
        The total number of pixels in the pixelization that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    sub_fraction = 1.0 / (sub_size ** 2.0)

    data_to_pix_unique = -1 * np.ones((data_pixels, sub_size ** 2))
    data_weights = np.zeros((data_pixels, sub_size ** 2))
    pix_lengths = np.zeros(data_pixels)

    for ip in range(data_pixels):

        pix_size = 0

        ip_sub_start = ip * sub_size ** 2
        ip_sub_end = ip_sub_start + sub_size ** 2

        for ip_sub in range(ip_sub_start, ip_sub_end):

            pix = pixelization_index_for_sub_slim_index[ip_sub]

            stored_already = False

            for i in range(pix_size):

                if data_to_pix_unique[ip, i] == pix:

                    data_weights[ip, i] += sub_fraction
                    stored_already = True

            if not stored_already:

                data_to_pix_unique[ip, pix_size] = pix
                data_weights[ip, pix_size] += sub_fraction

                pix_size += 1

        pix_lengths[ip] = pix_size

    return data_to_pix_unique, data_weights, pix_lengths


@decorator_util.jit()
def mapping_matrix_from(
    pixelization_index_for_sub_slim_index: np.ndarray,
    pixels: int,
    total_mask_pixels: int,
    slim_index_for_sub_slim_index: np.ndarray,
    sub_fraction: float,
) -> np.ndarray:
    """
    Returns the mapping matrix, by iterating over the known mappings between the sub-grid and pixelization.

    Parameters
    -----------
    pixelization_index_for_sub_slim_index
        The mappings between the pixelization grid's pixels and the data's slimmed pixels.
    pixels
        The number of pixels in the pixelization.
    total_mask_pixels
        The number of datas pixels in the observed datas and thus on the grid.
    slim_index_for_sub_slim_index
        The mappings between the data's sub slimmed indexes and the slimmed indexes on the non sub-sized indexes.
    sub_fraction : float
        The fractional area each sub-pixel takes up in an pixel.
    """

    mapping_matrix = np.zeros((total_mask_pixels, pixels))

    for sub_slim_index in range(slim_index_for_sub_slim_index.shape[0]):
        mapping_matrix[
            slim_index_for_sub_slim_index[sub_slim_index],
            pixelization_index_for_sub_slim_index[sub_slim_index],
        ] += sub_fraction

    return mapping_matrix


@decorator_util.jit()
def pixelization_index_for_voronoi_sub_slim_index_from(
    grid: np.ndarray,
    nearest_pixelization_index_for_slim_index: np.ndarray,
    slim_index_for_sub_slim_index: np.ndarray,
    pixelization_grid: np.ndarray,
    pixel_neighbors: np.ndarray,
    pixel_neighbors_size: np.ndarray,
) -> np.ndarray:
    """
    Returns the mappings between a set of slimmed sub-grid pixels and pixelization pixels, using information on
    how the pixels hosting each sub-pixel map to their closest pixelization pixel on the slim grid in the data-plane
    and the pixelization's pixel centres.

    To determine the complete set of slim sub-pixel to pixelization pixel mappings, we must pair every sub-pixel to
    its nearest pixel. Using a full nearest neighbor search to do this is slow, thus the pixel neighbors (derived via
    the Voronoi grid) are used to localize each nearest neighbor search by using a graph search.

    Parameters
    ----------
    grid : Grid2D
        The grid of (y,x) scaled coordinates at the centre of every unmasked pixel, which has been traced to
        to an irgrid via lens.
    nearest_pixelization_index_for_slim_index
        A 1D array that maps every slimmed data-plane pixel to its nearest pixelization pixel.
    slim_index_for_sub_slim_index
        The mappings between the data slimmed sub-pixels and their regular pixels.
    pixelization_grid
        The (y,x) centre of every Voronoi pixel in arc-seconds.
    pixel_neighbors
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_size
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.
    """

    pixelization_index_for_voronoi_sub_slim_index = np.zeros(grid.shape[0])

    for sub_slim_index in range(grid.shape[0]):

        nearest_pixelization_index = nearest_pixelization_index_for_slim_index[
            slim_index_for_sub_slim_index[sub_slim_index]
        ]

        whiletime = 0

        while True:

            if whiletime > 1000000:
                raise exc.PixelizationException

            nearest_pixelization_pixel_center = pixelization_grid[
                nearest_pixelization_index
            ]

            sub_pixel_to_nearest_pixelization_distance = (
                (grid[sub_slim_index, 0] - nearest_pixelization_pixel_center[0]) ** 2
                + (grid[sub_slim_index, 1] - nearest_pixelization_pixel_center[1]) ** 2
            )

            closest_separation_from_pixelization_to_neighbor = 1.0e8

            for neighbor_pixelization_index in range(
                pixel_neighbors_size[nearest_pixelization_index]
            ):

                neighbor = pixel_neighbors[
                    nearest_pixelization_index, neighbor_pixelization_index
                ]

                separation_from_neighbor = (
                    grid[sub_slim_index, 0] - pixelization_grid[neighbor, 0]
                ) ** 2 + (grid[sub_slim_index, 1] - pixelization_grid[neighbor, 1]) ** 2

                if (
                    separation_from_neighbor
                    < closest_separation_from_pixelization_to_neighbor
                ):
                    closest_separation_from_pixelization_to_neighbor = (
                        separation_from_neighbor
                    )
                    closest_neighbor_pixelization_index = neighbor_pixelization_index

            neighboring_pixelization_index = pixel_neighbors[
                nearest_pixelization_index, closest_neighbor_pixelization_index
            ]
            sub_pixel_to_neighboring_pixelization_distance = (
                closest_separation_from_pixelization_to_neighbor
            )

            whiletime += 1

            if (
                sub_pixel_to_nearest_pixelization_distance
                <= sub_pixel_to_neighboring_pixelization_distance
            ):
                pixelization_index_for_voronoi_sub_slim_index[
                    sub_slim_index
                ] = nearest_pixelization_index
                break
            else:
                nearest_pixelization_index = neighboring_pixelization_index

    return pixelization_index_for_voronoi_sub_slim_index


@decorator_util.jit()
def adaptive_pixel_signals_from(
    pixels: int,
    signal_scale: float,
    pixelization_index_for_sub_slim_index: np.ndarray,
    slim_index_for_sub_slim_index: np.ndarray,
    hyper_image: np.ndarray,
) -> np.ndarray:
    """
    Returns the (hyper) signal in each pixel, where the signal is the sum of its mapped data values.
    These pixel-signals are used to compute the effective regularization weight of each pixel.

    The pixel signals are computed as follows:

    1) Divide by the number of mappe data points in the pixel, to ensure all pixels have the same
    'relative' signal (i.e. a pixel with 10 pixels doesn't have x2 the signal of one with 5).

    2) Divided by the maximum pixel-signal, so that all signals vary between 0 and 1. This ensures that the
    regularization weight_list are defined identically for any data quantity or signal-to-noise_map ratio.

    3) Raised to the power of the hyper-parameter *signal_scale*, so the method can control the relative
    contribution regularization in different regions of pixelization.

    Parameters
    -----------
    pixels
        The total number of pixels in the pixelization the regularization scheme is applied to.
    signal_scale : float
        A factor which controls how rapidly the smoothness of regularization varies from high signal regions to
        low signal regions.
    regular_to_pix
        A 1D array util every pixel on the grid to a pixel on the pixelization.
    hyper_image
        The image of the galaxy which is used to compute the weigghted pixel signals.
    """

    pixel_signals = np.zeros((pixels,))
    pixel_sizes = np.zeros((pixels,))

    for sub_slim_index in range(len(pixelization_index_for_sub_slim_index)):
        mask_1d_index = slim_index_for_sub_slim_index[sub_slim_index]
        pixel_signals[
            pixelization_index_for_sub_slim_index[sub_slim_index]
        ] += hyper_image[mask_1d_index]
        pixel_sizes[pixelization_index_for_sub_slim_index[sub_slim_index]] += 1

    pixel_sizes[pixel_sizes == 0] = 1
    pixel_signals /= pixel_sizes
    pixel_signals /= np.max(pixel_signals)

    return pixel_signals ** signal_scale
