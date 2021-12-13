import numpy as np
from typing import Tuple

from autoarray import numba_util
from autoarray import exc

from scipy.spatial import Delaunay


@numba_util.jit()
def data_slim_to_pixelization_unique_from(
    data_pixels, pixelization_index_for_sub_slim_index: np.ndarray, sub_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create an array describing the unique mappings between the sub-pixels of every slim data pixel and the pixelization
    pixels, which is used to perform efficiently linear algebra calculations.

    For example, assuming `sub_size=2`:

    - If 3 sub-pixels in image pixel 0 map to pixelization pixel 2 then `data_pix_to_unique[0, 0] = 2`.
    - If the fourth sub-pixel maps to pixelizaiton pixel 4, then `data_to_pix_unique[0, 1] = 4`.

    The size of the second index depends on the number of unique sub-pixel to pixelization pixels mappings in a given
    data pixel. In the example above, there were only two unique sets of mapping, but for high levels of sub-gridding
    there could be many more unique mappings all of which must be stored.

    The array `data_to_pix_unique` does not describe how many sub-pixels uniquely map to each pixelization pixel for
    a given data pixel. This information is contained in the array `data_weights`. For the example above,
    where `sub_size=2` and therefore `sub_fraction=0.25`:

    - `data_weights[0, 0] = 0.75` (because 3 sub-pixels mapped to this pixelization pixel).
    - `data_weights[0, 1] = 0.25` (because 1 sub-pixel mapped to this pixelization pixel).

    The `sub_fractions` are stored as opposed to the number of sub-pixels, because these values are used directly
    when performing the linear algebra calculation.

    The array `pix_lengths` in a 1D array of dimensions [data_pixels] describing how many unique pixelization pixels
    each data pixel's set of sub-pixels maps too.

    Parameters
    ----------
    data_pixels
        The total number of data pixels in the dataset.
    pixelization_index_for_sub_slim_index
        Maps an unmasked data sub pixel to its corresponding pixelization pixel.
    sub_size
        The size of the sub-grid defining the number of sub-pixels in every data pixel.

    Returns
    -------
    ndarray
        The unique mappings between the sub-pixels of every data pixel and the pixelization pixels, alongside arrays
        that give the weights and total number of mappings.
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


@numba_util.jit()
def data_slim_to_pixelization_unique_2_from(
    data_pixels,
    pixelization_indexes_for_sub_slim_index: np.ndarray,
    pixelization_indexes_for_sub_slim_sizes: np.ndarray,
    pixel_weights_for_sub_slim_index,
    sub_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create an array describing the unique mappings between the sub-pixels of every slim data pixel and the pixelization
    pixels, which is used to perform efficiently linear algebra calculations.

    For example, assuming `sub_size=2`:

    - If 3 sub-pixels in image pixel 0 map to pixelization pixel 2 then `data_pix_to_unique[0, 0] = 2`.
    - If the fourth sub-pixel maps to pixelizaiton pixel 4, then `data_to_pix_unique[0, 1] = 4`.

    The size of the second index depends on the number of unique sub-pixel to pixelization pixels mappings in a given
    data pixel. In the example above, there were only two unique sets of mapping, but for high levels of sub-gridding
    there could be many more unique mappings all of which must be stored.

    The array `data_to_pix_unique` does not describe how many sub-pixels uniquely map to each pixelization pixel for
    a given data pixel. This information is contained in the array `data_weights`. For the example above,
    where `sub_size=2` and therefore `sub_fraction=0.25`:

    - `data_weights[0, 0] = 0.75` (because 3 sub-pixels mapped to this pixelization pixel).
    - `data_weights[0, 1] = 0.25` (because 1 sub-pixel mapped to this pixelization pixel).

    The `sub_fractions` are stored as opposed to the number of sub-pixels, because these values are used directly
    when performing the linear algebra calculation.

    The array `pix_lengths` in a 1D array of dimensions [data_pixels] describing how many unique pixelization pixels
    each data pixel's set of sub-pixels maps too.

    Parameters
    ----------
    data_pixels
        The total number of data pixels in the dataset.
    pixelization_index_for_sub_slim_index
        Maps an unmasked data sub pixel to its corresponding pixelization pixel.
    sub_size
        The size of the sub-grid defining the number of sub-pixels in every data pixel.

    Returns
    -------
    ndarray
        The unique mappings between the sub-pixels of every data pixel and the pixelization pixels, alongside arrays
        that give the weights and total number of mappings.
    """

    sub_fraction = 1.0 / (sub_size ** 2.0)

    max_pix_mappings = int(np.max(pixelization_indexes_for_sub_slim_sizes))

    data_to_pix_unique = -1 * np.ones((data_pixels, max_pix_mappings * sub_size ** 2))
    data_weights = np.zeros((data_pixels, max_pix_mappings * sub_size ** 2))
    pix_lengths = np.zeros(data_pixels)

    for ip in range(data_pixels):

        pix_size = 0

        ip_sub_start = ip * sub_size ** 2
        ip_sub_end = ip_sub_start + sub_size ** 2

        for ip_sub in range(ip_sub_start, ip_sub_end):

            for pix_to_slim_index in range(
                pixelization_indexes_for_sub_slim_sizes[ip_sub]
            ):

                pix = pixelization_indexes_for_sub_slim_index[ip_sub, pix_to_slim_index]
                pixel_weight = pixel_weights_for_sub_slim_index[
                    ip_sub, pix_to_slim_index
                ]

                stored_already = False

                for i in range(pix_size):

                    if data_to_pix_unique[ip, i] == pix:

                        data_weights[ip, i] += sub_fraction * pixel_weight
                        stored_already = True

                if not stored_already:

                    data_to_pix_unique[ip, pix_size] = pix
                    data_weights[ip, pix_size] += sub_fraction * pixel_weight

                    pix_size += 1

        pix_lengths[ip] = pix_size

    return data_to_pix_unique, data_weights, pix_lengths


@numba_util.jit()
def mapping_matrix_from(
    pixelization_index_for_sub_slim_index: np.ndarray,
    pixels: int,
    total_mask_sub_pixels: int,
    slim_index_for_sub_slim_index: np.ndarray,
    sub_fraction: float,
) -> np.ndarray:
    """
    The `mapping_matrix` is a matrix that represents mapping between every unmasked pixel in a dataset and the pixels
    of a pixelization as a 2D matrix. It in the following paper as 
    matrix `f` https://arxiv.org/pdf/astro-ph/0302587.pdf.

    If the mappings between pixels in masked data and pixels in a pixelization are as follows:
    
    data_pixel_0 -> pixelization_pixel_0
    data_pixel_1 -> pixelization_pixel_0
    data_pixel_2 -> pixelization_pixel_1
    data_pixel_3 -> pixelization_pixel_1
    data_pixel_4 -> pixelization_pixel_2

    The `mapping_matrix` (which is of dimensions [data_pixels, pixelization_pixels] appears as follows:

    [1, 0, 0] [0->0]
    [1, 0, 0] [1->0]
    [0, 1, 0] [2->1]
    [0, 1, 0] [3->1]
    [0, 0, 1] [4->2]

    The `mapping_matrix` can be constructed using sub-pixel mappings between the data and pixelization, whereby each 
    masked data pixel is divided into sub-pixels which are paired to pixels in the pixelization. The entries
    in the `mapping_matrix` now become fractional values dependent on the sub-pixel sizes. For example, for 2x2
    sub-pixels each pixel maps with the fractional value is 1.0/(2.0^2) = 0.25. 
    
    If the mappings between sub-pixels in masked data and pixels in a pixelization are as follows:

    data_pixel_0 -> data_sub_pixel_0 -> pixelization pixel_0
    data_pixel_0 -> data_sub_pixel_1 -> pixelization pixel_1
    data_pixel_0 -> data_sub_pixel_2 -> pixelization pixel_1
    data_pixel_0 -> data_sub_pixel_3 -> pixelization pixel_1
    data_pixel_1 -> data_sub_pixel_0 -> pixelization pixel_1
    data_pixel_1 -> data_sub_pixel_1 -> pixelization pixel_1
    data_pixel_1 -> data_sub_pixel_2 -> pixelization pixel_1
    data_pixel_1 -> data_sub_pixel_3 -> pixelization pixel_1
    data_pixel_2 -> data_sub_pixel_0 -> pixelization pixel_2
    data_pixel_2 -> data_sub_pixel_1 -> pixelization pixel_2
    data_pixel_2 -> data_sub_pixel_2 -> pixelization pixel_3
    data_pixel_2 -> data_sub_pixel_3 -> pixelization pixel_3

    The `mapping_matrix` (which is still of dimensions [data_pixels, pixelization_pixels] appears as follows:

    [0.25, 0.75, 0.0, 0.0] [1 sub-pixel maps to pixel 0, 3 map to pixel 1]
    [ 0.0,  1.0, 0.0, 0.0] [All sub-pixels map to pixel 1]
    [ 0.0,  0.0, 0.5, 0.5] [2 sub-pixels map to pixel 2, 2 map to pixel 3]

    Parameters
    -----------
    pixelization_index_for_sub_slim_index
        Maps an unmasked data sub pixel to its corresponding pixelization pixel.
    pixels
        The total number of pixels in the pixelization.
    total_mask_sub_pixels
        The number of unmasked sub-pixels in the data all of which map to pixelization pixels.
    slim_index_for_sub_slim_index
        The mappings between the data's sub-pixels and their indexes without sub-pixel divisions (e.g. for
        a `sub_size=1).
    sub_fraction
        The fractional area each sub-pixel takes up in an pixel.
    """

    mapping_matrix = np.zeros((total_mask_sub_pixels, pixels))

    for sub_slim_index in range(slim_index_for_sub_slim_index.shape[0]):

        mapping_matrix[
            slim_index_for_sub_slim_index[sub_slim_index],
            pixelization_index_for_sub_slim_index[sub_slim_index],
        ] += sub_fraction

    return mapping_matrix


@numba_util.jit()
def triangle_area(pa, pb, pc):

    x1 = pa[0]
    y1 = pa[1]
    x2 = pb[0]
    y2 = pb[1]
    x3 = pc[0]
    y3 = pc[1]

    return 0.5 * np.abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)


def get_triangle_points_neighbours(relocated_grid, relocated_pixelization_grid):

    tri = Delaunay(relocated_pixelization_grid)
    triangle_ids_of_points = tri.find_simplex(relocated_grid)
    tri_simplices = tri.simplices

    return triangle_ids_of_points, tri_simplices


@numba_util.jit()
def pixel_weights_from(
    source_grid_slim,
    source_pixelization_grid,
    slim_index_for_sub_slim_index: np.ndarray,
    pixelization_indexes_for_sub_slim_index,
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
    sub_fraction
        The fractional area each sub-pixel takes up in an pixel.
    """

    pixel_weights = np.zeros(pixelization_indexes_for_sub_slim_index.shape)

    for sub_slim_index in range(slim_index_for_sub_slim_index.shape[0]):

        vertices_indexes = pixelization_indexes_for_sub_slim_index[sub_slim_index]

        if vertices_indexes[1] != -1:

            vertices_of_the_simplex = source_pixelization_grid[vertices_indexes]

            sub_gird_coordinate_on_source_place = source_grid_slim[sub_slim_index]

            term0 = triangle_area(
                pa=vertices_of_the_simplex[1],
                pb=vertices_of_the_simplex[2],
                pc=sub_gird_coordinate_on_source_place,
            )
            term1 = triangle_area(
                pa=vertices_of_the_simplex[0],
                pb=vertices_of_the_simplex[2],
                pc=sub_gird_coordinate_on_source_place,
            )
            term2 = triangle_area(
                pa=vertices_of_the_simplex[0],
                pb=vertices_of_the_simplex[1],
                pc=sub_gird_coordinate_on_source_place,
            )

            norm = term0 + term1 + term2

            weight_abc = np.array([term0, term1, term2]) / norm

            pixel_weights[sub_slim_index] = weight_abc

        else:
            pixel_weights[sub_slim_index][0] = 1.0

    return pixel_weights


@numba_util.jit()
def mapping_matrix_Delaunay_baricentric_interpolation_from(
    pixel_weights,
    pixels: int,
    total_mask_pixels: int,
    slim_index_for_sub_slim_index: np.ndarray,
    pixelization_indexes_for_sub_slim_index,
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
    sub_fraction
        The fractional area each sub-pixel takes up in an pixel.
    """

    mapping_matrix = np.zeros((total_mask_pixels, pixels))

    for sub_slim_index in range(slim_index_for_sub_slim_index.shape[0]):

        vertices_indexes = pixelization_indexes_for_sub_slim_index[sub_slim_index]

        if vertices_indexes[1] != -1:
            mapping_matrix[slim_index_for_sub_slim_index[sub_slim_index]][
                vertices_indexes
            ] += (sub_fraction * pixel_weights[sub_slim_index])
        else:
            mapping_matrix[slim_index_for_sub_slim_index[sub_slim_index]][
                vertices_indexes[0]
            ] += sub_fraction

    return mapping_matrix


# @numba_util.jit()
def pixelization_indexes_for_sub_slim_index_delaunay_from(
    delaunay, source_grid_slim
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The indexes mappings between the sub pixels and Voronoi pixelization pixels.
    For Delaunay tessellation, most sub pixels should have contribution of 3 pixelization pixels. However,
    for those ones not belonging to any triangle, we link its value to its closest point. 

    The returning result is a matrix of (len(sub_pixels, 3)) where the entries mark the relevant source pixel indexes.
    A row like [A, -1, -1] means that sub pixel only links to source pixel A.
    """

    simplex_index_for_sub_slim_index = delaunay.find_simplex(source_grid_slim)
    pixelization_indexes_for_simplex_index = delaunay.simplices

    tem_list = -1 * np.ones((len(source_grid_slim), 3), dtype="int")

    for i in range(len(source_grid_slim)):
        simplex_index = simplex_index_for_sub_slim_index[i]
        if simplex_index != -1:
            tem_list[i] = pixelization_indexes_for_simplex_index[
                simplex_index_for_sub_slim_index[i]
            ]
        else:
            tem_list[i][0] = np.argmin(
                np.sum((delaunay.points - source_grid_slim[i]) ** 2.0, axis=1)
            )
            # print(tem_list[i])

    # TODO : This can be much more cleanly combined with the for loop above, I just cant figure out how...

    pixelization_indexes_for_sub_slim_index_sizes = np.zeros(len(source_grid_slim))

    for i in range(len(source_grid_slim)):

        for pix_index in range(0, 3):

            if tem_list[i, pix_index] >= 0:

                pixelization_indexes_for_sub_slim_index_sizes[i] += 1

    return tem_list, pixelization_indexes_for_sub_slim_index_sizes


@numba_util.jit()
def pixelization_index_for_sub_slim_index_voronoi_from(
    grid: np.ndarray,
    nearest_pixelization_index_for_slim_index: np.ndarray,
    slim_index_for_sub_slim_index: np.ndarray,
    pixelization_grid: np.ndarray,
    pixel_neighbors: np.ndarray,
    pixel_neighbors_sizes: np.ndarray,
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
    grid
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
    pixel_neighbors_sizes
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

            closest_separation_pixelization_to_neighbor = 1.0e8

            for neighbor_pixelization_index in range(
                pixel_neighbors_sizes[nearest_pixelization_index]
            ):

                neighbor = pixel_neighbors[
                    nearest_pixelization_index, neighbor_pixelization_index
                ]

                distance_to_neighbor = (
                    grid[sub_slim_index, 0] - pixelization_grid[neighbor, 0]
                ) ** 2 + (grid[sub_slim_index, 1] - pixelization_grid[neighbor, 1]) ** 2

                if distance_to_neighbor < closest_separation_pixelization_to_neighbor:
                    closest_separation_pixelization_to_neighbor = distance_to_neighbor
                    closest_neighbor_pixelization_index = neighbor_pixelization_index

            neighboring_pixelization_index = pixel_neighbors[
                nearest_pixelization_index, closest_neighbor_pixelization_index
            ]
            sub_pixel_to_neighboring_pixelization_distance = (
                closest_separation_pixelization_to_neighbor
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


@numba_util.jit()
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

    1) Divide by the number of mapped data points in the pixel, to ensure all pixels have the same
    'relative' signal (i.e. a pixel with 10 pixels doesn't have x2 the signal of one with 5).

    2) Divide by the maximum pixel-signal, so that all signals vary between 0 and 1. This ensures that the
    regularization weights are defined identically for any data quantity or signal-to-noise_map ratio.

    3) Raise to the power of the hyper-parameter `signal_scale`, so the method can control the relative
    contribution of regularization in different regions of pixelization.

    Expressed differently, this function quantifies the expected value of flux every source pixel will reconstruct in
    an inversion, before the inversion is performed. Properties of the inversion associated with each source pixel
    can then be adapted to the reconstructed source's surface brightness, notably an adaptive regularization scheme
    which regularizes the brightest source pixels least (these pixels may otherwise be over smoothed and fail to
    reconstruct the data accurately).

    Parameters
    -----------
    pixels
        The total number of pixels in the pixelization the regularization scheme is applied to.
    signal_scale
        A factor which controls how rapidly the smoothness of regularization varies from high signal regions to
        low signal regions.
    pixelization_index_for_sub_slim_index
        A 1D array mapping every pixel on the grid to a pixel on the pixelization.
    slim_index_for_sub_slim_index
        The mappings between the data's sub-pixels and their indexes without sub-pixel divisions (e.g. for
        a `sub_size=1). This is used for efficiently mapping each sub pixel with its host image pixel in order to
        extract the correct value from the `hyper_image`.
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


@numba_util.jit()
def adaptive_pixel_signals_Delaunay_version_from(
    pixels: int,
    pixel_weights: float,
    signal_scale: float,
    pixelization_indexes_for_sub_slim_index: np.ndarray,
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
    signal_scale
        A factor which controls how rapidly the smoothness of regularization varies from high signal regions to
        low signal regions.
    regular_to_pix
        A 1D array util every pixel on the grid to a pixel on the pixelization.
    hyper_image
        The image of the galaxy which is used to compute the weigghted pixel signals.
    """

    pixel_signals = np.zeros((pixels,))
    pixel_sizes = np.zeros((pixels,))

    for sub_slim_index in range(len(pixelization_indexes_for_sub_slim_index)):

        vertices_indexes = pixelization_indexes_for_sub_slim_index[sub_slim_index]

        mask_1d_index = slim_index_for_sub_slim_index[sub_slim_index]

        if vertices_indexes[1] != -1:

            pixel_signals[vertices_indexes] += (
                hyper_image[mask_1d_index] * pixel_weights[sub_slim_index]
            )
            pixel_sizes[vertices_indexes] += 1
        else:
            pixel_signals[vertices_indexes[0]] += hyper_image[mask_1d_index]
            pixel_sizes[vertices_indexes[0]] += 1

    pixel_sizes[pixel_sizes == 0] = 1
    pixel_signals /= pixel_sizes
    pixel_signals /= np.max(pixel_signals)

    return pixel_signals ** signal_scale
