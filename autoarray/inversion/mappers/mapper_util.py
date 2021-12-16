import numpy as np
from typing import Tuple

from autoarray import numba_util
from autoarray import exc

from scipy.spatial import Delaunay


@numba_util.jit()
def data_slim_to_pixelization_unique_from(
    data_pixels,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_indexes_for_sub_slim_sizes: np.ndarray,
    pix_weights_for_sub_slim_index,
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
    pix_indexes_for_sub_slim_index
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

    max_pix_mappings = int(np.max(pix_indexes_for_sub_slim_sizes))

    data_to_pix_unique = -1 * np.ones((data_pixels, max_pix_mappings * sub_size ** 2))
    data_weights = np.zeros((data_pixels, max_pix_mappings * sub_size ** 2))
    pix_lengths = np.zeros(data_pixels)

    for ip in range(data_pixels):

        pix_size = 0

        ip_sub_start = ip * sub_size ** 2
        ip_sub_end = ip_sub_start + sub_size ** 2

        for ip_sub in range(ip_sub_start, ip_sub_end):

            for pix_to_slim_index in range(pix_indexes_for_sub_slim_sizes[ip_sub]):

                pix = pix_indexes_for_sub_slim_index[ip_sub, pix_to_slim_index]
                pixel_weight = pix_weights_for_sub_slim_index[ip_sub, pix_to_slim_index]

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
    pix_indexes_for_sub_slim_index,
) -> np.ndarray:
    """
    Returns the mapping matrix, by iterating over the known mappings between the sub-grid and pixelization.

    Parameters
    -----------
    pixels
        The number of pixels in the pixelization.
    total_mask_pixels
        The number of datas pixels in the observed datas and thus on the grid.
    slim_index_for_sub_slim_index
        The mappings between the data's sub slimmed indexes and the slimmed indexes on the non sub-sized indexes.
    pixelizationes_index_for_sub_slim_index
        The mappings between the pixelization grid's pixels and the data's slimmed pixels.
    sub_fraction
        The fractional area each sub-pixel takes up in an pixel.
    """

    pixel_weights = np.zeros(pix_indexes_for_sub_slim_index.shape)

    for sub_slim_index in range(slim_index_for_sub_slim_index.shape[0]):

        vertices_indexes = pix_indexes_for_sub_slim_index[sub_slim_index]

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
def mapping_matrix_from(
    pixel_weights,
    pixels: int,
    total_mask_sub_pixels: int,
    slim_index_for_sub_slim_index: np.ndarray,
    pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index,
    sub_fraction: float,
) -> np.ndarray:
    """
    Returns the mapping matrix, by iterating over the known mappings between the sub-grid and pixelization.

    Parameters
    -----------
    pix_indexes_for_sub_slim_index
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

    mapping_matrix = np.zeros((total_mask_sub_pixels, pixels))

    for sub_slim_index in range(slim_index_for_sub_slim_index.shape[0]):

        vertices_indexes = pix_indexes_for_sub_slim_index[sub_slim_index]

        if pix_size_for_sub_slim_index[sub_slim_index] > 1:
            mapping_matrix[slim_index_for_sub_slim_index[sub_slim_index]][
                vertices_indexes
            ] += (sub_fraction * pixel_weights[sub_slim_index])
        else:
            mapping_matrix[slim_index_for_sub_slim_index[sub_slim_index]][
                vertices_indexes[0]
            ] += sub_fraction

    return mapping_matrix


#  @numba_util.jit()
def pix_indexes_for_sub_slim_index_delaunay_from(
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
    pix_indexes_for_simplex_index = delaunay.simplices

    pix_indexes_for_sub_slim_index = -1 * np.ones(
        (len(source_grid_slim), 3), dtype="int"
    )

    for i in range(len(source_grid_slim)):
        simplex_index = simplex_index_for_sub_slim_index[i]
        if simplex_index != -1:
            pix_indexes_for_sub_slim_index[i] = pix_indexes_for_simplex_index[
                simplex_index_for_sub_slim_index[i]
            ]
        else:
            pix_indexes_for_sub_slim_index[i][0] = np.argmin(
                np.sum((delaunay.points - source_grid_slim[i]) ** 2.0, axis=1)
            )

    pix_indexes_for_sub_slim_index_sizes = np.sum(
        pix_indexes_for_sub_slim_index >= 0, axis=1
    )

    return pix_indexes_for_sub_slim_index, pix_indexes_for_sub_slim_index_sizes


@numba_util.jit()
def pix_indexes_for_sub_slim_index_voronoi_from(
    grid: np.ndarray,
    nearest_pix_index_for_slim_index: np.ndarray,
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
    nearest_pix_index_for_slim_index
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

    pix_index_for_sub_slim_index = np.zeros(shape=(grid.shape[0], 1))

    for sub_slim_index in range(grid.shape[0]):

        nearest_pix_index = nearest_pix_index_for_slim_index[
            slim_index_for_sub_slim_index[sub_slim_index]
        ]

        whiletime = 0

        while True:

            if whiletime > 1000000:
                raise exc.PixelizationException

            nearest_pix_center = pixelization_grid[nearest_pix_index]

            sub_pixel_to_nearest_pix_distance = (
                grid[sub_slim_index, 0] - nearest_pix_center[0]
            ) ** 2 + (grid[sub_slim_index, 1] - nearest_pix_center[1]) ** 2

            closest_separation_pix_to_neighbor = 1.0e8

            for neighbor_pix_index in range(pixel_neighbors_sizes[nearest_pix_index]):

                neighbor = pixel_neighbors[nearest_pix_index, neighbor_pix_index]

                distance_to_neighbor = (
                    grid[sub_slim_index, 0] - pixelization_grid[neighbor, 0]
                ) ** 2 + (grid[sub_slim_index, 1] - pixelization_grid[neighbor, 1]) ** 2

                if distance_to_neighbor < closest_separation_pix_to_neighbor:
                    closest_separation_pix_to_neighbor = distance_to_neighbor
                    closest_neighbor_pix_index = neighbor_pix_index

            neighboring_pix_index = pixel_neighbors[
                nearest_pix_index, closest_neighbor_pix_index
            ]
            sub_pixel_to_neighboring_pix_distance = closest_separation_pix_to_neighbor

            whiletime += 1

            if (
                sub_pixel_to_nearest_pix_distance
                <= sub_pixel_to_neighboring_pix_distance
            ):
                pix_index_for_sub_slim_index[sub_slim_index, 0] = nearest_pix_index
                break
            else:
                nearest_pix_index = neighboring_pix_index

    return pix_index_for_sub_slim_index


@numba_util.jit()
def adaptive_pixel_signals_from(
    pixels: int,
    pixel_weights: np.ndarray,
    signal_scale: float,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
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

    for sub_slim_index in range(len(pix_indexes_for_sub_slim_index)):

        vertices_indexes = pix_indexes_for_sub_slim_index[sub_slim_index]

        mask_1d_index = slim_index_for_sub_slim_index[sub_slim_index]

        if pix_size_for_sub_slim_index[sub_slim_index] > 1:

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
