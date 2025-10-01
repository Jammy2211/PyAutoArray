import numpy as np
from typing import Tuple

from autoarray import numba_util
from autoarray.inversion.pixelization.mesh import mesh_numba_util


@numba_util.jit()
def data_slim_to_pixelization_unique_from(
    data_pixels,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_sizes_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index,
    pix_pixels: int,
    sub_size: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create an array describing the unique mappings between the sub-pixels of every slim data pixel and the pixelization
    pixels, which is used to perform efficiently linear algebra calculations.

    For example, assuming `sub_size=2`:

    - If 3 sub-pixels in image pixel 0 map to pixelization pixel 2 then `data_pix_to_unique[0, 0] = 2`.
    - If the fourth sub-pixel maps to pixelization pixel 4, then `data_to_pix_unique[0, 1] = 4`.

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

    sub_fraction = 1.0 / (sub_size**2.0)

    max_pix_mappings = int(np.max(pix_sizes_for_sub_slim_index))

    # TODO : Work out if we can reduce size from np.max(sub_size) using sub_size of max_pix_mappings.

    data_to_pix_unique = -1 * np.ones(
        (data_pixels, max_pix_mappings * np.max(sub_size) ** 2)
    )
    data_weights = np.zeros((data_pixels, max_pix_mappings * np.max(sub_size) ** 2))
    pix_lengths = np.zeros(data_pixels)
    pix_check = -1 * np.ones(shape=pix_pixels)

    ip_sub_start = 0

    for ip in range(data_pixels):
        pix_check[:] = -1

        pix_size = 0

        ip_sub_end = ip_sub_start + sub_size[ip] ** 2

        for ip_sub in range(ip_sub_start, ip_sub_end):
            for pix_interp_index in range(pix_sizes_for_sub_slim_index[ip_sub]):
                pix = pix_indexes_for_sub_slim_index[ip_sub, pix_interp_index]
                pixel_weight = pix_weights_for_sub_slim_index[ip_sub, pix_interp_index]

                if pix_check[pix] > -0.5:
                    data_weights[ip, int(pix_check[pix])] += (
                        sub_fraction[ip] * pixel_weight
                    )

                else:
                    data_to_pix_unique[ip, pix_size] = pix
                    data_weights[ip, pix_size] += sub_fraction[ip] * pixel_weight
                    pix_check[pix] = pix_size
                    pix_size += 1

        ip_sub_start = ip_sub_end

        pix_lengths[ip] = pix_size

    return data_to_pix_unique, data_weights, pix_lengths


@numba_util.jit()
def pix_indexes_for_sub_slim_index_delaunay_from(
    source_plane_data_grid,
    simplex_index_for_sub_slim_index,
    pix_indexes_for_simplex_index,
    delaunay_points,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The indexes mappings between the sub pixels and Voronoi mesh pixels.
    For Delaunay tessellation, most sub pixels should have contribution of 3 pixelization pixels. However,
    for those ones not belonging to any triangle, we link its value to its closest point.

    The returning result is a matrix of (len(sub_pixels, 3)) where the entries mark the relevant source pixel indexes.
    A row like [A, -1, -1] means that sub pixel only links to source pixel A.
    """

    pix_indexes_for_sub_slim_index = -1 * np.ones(
        shape=(source_plane_data_grid.shape[0], 3)
    )

    for i in range(len(source_plane_data_grid)):
        simplex_index = simplex_index_for_sub_slim_index[i]
        if simplex_index != -1:
            pix_indexes_for_sub_slim_index[i] = pix_indexes_for_simplex_index[
                simplex_index_for_sub_slim_index[i]
            ]
        else:
            pix_indexes_for_sub_slim_index[i][0] = np.argmin(
                np.sum((delaunay_points - source_plane_data_grid[i]) ** 2.0, axis=1)
            )

    pix_indexes_for_sub_slim_index_sizes = np.sum(
        pix_indexes_for_sub_slim_index >= 0, axis=1
    )

    return pix_indexes_for_sub_slim_index, pix_indexes_for_sub_slim_index_sizes


@numba_util.jit()
def pixel_weights_delaunay_from(
    source_plane_data_grid,
    source_plane_mesh_grid,
    slim_index_for_sub_slim_index: np.ndarray,
    pix_indexes_for_sub_slim_index,
) -> np.ndarray:
    """
    Returns the weights of the mappings between the masked sub-pixels and the Delaunay pixelization.

    Weights are determiend via a nearest neighbor interpolation scheme, whereby every data-sub pixel maps to three
    Delaunay pixel vertexes (in the source frame). The weights of these 3 mappings depends on the distance of the
    coordinate to each vertex, with the highest weight being its closest neighbor,

    Parameters
    ----------
    source_plane_data_grid
        A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
        `source` reference frame.
    source_plane_mesh_grid
        The 2D grid of (y,x) centres of every pixelization pixel in the `source` frame.
    slim_index_for_sub_slim_index
        The mappings between the data's sub slimmed indexes and the slimmed indexes on the non sub-sized indexes.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    """

    pixel_weights = np.zeros(pix_indexes_for_sub_slim_index.shape)

    for sub_slim_index in range(slim_index_for_sub_slim_index.shape[0]):
        pix_indexes = pix_indexes_for_sub_slim_index[sub_slim_index]

        if pix_indexes[1] != -1:
            vertices_of_the_simplex = source_plane_mesh_grid[pix_indexes]

            sub_gird_coordinate_on_source_place = source_plane_data_grid[sub_slim_index]

            area_0 = mesh_numba_util.delaunay_triangle_area_from(
                corner_0=vertices_of_the_simplex[1],
                corner_1=vertices_of_the_simplex[2],
                corner_2=sub_gird_coordinate_on_source_place,
            )
            area_1 = mesh_numba_util.delaunay_triangle_area_from(
                corner_0=vertices_of_the_simplex[0],
                corner_1=vertices_of_the_simplex[2],
                corner_2=sub_gird_coordinate_on_source_place,
            )
            area_2 = mesh_numba_util.delaunay_triangle_area_from(
                corner_0=vertices_of_the_simplex[0],
                corner_1=vertices_of_the_simplex[1],
                corner_2=sub_gird_coordinate_on_source_place,
            )

            norm = area_0 + area_1 + area_2

            weight_abc = np.array([area_0, area_1, area_2]) / norm

            pixel_weights[sub_slim_index] = weight_abc

        else:
            pixel_weights[sub_slim_index][0] = 1.0

    return pixel_weights


@numba_util.jit()
def remove_bad_entries_voronoi_nn(
    bad_indexes,
    pix_weights_for_sub_slim_index,
    pix_indexes_for_sub_slim_index,
    grid,
    mesh_grid,
):
    """
    The nearest neighbor interpolation can return invalid or bad entries which are removed from the mapping arrays. The
    current circumstances this arises are:

    1) If a point is outside the whole Voronoi region, some weights have negative values. In this case, we reset its
       neighbor to its closest neighbor.

    2) The nearest neighbor interpolation code may not return even a single neighbor. We mark these as a bad grid by
       settings their neighbors to the closest ones.

    Parameters
    ----------
    bad_indexes
    pix_weights_for_sub_slim_index
    pix_indexes_for_sub_slim_index
    grid
    mesh_grid

    Returns
    -------

    """

    for item in bad_indexes:
        ind = item[0]
        pix_indexes_for_sub_slim_index[ind] = -1
        pix_indexes_for_sub_slim_index[ind][0] = np.argmin(
            np.sum((grid[ind] - mesh_grid) ** 2.0, axis=1)
        )
        pix_weights_for_sub_slim_index[ind] = 0.0
        pix_weights_for_sub_slim_index[ind][0] = 1.0

    return pix_weights_for_sub_slim_index, pix_indexes_for_sub_slim_index
