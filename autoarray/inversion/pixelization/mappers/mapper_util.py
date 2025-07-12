import jax.numpy as jnp
import numpy as np
from typing import Tuple

from autoconf import conf

from autoarray import numba_util
from autoarray import exc
from autoarray.inversion.pixelization.mesh import mesh_util


@numba_util.jit()
def sub_slim_indexes_for_pix_index(
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    pix_pixels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub_slim_sizes_for_pix_index = np.zeros(pix_pixels)

    for pix_indexes in pix_indexes_for_sub_slim_index:
        for pix_index in pix_indexes:
            sub_slim_sizes_for_pix_index[pix_index] += 1

    max_pix_size = np.max(sub_slim_sizes_for_pix_index)

    sub_slim_indexes_for_pix_index = -1 * np.ones(shape=(pix_pixels, int(max_pix_size)))
    sub_slim_weights_for_pix_index = -1 * np.ones(shape=(pix_pixels, int(max_pix_size)))
    sub_slim_sizes_for_pix_index = np.zeros(pix_pixels)

    for slim_index, pix_indexes in enumerate(pix_indexes_for_sub_slim_index):
        pix_weights = pix_weights_for_sub_slim_index[slim_index]

        for pix_index, pix_weight in zip(pix_indexes, pix_weights):
            sub_slim_indexes_for_pix_index[
                pix_index, int(sub_slim_sizes_for_pix_index[pix_index])
            ] = slim_index

            sub_slim_weights_for_pix_index[
                pix_index, int(sub_slim_sizes_for_pix_index[pix_index])
            ] = pix_weight

            sub_slim_sizes_for_pix_index[pix_index] += 1

    return (
        sub_slim_indexes_for_pix_index,
        sub_slim_sizes_for_pix_index,
        sub_slim_weights_for_pix_index,
    )


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


def rectangular_mappings_weights_via_interpolation_from(
    shape_native: Tuple[int, int],
    source_plane_data_grid: jnp.ndarray,
    source_plane_mesh_grid: jnp.ndarray,
):
    """
    Compute bilinear interpolation weights and corresponding rectangular mesh indices for an irregular grid.

    Given a flattened regular rectangular mesh grid and an irregular grid of data points, this function
    determines for each irregular point:
    - the indices of the 4 nearest rectangular mesh pixels (top-left, top-right, bottom-left, bottom-right), and
    - the bilinear interpolation weights with respect to those pixels.

    The function supports JAX and is compatible with JIT compilation.

    Parameters
    ----------
    shape_native
        The shape (Ny, Nx) of the original rectangular mesh grid before flattening.
    source_plane_data_grid
        The irregular grid of (y, x) points to interpolate.
    source_plane_mesh_grid
        The flattened regular rectangular mesh grid of (y, x) coordinates.

    Returns
    -------
    mappings : jnp.ndarray of shape (N, 4)
        Indices of the four nearest rectangular mesh pixels in the flattened mesh grid.
        Order is: top-left, top-right, bottom-left, bottom-right.
    weights : jnp.ndarray of shape (N, 4)
        Bilinear interpolation weights corresponding to the four nearest mesh pixels.

    Notes
    -----
    - Assumes the mesh grid is uniformly spaced.
    - The weights sum to 1 for each irregular point.
    - Uses bilinear interpolation in the (y, x) coordinate system.
    """
    source_plane_mesh_grid = source_plane_mesh_grid.reshape(*shape_native, 2)

    # Assume mesh is shaped (Ny, Nx, 2)
    Ny, Nx = source_plane_mesh_grid.shape[:2]

    # Get mesh spacings and lower corner
    y_coords = source_plane_mesh_grid[:, 0, 0]  # shape (Ny,)
    x_coords = source_plane_mesh_grid[0, :, 1]  # shape (Nx,)

    dy = y_coords[1] - y_coords[0]
    dx = x_coords[1] - x_coords[0]

    y_min = y_coords[0]
    x_min = x_coords[0]

    # shape (N_irregular, 2)
    irregular = source_plane_data_grid

    # Compute normalized mesh coordinates (floating indices)
    fy = (irregular[:, 0] - y_min) / dy
    fx = (irregular[:, 1] - x_min) / dx

    # Integer indices of top-left corners
    ix = jnp.floor(fx).astype(jnp.int32)
    iy = jnp.floor(fy).astype(jnp.int32)

    # Clip to stay within bounds
    ix = jnp.clip(ix, 0, Nx - 2)
    iy = jnp.clip(iy, 0, Ny - 2)

    # Local coordinates inside the cell (0 <= tx, ty <= 1)
    tx = fx - ix
    ty = fy - iy

    # Bilinear weights
    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty

    weights = jnp.stack([w00, w10, w01, w11], axis=1)  # shape (N_irregular, 4)

    # Compute indices of 4 surrounding pixels in the flattened mesh
    i00 = iy * Nx + ix
    i10 = iy * Nx + (ix + 1)
    i01 = (iy + 1) * Nx + ix
    i11 = (iy + 1) * Nx + (ix + 1)

    mappings = jnp.stack([i00, i10, i01, i11], axis=1)  # shape (N_irregular, 4)

    return mappings, weights


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


def nearest_pixelization_index_for_slim_index_from_kdtree(grid, mesh_grid):
    from scipy.spatial import cKDTree

    kdtree = cKDTree(mesh_grid)

    sparse_index_for_slim_index = []

    for i in range(grid.shape[0]):
        input_point = [grid[i, [0]], grid[i, 1]]
        index = kdtree.query(input_point)[1]
        sparse_index_for_slim_index.append(index)

    return sparse_index_for_slim_index


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

            area_0 = mesh_util.delaunay_triangle_area_from(
                corner_0=vertices_of_the_simplex[1],
                corner_1=vertices_of_the_simplex[2],
                corner_2=sub_gird_coordinate_on_source_place,
            )
            area_1 = mesh_util.delaunay_triangle_area_from(
                corner_0=vertices_of_the_simplex[0],
                corner_1=vertices_of_the_simplex[2],
                corner_2=sub_gird_coordinate_on_source_place,
            )
            area_2 = mesh_util.delaunay_triangle_area_from(
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


def pix_size_weights_voronoi_nn_from(
    grid: np.ndarray, mesh_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    slim_index_for_sub_slim_index
        The mappings between the data slimmed sub-pixels and their regular pixels.
    mesh_grid
        The (y,x) centre of every Voronoi pixel in arc-seconds.
    neighbors
        An array of length (voronoi_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of length (voronoi_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.
    """

    try:
        from autoarray.util.nn import nn_py
    except ImportError as e:
        raise ImportError(
            "In order to use the Voronoi pixelization you must install the "
            "Natural Neighbor Interpolation c package.\n\n"
            ""
            "See: https://github.com/Jammy2211/PyAutoArray/tree/main/autoarray/util/nn"
        ) from e

    max_nneighbours = conf.instance["general"]["pixelization"][
        "voronoi_nn_max_interpolation_neighbors"
    ]

    (
        pix_weights_for_sub_slim_index,
        pix_indexes_for_sub_slim_index,
    ) = nn_py.natural_interpolation_weights(
        x_in=mesh_grid[:, 1],
        y_in=mesh_grid[:, 0],
        x_target=grid[:, 1],
        y_target=grid[:, 0],
        max_nneighbours=max_nneighbours,
    )

    bad_indexes = np.argwhere(np.sum(pix_weights_for_sub_slim_index < 0.0, axis=1) > 0)

    (
        pix_weights_for_sub_slim_index,
        pix_indexes_for_sub_slim_index,
    ) = remove_bad_entries_voronoi_nn(
        bad_indexes=bad_indexes,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        grid=np.array(grid),
        mesh_grid=np.array(mesh_grid),
    )

    bad_indexes = np.argwhere(pix_indexes_for_sub_slim_index[:, 0] == -1)

    (
        pix_weights_for_sub_slim_index,
        pix_indexes_for_sub_slim_index,
    ) = remove_bad_entries_voronoi_nn(
        bad_indexes=bad_indexes,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        grid=np.array(grid),
        mesh_grid=np.array(mesh_grid),
    )

    pix_indexes_for_sub_slim_index_sizes = np.sum(
        pix_indexes_for_sub_slim_index != -1, axis=1
    )

    if np.max(pix_indexes_for_sub_slim_index_sizes) > max_nneighbours:
        raise exc.MeshException(
            f"""
            The number of Voronoi natural neighbours interpolations in one or more pixelization pixel's 
            exceeds the maximum allowed: max_nneighbors = {max_nneighbours}.

            To fix this, increase the value of `voronoi_nn_max_interpolation_neighbors` in the [pixelization]
            section of the `general.ini` config file.
            """
        )

    return (
        pix_indexes_for_sub_slim_index,
        pix_indexes_for_sub_slim_index_sizes,
        pix_weights_for_sub_slim_index,
    )


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


@numba_util.jit()
def adaptive_pixel_signals_from(
    pixels: int,
    pixel_weights: np.ndarray,
    signal_scale: float,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    slim_index_for_sub_slim_index: np.ndarray,
    adapt_data: np.ndarray,
) -> np.ndarray:
    """
    Returns the signal in each pixel, where the signal is the sum of its mapped data values.
    These pixel-signals are used to compute the effective regularization weight of each pixel.

    The pixel signals are computed as follows:

    1) Divide by the number of mappe data points in the pixel, to ensure all pixels have the same
       'relative' signal (i.e. a pixel with 10 pixels doesn't have x2 the signal of one with 5).

    2) Divided by the maximum pixel-signal, so that all signals vary between 0 and 1. This ensures that the
       regularization weight_list are defined identically for any data quantity or signal-to-noise_map ratio.

    3) Raised to the power of the parameter *signal_scale*, so the method can control the relative
       contribution regularization in different regions of pixelization.

    Parameters
    ----------
    pixels
        The total number of pixels in the pixelization the regularization scheme is applied to.
    signal_scale
        A factor which controls how rapidly the smoothness of regularization varies from high signal regions to
        low signal regions.
    regular_to_pix
        A 1D array util every pixel on the grid to a pixel on the pixelization.
    adapt_data
        The image of the galaxy which is used to compute the weigghted pixel signals.
    """

    pixel_signals = np.zeros((pixels,))
    pixel_sizes = np.zeros((pixels,))

    for sub_slim_index in range(len(pix_indexes_for_sub_slim_index)):
        vertices_indexes = pix_indexes_for_sub_slim_index[sub_slim_index]

        mask_1d_index = slim_index_for_sub_slim_index[sub_slim_index]

        pix_size_tem = pix_size_for_sub_slim_index[sub_slim_index]

        if pix_size_tem > 1:
            pixel_signals[vertices_indexes[:pix_size_tem]] += (
                adapt_data[mask_1d_index] * pixel_weights[sub_slim_index]
            )
            pixel_sizes[vertices_indexes] += 1
        else:
            pixel_signals[vertices_indexes[0]] += adapt_data[mask_1d_index]
            pixel_sizes[vertices_indexes[0]] += 1

    pixel_sizes[pixel_sizes == 0] = 1
    pixel_signals /= pixel_sizes
    pixel_signals /= np.max(pixel_signals)

    return pixel_signals**signal_scale


@numba_util.jit()
def mapping_matrix_from(
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    pixels: int,
    total_mask_pixels: int,
    slim_index_for_sub_slim_index: np.ndarray,
    sub_fraction: np.ndarray,
) -> np.ndarray:
    """
    Returns the mapping matrix, which is a matrix representing the mapping between every unmasked sub-pixel of the data
    and the pixels of a pixelization. Non-zero entries signify a mapping, whereas zeros signify no mapping.

    For example, if the data has 5 unmasked pixels (with `sub_size=1` so there are not sub-pixels) and the pixelization
    3 pixels, with the following mappings:

    data pixel 0 -> pixelization pixel 0
    data pixel 1 -> pixelization pixel 0
    data pixel 2 -> pixelization pixel 1
    data pixel 3 -> pixelization pixel 1
    data pixel 4 -> pixelization pixel 2

    The mapping matrix (which is of dimensions [data_pixels, pixelization_pixels]) would appear as follows:

    [1, 0, 0] [0->0]
    [1, 0, 0] [1->0]
    [0, 1, 0] [2->1]
    [0, 1, 0] [3->1]
    [0, 0, 1] [4->2]

    The mapping matrix is actually built using the sub-grid of the grid, whereby each pixel is divided into a grid of
    sub-pixels which are all paired to pixels in the pixelization. The entries in the mapping matrix now become
    fractional values dependent on the sub-pixel sizes.

    For example, for a 2x2 sub-pixels in each pixel means the fractional value is 1.0/(2.0^2) = 0.25, if we have the
    following mappings:

    data pixel 0 -> data sub pixel 0 -> pixelization pixel 0
    data pixel 0 -> data sub pixel 1 -> pixelization pixel 1
    data pixel 0 -> data sub pixel 2 -> pixelization pixel 1
    data pixel 0 -> data sub pixel 3 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 0 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 1 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 2 -> pixelization pixel 1
    data pixel 1 -> data sub pixel 3 -> pixelization pixel 1
    data pixel 2 -> data sub pixel 0 -> pixelization pixel 2
    data pixel 2 -> data sub pixel 1 -> pixelization pixel 2
    data pixel 2 -> data sub pixel 2 -> pixelization pixel 3
    data pixel 2 -> data sub pixel 3 -> pixelization pixel 3

    The mapping matrix (which is still of dimensions [data_pixels, pixelization_pixels]) appears as follows:

    [0.25, 0.75, 0.0, 0.0] [1 sub-pixel maps to pixel 0, 3 map to pixel 1]
    [ 0.0,  1.0, 0.0, 0.0] [All sub-pixels map to pixel 1]
    [ 0.0,  0.0, 0.5, 0.5] [2 sub-pixels map to pixel 2, 2 map to pixel 3]

    For certain pixelizations each data sub-pixel maps to multiple pixelization pixels in a weighted fashion, for
    example a Delaunay pixelization where there are 3 mappings per sub-pixel whose weights are determined via a
    nearest neighbor interpolation scheme.

    In this case, each mapping value is multiplied by this interpolation weight (which are in the array
    `pix_weights_for_sub_slim_index`) when the mapping matrix is constructed.

    Parameters
    ----------
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelization pixel.
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
        slim_index = slim_index_for_sub_slim_index[sub_slim_index]

        for pix_count in range(pix_size_for_sub_slim_index[sub_slim_index]):
            pix_index = pix_indexes_for_sub_slim_index[sub_slim_index, pix_count]
            pix_weight = pix_weights_for_sub_slim_index[sub_slim_index, pix_count]

            mapping_matrix[slim_index][pix_index] += (
                sub_fraction[slim_index] * pix_weight
            )

    return mapping_matrix


@numba_util.jit()
def mapped_to_source_via_mapping_matrix_from(
    mapping_matrix: np.ndarray, array_slim: np.ndarray
) -> np.ndarray:
    """
    Map a masked 2d image in the image domain to the source domain and sum up all mappings on the source-pixels.

    For example, suppose we have an image and a mapper. We can map every image-pixel to its corresponding mapper's
    source pixel and sum the values based on these mappings.

    This will produce something similar to a `reconstruction`, albeit it bypasses the linear algebra / inversion.

    Parameters
    ----------
    mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    array_slim
        The masked 2D array of values in its slim representation (e.g. the image data) which are mapped to the
        source domain in order to compute their average values.
    """

    mapped_to_source = np.zeros(mapping_matrix.shape[1])

    source_pixel_count = np.zeros(mapping_matrix.shape[1])

    for i in range(mapping_matrix.shape[0]):
        for j in range(mapping_matrix.shape[1]):
            if mapping_matrix[i, j] > 0:
                mapped_to_source[j] += array_slim[i] * mapping_matrix[i, j]
                source_pixel_count[j] += 1

    for j in range(mapping_matrix.shape[1]):
        if source_pixel_count[j] > 0:
            mapped_to_source[j] /= source_pixel_count[j]

    return mapped_to_source


@numba_util.jit()
def data_weight_total_for_pix_from(
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    pixels: int,
) -> np.ndarray:
    """
    Returns the total weight of every pixelization pixel, which is the sum of the weights of all data-points that
    map to that pixel.

    Parameters
    ----------
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub-pixel and pixelization pixel.
    pixels
        The number of pixels in the pixelization.
    """

    pix_weight_total = np.zeros(pixels)

    for slim_index, pix_indexes in enumerate(pix_indexes_for_sub_slim_index):
        for pix_index, weight in zip(
            pix_indexes, pix_weights_for_sub_slim_index[slim_index]
        ):
            pix_weight_total[int(pix_index)] += weight

    return pix_weight_total
