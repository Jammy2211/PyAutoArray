import jax.numpy as jnp
import numpy as np
from typing import Tuple

from autoconf import conf

from autoarray import numba_util
from autoarray import exc
from autoarray.inversion.pixelization.mesh import mesh_util


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


import jax
import jax.numpy as jnp

from functools import partial


def forward_interp(xp, yp, x):
    return jax.vmap(jnp.interp, in_axes=(1, 1, None, None, None))(x, xp, yp, 0, 1).T


def reverse_interp(xp, yp, x):
    return jax.vmap(jnp.interp, in_axes=(1, None, 1))(x, xp, yp).T


def create_transforms(traced_points):
    # make functions that takes a set of traced points
    # stored in a (N, 2) array and return functions that
    # take in (N, 2) arrays and transform the values into
    # the range (0, 1) and the inverse transform
    N = traced_points.shape[0]  # // 2
    t = jnp.arange(1, N + 1) / (N + 1)

    sort_points = jnp.sort(traced_points, axis=0)  # [::2]

    transform = partial(forward_interp, sort_points, t)
    inv_transform = partial(reverse_interp, t, sort_points)
    return transform, inv_transform


def adaptive_rectangular_transformed_grid_from(source_plane_data_grid, grid):
    mu = source_plane_data_grid.mean(axis=0)
    scale = source_plane_data_grid.std(axis=0).min()
    source_grid_scaled = (source_plane_data_grid - mu) / scale

    transform, inv_transform = create_transforms(source_grid_scaled)

    def inv_full(U):
        return inv_transform(U) * scale + mu

    return inv_full(grid)


def adaptive_rectangular_areas_from(source_grid_size, source_plane_data_grid):

    pixel_edges_1d = jnp.linspace(0, 1, source_grid_size + 1)

    mu = source_plane_data_grid.mean(axis=0)
    scale = source_plane_data_grid.std(axis=0).min()
    source_grid_scaled = (source_plane_data_grid - mu) / scale

    transform, inv_transform = create_transforms(source_grid_scaled)

    def inv_full(U):
        return inv_transform(U) * scale + mu

    pixel_edges = inv_full(jnp.stack([pixel_edges_1d, pixel_edges_1d]).T)
    pixel_lengths = jnp.diff(pixel_edges, axis=0).squeeze()  # shape (N_source, 2)

    dy = pixel_lengths[:, 0]
    dx = pixel_lengths[:, 1]

    return jnp.outer(dy, dx).flatten()


def adaptive_rectangular_mappings_weights_via_interpolation_from(
    source_grid_size: int,
    source_plane_data_grid,
    source_plane_data_grid_over_sampled,
):
    """
    Compute bilinear interpolation indices and weights for mapping an oversampled
    source-plane grid onto a regular rectangular pixelization.

    This function takes a set of irregularly-sampled source-plane coordinates and
    builds an adaptive mapping onto a `source_grid_size x source_grid_size` rectangular
    pixelization using bilinear interpolation. The interpolation is expressed as:

        f(x, y) ≈ w_bl * f(ix_down, iy_down) +
                  w_br * f(ix_up,   iy_down) +
                  w_tl * f(ix_down, iy_up) +
                  w_tr * f(ix_up,   iy_up)

    where `(ix_down, ix_up, iy_down, iy_up)` are the integer grid coordinates
    surrounding the continuous position `(x, y)`.

    Steps performed:
      1. Normalize the source-plane grid by subtracting its mean and dividing by
         the minimum axis standard deviation (to balance scaling).
      2. Construct forward/inverse transforms which map the grid into the unit square [0,1]^2.
      3. Transform the oversampled source-plane grid into [0,1]^2, then scale it
         to index space `[0, source_grid_size)`.
      4. Compute floor/ceil along x and y axes to find the enclosing rectangular cell.
      5. Build the four corner indices: bottom-left (bl), bottom-right (br),
         top-left (tl), and top-right (tr).
      6. Flatten the 2D indices into 1D indices suitable for scatter operations,
         with a flipped row-major convention: row = source_grid_size - i, col = j.
      7. Compute bilinear interpolation weights (`w_bl, w_br, w_tl, w_tr`).
      8. Return arrays of flattened indices and weights of shape `(N, 4)`, where
         `N` is the number of oversampled coordinates.

    Parameters
    ----------
    source_grid_size : int
        The number of pixels along one dimension of the rectangular pixelization.
        The grid is square: (source_grid_size x source_grid_size).
    source_plane_data_grid : (M, 2) ndarray
        The base source-plane coordinates, used to define normalization and transforms.
    source_plane_data_grid_over_sampled : (N, 2) ndarray
        Oversampled source-plane coordinates to be interpolated onto the rectangular grid.

    Returns
    -------
    flat_indices : (N, 4) int ndarray
        The flattened indices of the four neighboring pixel corners for each oversampled point.
        Order: [bl, br, tl, tr].
    weights : (N, 4) float ndarray
        The bilinear interpolation weights for each of the four neighboring pixels.
        Order: [w_bl, w_br, w_tl, w_tr].
    """

    # --- Step 1. Normalize grid ---
    mu = source_plane_data_grid.mean(axis=0)
    scale = source_plane_data_grid.std(axis=0).min()
    source_grid_scaled = (source_plane_data_grid - mu) / scale

    # --- Step 2. Build transforms ---
    transform, inv_transform = create_transforms(source_grid_scaled)

    # --- Step 3. Transform oversampled grid into index space ---
    grid_over_sampled_scaled = (source_plane_data_grid_over_sampled - mu) / scale
    grid_over_sampled_transformed = transform(grid_over_sampled_scaled)
    grid_over_index = (source_grid_size - 3) * grid_over_sampled_transformed + 1

    # --- Step 4. Floor/ceil indices ---
    ix_down = jnp.floor(grid_over_index[:, 0])
    ix_up = jnp.ceil(grid_over_index[:, 0])
    iy_down = jnp.floor(grid_over_index[:, 1])
    iy_up = jnp.ceil(grid_over_index[:, 1])

    # --- Step 5. Four corners ---
    idx_tl = jnp.stack([ix_up, iy_down], axis=1)
    idx_tr = jnp.stack([ix_up, iy_up], axis=1)
    idx_br = jnp.stack([ix_down, iy_up], axis=1)
    idx_bl = jnp.stack([ix_down, iy_down], axis=1)

    # --- Step 6. Flatten indices ---
    def flatten(idx, n):
        row = n - idx[:, 0]
        col = idx[:, 1]
        return row * n + col

    flat_tl = flatten(idx_tl, source_grid_size)
    flat_tr = flatten(idx_tr, source_grid_size)
    flat_bl = flatten(idx_bl, source_grid_size)
    flat_br = flatten(idx_br, source_grid_size)

    flat_indices = jnp.stack([flat_tl, flat_tr, flat_bl, flat_br], axis=1).astype(
        "int64"
    )

    # --- Step 7. Bilinear interpolation weights ---
    t_row = (grid_over_index[:, 0] - ix_down) / (ix_up - ix_down + 1e-12)
    t_col = (grid_over_index[:, 1] - iy_down) / (iy_up - iy_down + 1e-12)

    # Weights
    w_tl = (1 - t_row) * (1 - t_col)
    w_tr = (1 - t_row) * t_col
    w_bl = t_row * (1 - t_col)
    w_br = t_row * t_col
    weights = jnp.stack([w_tl, w_tr, w_bl, w_br], axis=1)

    return flat_indices, weights


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

    M_sub, B = pix_indexes_for_sub_slim_index.shape

    # 1) Flatten the per‐mapping tables:
    flat_pixidx = pix_indexes_for_sub_slim_index.reshape(-1)  # (M_sub*B,)
    flat_weights = pixel_weights.reshape(-1)  # (M_sub*B,)

    # 2) Build a matching “parent‐slim” index for each flattened entry:
    I_sub = jnp.repeat(jnp.arange(M_sub), B)  # (M_sub*B,)

    # 3) Mask out any k >= pix_size_for_sub_slim_index[i]
    valid = I_sub < 0  # dummy to get shape
    # better:
    valid = (jnp.arange(B)[None, :] < pix_size_for_sub_slim_index[:, None]).reshape(-1)

    flat_weights = jnp.where(valid, flat_weights, 0.0)
    flat_pixidx = jnp.where(
        valid, flat_pixidx, pixels
    )  # send invalid indices to an out-of-bounds slot

    # 4) Look up data & multiply by mapping weights:
    flat_data_vals = adapt_data[slim_index_for_sub_slim_index][I_sub]  # (M_sub*B,)
    flat_contrib = flat_data_vals * flat_weights  # (M_sub*B,)

    # 5) Scatter‐add into signal sums and counts:
    pixel_signals = jnp.zeros((pixels + 1,)).at[flat_pixidx].add(flat_contrib)
    pixel_counts = jnp.zeros((pixels + 1,)).at[flat_pixidx].add(valid.astype(float))

    # 6) Drop the extra “out-of-bounds” slot:
    pixel_signals = pixel_signals[:pixels]
    pixel_counts = pixel_counts[:pixels]

    # 7) Normalize
    pixel_counts = jnp.where(pixel_counts > 0, pixel_counts, 1.0)
    pixel_signals = pixel_signals / pixel_counts
    max_sig = jnp.max(pixel_signals)
    pixel_signals = jnp.where(max_sig > 0, pixel_signals / max_sig, pixel_signals)

    # 8) Exponentiate
    return pixel_signals**signal_scale


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
    M_sub, B = pix_indexes_for_sub_slim_index.shape
    M = total_mask_pixels
    S = pixels

    # 1) Flatten
    flat_pixidx = pix_indexes_for_sub_slim_index.reshape(-1)  # (M_sub*B,)
    flat_w = pix_weights_for_sub_slim_index.reshape(-1)  # (M_sub*B,)
    flat_parent = jnp.repeat(slim_index_for_sub_slim_index, B)  # (M_sub*B,)
    flat_count = jnp.repeat(pix_size_for_sub_slim_index, B)  # (M_sub*B,)

    # 2) Build valid mask: k < pix_size[i]
    k = jnp.tile(jnp.arange(B), M_sub)  # (M_sub*B,)
    valid = k < flat_count  # (M_sub*B,)

    # 3) Zero out invalid weights
    flat_w = flat_w * valid.astype(flat_w.dtype)

    # 4) Redirect -1 indices to extra bin S
    OUT = S
    flat_pixidx = jnp.where(flat_pixidx < 0, OUT, flat_pixidx)

    # 5) Multiply by sub_fraction of the slim row
    flat_frac = sub_fraction[flat_parent]  # (M_sub*B,)
    flat_contrib = flat_w * flat_frac  # (M_sub*B,)

    # 6) Scatter into (M × (S+1)), summing duplicates
    mat = jnp.zeros((M, S + 1), dtype=flat_contrib.dtype)
    mat = mat.at[flat_parent, flat_pixidx].add(flat_contrib)

    # 7) Drop the extra column and return
    return mat[:, :S]


def mapped_to_source_via_mapping_matrix_from(
    mapping_matrix: np.ndarray, array_slim: np.ndarray
) -> np.ndarray:
    """
    Map a masked 2D image (in slim form) into the source plane by summing and averaging
    each image-pixel's contribution to its mapped source-pixels.

    Each row i of `mapping_matrix` describes how image-pixel i is distributed (with
    weights) across the source-pixels j.  `array_slim[i]` is then multiplied by those
    weights and summed over i to give each source-pixel’s total mapped value; finally,
    we divide by the number of nonzero contributions to form an average.

    Parameters
    ----------
    mapping_matrix : ndarray of shape (M, N)
        mapping_matrix[i, j] ≥ 0 is the weight by which image-pixel i contributes to
        source-pixel j.  Zero means “no contribution.”
    array_slim : ndarray of shape (M,)
        The slimmed image values for each image-pixel i.

    Returns
    -------
    mapped_to_source : ndarray of shape (N,)
        The averaged, mapped values on each of the N source-pixels.
    """
    # weighted sums: sum over i of array_slim[i] * mapping_matrix[i, j]
    # ==> vector‐matrix multiply: (1×M) dot (M×N) → (N,)
    mapped_to_source = array_slim @ mapping_matrix

    # count how many nonzero contributions each source-pixel j received
    counts = np.count_nonzero(mapping_matrix > 0.0, axis=0)

    # avoid division by zero: only divide where counts > 0
    nonzero = counts > 0
    mapped_to_source[nonzero] /= counts[nonzero]

    return mapped_to_source


def data_weight_total_for_pix_from(
    pix_indexes_for_sub_slim_index: np.ndarray,  # shape (M, B)
    pix_weights_for_sub_slim_index: np.ndarray,  # shape (M, B)
    pixels: int,
) -> np.ndarray:
    """
    Returns the total weight of every pixelization pixel, which is the sum of
    the weights of all data‐points (sub‐pixels) that map to that pixel.

    Parameters
    ----------
    pix_indexes_for_sub_slim_index : np.ndarray, shape (M, B), int
        For each of M sub‐slim indexes, the B pixelization‐pixel indices it maps to.
    pix_weights_for_sub_slim_index : np.ndarray, shape (M, B), float
        For each of those mappings, the corresponding interpolation weight.
    pixels : int
        The total number of pixelization pixels N.

    Returns
    -------
    np.ndarray, shape (N,)
        The per‐pixel total weight: for each j in [0..N-1], the sum of all
        pix_weights_for_sub_slim_index[i,k] such that pix_indexes_for_sub_slim_index[i,k] == j.
    """
    # Flatten arrays
    flat_idxs = pix_indexes_for_sub_slim_index.ravel()
    flat_weights = pix_weights_for_sub_slim_index.ravel()

    # Filter out -1 (invalid mappings)
    valid_mask = flat_idxs >= 0
    flat_idxs = flat_idxs[valid_mask]
    flat_weights = flat_weights[valid_mask]

    # Sum weights by pixel index
    return np.bincount(flat_idxs, weights=flat_weights, minlength=pixels)
