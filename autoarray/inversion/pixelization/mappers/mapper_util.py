from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

from autoconf import conf

from autoarray import exc


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


def nearest_pixelization_index_for_slim_index_from_kdtree(grid, mesh_grid):
    from scipy.spatial import cKDTree

    kdtree = cKDTree(mesh_grid)

    sparse_index_for_slim_index = []

    for i in range(grid.shape[0]):
        input_point = [grid[i, [0]], grid[i, 1]]
        index = kdtree.query(input_point)[1]
        sparse_index_for_slim_index.append(index)

    return sparse_index_for_slim_index


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
