import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from autoarray.inversion.pixelization.mesh.delaunay import Delaunay
from autoarray.structures.mesh.knn_delaunay_2d import Mesh2DDelaunayKNN


def wendland_c4(r, h):
    """
    Wendland C4: (1 - r/h)^6 * (35*(r/h)^2 + 18*r/h + 3)
    C4 continuous, smoother, compact support
    """
    s = r / (h + 1e-10)
    w = jnp.where(s < 1.0, (1.0 - s) ** 6 * (35.0 * s**2 + 18.0 * s + 3.0), 0.0)
    return w


def get_interpolation_weights(
    points, query_points, k_neighbors, radius_scale, point_block=128
):
    """
    Compute interpolation weights between source points and query points.

    This is a standalone function to get the weights used in kernel interpolation,
    useful when you want to analyze or reuse weights separately from interpolation.

    Low-VRAM compute_weights:
    - does NOT form (M, N, 2) diff
    - does NOT form full (M, N) distances
    - streams points in blocks, maintaining running per-row top-k
    - sqrt only on the selected k

    Args:
        points:       (N, 2) source point coordinates
        query_points: (M, 2) query point coordinates
        k_neighbors:  number of nearest neighbors (default: 10)
        radius_scale: multiplier for auto-computed radius (default: 1.5)

    Returns:
        weights:   (M, k) normalized weights for each query point
        indices:   (M, k) indices of K nearest neighbors in points array
        distances: (M, k) distances to K nearest neighbors

    Example:
        >>> weights, indices, distances = get_interpolation_weights(src_pts, query_pts)
        >>> # Now you can use weights and indices for custom interpolation
        >>> interpolated = jnp.sum(weights * values[indices], axis=1)
    """
    points = jnp.asarray(points)
    query_points = jnp.asarray(query_points)

    M = query_points.shape[0]
    N = points.shape[0]
    k = int(k_neighbors)
    B = int(point_block)

    # Precompute ||q||^2 once (M, 1)
    q2 = jnp.sum(query_points * query_points, axis=1, keepdims=True)

    # Running best: store NEGATIVE squared distances so we can use lax.top_k (largest)
    best_vals = -jnp.inf * jnp.ones((M, k), dtype=query_points.dtype)
    best_idx = jnp.zeros((M, k), dtype=jnp.int32)

    # How many blocks (pad last block logically)
    n_blocks = (N + B - 1) // B

    def body_fun(bi, carry):
        best_vals, best_idx = carry
        start = bi * B
        block_n = jnp.minimum(B, N - start)

        # Slice points block (block_n, 2); pad to (B, 2) to keep shapes static for JIT
        p_block = jax.lax.dynamic_slice(points, (start, 0), (B, points.shape[1]))
        # Mask out padded rows in last block
        mask = jnp.arange(B) < block_n  # (B,)

        # Compute squared distances for this block WITHOUT (M,B,2):
        # dist_sq = ||q||^2 + ||p||^2 - 2 q·p
        p2 = jnp.sum(p_block * p_block, axis=1, keepdims=True).T  # (1, B)
        qp = query_points @ p_block.T  # (M, B)
        dist_sq = q2 + p2 - 2.0 * qp  # (M, B)
        dist_sq = jnp.maximum(dist_sq, 0.0)

        # Invalidate padded points by setting dist_sq to +inf (so -dist_sq = -inf)
        dist_sq = jnp.where(mask[None, :], dist_sq, jnp.inf)

        vals = -dist_sq  # (M, B)  negative squared distances

        # Indices for this block (M, B)
        idx_block = (start + jnp.arange(B, dtype=jnp.int32))[None, :]  # (1,B)
        idx_block = jnp.broadcast_to(idx_block, (M, B))

        # Merge candidates with current best, then take top-k
        merged_vals = jnp.concatenate([best_vals, vals], axis=1)  # (M, k+B)
        merged_idx = jnp.concatenate([best_idx, idx_block], axis=1)  # (M, k+B)

        new_vals, new_pos = jax.lax.top_k(merged_vals, k)  # (M, k), (M, k)
        new_idx = jnp.take_along_axis(merged_idx, new_pos, axis=1)  # (M, k)

        return (new_vals, new_idx)

    best_vals, best_idx = jax.lax.fori_loop(
        0, n_blocks, body_fun, (best_vals, best_idx)
    )

    # Convert back to positive distances
    knn_dist_sq = -best_vals  # (M, k)
    knn_distances = jnp.sqrt(knn_dist_sq + 1e-20)  # (M, k)

    # Radius per query
    h = jnp.max(knn_distances, axis=1, keepdims=True) * radius_scale  # (M, 1)

    # Kernel weights + partition-of-unity normalisation
    weights = wendland_c4(knn_distances, h)  # (M, k)
    weights_sum = jnp.sum(weights, axis=1, keepdims=True) + 1e-10
    weights_normalized = weights / weights_sum

    return weights_normalized, best_idx, knn_distances


def kernel_interpolate_points(query_chunk, points, values, k, radius_scale):
    """
    Compute kernel interpolation for a chunk of query points using K nearest neighbors.

    Args:
        query_chunk: (M, 2) query points
        points: (N, 2) source points
        values: (N,) values at source points
        k: number of nearest neighbors
        radius_scale: multiplier for radius

    Returns:
        (M,) interpolated values
    """
    # Compute weights using the intermediate function
    weights_normalized, top_k_indices, _ = get_interpolation_weights(
        points,
        query_chunk,
        k,
        radius_scale,
    )

    # Get neighbor values
    neighbor_values = values[top_k_indices]  # (M, k)

    # Interpolate: weighted sum
    interpolated = jnp.sum(weights_normalized * neighbor_values, axis=1)  # (M,)

    return interpolated


class KNNInterpolator(Delaunay):

    def __init__(self, k_neighbors=10, radius_scale=1.5):

        self.k_neighbors = k_neighbors
        self.radius_scale = radius_scale

        super().__init__()

    def mesh_grid_from(
        self,
        source_plane_data_grid=None,
        source_plane_mesh_grid=None,
        preloads=None,
        xp=np,
    ):
        """
        Return the Delaunay ``source_plane_mesh_grid`` as a ``Mesh2DDelaunay`` object, which provides additional
        functionality for performing operations that exploit the geometry of a Delaunay mesh.

        Parameters
        ----------
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            ``source`` reference frame.
        source_plane_mesh_grid
            The centres of every Delaunay pixel in the ``source`` frame, which are initially derived by computing a sparse
            set of (y,x) coordinates computed from the unmasked data in the image-plane and applying a transformation
            to this.
        settings
            Settings controlling the pixelization for example if a border is used to relocate its exterior coordinates.
        """

        return Mesh2DDelaunayKNN(
            values=source_plane_mesh_grid,
            source_plane_data_grid_over_sampled=source_plane_data_grid,
            preloads=preloads,
            _xp=xp,
        )
