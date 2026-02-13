"""
Optimized Kernel-Based Interpolation in JAX
Uses Wendland compactly supported kernels with normalized weights (partition of unity).
More robust and faster than MLS, better accuracy than simple IDW.
"""
import jax
import jax.numpy as jnp
from functools import partial


def get_interpolation_weights(points, query_points, k_neighbors=10, kernel='wendland_c4',
                              radius_scale=1.5):
    """
    Compute interpolation weights between source points and query points.

    This is a standalone function to get the weights used in kernel interpolation,
    useful when you want to analyze or reuse weights separately from interpolation.

    Args:
        points:       (N, 2) source point coordinates
        query_points: (M, 2) query point coordinates
        k_neighbors:  number of nearest neighbors (default: 10)
        kernel:       'wendland_c2', 'wendland_c4', or 'wendland_c6' (default: 'wendland_c4')
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

    # Select kernel function
    if kernel == 'wendland_c2':
        kernel_fn = wendland_c2
    elif kernel == 'wendland_c4':
        kernel_fn = wendland_c4
    elif kernel == 'wendland_c6':
        kernel_fn = wendland_c6
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return compute_weights(points, query_points, k_neighbors, radius_scale, kernel_fn)


def kernel_interpolate(points, values, query_points, k_neighbors=10, kernel='wendland_c4',
                       radius_scale=1.5, chunk_size=None):
    """
    Kernel-based interpolation using K-nearest neighbors with Wendland kernels.

    Uses normalized kernel weights ensuring partition of unity for better accuracy.
    More robust than MLS (no linear solve) and more accurate than simple 1/d^p.

    Args:
        points:       (N, 2) source point coordinates
        values:       (N,) values at source points
        query_points: (M, 2) query point coordinates
        k_neighbors:  number of nearest neighbors (default: 10)
        kernel:       'wendland_c2', 'wendland_c4', or 'wendland_c6' (default: 'wendland_c4')
        radius_scale: multiplier for auto-computed radius (default: 1.5)
        chunk_size:   if provided, process queries in chunks

    Returns:
        (M,) interpolated values
    """
    points = jnp.asarray(points)
    values = jnp.asarray(values)
    query_points = jnp.asarray(query_points)

    # Select kernel function
    if kernel == 'wendland_c2':
        kernel_fn = wendland_c2
    elif kernel == 'wendland_c4':
        kernel_fn = wendland_c4
    elif kernel == 'wendland_c6':
        kernel_fn = wendland_c6
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    if chunk_size is None:
        return _kernel_knn_jit(points, values, query_points, k_neighbors,
                               radius_scale, kernel_fn)
    else:
        return _kernel_chunked(points, values, query_points, k_neighbors,
                               radius_scale, kernel_fn, int(chunk_size))


def wendland_c2(r, h):
    """
    Wendland C2: (1 - r/h)^4 * (4*r/h + 1)
    C2 continuous, compact support
    """
    s = r / (h + 1e-10)
    w = jnp.where(s < 1.0, (1.0 - s) ** 4 * (4.0 * s + 1.0), 0.0)
    return w


def wendland_c4(r, h):
    """
    Wendland C4: (1 - r/h)^6 * (35*(r/h)^2 + 18*r/h + 3)
    C4 continuous, smoother, compact support
    """
    s = r / (h + 1e-10)
    w = jnp.where(s < 1.0, (1.0 - s) ** 6 * (35.0 * s ** 2 + 18.0 * s + 3.0), 0.0)
    return w


def wendland_c6(r, h):
    """
    Wendland C6: (1 - r/h)^8 * (32*(r/h)^3 + 25*(r/h)^2 + 8*r/h + 1)
    C6 continuous, very smooth, compact support
    """
    s = r / (h + 1e-10)
    w = jnp.where(s < 1.0, (1.0 - s) ** 8 * (32.0 * s ** 3 + 25.0 * s ** 2 + 8.0 * s + 1.0), 0.0)
    return w


def compute_weights(points, query_points, k_neighbors, radius_scale, kernel_fn):
    """
    Compute normalized kernel weights for interpolation.

    This function computes the weights between source points and
    query points using K-nearest neighbors and Wendland kernels.

    Args:
        points:       (N, 2) source point coordinates
        query_points: (M, 2) query point coordinates
        k_neighbors:  number of nearest neighbors
        radius_scale: multiplier for auto-computed radius
        kernel_fn:    kernel function (wendland_c2/c4/c6)

    Returns:
        weights:      (M, k) normalized weights for each query point
        indices:      (M, k) indices of K nearest neighbors for each query point
        distances:    (M, k) distances to K nearest neighbors
    """
    # Compute pairwise distances
    diff = query_points[:, None, :] - points[None, :, :]  # (M, N, 2)
    dist_sq = jnp.sum(diff * diff, axis=-1)  # (M, N)
    dist = jnp.sqrt(dist_sq)  # (M, N)

    # Find K nearest neighbors
    top_k_vals, top_k_indices = jax.lax.top_k(-dist, k_neighbors)  # negative for smallest
    knn_distances = -top_k_vals  # (M, k)

    # Auto-compute radius: use max KNN distance + margin
    h = jnp.max(knn_distances, axis=1, keepdims=True) * radius_scale  # (M, 1)

    # Compute kernel weights
    weights = kernel_fn(knn_distances, h)  # (M, k)

    # Normalize weights (partition of unity)
    # Add small epsilon to avoid division by zero
    weight_sum = jnp.sum(weights, axis=1, keepdims=True) + 1e-10  # (M, 1)
    weights_normalized = weights / weight_sum  # (M, k)

    return weights_normalized, top_k_indices, knn_distances


def _compute_kernel_knn(query_chunk, points, values, k, radius_scale, kernel_fn):
    """
    Compute kernel interpolation for a chunk of query points using K nearest neighbors.

    Args:
        query_chunk: (M, 2) query points
        points: (N, 2) source points
        values: (N,) values at source points
        k: number of nearest neighbors
        radius_scale: multiplier for radius
        kernel_fn: kernel function

    Returns:
        (M,) interpolated values
    """
    # Compute weights using the intermediate function
    weights_normalized, top_k_indices, _ = compute_weights(
        points, query_chunk, k, radius_scale, kernel_fn
    )

    # Get neighbor values
    neighbor_values = values[top_k_indices]  # (M, k)

    # Interpolate: weighted sum
    interpolated = jnp.sum(weights_normalized * neighbor_values, axis=1)  # (M,)

    return interpolated


@partial(jax.jit, static_argnames=("k_neighbors", "kernel_fn"))
def _kernel_knn_jit(points, values, query_points, k_neighbors, radius_scale, kernel_fn):
    """
    JIT-compiled kernel interpolation.
    """
    return _compute_kernel_knn(query_points, points, values, k_neighbors,
                               radius_scale, kernel_fn)


def _kernel_chunked(points, values, query_points, k_neighbors, radius_scale,
                    kernel_fn, chunk_size):
    """
    Chunked kernel interpolation for memory efficiency.
    """
    M = query_points.shape[0]
    D = query_points.shape[1]

    # Pad queries
    remainder = M % chunk_size
    pad = 0 if remainder == 0 else (chunk_size - remainder)
    if pad:
        qp_pad = jnp.pad(query_points, ((0, pad), (0, 0)))
    else:
        qp_pad = query_points

    out_pad = _kernel_chunked_jit(points, values, qp_pad, k_neighbors,
                                  radius_scale, kernel_fn, chunk_size)
    return out_pad[:M]


@partial(jax.jit, static_argnames=("k_neighbors", "kernel_fn", "chunk_size"))
def _kernel_chunked_jit(points, values, query_points_padded, k_neighbors,
                        radius_scale, kernel_fn, chunk_size):
    """
    JIT-compiled chunked kernel interpolation.
    """
    M_pad = query_points_padded.shape[0]
    D = points.shape[1]
    n_chunks = M_pad // chunk_size

    out = jnp.zeros((M_pad,), dtype=values.dtype)

    def body_fun(i, out_acc):
        start = i * chunk_size

        # Extract chunk
        q_chunk = jax.lax.dynamic_slice(
            query_points_padded, (start, 0), (chunk_size, D)
        )

        # Compute kernel interpolation for this chunk
        result_chunk = _compute_kernel_knn(q_chunk, points, values, k_neighbors,
                                           radius_scale, kernel_fn)

        # Update output
        out_acc = jax.lax.dynamic_update_slice(out_acc, result_chunk, (start,))

        return out_acc

    out = jax.lax.fori_loop(0, n_chunks, body_fun, out)
    return out


from autoarray.inversion.pixelization.mesh.delaunay import Delaunay

class KNNInterpolator(Delaunay):

    def __init__(self):

        super().__init__()

