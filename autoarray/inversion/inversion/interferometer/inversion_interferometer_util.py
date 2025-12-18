import logging
import numpy as np
import time
import multiprocessing as mp
import os
from typing import Tuple

from autoarray import numba_util

logger = logging.getLogger(__name__)


def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray[tuple[int], np.float64],
    uv_wavelengths: np.ndarray[tuple[int, int], np.float64],
    grid_radians_slim: np.ndarray[tuple[int, int], np.float64],
) -> np.ndarray[tuple[int, int], np.float64]:
    r"""
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Note that the current implementation does not take advantage of the fact that w_tilde is symmetric,
    due to the use of vectorized operations.

    .. math::
        \tilde{W}_{ij} = \sum_{k=1}^N \frac{1}{n_k^2} \cos(2\pi[(g_{i1} - g_{j1})u_{k0} + (g_{i0} - g_{j0})u_{k1}])

    The function is written in a way that the memory use does not depend on size of data K.

    Parameters
    ----------
    noise_map_real : ndarray, shape (K,), dtype=float64
        The real noise-map values of the interferometer data.
    uv_wavelengths : ndarray, shape (K, 2), dtype=float64
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim : ndarray, shape (M, 2), dtype=float64
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    curvature_matrix : ndarray, shape (M, M), dtype=float64
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    import jax
    import jax.numpy as jnp

    TWO_PI = 2.0 * jnp.pi

    M = grid_radians_slim.shape[0]
    g_2pi = TWO_PI * grid_radians_slim
    δg_2pi = g_2pi.reshape(M, 1, 2) - g_2pi.reshape(1, M, 2)
    δg_2pi_y = δg_2pi[:, :, 0]
    δg_2pi_x = δg_2pi[:, :, 1]

    def f_k(
        noise_map_real: float,
        uv_wavelengths: np.ndarray[tuple[int], np.float64],
    ) -> np.ndarray[tuple[int, int], np.float64]:
        return jnp.cos(δg_2pi_x * uv_wavelengths[0] + δg_2pi_y * uv_wavelengths[1]) * jnp.reciprocal(
            jnp.square(noise_map_real)
        )

    def f_scan(
        sum_: np.ndarray[tuple[int, int], np.float64],
        args: tuple[float, np.ndarray[tuple[int], np.float64]],
    ) -> tuple[np.ndarray[tuple[int, int], np.float64], None]:
        noise_map_real, uv_wavelengths = args
        return sum_ + f_k(noise_map_real, uv_wavelengths), None

    res, _ = jax.lax.scan(
        f_scan,
        jnp.zeros((M, M)),
        (
            noise_map_real,
            uv_wavelengths,
        ),
    )
    return res


def data_vector_via_transformed_mapping_matrix_from(
    transformed_mapping_matrix: np.ndarray,
    visibilities: np.ndarray,
    noise_map: np.ndarray,
) -> np.ndarray:
    """
    Returns the data vector `D` from a transformed mapping matrix `f` and the 1D image `d` and 1D noise-map `sigma`
    (see Warren & Dye 2003).

    Parameters
    ----------
    transformed_mapping_matrix
        The matrix representing the transformed mappings between sub-grid pixels and pixelization pixels.
    image
        Flattened 1D array of the observed image the inversion is fitting.
    noise_map
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """
    # Extract components
    vis_real = visibilities.real
    vis_imag = visibilities.imag
    f_real = transformed_mapping_matrix.real
    f_imag = transformed_mapping_matrix.imag
    noise_real = noise_map.real
    noise_imag = noise_map.imag

    # Square noise components
    inv_var_real = 1.0 / (noise_real**2)
    inv_var_imag = 1.0 / (noise_imag**2)

    # Real and imaginary contributions
    weighted_real = (vis_real * inv_var_real)[:, None] * f_real
    weighted_imag = (vis_imag * inv_var_imag)[:, None] * f_imag

    # Sum over visibilities
    return np.sum(weighted_real + weighted_imag, axis=0)


def mapped_reconstructed_visibilities_from(
    transformed_mapping_matrix: np.ndarray, reconstruction: np.ndarray
) -> np.ndarray:
    """
    Returns the reconstructed data vector from the blurrred mapping matrix `f` and solution vector *S*.

    Parameters
    ----------
    transformed_mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """
    return transformed_mapping_matrix @ reconstruction




import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.ops import segment_sum
except ImportError as e:
    raise ImportError("This function requires JAX. Install jax + jaxlib.") from e


def extract_curvature_for_mask(
    C_rect,
    rect_index_for_mask_index,
):
    """
    Extract curvature matrix for an arbitrary mask from a rectangular curvature matrix.

    Parameters
    ----------
    C_rect : array, shape (S_rect, S_rect)
        Curvature matrix computed on the rectangular grid.
    rect_index_for_mask_index : array, shape (S_mask,)
        For each masked pixel index, gives its index in the rectangular grid.

    Returns
    -------
    C_mask : array, shape (S_mask, S_mask)
        Curvature matrix for the arbitrary mask.
    """
    xp = type(C_rect)  # works for np and jnp via duck typing

    idx = rect_index_for_mask_index
    return C_rect[idx[:, None], idx[None, :]]

# -----------------------------------------------------------------------------
# Public API: replacement for the numba interferometer curvature via W~ preload
# -----------------------------------------------------------------------------
def curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
    curvature_preload: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_sizes_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    pix_pixels: int,
    mask_rectangular: np.ndarray,
    rect_index_for_mask_index: np.ndarray,
    *,
    batch_size: int = 128,
    enable_x64: bool = True,
    return_numpy: bool = True,
):
    """
    Compute the curvature matrix for an interferometer inversion using a preloaded
    W-tilde curvature kernel on a *rectangular* real-space grid, but a mapping matrix
    defined on an *arbitrary* (non-rectangular) mask.

    This is the JAX replacement for:
        inversion_interferometer_numba_util.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(...)

    Key idea
    --------
    The FFT-based W~ convolution assumes a full rectangular grid of shape (y_shape, x_shape),
    where y_shape/x_shape are inferred from curvature_preload.shape == (2*y_shape, 2*x_shape).

    The mapper arrays (pix_indexes/pix_sizes/pix_weights) are defined for the masked image
    (slim indexing). We embed that masked mapping into the rectangular grid using
    rect_index_for_mask_index:
        rows_rect = rect_index_for_mask_index[rows_mask]

    Any rectangular pixels outside the mask implicitly have zero mapping entries.

    Parameters
    ----------
    curvature_preload
        The W-tilde curvature preload kernel, shape (2*y, 2*x), real-valued.
        (This is typically `self.w_tilde.curvature_preload`.)
    pix_indexes_for_sub_slim_index
        Mapper indices, shape (M_masked, Pmax), with -1 padding for unused entries.
    pix_sizes_for_sub_slim_index
        Number of active entries per masked image pixel, shape (M_masked,).
    pix_weights_for_sub_slim_index
        Mapper weights, shape (M_masked, Pmax).
    native_index_for_slim_index
        Native indices for slim pixels. Kept for interface parity / debugging.
        Not required if rect_index_for_mask_index is provided correctly.
    pix_pixels
        Number of source pixels (S).
    mask_rectangular
        Boolean mask array for the rectangular grid (True=masked), shape (y_shape, x_shape).
        Used for sanity-checking only (the W~ kernel already defines the rectangle).
    rect_index_for_mask_index
        Array mapping masked slim index -> rectangular slim index, shape (M_masked,).
        Values must be in [0, y_shape*x_shape).
    batch_size
        Column-block size in source space (static shape inside JIT).
    enable_x64
        Enable float64 in JAX (recommended for numerical parity).
    return_numpy
        If True, returns a NumPy array. Otherwise returns a JAX DeviceArray.

    Returns
    -------
    curvature_matrix : (S, S)
        The curvature matrix.
    """

    # -------------------------
    # JAX precision config
    # -------------------------
    if enable_x64:
        jax.config.update("jax_enable_x64", True)

    # -------------------------
    # Infer rectangle from preload
    # -------------------------
    w = np.asarray(curvature_preload, dtype=np.float64)
    H2, W2 = w.shape
    if (H2 % 2) != 0 or (W2 % 2) != 0:
        raise ValueError(
            f"curvature_preload must have even shape (2y,2x). Got {w.shape}."
        )
    y_shape = H2 // 2
    x_shape = W2 // 2
    M_rect = y_shape * x_shape

    # Optional sanity check against provided rectangular mask
    if mask_rectangular is not None:
        mask_rectangular = np.asarray(mask_rectangular, dtype=bool)
        if mask_rectangular.shape != (y_shape, x_shape):
            raise ValueError(
                f"mask_rectangular has shape {mask_rectangular.shape} but expected {(y_shape, x_shape)} "
                f"from curvature_preload."
            )

    # -------------------------
    # Build COO for masked mapping and embed into rectangular rows
    # -------------------------
    pix_idx = np.asarray(pix_indexes_for_sub_slim_index, dtype=np.int32)
    pix_wts = np.asarray(pix_weights_for_sub_slim_index, dtype=np.float64)
    pix_sizes = np.asarray(pix_sizes_for_sub_slim_index, dtype=np.int32)

    M_masked, Pmax = pix_idx.shape
    S = int(pix_pixels)

    rect_index_for_mask_index = np.asarray(rect_index_for_mask_index, dtype=np.int32)
    if rect_index_for_mask_index.shape != (M_masked,):
        raise AssertionError(
            f"rect_index_for_mask_index must have shape (M_masked,) == ({M_masked},), "
            f"got {rect_index_for_mask_index.shape}."
        )
    if rect_index_for_mask_index.min() < 0 or rect_index_for_mask_index.max() >= M_rect:
        raise AssertionError(
            "rect_index_for_mask_index contains out-of-range rectangular indices."
        )

    # COO over masked rows
    # mask_valid selects only first pix_sizes[m] entries in each row (and valid source cols)
    mask_valid = (np.arange(Pmax)[None, :] < pix_sizes[:, None])
    rows_mask = np.repeat(np.arange(M_masked, dtype=np.int32), Pmax)[mask_valid.ravel()]
    cols = pix_idx[mask_valid].astype(np.int32)
    vals = pix_wts[mask_valid].astype(np.float64)

    # Guard cols (some pipelines keep -1 even inside mask_valid if pix_sizes not perfectly clean)
    keep = (cols >= 0) & (cols < S)
    rows_mask = rows_mask[keep]
    cols = cols[keep]
    vals = vals[keep]

    # Embed masked rows into rectangular rows
    rows_rect = rect_index_for_mask_index[rows_mask].astype(np.int32)

    # -------------------------
    # JAX core: curvature from rectangular W~ preload
    # -------------------------
    def _curvature_from_preload_jax(
        w_preload_jax: jnp.ndarray,   # (2y,2x)
        rows_jax: jnp.ndarray,        # (nnz,)
        cols_jax: jnp.ndarray,        # (nnz,)
        vals_jax: jnp.ndarray,        # (nnz,)
        *,
        y_shape: int,
        x_shape: int,
        S: int,
        batch_size: int,
    ) -> jnp.ndarray:
        """
        Returns curvature matrix C (S,S) using:
            C = F^T W F
        where W is linear convolution by w_preload on the rectangular grid.
        """
        M = y_shape * x_shape

        # Precompute FFT of kernel once
        Khat = jnp.fft.fft2(w_preload_jax)  # (2y,2x)

        def apply_W_fft_batch(Fbatch_flat: jnp.ndarray) -> jnp.ndarray:
            # Fbatch_flat: (M, B)
            B = Fbatch_flat.shape[1]
            F_img = Fbatch_flat.T.reshape((B, y_shape, x_shape))
            F_pad = jnp.pad(F_img, ((0, 0), (0, y_shape), (0, x_shape)))  # -> (B,2y,2x)

            Fhat = jnp.fft.fft2(F_pad)
            Ghat = Fhat * Khat[None, :, :]
            G_pad = jnp.fft.ifft2(Ghat)
            G = jnp.real(G_pad[:, :y_shape, :x_shape])  # back to (B,y,x)
            return G.reshape((B, M)).T  # (M,B)

        @jax.jit
        def compute_block(start_col: jnp.ndarray) -> jnp.ndarray:
            """
            Always returns (S, batch_size). Tail handled outside by slicing.
            """
            in_block = (cols_jax >= start_col) & (cols_jax < start_col + batch_size)

            bc = jnp.where(in_block, cols_jax - start_col, 0).astype(jnp.int32)
            v = jnp.where(in_block, vals_jax, 0.0)

            Fbatch = jnp.zeros((M, batch_size), dtype=vals_jax.dtype)
            Fbatch = Fbatch.at[rows_jax, bc].add(v)

            Gbatch = apply_W_fft_batch(Fbatch)            # (M, B)
            G_at_rows = Gbatch[rows_jax, :]               # (nnz, B)
            contrib = vals_jax[:, None] * G_at_rows       # (nnz, B)

            return segment_sum(contrib, cols_jax, num_segments=S)  # (S, B)

        C = jnp.zeros((S, S), dtype=vals_jax.dtype)
        for start in range(0, S, batch_size):
            Cblock = compute_block(jnp.asarray(start, dtype=jnp.int32))
            width = min(batch_size, S - start)
            C = C.at[:, start : start + width].set(Cblock[:, :width])

        return 0.5 * (C + C.T)

    # JIT the *outer* with static args (shape constants)
    curvature_jit = jax.jit(
        _curvature_from_preload_jax,
        static_argnames=("y_shape", "x_shape", "S", "batch_size"),
    )

    # Move inputs once (static-ish)
    w_jax = jnp.asarray(w)
    rows_jax = jnp.asarray(rows_rect)
    cols_jax = jnp.asarray(cols)
    vals_jax = jnp.asarray(vals)

    C_rect = curvature_jit(
        w_jax,
        rows_jax,
        cols_jax,
        vals_jax,
        y_shape=y_shape,
        x_shape=x_shape,
        S=S,
        batch_size=int(batch_size),
    )

    C_mask = extract_curvature_for_mask(
        C_rect=C_rect,
        rect_index_for_mask_index=rect_index_for_mask_index,
    )

    return np.asarray(C) if return_numpy else C
