import logging
import numpy as np

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
        return jnp.cos(
            δg_2pi_x * uv_wavelengths[0] + δg_2pi_y * uv_wavelengths[1]
        ) * jnp.reciprocal(jnp.square(noise_map_real))

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


from dataclasses import dataclass


@dataclass(frozen=True)
class WTildeFFTState:
    """
    Fully static FFT / geometry state for W~ curvature.

    Safe to cache as long as:
      - curvature_preload is fixed
      - mask / rectangle definition is fixed
      - dtype is fixed
      - batch_size is fixed
    """

    y_shape: int
    x_shape: int
    M: int
    batch_size: int
    w_dtype: "jax.numpy.dtype"
    Khat: "jax.Array"  # (2y, 2x), complex


def w_tilde_fft_state_from(
    curvature_preload: np.ndarray,
    *,
    batch_size: int = 128,
) -> WTildeFFTState:
    import jax.numpy as jnp

    H2, W2 = curvature_preload.shape
    if (H2 % 2) != 0 or (W2 % 2) != 0:
        raise ValueError(
            f"curvature_preload must have even shape (2y,2x). Got {curvature_preload.shape}."
        )

    y_shape = H2 // 2
    x_shape = W2 // 2
    M = y_shape * x_shape

    Khat = jnp.fft.fft2(curvature_preload)

    return WTildeFFTState(
        y_shape=y_shape,
        x_shape=x_shape,
        M=M,
        batch_size=int(batch_size),
        w_dtype=curvature_preload.dtype,
        Khat=Khat,
    )


def curvature_matrix_via_w_tilde_interferometer_from(
    *,
    fft_state: WTildeFFTState,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    pix_pixels: int,
    rect_index_for_mask_index: np.ndarray,
):
    """
    Compute curvature matrix for an interferometer inversion using a precomputed FFT state.

    IMPORTANT
    ---------
    - COO construction is unchanged from the known-working implementation
    - Only FFT- and geometry-related quantities are taken from `fft_state`
    """
    import jax.numpy as jnp
    from jax.ops import segment_sum

    # -------------------------
    # Pull static quantities from state
    # -------------------------
    y_shape = fft_state.y_shape
    x_shape = fft_state.x_shape
    M = fft_state.M
    batch_size = fft_state.batch_size
    Khat = fft_state.Khat
    w_dtype = fft_state.w_dtype

    # -------------------------
    # Basic shape checks (NumPy side, safe)
    # -------------------------
    M_masked, Pmax = pix_indexes_for_sub_slim_index.shape
    S = int(pix_pixels)

    # -------------------------
    # JAX core (unchanged COO logic)
    # -------------------------
    def _curvature_rect_jax(
        pix_idx: jnp.ndarray,  # (M_masked, Pmax)
        pix_wts: jnp.ndarray,  # (M_masked, Pmax)
        rect_map: jnp.ndarray,  # (M_masked,)
    ) -> jnp.ndarray:

        rect_map = jnp.asarray(rect_map)

        nnz_full = M_masked * Pmax

        # Flatten mapping arrays into a fixed-length COO stream
        rows_mask = jnp.repeat(
            jnp.arange(M_masked, dtype=jnp.int32), Pmax
        )  # (nnz_full,)
        cols = pix_idx.reshape((nnz_full,)).astype(jnp.int32)
        vals = pix_wts.reshape((nnz_full,)).astype(w_dtype)

        # Validity mask
        valid = (cols >= 0) & (cols < S)

        # Embed masked rows into rectangular rows
        rows_rect = rect_map[rows_mask].astype(jnp.int32)

        # Make cols / vals safe
        cols_safe = jnp.where(valid, cols, 0)
        vals_safe = jnp.where(valid, vals, 0.0)

        def apply_W_fft_batch(Fbatch_flat: jnp.ndarray) -> jnp.ndarray:
            B = Fbatch_flat.shape[1]
            F_img = Fbatch_flat.T.reshape((B, y_shape, x_shape))
            F_pad = jnp.pad(F_img, ((0, 0), (0, y_shape), (0, x_shape)))  # (B,2y,2x)
            Fhat = jnp.fft.fft2(F_pad)
            Ghat = Fhat * Khat[None, :, :]
            G_pad = jnp.fft.ifft2(Ghat)
            G = jnp.real(G_pad[:, :y_shape, :x_shape])
            return G.reshape((B, M)).T  # (M,B)

        def compute_block(start_col: int) -> jnp.ndarray:
            in_block = (cols_safe >= start_col) & (cols_safe < start_col + batch_size)
            in_use = valid & in_block

            bc = jnp.where(in_use, cols_safe - start_col, 0).astype(jnp.int32)
            v = jnp.where(in_use, vals_safe, 0.0)

            Fbatch = jnp.zeros((M, batch_size), dtype=w_dtype)
            Fbatch = Fbatch.at[rows_rect, bc].add(v)

            Gbatch = apply_W_fft_batch(Fbatch)
            G_at_rows = Gbatch[rows_rect, :]

            contrib = vals_safe[:, None] * G_at_rows
            return segment_sum(contrib, cols_safe, num_segments=S)

        # Assemble curvature
        C = jnp.zeros((S, S), dtype=w_dtype)
        for start in range(0, S, batch_size):
            Cblock = compute_block(start)
            width = min(batch_size, S - start)
            C = C.at[:, start : start + width].set(Cblock[:, :width])

        return 0.5 * (C + C.T)

    return _curvature_rect_jax(
        pix_indexes_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        rect_index_for_mask_index,
    )
