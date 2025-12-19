from dataclasses import dataclass
import logging
import numpy as np
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)


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


def _report_memory(arr):
    """
    Report array memory + process RSS (best-effort).
    Safe to call inside a tqdm loop.
    """
    try:
        import resource

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        arr_mb = arr.nbytes / 1024**2
        from tqdm import tqdm

        tqdm.write(f"    Memory: array={arr_mb:.1f} MB, RSS≈{rss_mb:.1f} MB")
    except Exception:
        pass


def w_tilde_curvature_preload_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d,
    grid_radians_2d: np.ndarray,
    *,
    chunk_k: int = 2048,
    show_progress: bool = False,
    show_memory: bool = False,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [unmasked_image_pixels, unmasked_image_pixels] that encodes the
    NUFFT of every pair of image pixels given the noise map. This can be used to efficiently compute the curvature
    matrix via the mapping matrix, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. This methods creates
    a preload matrix that can compute the matrix w_tilde via an efficient preloading scheme which exploits the
    symmetries in the NUFFT.

    To compute w_tilde, one first defines a real space mask where every False entry is an unmasked pixel which is
    used in the calculation, for example:

        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     This is an imaging.Mask2D, where:
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI     x = `True` (Pixel is masked and excluded from lens)
        IxIxIxIoIoIoIxIxIxIxI     o = `False` (Pixel is not masked and included in lens)
        IxIxIxIoIoIoIxIxIxIxI
        IxIxIxIoIoIoIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI


    Here, there are 9 unmasked pixels. Indexing of each unmasked pixel goes from the top-left corner right and
    downwards, therefore:

        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxI0I1I2IxIxIxIxI
        IxIxIxI3I4I5IxIxIxIxI
        IxIxIxI6I7I8IxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI
        IxIxIxIxIxIxIxIxIxIxI

    In the standard calculation of `w_tilde` it is a matrix of
    dimensions [unmasked_image_pixels, unmasked_pixel_images], therefore for the example mask above it would be
    dimensions [9, 9]. One performs a double for loop over `unmasked_image_pixels`, using the (y,x) spatial offset
    between every possible pair of unmasked image pixels to precompute values that depend on the properties of the NUFFT.

    This calculation has a lot of redundancy, because it uses the (y,x) *spatial offset* between the image pixels. For
    example, if two image pixel are next to one another by the same spacing the same value will be computed via the
    NUFFT. For the example mask above:

    - The value precomputed for pixel pair [0,1] is the same as pixel pairs [1,2], [3,4], [4,5], [6,7] and [7,9].
    - The value precomputed for pixel pair [0,3] is the same as pixel pairs [1,4], [2,5], [3,6], [4,7] and [5,8].
    - The values of pixels paired with themselves are also computed repeatedly for the standard calculation (e.g. 9
    times using the mask above).

    The `curvature_preload` method instead only computes each value once. To do this, it stores the preload values in a
    matrix of dimensions [shape_masked_pixels_y, shape_masked_pixels_x, 2], where `shape_masked_pixels` is the (y,x)
    size of the vertical and horizontal extent of unmasked pixels, e.g. the spatial extent over which the real space
    grid extends.

    Each entry in the matrix `curvature_preload[:,:,0]` provides the the precomputed NUFFT value mapping an image pixel
    to a pixel offset by that much in the y and x directions, for example:

    - curvature_preload[0,0,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
    in the x direction by 0 - the values of pixels paired with themselves.
    - curvature_preload[1,0,0] gives the precomputed values of image pixels that are offset in the y direction by 1 and
    in the x direction by 0 - the values of pixel pairs [0,3], [1,4], [2,5], [3,6], [4,7] and [5,8]
    - curvature_preload[0,1,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
    in the x direction by 1 - the values of pixel pairs [0,1], [1,2], [3,4], [4,5], [6,7] and [7,9].

    Flipped pairs:

    The above preloaded values pair all image pixel NUFFT values when a pixel is to the right and / or down of the
    first image pixel. However, one must also precompute pairs where the paired pixel is to the left of the host
    pixels. These pairings are stored in `curvature_preload[:,:,1]`, and the ordering of these pairings is flipped in the
    x direction to make it straight forward to use this matrix when computing w_tilde.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    shape_masked_pixels_2d
        The (y,x) shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask.
    grid_radians_2d
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray
        A matrix that precomputes the values for fast computation of w_tilde.
    """
    # -----------------------------
    # Enforce float64 everywhere
    # -----------------------------
    noise_map_real = np.asarray(noise_map_real, dtype=np.float64)
    uv_wavelengths = np.asarray(uv_wavelengths, dtype=np.float64)
    grid_radians_2d = np.asarray(grid_radians_2d, dtype=np.float64)

    y_shape, x_shape = shape_masked_pixels_2d
    grid = grid_radians_2d[:y_shape, :x_shape]
    gy = grid[..., 0]
    gx = grid[..., 1]

    K = uv_wavelengths.shape[0]

    w = 1.0 / (noise_map_real**2)
    ku = 2.0 * np.pi * uv_wavelengths[:, 0]
    kv = 2.0 * np.pi * uv_wavelengths[:, 1]

    out = np.zeros((2 * y_shape, 2 * x_shape), dtype=np.float64)

    # Corner coordinates
    y00, x00 = gy[0, 0], gx[0, 0]
    y0m, x0m = gy[0, x_shape - 1], gx[0, x_shape - 1]
    ym0, xm0 = gy[y_shape - 1, 0], gx[y_shape - 1, 0]
    ymm, xmm = gy[y_shape - 1, x_shape - 1], gx[y_shape - 1, x_shape - 1]

    def accum_from_corner(y_ref, x_ref, gy_block, gx_block, label=""):
        dy = y_ref - gy_block
        dx = x_ref - gx_block

        acc = np.zeros(gy_block.shape, dtype=np.float64)

        iterator = range(0, K, chunk_k)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=f"Accumulating visibilities {label}",
                total=(K + chunk_k - 1) // chunk_k,
            )

        for k0 in iterator:
            k1 = min(K, k0 + chunk_k)

            phase = dx[..., None] * ku[k0:k1] + dy[..., None] * kv[k0:k1]
            acc += np.sum(
                np.cos(phase) * w[k0:k1],
                axis=2,
            )

            if show_memory and show_progress:
                _report_memory(acc)

        return acc

    # -----------------------------
    # Main quadrant (+,+)
    # -----------------------------
    out[:y_shape, :x_shape] = accum_from_corner(y00, x00, gy, gx, label="(+,+)")

    # -----------------------------
    # Flip in x (+,-)
    # -----------------------------
    if x_shape > 1:
        block = accum_from_corner(y0m, x0m, gy[:, ::-1], gx[:, ::-1], label="(+,-)")
        out[:y_shape, -1:-(x_shape):-1] = block[:, 1:]

    # -----------------------------
    # Flip in y (-,+)
    # -----------------------------
    if y_shape > 1:
        block = accum_from_corner(ym0, xm0, gy[::-1, :], gx[::-1, :], label="(-,+)")
        out[-1:-(y_shape):-1, :x_shape] = block[1:, :]

    # -----------------------------
    # Flip in x and y (-,-)
    # -----------------------------
    if (y_shape > 1) and (x_shape > 1):
        block = accum_from_corner(
            ymm, xmm, gy[::-1, ::-1], gx[::-1, ::-1], label="(-,-)"
        )
        out[-1:-(y_shape):-1, -1:-(x_shape):-1] = block[1:, 1:]

    return out


def w_tilde_via_preload_from(curvature_preload, native_index_for_slim_index):
    """
    Use the preloaded w_tilde matrix (see `curvature_preload_interferometer_from`) to compute
    w_tilde (see `w_tilde_interferometer_from`) efficiently.

    Parameters
    ----------
    curvature_preload
        The preloaded values of the NUFFT that enable efficient computation of w_tilde.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    slim_size = len(native_index_for_slim_index)

    w_tilde_via_preload = np.zeros((slim_size, slim_size))

    for i in range(slim_size):
        i_y, i_x = native_index_for_slim_index[i]

        for j in range(i, slim_size):
            j_y, j_x = native_index_for_slim_index[j]

            y_diff = j_y - i_y
            x_diff = j_x - i_x

            w_tilde_via_preload[i, j] = curvature_preload[y_diff, x_diff]

    for i in range(slim_size):
        for j in range(i, slim_size):
            w_tilde_via_preload[j, i] = w_tilde_via_preload[i, j]

    return w_tilde_via_preload


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
    import jax
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
