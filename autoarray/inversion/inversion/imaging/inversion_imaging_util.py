import numpy as np
from functools import partial

def psf_operator_matrix_dense_from(
    kernel_native: np.ndarray,
    native_index_for_slim_index: np.ndarray,  # shape (N_pix, 2), native (y,x) coords of masked pixels
    native_shape: tuple[int, int],
    correlate: bool = True,
) -> np.ndarray:
    """
    Construct a dense PSF operator W (N_pix x N_pix) that maps masked image pixels to masked image pixels.

    Parameters
    ----------
    kernel_native : (Ky, Kx) PSF kernel.
    native_index_for_slim_index : (N_pix, 2) array of int
        Native (y, x) coords for each masked pixel.
    native_shape : (Ny, Nx)
        Native 2D image shape.
    correlate : bool, default True
        If True, use correlation convention (no kernel flip).
        If False, use convolution convention (flip kernel).

    Returns
    -------
    W : ndarray, shape (N_pix, N_pix)
        Dense PSF operator.
    """
    Ky, Kx = kernel_native.shape
    ph, pw = Ky // 2, Kx // 2
    Ny, Nx = native_shape
    N_pix = native_index_for_slim_index.shape[0]

    ker = kernel_native if correlate else kernel_native[::-1, ::-1]

    # Padded index grid: -1 everywhere, slim index where masked
    index_padded = -np.ones((Ny + 2 * ph, Nx + 2 * pw), dtype=np.int64)
    for p, (y, x) in enumerate(native_index_for_slim_index):
        index_padded[y + ph, x + pw] = p

    # Neighborhood offsets
    dy = np.arange(Ky) - ph
    dx = np.arange(Kx) - pw

    W = np.zeros((N_pix, N_pix), dtype=float)

    for i, (y, x) in enumerate(native_index_for_slim_index):
        yp = y + ph
        xp = x + pw
        for j, dy_ in enumerate(dy):
            for k, dx_ in enumerate(dx):
                neigh = index_padded[yp + dy_, xp + dx_]
                if neigh >= 0:
                    W[i, neigh] += ker[j, k]

    return W


def w_tilde_data_imaging_from(
    image_native: np.ndarray,
    noise_map_native: np.ndarray,
    kernel_native: np.ndarray,
    native_index_for_slim_index,
    xp=np,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    When w_tilde is used to perform an inversion, the mapping matrices are not computed, meaning that they cannot be
    used to compute the data vector. This method creates the vector `w_tilde_data` which allows for the data
    vector to be computed efficiently without the mapping matrix.

    The matrix w_tilde_data is dimensions [image_pixels] and encodes the PSF convolution with the `weight_map`,
    where the weights are the image-pixel values divided by the noise-map values squared:

    weight = image / noise**2.0

    Parameters
    ----------
    image_native
        The two dimensional masked image of values which `w_tilde_data` is computed from.
    noise_map_native
        The two dimensional masked noise-map of values which `w_tilde_data` is computed from.
    kernel_native
        The two dimensional PSF kernel that `w_tilde_data` encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    """

    # 1) weight map = image / noise^2 (safe where noise==0)
    weight_map = xp.where(
        noise_map_native > 0.0, image_native / (noise_map_native**2), 0.0
    )

    Ky, Kx = kernel_native.shape
    ph, pw = Ky // 2, Kx // 2

    # 2) pad so neighbourhood gathers never go OOB
    padded = xp.pad(
        weight_map, ((ph, ph), (pw, pw)), mode="constant", constant_values=0.0
    )

    # 3) build broadcasted neighbourhood indices for all requested pixels
    # shift pixel coords into the padded frame
    ys = native_index_for_slim_index[:, 0] + ph  # (N,)
    xs = native_index_for_slim_index[:, 1] + pw  # (N,)

    # kernel-relative offsets
    dy = xp.arange(Ky) - ph  # (Ky,)
    dx = xp.arange(Kx) - pw  # (Kx,)

    # broadcast to (N, Ky, Kx)
    Y = ys[:, None, None] + dy[None, :, None]
    X = xs[:, None, None] + dx[None, None, :]

    # 4) gather patches and correlate (no kernel flip)
    patches = padded[Y, X]  # (N, Ky, Kx)
    return xp.sum(patches * kernel_native[None, :, :], axis=(1, 2))  # (N,)


def data_vector_via_w_tilde_from(
    w_tilde_data: np.ndarray,     # (M_pix,) float64
    rows: np.ndarray,             # (nnz,) int32  each triplet's data pixel (slim index)
    cols: np.ndarray,                  # (nnz,) int32  source pixel index
    vals: np.ndarray,                  # (nnz,) float64 mapping weights incl sub_fraction
    S: int,                             # number of source pixels
) -> np.ndarray:
    """
    Replacement for numba data_vector_via_w_tilde_data_imaging_from using triplets.

    Computes:
        D[p] = sum_{triplets t with col_t=p} vals[t] * w_tilde_data_slim[slim_rows[t]]

    Returns:
        (S,) float64
    """
    from jax.ops import segment_sum

    w = w_tilde_data[rows]          # (nnz,)
    contrib = vals * w                         # (nnz,)
    return segment_sum(contrib, cols, num_segments=S)  # (S,)


def data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix: np.ndarray, image: np.ndarray, noise_map: np.ndarray
) -> np.ndarray:
    """
    Returns the data vector `D` from a blurred mapping matrix `f` and the 1D image `d` and 1D noise-map $\sigma$`
    (see Warren & Dye 2003).

    Parameters
    ----------
    blurred_mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    image
        Flattened 1D array of the observed image the inversion is fitting.
    noise_map
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """
    return (image / noise_map**2.0) @ blurred_mapping_matrix


def data_linear_func_matrix_from(
    curvature_weights_matrix: np.ndarray, kernel_native, mask
) -> np.ndarray:
    """
    Returns a matrix that for each data pixel, maps it to the sum of the values of a linear object function convolved
    with the PSF kernel at the data pixel.

    If a linear function in an inversion is fixed, its values can be evaluated and preloaded beforehand. For every
    data pixel, the PSF convolution with this preloaded linear function can also be preloaded, in a matrix of
    shape [data_pixels, 1].

    Given that multiple linear functions can be used and fixed in an inversion, this matrix is extended to have
    dimensions [data_pixels, total_fixed_linear_functions].

    When mapper objects and linear functions are used simultaneously in an inversion, this preloaded matrix
    significantly speed up the computation of their off-diagonal terms in the curvature matrix.

    This is similar to the preloading performed via the w-tilde formalism, except that there it is the PSF convolved
    values of each noise-map value pair that are preloaded.

    In **PyAutoGalaxy** and **PyAutoLens**, this preload is used when linear light profiles are fixed in the model.
    For example, when using a multi Gaussian expansion, the values defining how those Gaussians are evaluated
    (e.g. `centre`, `ell_comps` and `sigma`) are often fixed in a model, meaning this matrix can be preloaded and
    used for speed up.

    Parameters
    ----------
    curvature_weights_matrix
        The operated values of each linear function divided by the noise-map squared, in a matrix of shape
        [data_pixels, total_fixed_linear_functions].
    kernel_native
        The 2D PSf kernel.

    Returns
    -------
    ndarray
        A matrix of shape [data_pixels, total_fixed_linear_functions] that for each data pixel, maps it to the sum of
        the values of a linear object function convolved with the PSF kernel at the data pixel.
    """

    ny, nx = mask.shape_native
    n_unmasked, n_funcs = curvature_weights_matrix.shape

    # Expand masked -> native grid
    native = np.zeros((ny, nx, n_funcs))
    native[~mask] = curvature_weights_matrix  # put values into unmasked positions

    # Convolve each function with PSF kernel
    from scipy.signal import fftconvolve

    blurred_list = []
    for i in range(n_funcs):
        blurred = fftconvolve(native[..., i], kernel_native, mode="same")
        # Re-mask: only keep unmasked pixels
        blurred_list.append(blurred[~mask])

    return np.stack(blurred_list, axis=1)  # shape (n_unmasked, n_funcs)


def curvature_matrix_mirrored_from(
    curvature_matrix,
    *,
    xp=np,
):
    """
    Mirror a curvature matrix so that any non-zero entry C[i,j]
    is copied to C[j,i].

    Supports:
      - NumPy (xp=np)
      - JAX  (xp=jax.numpy)

    Parameters
    ----------
    curvature_matrix : (N, N) array
        Possibly triangular / partially-filled curvature matrix.

    xp : module
        np or jax.numpy

    Returns
    -------
    curvature_matrix_mirrored : (N, N) array
        Symmetric curvature matrix.
    """

    # Ensure array type
    C = curvature_matrix

    # Boolean mask of where entries exist
    mask = C != 0

    # Copy entries symmetrically
    C_T = xp.swapaxes(C, 0, 1)

    # Prefer non-zero values from either side
    curvature_matrix_mirrored = xp.where(mask, C, C_T)

    return curvature_matrix_mirrored

from typing import Optional, List

def curvature_matrix_with_added_to_diag_from(
    curvature_matrix,
    value: float,
    no_regularization_index_list: Optional[List[int]] = None,
    *,
    xp=np,
):
    """
    Add a small stabilizing value to the diagonal entries of the curvature matrix.

    Supports:
      - NumPy (xp=np): in-place update
      - JAX  (xp=jax.numpy): functional `.at[].add()`

    Parameters
    ----------
    curvature_matrix : (N, N) array
        Curvature matrix to modify.

    value : float
        Value added to selected diagonal entries.

    no_regularization_index_list : list of int
        Indices where diagonal should be boosted.

    xp : module
        np or jax.numpy

    Returns
    -------
    curvature_matrix : array
        Updated matrix (new array in JAX, modified in NumPy).
    """

    if no_regularization_index_list is None:
        return curvature_matrix

    inds = xp.asarray(no_regularization_index_list, dtype=xp.int32)

    if xp is np:
        # -----------------------
        # NumPy: in-place update
        # -----------------------
        curvature_matrix[inds, inds] += value
        return curvature_matrix

    else:
        # -----------------------
        # JAX: functional update
        # -----------------------
        return curvature_matrix.at[inds, inds].add(value)


def build_inv_noise_var(noise):
    inv = np.zeros_like(noise, dtype=np.float64)
    good = np.isfinite(noise) & (noise > 0)
    inv[good] = 1.0 / noise[good]**2
    return inv


def precompute_Khat_rfft(kernel_2d: np.ndarray, fft_shape):
    """
    kernel_2d: (Ky, Kx) real
    fft_shape: (Fy, Fx) where Fy = Hy+Ky-1, Fx = Hx+Kx-1
    returns: rfft2(padded_kernel) with shape (Fy, Fx//2+1), complex128 if input float64
    """

    import jax.numpy as jnp

    Ky, Kx = kernel_2d.shape
    Fy, Fx = fft_shape
    kernel_pad = jnp.pad(kernel_2d, ((0, Fy - Ky), (0, Fx - Kx)))
    return jnp.fft.rfft2(kernel_pad, s=(Fy, Fx))


def rfft_convolve2d_same(images: np.ndarray, Khat_r: np.ndarray, Ky: int, Kx: int, fft_shape):
    """
    Batched real FFT convolution, returning 'same' output.

    images: (B, Hy, Hx) real float64
    Khat_r: (Fy, Fx//2+1) complex128  (rfft2 of padded kernel)
    fft_shape: (Fy, Fx) must equal (Hy+Ky-1, Hx+Kx-1)
    """

    import jax.numpy as jnp

    B, Hy, Hx = images.shape
    Fy, Fx = fft_shape

    images_pad = jnp.pad(images, ((0, 0), (0, Fy - Hy), (0, Fx - Hx)))
    Fhat = jnp.fft.rfft2(images_pad, s=(Fy, Fx))                 # (B, Fy, Fx//2+1)
    out_pad = jnp.fft.irfft2(Fhat * Khat_r[None, :, :], s=(Fy, Fx))  # (B, Fy, Fx), real

    cy, cx = Ky // 2, Kx // 2
    return out_pad[:, cy:cy + Hy, cx:cx + Hx]

def curvature_matrix_via_w_tilde_from(
    inv_noise_var,     # (Hy, Hx) float64
    rows, cols, vals,  # COO mapping arrays
    y_shape: int,
    x_shape: int,
    S: int,
    Khat_r,            # (Fy, Fx//2+1) complex128
    Khat_flip_r,       # (Fy, Fx//2+1) complex128
    Ky: int,
    Kx: int,
    batch_size: int = 32,
):
    from jax.ops import segment_sum
    from jax import lax
    import jax.numpy as jnp

    inv_noise_var = jnp.asarray(inv_noise_var, dtype=jnp.float64)
    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)

    M = y_shape * x_shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    def apply_W(Fbatch_flat: jnp.ndarray) -> jnp.ndarray:
        B = Fbatch_flat.shape[1]
        Fimg = Fbatch_flat.T.reshape((B, y_shape, x_shape))
        blurred = rfft_convolve2d_same(Fimg, Khat_r, Ky, Kx, fft_shape)
        weighted = blurred * inv_noise_var[None, :, :]
        back = rfft_convolve2d_same(weighted, Khat_flip_r, Ky, Kx, fft_shape)
        return back.reshape((B, M)).T  # (M,B)

    n_blocks = (S + batch_size - 1) // batch_size
    C0 = jnp.zeros((S, S), dtype=jnp.float64)

    # Precompute a [0..batch_size-1] vector once (static size)
    col_offsets = jnp.arange(batch_size, dtype=jnp.int32)

    def body(block_i, C):
        start = block_i * batch_size  # dynamic scalar, OK

        # IMPORTANT: keep the "in block" test using static width batch_size
        in_block = (cols >= start) & (cols < (start + batch_size))

        bc = jnp.where(in_block, cols - start, 0).astype(jnp.int32)
        v  = jnp.where(in_block, vals, 0.0)

        # Build Fbatch on pixel grid
        F = jnp.zeros((M, batch_size), dtype=jnp.float64)
        F = F.at[rows, bc].add(v)

        # Apply W
        G = apply_W(F)  # (M, batch_size)

        # Accumulate into curvature columns for this block
        contrib = vals[:, None] * G[rows, :]  # (nnz, batch_size)

        # ---- fix: segment over source-pixel index, not cols ----
        # In your earlier diagonal code you did: segment_sum(contrib, cols, num_segments=S)
        # That only makes sense if `cols` are the "left" index of curvature (i.e. same mapper)
        # and you are building full C[:, start:start+B]. Keep it as you had:
        Cblock = segment_sum(contrib, cols, num_segments=S)  # (S, batch_size)

        # Mask out columns beyond S in the last block
        width = jnp.maximum(0, S - start)         # dynamic scalar
        width = jnp.minimum(width, batch_size)    # dynamic scalar in [0, batch_size]
        mask = (col_offsets < width).astype(jnp.float64)  # (batch_size,)
        Cblock = Cblock * mask[None, :]  # (S, batch_size)

        # Update full (S, batch_size) slice; always legal
        C = lax.dynamic_update_slice(C, Cblock, (0, start))
        return C

    C = lax.fori_loop(0, n_blocks, body, C0)
    return 0.5 * (C + C.T)


def build_curvature_rfft_fn(psf: np.ndarray, y_shape: int, x_shape: int):

    import jax
    import jax.numpy as jnp

    """
    Precompute Khat_r and Khat_flip_r once (float64), return a curvature function
    that can be jitted and called repeatedly.
    """
    Ky, Kx = psf.shape
    fft_shape = (y_shape + Ky - 1, x_shape + Kx - 1)

    Khat_r = precompute_Khat_rfft(psf, fft_shape)
    Khat_flip_r = precompute_Khat_rfft(jnp.flip(psf, axis=(0, 1)), fft_shape)

    # Jit wrapper with static shapes
    curvature_jit = jax.jit(
        partial(curvature_matrix_via_w_tilde_from, Khat_r=Khat_r, Khat_flip_r=Khat_flip_r, Ky=Ky, Kx=Kx),
        static_argnames=("y_shape", "x_shape", "S", "batch_size"),
    )
    return curvature_jit




def mapped_image_rect_from_triplets(
    reconstruction,   # (S,)
    rows,
    cols,
    vals, # (nnz,)
    fft_index_for_masked_pixel,
    data_shape: int,      # y_shape * x_shape
):
    import jax.numpy as jnp
    from jax.ops import segment_sum

    reconstruction = jnp.asarray(reconstruction, dtype=jnp.float64)
    rows = jnp.asarray(rows, dtype=jnp.int32)
    cols = jnp.asarray(cols, dtype=jnp.int32)
    vals = jnp.asarray(vals, dtype=jnp.float64)

    contrib = vals * reconstruction[cols]     # (nnz,)
    image_rect = segment_sum(contrib, rows, num_segments=data_shape[0] * data_shape[1])  # (M_rect,)

    image_slim = image_rect[fft_index_for_masked_pixel]            # (M_pix,)
    return image_slim
