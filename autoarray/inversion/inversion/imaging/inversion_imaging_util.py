import numpy as np


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
    xp=np
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
