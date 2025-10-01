import jax.numpy as jnp
from typing import Tuple

from autoarray import numba_util

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
    weight_map = jnp.where(
        noise_map_native > 0.0, image_native / (noise_map_native**2), 0.0
    )

    Ky, Kx = kernel_native.shape
    ph, pw = Ky // 2, Kx // 2

    # 2) pad so neighbourhood gathers never go OOB
    padded = jnp.pad(
        weight_map, ((ph, ph), (pw, pw)), mode="constant", constant_values=0.0
    )

    # 3) build broadcasted neighbourhood indices for all requested pixels
    # shift pixel coords into the padded frame
    ys = native_index_for_slim_index[:, 0] + ph  # (N,)
    xs = native_index_for_slim_index[:, 1] + pw  # (N,)

    # kernel-relative offsets
    dy = jnp.arange(Ky) - ph  # (Ky,)
    dx = jnp.arange(Kx) - pw  # (Kx,)

    # broadcast to (N, Ky, Kx)
    Y = ys[:, None, None] + dy[None, :, None]
    X = xs[:, None, None] + dx[None, None, :]

    # 4) gather patches and correlate (no kernel flip)
    patches = padded[Y, X]  # (N, Ky, Kx)
    return jnp.sum(patches * kernel_native[None, :, :], axis=(1, 2))  # (N,)


@numba_util.jit()
def data_vector_via_w_tilde_data_imaging_from(
    w_tilde_data: np.ndarray,
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    pix_pixels: int,
) -> np.ndarray:
    """
    Returns the data vector `D` from the `w_tilde_data` matrix (see `w_tilde_data_imaging_from`), which encodes the
    the 1D image `d` and 1D noise-map values `\sigma` (see Warren & Dye 2003).

    This uses the array `data_to_pix_unique`, which describes the unique mappings of every set of image sub-pixels to
    pixelization pixels and `data_weights`, which describes how many sub-pixels uniquely map to each pixelization
    pixels (see `data_slim_to_pixelization_unique_from`).

    Parameters
    ----------
    w_tilde_data
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    data_to_pix_unique
        An array that maps every data pixel index (e.g. the masked image pixel indexes in 1D) to its unique set of
        pixelization pixel indexes (see `data_slim_to_pixelization_unique_from`).
    data_weights
        For every unique mapping between a set of data sub-pixels and a pixelization pixel, the weight of these mapping
        based on the number of sub-pixels that map to pixelization pixel.
    pix_lengths
        A 1D array describing how many unique pixels each data pixel maps too, which is used to iterate over
        `data_to_pix_unique` and `data_weights`.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.
    """

    data_pixels = w_tilde_data.shape[0]

    data_vector = np.zeros(pix_pixels)

    for data_0 in range(data_pixels):
        for pix_0_index in range(pix_lengths[data_0]):
            data_0_weight = data_weights[data_0, pix_0_index]
            pix_0 = data_to_pix_unique[data_0, pix_0_index]

            data_vector[pix_0] += data_0_weight * w_tilde_data[data_0]

    return data_vector


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


@numba_util.jit()
def curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
    curvature_preload: np.ndarray,
    curvature_indexes: np.ndarray,
    curvature_lengths: np.ndarray,
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    pix_pixels: int,
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) by computing it using `w_tilde_preload`
    (see `w_tilde_preload_interferometer_from`) for an imaging inversion.

    To compute the curvature matrix via w_tilde the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * w_tilde * mapping matrix

    This function speeds this calculation up in two ways:

    1) Instead of using `w_tilde` (dimensions [image_pixels, image_pixels] it uses `w_tilde_preload` (dimensions
    [image_pixels, kernel_overlap]). The massive reduction in the size of this matrix in memory allows for much fast
    computation.

    2) It omits the `mapping_matrix` and instead uses directly the 1D vector that maps every image pixel to a source
    pixel `native_index_for_slim_index`. This exploits the sparsity in the `mapping_matrix` to directly
    compute the `curvature_matrix` (e.g. it condenses the triple matrix multiplication into a double for loop!).

    Parameters
    ----------
    curvature_preload
        A matrix that precomputes the values for fast computation of the curvature matrix in a memory efficient way.
    curvature_indexes
        The image-pixel indexes of the values stored in the w tilde preload matrix, which are used to compute
        the weights of the data values when computing the curvature matrix.
    curvature_lengths
        The number of image pixels in every row of `w_tilde_curvature`, which is iterated over when computing the
        curvature matrix.
    data_to_pix_unique
        An array that maps every data pixel index (e.g. the masked image pixel indexes in 1D) to its unique set of
        pixelization pixel indexes (see `data_slim_to_pixelization_unique_from`).
    data_weights
        For every unique mapping between a set of data sub-pixels and a pixelization pixel, the weight of these mapping
        based on the number of sub-pixels that map to pixelization pixel.
    pix_lengths
        A 1D array describing how many unique pixels each data pixel maps too, which is used to iterate over
        `data_to_pix_unique` and `data_weights`.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    data_pixels = curvature_lengths.shape[0]

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    curvature_index = 0

    for data_0 in range(data_pixels):
        for data_1_index in range(curvature_lengths[data_0]):
            data_1 = curvature_indexes[curvature_index]
            w_tilde_value = curvature_preload[curvature_index]

            for pix_0_index in range(pix_lengths[data_0]):
                data_0_weight = data_weights[data_0, pix_0_index]
                pix_0 = data_to_pix_unique[data_0, pix_0_index]

                for pix_1_index in range(pix_lengths[data_1]):
                    data_1_weight = data_weights[data_1, pix_1_index]
                    pix_1 = data_to_pix_unique[data_1, pix_1_index]

                    curvature_matrix[pix_0, pix_1] += (
                        data_0_weight * data_1_weight * w_tilde_value
                    )

            curvature_index += 1

    for i in range(pix_pixels):
        for j in range(i, pix_pixels):
            curvature_matrix[i, j] += curvature_matrix[j, i]

    for i in range(pix_pixels):
        for j in range(i, pix_pixels):
            curvature_matrix[j, i] = curvature_matrix[i, j]

    return curvature_matrix


@numba_util.jit()
def curvature_matrix_off_diags_via_w_tilde_curvature_preload_imaging_from(
    curvature_preload: np.ndarray,
    curvature_indexes: np.ndarray,
    curvature_lengths: np.ndarray,
    data_to_pix_unique_0: np.ndarray,
    data_weights_0: np.ndarray,
    pix_lengths_0: np.ndarray,
    pix_pixels_0: int,
    data_to_pix_unique_1: np.ndarray,
    data_weights_1: np.ndarray,
    pix_lengths_1: np.ndarray,
    pix_pixels_1: int,
) -> np.ndarray:
    """
    Returns the off diagonal terms in the curvature matrix `F` (see Warren & Dye 2003) by computing them
    using `w_tilde_preload` (see `w_tilde_preload_interferometer_from`) for an imaging inversion.

    When there is more than one mapper in the inversion, its `mapping_matrix` is extended to have dimensions
    [data_pixels, sum(source_pixels_in_each_mapper)]. The curvature matrix therefore will have dimensions
    [sum(source_pixels_in_each_mapper), sum(source_pixels_in_each_mapper)].

    To compute the curvature matrix via w_tilde the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * w_tilde * mapping matrix

    When the `mapping_matrix` consists of multiple mappers from different planes, this means that shared data mappings
    between source-pixels in different mappers must be accounted for when computing the `curvature_matrix`. These
    appear as off-diagonal terms in the overall curvature matrix.

    This function evaluates these off-diagonal terms, by using the w-tilde curvature preloads and the unique
    data-to-pixelization mappings of each mapper. It behaves analogous to the
    function `curvature_matrix_via_w_tilde_curvature_preload_imaging_from`.

    Parameters
    ----------
    curvature_preload
        A matrix that precomputes the values for fast computation of the curvature matrix in a memory efficient way.
    curvature_indexes
        The image-pixel indexes of the values stored in the w tilde preload matrix, which are used to compute
        the weights of the data values when computing the curvature matrix.
    curvature_lengths
        The number of image pixels in every row of `w_tilde_curvature`, which is iterated over when computing the
        curvature matrix.
    data_to_pix_unique
        An array that maps every data pixel index (e.g. the masked image pixel indexes in 1D) to its unique set of
        pixelization pixel indexes (see `data_slim_to_pixelization_unique_from`).
    data_weights
        For every unique mapping between a set of data sub-pixels and a pixelization pixel, the weight of these mapping
        based on the number of sub-pixels that map to pixelization pixel.
    pix_lengths
        A 1D array describing how many unique pixels each data pixel maps too, which is used to iterate over
        `data_to_pix_unique` and `data_weights`.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    data_pixels = curvature_lengths.shape[0]

    curvature_matrix = np.zeros((pix_pixels_0, pix_pixels_1))

    curvature_index = 0

    for data_0 in range(data_pixels):
        for data_1_index in range(curvature_lengths[data_0]):
            data_1 = curvature_indexes[curvature_index]
            w_tilde_value = curvature_preload[curvature_index]

            for pix_0_index in range(pix_lengths_0[data_0]):
                data_0_weight = data_weights_0[data_0, pix_0_index]
                pix_0 = data_to_pix_unique_0[data_0, pix_0_index]

                for pix_1_index in range(pix_lengths_1[data_1]):
                    data_1_weight = data_weights_1[data_1, pix_1_index]
                    pix_1 = data_to_pix_unique_1[data_1, pix_1_index]

                    curvature_matrix[pix_0, pix_1] += (
                        data_0_weight * data_1_weight * w_tilde_value
                    )

            curvature_index += 1

    return curvature_matrix


@numba_util.jit()
def data_linear_func_matrix_from(
    curvature_weights_matrix: np.ndarray,
    image_frame_1d_lengths: np.ndarray,
    image_frame_1d_indexes: np.ndarray,
    image_frame_1d_kernels: np.ndarray,
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
    image_frame_indexes
        The indexes of all masked pixels that the PSF blurs light into (see the `Convolver` object).
    image_frame_kernels
        The kernel values of all masked pixels that the PSF blurs light into (see the `Convolver` object).
    image_frame_length
        The number of masked pixels it will blur light into (unmasked pixels are excluded, see the `Convolver` object).

    Returns
    -------
    ndarray
        A matrix of shape [data_pixels, total_fixed_linear_functions] that for each data pixel, maps it to the sum of
        the values of a linear object function convolved with the PSF kernel at the data pixel.
    """
    data_pixels = curvature_weights_matrix.shape[0]
    linear_func_pixels = curvature_weights_matrix.shape[1]

    data_linear_func_matrix_dict = np.zeros(shape=(data_pixels, linear_func_pixels))

    for data_0 in range(data_pixels):
        for psf_index in range(image_frame_1d_lengths[data_0]):
            data_index = image_frame_1d_indexes[data_0, psf_index]
            kernel_value = image_frame_1d_kernels[data_0, psf_index]

            for linear_index in range(linear_func_pixels):
                data_linear_func_matrix_dict[data_0, linear_index] += (
                    kernel_value * curvature_weights_matrix[data_index, linear_index]
                )

    return data_linear_func_matrix_dict


@numba_util.jit()
def curvature_matrix_off_diags_via_data_linear_func_matrix_from(
    data_linear_func_matrix: np.ndarray,
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    pix_pixels: int,
):
    """
    Returns the off diagonal terms in the curvature matrix `F` (see Warren & Dye 2003) between a mapper object
    and a linear func object, using the preloaded `data_linear_func_matrix` of the values of the linear functions.


    If a linear function in an inversion is fixed, its values can be evaluated and preloaded beforehand. For every
    data pixel, the PSF convolution with this preloaded linear function can also be preloaded, in a matrix of
    shape [data_pixels, 1].

    When mapper objects and linear functions are used simultaneously in an inversion, this preloaded matrix
    significantly speed up the computation of their off-diagonal terms in the curvature matrix.

    This function performs this efficient calcluation via the preloaded `data_linear_func_matrix`.

    Parameters
    ----------
    data_linear_func_matrix
        A matrix of shape [data_pixels, total_fixed_linear_functions] that for each data pixel, maps it to the sum of
        the values of a linear object function convolved with the PSF kernel at the data pixel.
    data_to_pix_unique
        The indexes of all pixels that each data pixel maps to (see the `Mapper` object).
    data_weights
        The weights of all pixels that each data pixel maps to (see the `Mapper` object).
    pix_lengths
        The number of pixelization pixels that each data pixel maps to (see the `Mapper` object).
    pix_pixels
        The number of pixelization pixels in the pixelization (see the `Mapper` object).
    """

    linear_func_pixels = data_linear_func_matrix.shape[1]

    off_diag = np.zeros((pix_pixels, linear_func_pixels))

    data_pixels = data_weights.shape[0]

    for data_0 in range(data_pixels):
        for pix_0_index in range(pix_lengths[data_0]):
            data_0_weight = data_weights[data_0, pix_0_index]
            pix_0 = data_to_pix_unique[data_0, pix_0_index]

            for linear_index in range(linear_func_pixels):
                off_diag[pix_0, linear_index] += (
                    data_linear_func_matrix[data_0, linear_index] * data_0_weight
                )

    return off_diag


@numba_util.jit()
def curvature_matrix_off_diags_via_mapper_and_linear_func_curvature_vector_from(
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    pix_pixels: int,
    curvature_weights: np.ndarray,
    image_frame_1d_lengths: np.ndarray,
    image_frame_1d_indexes: np.ndarray,
    image_frame_1d_kernels: np.ndarray,
) -> np.ndarray:
    """
    Returns the off diagonal terms in the curvature matrix `F` (see Warren & Dye 2003) between a mapper object
    and a linear func object, using the unique mappings between data pixels and pixelization pixels.

    This takes as input the curvature weights of the linear function object, which are the values of the linear
    function convolved with the PSF and divided by the noise-map squared.

    For each unique mapping between a data pixel and a pixelization pixel, the pixels which that pixel convolves
    light into are computed, multiplied by their corresponding curvature weights and summed. This process also
    accounts the sub-pixel mapping of each data pixel to the pixelization pixel

    This is done for every unique mapping of a data pixel to a pixelization pixel, giving the off-diagonal terms in
    the curvature matrix.

    Parameters
    ----------
    data_to_pix_unique
        An array that maps every data pixel index (e.g. the masked image pixel indexes in 1D) to its unique set of
        pixelization pixel indexes (see `data_slim_to_pixelization_unique_from`).
    data_weights
        For every unique mapping between a set of data sub-pixels and a pixelization pixel, the weight of these mapping
        based on the number of sub-pixels that map to pixelization pixel.
    pix_lengths
        A 1D array describing how many unique pixels each data pixel maps too, which is used to iterate over
        `data_to_pix_unique` and `data_weights`.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.
    curvature_weights
        The operated values of the linear func divided by the noise-map squared.
    image_frame_indexes
        The indexes of all masked pixels that the PSF blurs light into (see the `Convolver` object).
    image_frame_kernels
        The kernel values of all masked pixels that the PSF blurs light into (see the `Convolver` object).
    image_frame_length
        The number of masked pixels it will blur light into (unmasked pixels are excluded, see the `Convolver` object).

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    data_pixels = data_weights.shape[0]
    linear_func_pixels = curvature_weights.shape[1]

    off_diag = np.zeros((pix_pixels, linear_func_pixels))

    for data_0 in range(data_pixels):
        for pix_0_index in range(pix_lengths[data_0]):
            data_0_weight = data_weights[data_0, pix_0_index]
            pix_0 = data_to_pix_unique[data_0, pix_0_index]

            for psf_index in range(image_frame_1d_lengths[data_0]):
                data_index = image_frame_1d_indexes[data_0, psf_index]
                kernel_value = image_frame_1d_kernels[data_0, psf_index]

                off_diag[pix_0, :] += (
                    data_0_weight * curvature_weights[data_index, :] * kernel_value
                )

    return off_diag
