from typing import Tuple

from autoarray import numba_util

import numpy as np

@numba_util.jit()
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

    kernel_shift_y = -(kernel_native.shape[1] // 2)
    kernel_shift_x = -(kernel_native.shape[0] // 2)

    image_pixels = len(native_index_for_slim_index)

    w_tilde_data = np.zeros((image_pixels,))

    weight_map_native = image_native / noise_map_native**2.0

    for ip0 in range(image_pixels):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        value = 0.0

        for k0_y in range(kernel_native.shape[0]):
            for k0_x in range(kernel_native.shape[1]):
                weight_value = weight_map_native[
                    ip0_y + k0_y + kernel_shift_y, ip0_x + k0_x + kernel_shift_x
                ]

                if not np.isnan(weight_value):
                    value += kernel_native[k0_y, k0_x] * weight_value

        w_tilde_data[ip0] = value

    return w_tilde_data

@numba_util.jit()
def w_tilde_curvature_imaging_from(
    noise_map_native: np.ndarray, kernel_native: np.ndarray, native_index_for_slim_index
) -> np.ndarray:
    """
    The matrix `w_tilde_curvature` is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF
    convolution of every pair of image pixels given the noise map. This can be used to efficiently compute the
    curvature matrix via the mappings between image and source pixels, in a way that omits having to perform the
    PSF convolution on every individual source pixel. This provides a significant speed up for inversions of imaging
    datasets.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_curvature_preload_imaging_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Parameters
    ----------
    noise_map_native
        The two dimensional masked noise-map of values which w_tilde is computed from.
    kernel_native
        The two dimensional PSF kernel that w_tilde encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the noise map that enables efficient calculation of
        the curvature matrix.
    """
    image_pixels = len(native_index_for_slim_index)

    w_tilde_curvature = np.zeros((image_pixels, image_pixels))

    for ip0 in range(w_tilde_curvature.shape[0]):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        for ip1 in range(ip0, w_tilde_curvature.shape[1]):
            ip1_y, ip1_x = native_index_for_slim_index[ip1]

            w_tilde_curvature[ip0, ip1] += w_tilde_curvature_value_from(
                value_native=noise_map_native,
                kernel_native=kernel_native,
                ip0_y=ip0_y,
                ip0_x=ip0_x,
                ip1_y=ip1_y,
                ip1_x=ip1_x,
            )

    for ip0 in range(w_tilde_curvature.shape[0]):
        for ip1 in range(ip0, w_tilde_curvature.shape[1]):
            w_tilde_curvature[ip1, ip0] = w_tilde_curvature[ip0, ip1]

    return w_tilde_curvature


@numba_util.jit()
def w_tilde_curvature_preload_imaging_from(
    noise_map_native: np.ndarray, kernel_native: np.ndarray, native_index_for_slim_index
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The matrix `w_tilde_curvature` is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF
    convolution of every pair of image pixels on the noise map. This can be used to efficiently compute the
    curvature matrix via the mappings between image and source pixels, in a way that omits having to repeat the PSF
    convolution on every individual source pixel. This provides a significant speed up for inversions of imaging
    datasets.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations slow. This methods creates
    a sparse matrix that can compute the matrix `w_tilde_curvature` efficiently, albeit the linear algebra calculations
    in PyAutoArray bypass this matrix entirely to go straight to the curvature matrix.

    for dataset data, w_tilde is a sparse matrix, whereby non-zero entries are only contained for pairs of image pixels
    where the two pixels overlap due to the kernel size. For example, if the kernel size is (11, 11) and two image
    pixels are separated by more than 20 pixels, the kernel will never convolve flux between the two pixels. Two image
    pixels will only share a convolution if they are within `kernel_overlap_size = 2 * kernel_shape - 1` pixels within
    one another.

    Thus, a `w_tilde_curvature_preload` matrix of dimensions [image_pixels, kernel_overlap_size ** 2] can be computed
    which significantly reduces the memory consumption by removing the sparsity. Because the dimensions of the second
    axes is no longer `image_pixels`, a second matrix `w_tilde_indexes` must also be computed containing the slim image
    pixel indexes of every entry of `w_tilde_preload`.

    In order for the preload to store half the number of values, owing to the symmetry of the `w_tilde_curvature`
    matrix, the image pixel pairs corresponding to the same image pixel are divided by two. This ensures that when the
    curvature matrix is computed these pixels are not double-counted.

    The values stored in `w_tilde_curvature_preload` represent the convolution of overlapping noise-maps given the
    PSF kernel. It is common for many values to be neglibly small. Removing these values can speed up the inversion
    and reduce memory at the expense of a numerically irrelevent change of solution.

    This matrix can then be used to compute the `curvature_matrix` in a memory efficient way that exploits the sparsity
    of the linear algebra.

    Parameters
    ----------
    noise_map_native
        The two dimensional masked noise-map of values which `w_tilde_curvature` is computed from.
    signal_to_noise_map_native
        The two dimensional masked signal-to-noise-map from which the threshold discarding low S/N image pixel
        pairs is used.
    kernel_native
        The two dimensional PSF kernel that `w_tilde_curvature` encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the noise map that enables efficient calculation of
        the curvature matrix, where the dimensions are reduced to save memory.
    """

    image_pixels = len(native_index_for_slim_index)

    kernel_overlap_size = (2 * kernel_native.shape[0] - 1) * (
        2 * kernel_native.shape[1] - 1
    )

    curvature_preload_tmp = np.zeros((image_pixels, kernel_overlap_size))
    curvature_indexes_tmp = np.zeros((image_pixels, kernel_overlap_size))
    curvature_lengths = np.zeros(image_pixels)

    for ip0 in range(image_pixels):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        kernel_index = 0

        for ip1 in range(ip0, curvature_preload_tmp.shape[0]):
            ip1_y, ip1_x = native_index_for_slim_index[ip1]

            noise_value = w_tilde_curvature_value_from(
                value_native=noise_map_native,
                kernel_native=kernel_native,
                ip0_y=ip0_y,
                ip0_x=ip0_x,
                ip1_y=ip1_y,
                ip1_x=ip1_x,
            )

            if ip0 == ip1:
                noise_value /= 2.0

            if noise_value > 0.0:
                curvature_preload_tmp[ip0, kernel_index] = noise_value
                curvature_indexes_tmp[ip0, kernel_index] = ip1
                kernel_index += 1

        curvature_lengths[ip0] = kernel_index

    curvature_total_pairs = int(np.sum(curvature_lengths))

    curvature_preload = np.zeros((curvature_total_pairs))
    curvature_indexes = np.zeros((curvature_total_pairs))

    index = 0

    for i in range(image_pixels):
        for data_index in range(int(curvature_lengths[i])):
            curvature_preload[index] = curvature_preload_tmp[i, data_index]
            curvature_indexes[index] = curvature_indexes_tmp[i, data_index]

            index += 1

    return (curvature_preload, curvature_indexes, curvature_lengths)


@numba_util.jit()
def w_tilde_curvature_value_from(
    value_native: np.ndarray,
    kernel_native: np.ndarray,
    ip0_y,
    ip0_x,
    ip1_y,
    ip1_x,
    renormalize=False,
) -> float:
    """
    Compute the value of an entry of the `w_tilde_curvature` matrix, where this entry encodes the PSF convolution of
    the noise-map between two image pixels.

    The calculation is performed by over-laying the PSF kernel over two noise-map pixels in 2D. For all pixels where
    the two overlaid PSF kernels overlap, the following calculation is performed for every noise map value:

    `value = kernel_value_0 * kernel_value_1 * (1.0 / noise_value) ** 2.0`

    This calculation infers the fraction of flux that every PSF convolution will move between each pair of noise-map
    pixels and can therefore be used to efficiently calculate the curvature_matrix that is used in the linear algebra
    calculation of an inversion.

    The sum of all values where kernel pixels overlap is returned to give the `w_tilde` value.

    Parameters
    ----------
    value_native
        A two dimensional masked array of values (e.g. a noise-map, signal to noise map) which the w_tilde curvature
        values are computed from.
    kernel_native
        The two dimensional PSF kernel that w_tilde encodes the convolution of.
    ip0_y
        The y index of the first image pixel in the image pixel pair.
    ip0_x
        The x index of the first image pixel in the image pixel pair.
    ip1_y
        The y index of the second image pixel in the image pixel pair.
    ip1_x
        The x index of the second image pixel in the image pixel pair.

    Returns
    -------
    float
        The w_tilde value that encodes the value of PSF convolution between a pair of image pixels.

    """

    curvature_value = 0.0

    kernel_shift_y = -(kernel_native.shape[1] // 2)
    kernel_shift_x = -(kernel_native.shape[0] // 2)

    ip_y_offset = ip0_y - ip1_y
    ip_x_offset = ip0_x - ip1_x

    if (
        ip_y_offset < 2 * kernel_shift_y
        or ip_y_offset > -2 * kernel_shift_y
        or ip_x_offset < 2 * kernel_shift_x
        or ip_x_offset > -2 * kernel_shift_x
    ):
        return curvature_value

    kernel_pixels = kernel_native.shape[0] * kernel_native.shape[1]
    kernel_count = 0

    for k0_y in range(kernel_native.shape[0]):
        for k0_x in range(kernel_native.shape[1]):
            value = value_native[
                ip0_y + k0_y + kernel_shift_y, ip0_x + k0_x + kernel_shift_x
            ]

            if value > 0.0:
                k1_y = k0_y + ip_y_offset
                k1_x = k0_x + ip_x_offset

                if (
                    k1_y >= 0
                    and k1_x >= 0
                    and k1_y < kernel_native.shape[0]
                    and k1_x < kernel_native.shape[1]
                ):
                    kernel_count += 1

                    kernel_value_0 = kernel_native[k0_y, k0_x]
                    kernel_value_1 = kernel_native[k1_y, k1_x]

                    curvature_value += (
                        kernel_value_0 * kernel_value_1 * (1.0 / value) ** 2.0
                    )

    if renormalize:
        if kernel_count > 0:
            curvature_value *= kernel_pixels / kernel_count

    return curvature_value

@numba_util.jit()
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

    data_shape = blurred_mapping_matrix.shape

    data_vector = np.zeros(data_shape[1])

    for data_index in range(data_shape[0]):
        for pix_index in range(data_shape[1]):
            data_vector[pix_index] += (
                image[data_index]
                * blurred_mapping_matrix[data_index, pix_index]
                / (noise_map[data_index] ** 2.0)
            )

    return data_vector

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
def convolve_with_kernel_native(curvature_native, psf_kernel):
    """
    Convolve each function slice of curvature_native with psf_kernel using direct sliding window.

    Parameters
    ----------
    curvature_native : ndarray (ny, nx, n_funcs)
        Curvature weights expanded to the native grid, 0 in masked regions.
    psf_kernel : ndarray (ky, kx)
        The PSF kernel.

    Returns
    -------
    blurred_native : ndarray (ny, nx, n_funcs)
        The curvature weights convolved with the PSF.
    """
    ny, nx, n_funcs = curvature_native.shape
    ky, kx = psf_kernel.shape
    cy, cx = ky // 2, kx // 2  # kernel center

    blurred_native = np.zeros_like(curvature_native)

    for f in range(n_funcs):  # parallelize over functions
        for y in range(ny):
            for x in range(nx):
                acc = 0.0
                for dy in range(ky):
                    for dx in range(kx):
                        yy = y + dy - cy
                        xx = x + dx - cx
                        if 0 <= yy < ny and 0 <= xx < nx:
                            acc += psf_kernel[dy, dx] * curvature_native[yy, xx, f]
                blurred_native[y, x, f] = acc
    return blurred_native

@numba_util.jit()
def curvature_matrix_off_diags_via_mapper_and_linear_func_curvature_vector_from(
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    pix_pixels: int,
    curvature_weights: np.ndarray,   # shape (n_unmasked, n_funcs)
    mask: np.ndarray,                # shape (ny, nx), bool
    psf_kernel: np.ndarray           # shape (ky, kx)
) -> np.ndarray:
    """
    Compute the off-diagonal curvature terms using the PSF kernel directly (Numba version).
    """
    data_pixels = data_weights.shape[0]
    n_funcs = curvature_weights.shape[1]
    ny, nx = mask.shape

    # Expand curvature weights into native grid
    curvature_native = np.zeros((ny, nx, n_funcs))
    unmasked_coords = np.argwhere(~mask)
    for i, (y, x) in enumerate(unmasked_coords):
        for f in range(n_funcs):
            curvature_native[y, x, f] = curvature_weights[i, f]

    # Convolve in native space
    blurred_native = convolve_with_kernel_native(curvature_native, psf_kernel)

    # Map back to slim representation
    blurred_slim = np.zeros((data_pixels, n_funcs))
    for i, (y, x) in enumerate(unmasked_coords):
        for f in range(n_funcs):
            blurred_slim[i, f] = blurred_native[y, x, f]

    # Accumulate into off_diag
    off_diag = np.zeros((pix_pixels, n_funcs))
    for data_0 in range(data_pixels):
        for pix_0_index in range(pix_lengths[data_0]):
            data_0_weight = data_weights[data_0, pix_0_index]
            pix_0 = data_to_pix_unique[data_0, pix_0_index]
            for f in range(n_funcs):
                off_diag[pix_0, f] += data_0_weight * blurred_slim[data_0, f]

    return off_diag