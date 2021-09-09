import numpy as np
from scipy.linalg import cho_solve
from typing import Tuple

from autoarray.inversion.inversion.settings import SettingsInversion

from autoarray import numba_util
from autoarray import exc


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

    weight_map_native = image_native / noise_map_native ** 2.0

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
    noise_map_native: np.ndarray,
    signal_to_noise_map_native: np.ndarray,
    kernel_native: np.ndarray,
    native_index_for_slim_index,
    snr_cut=-1.0e99,
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
    in PyAutoArray bypass this matrix entire to go straight to the curvature matrix.

    For imaging data, w_tilde is a sparse matrix, whereby non-zero entries are only contained for pairs of image pixels
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

    Removing values based on the noise-map depends on the units of the noise-map and is hard to define generically.
    Thus a `snr_cut` is used instead that removes all PSF convolved image-pixel pairs where the
    convolved S/N is below this value. Tests reveal that using a value of 1.0e-10 has neglible impact on the numerical
    solution of an inversion.

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

            # signal_to_noise_value = w_tilde_curvature_value_from(
            #     value_native=signal_to_noise_map_native,
            #     kernel_native=kernel_native,
            #     ip0_y=ip0_y,
            #     ip0_x=ip0_x,
            #     ip1_y=ip1_y,
            #     ip1_x=ip1_x,
            #     renormalize=True,
            # )
            #
            # if signal_to_noise_value > snr_cut:

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
    -----------
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
def data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix: np.ndarray, image: np.ndarray, noise_map: np.ndarray
) -> np.ndarray:
    """
    Returns the data vector `D` from a blurred mapping matrix `f` and the 1D image `d` and 1D noise-map $\sigma$`
    (see Warren & Dye 2003).

    Parameters
    -----------
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
def data_vector_via_transformed_mapping_matrix_from(
    transformed_mapping_matrix: np.ndarray,
    visibilities: np.ndarray,
    noise_map: np.ndarray,
) -> np.ndarray:
    """
    Returns the data vector `D` from a transformed mapping matrix `f` and the 1D image `d` and 1D noise-map `sigma`
    (see Warren & Dye 2003).

    Parameters
    -----------
    transformed_mapping_matrix
        The matrix representing the transformed mappings between sub-grid pixels and pixelization pixels.
    image
        Flattened 1D array of the observed image the inversion is fitting.
    noise_map
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """

    data_vector = np.zeros(transformed_mapping_matrix.shape[1])

    visibilities_real = visibilities.real
    visibilities_imag = visibilities.imag
    transformed_mapping_matrix_real = transformed_mapping_matrix.real
    transformed_mapping_matrix_imag = transformed_mapping_matrix.imag
    noise_map_real = noise_map.real
    noise_map_imag = noise_map.imag

    for vis_1d_index in range(transformed_mapping_matrix.shape[0]):
        for pix_1d_index in range(transformed_mapping_matrix.shape[1]):
            real_value = (
                visibilities_real[vis_1d_index]
                * transformed_mapping_matrix_real[vis_1d_index, pix_1d_index]
                / (noise_map_real[vis_1d_index] ** 2.0)
            )
            imag_value = (
                visibilities_imag[vis_1d_index]
                * transformed_mapping_matrix_imag[vis_1d_index, pix_1d_index]
                / (noise_map_imag[vis_1d_index] ** 2.0)
            )
            data_vector[pix_1d_index] += real_value + imag_value

    return data_vector


@numba_util.jit()
def curvature_matrix_via_w_tilde_from(
    w_tilde: np.ndarray, mapping_matrix: np.ndarray
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) from `w_tilde`.

    The dimensions of `w_tilde` are [image_pixels, image_pixels], meaning that for datasets with many image pixels
    this matrix can take up 10's of GB of memory. The calculation of the `curvature_matrix` via this function will
    therefore be very slow, and the method `curvature_matrix_via_w_tilde_curvature_preload_imaging_from` should be used
    instead.

    Parameters
    ----------
    w_tilde
        A matrix of dimensions [image_pixels, image_pixels] that encodes the convolution or NUFFT of every image pixel
        pair on the noise map.
    mapping_matrix
        The matrix representing the mappings between sub-grid pixels and pixelization pixels.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    return np.dot(mapping_matrix.T, np.dot(w_tilde, mapping_matrix))


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


def curvature_matrix_via_mapping_matrix_from(
    mapping_matrix: np.ndarray, noise_map: np.ndarray
) -> np.ndarray:
    """
    Returns the curvature matrix `F` from a blurred mapping matrix `f` and the 1D noise-map $\sigma$
     (see Warren & Dye 2003).

    Parameters
    -----------
    mapping_matrix
        The matrix representing the mappings (these could be blurred or transfomed) between sub-grid pixels and
        pixelization pixels.
    noise_map
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """
    array = mapping_matrix / noise_map[:, None]
    curvature_matrix = np.dot(array.T, array)
    return curvature_matrix


@numba_util.jit()
def curvature_matrix_sparse_preload_via_mapping_matrix_from(
    mapping_matrix: np.ndarray, mapping_matrix_threshold=1.0e-8
) -> np.ndarray:
    """
    Returns a matrix that expresses the non-zero entries of the blurred mapping matrix for an efficient construction of
    the curvature matrix `F` (see Warren & Dye 2003).

    This is used for models where the blurred mapping matrix does not change, but the noise-map of the values that
    are used to construct that blurred mapping matrix do change.

    Parameters
    -----------
    mapping_matrix
        The matrix representing the mappings (these could be blurred or transfomed) between sub-grid pixels and
        pixelization pixels.
    """

    curvature_matrix_preload_counts = np.zeros(mapping_matrix.shape[0])

    for mask_1d_index in range(mapping_matrix.shape[0]):
        for pix_index in range(mapping_matrix.shape[1]):
            if mapping_matrix[mask_1d_index, pix_index] > mapping_matrix_threshold:
                curvature_matrix_preload_counts[mask_1d_index] += 1

    preload_max = np.max(curvature_matrix_preload_counts)

    curvature_matrix_sparse_preload = np.zeros(
        (mapping_matrix.shape[0], int(preload_max))
    )

    for mask_1d_index in range(mapping_matrix.shape[0]):
        index = 0
        for pix_index in range(mapping_matrix.shape[1]):
            if mapping_matrix[mask_1d_index, pix_index] > mapping_matrix_threshold:
                curvature_matrix_sparse_preload[mask_1d_index, index] = pix_index
                index += 1

    return curvature_matrix_sparse_preload, curvature_matrix_preload_counts


@numba_util.jit()
def curvature_matrix_via_sparse_preload_from(
    mapping_matrix: np.ndarray,
    noise_map: np.ndarray,
    curvature_matrix_sparse_preload,
    curvature_matrix_preload_counts,
) -> np.ndarray:
    """
    Returns the curvature matrix `F` from a blurred mapping matrix `f` and the 1D noise-map $\sigma$
     (see Warren & Dye 2003) via a sparse preload matrix, which has already mapped out where all non-zero entries
     and multiplications take place.

    This is used for models where the blurred mapping matrix does not change, but the noise-map of the values that
    are used to construct that blurred mapping matrix do change.

    Parameters
    -----------
    mapping_matrix
        The matrix representing the mappings (these could be blurred or transfomed) between sub-grid pixels and
        pixelization pixels.
    noise_map
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """
    curvature_matrix = np.zeros((mapping_matrix.shape[1], mapping_matrix.shape[1]))

    for mask_1d_index in range(mapping_matrix.shape[0]):

        total_pix = curvature_matrix_preload_counts[mask_1d_index]

        for preload_index_0 in range(total_pix):
            for preload_index_1 in range(
                preload_index_0, curvature_matrix_preload_counts[mask_1d_index]
            ):

                pix_index_0 = curvature_matrix_sparse_preload[
                    mask_1d_index, preload_index_0
                ]
                pix_index_1 = curvature_matrix_sparse_preload[
                    mask_1d_index, preload_index_1
                ]

                curvature_matrix[pix_index_0, pix_index_1] += (
                    mapping_matrix[mask_1d_index, pix_index_0]
                    * mapping_matrix[mask_1d_index, pix_index_1]
                ) / noise_map[mask_1d_index] ** 2

    for i in range(mapping_matrix.shape[1]):
        for j in range(mapping_matrix.shape[1]):
            curvature_matrix[j, i] = curvature_matrix[i, j]

    return curvature_matrix


@numba_util.jit()
def curvature_reg_matrix_from(
    curvature_matrix: np.ndarray,
    regularization_matrix: np.ndarray,
    pixel_neighbors: np.ndarray,
    pixel_neighbors_sizes: np.ndarray,
):
    """
    Add together the `curvature_matrix` (F) and `regularization_matrix` (H).

    This function is faster than an np.add for the following reasons:

    1) The memory allocated to the `curvature_matrix` is overwritten, avoiding overhead due to copying.
    2) The pixel neighbors computed previously are used to speed up the calculation, by using the known sparsity of
    the regularization matrix.

    Parameters
    -----------
    curvature_matrix
        The matrix defining the correlations between image-pixels due to them being in the same source pixels, or
         correlated due to sub-gridding / PSF convolution / a Fourier Transform.
    regularization_matrix
        The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
    pixel_neighbors
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    pixel_neighbors_sizes
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.

    Returns
    -------
    curvature_reg_matrix
        The curvature_matrix plus regularization matrix, overwriting the curvature_matrix in memory.
    """
    for i in range(regularization_matrix.shape[0]):

        curvature_matrix[i, i] += regularization_matrix[i, i]

        for j in range(pixel_neighbors_sizes[i]):

            neighbor_index = pixel_neighbors[i, j]

            curvature_matrix[i, neighbor_index] += regularization_matrix[
                i, neighbor_index
            ]

    return curvature_matrix


def reconstruction_from(
    data_vector: np.ndarray,
    curvature_reg_matrix_cholesky: np.ndarray,
    settings: SettingsInversion = SettingsInversion(),
):
    """
    Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
    of https://arxiv.org/pdf/astro-ph/0302587.pdf

    S is the vector of reconstructed inversion values.

    This function checks that the solution does not give a linear algebra error (e.g. because the input matrix is
    not positive-definitive) and that it avoids solutions where all reconstructed values go to the same value. If these
    occur it raises an exception.

    Parameters
    ----------
    data_vector
        The `data_vector` D which is solved for.
    curvature_reg_matrix_cholesky
        The cholesky decomposition of the sum of the curvature and regularization matrices.
    settings
        Controls the settings of the inversion, for this function where the solution is checked to not be all
        the same values.

    Returns
    -------
    curvature_reg_matrix
        The curvature_matrix plus regularization matrix, overwriting the curvature_matrix in memory.
    """
    try:
        reconstruction = cho_solve((curvature_reg_matrix_cholesky, True), data_vector)
    except np.linalg.LinAlgError:
        raise exc.InversionException()

    if settings.check_solution:
        if np.isclose(a=reconstruction[0], b=reconstruction[1], atol=1e-4).all():
            if np.isclose(a=reconstruction[0], b=reconstruction, atol=1e-4).all():
                raise exc.InversionException()

    return reconstruction


@numba_util.jit()
def mapped_reconstructed_data_via_image_to_pix_unique_from(
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    reconstruction: np.ndarray,
) -> np.ndarray:
    """
    Returns the reconstructed data vector from the blurred mapping matrix `f` and solution vector *S*.

    Parameters
    -----------
    mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """

    data_pixels = data_to_pix_unique.shape[0]

    mapped_reconstructed_data = np.zeros(data_pixels)

    for data_0 in range(data_pixels):
        for pix_0 in range(pix_lengths[data_0]):

            pix_for_data = data_to_pix_unique[data_0, pix_0]

            mapped_reconstructed_data[data_0] += (
                data_weights[data_0, pix_0] * reconstruction[pix_for_data]
            )

    return mapped_reconstructed_data


@numba_util.jit()
def mapped_reconstructed_data_via_mapping_matrix_from(
    mapping_matrix: np.ndarray, reconstruction: np.ndarray
) -> np.ndarray:
    """
    Returns the reconstructed data vector from the blurred mapping matrix `f` and solution vector *S*.

    Parameters
    -----------
    mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """
    mapped_reconstructed_data = np.zeros(mapping_matrix.shape[0])
    for i in range(mapping_matrix.shape[0]):
        for j in range(reconstruction.shape[0]):
            mapped_reconstructed_data[i] += reconstruction[j] * mapping_matrix[i, j]

    return mapped_reconstructed_data


@numba_util.jit()
def mapped_reconstructed_visibilities_from(
    transformed_mapping_matrix: np.ndarray, reconstruction: np.ndarray
) -> np.ndarray:
    """
    Returns the reconstructed data vector from the blurrred mapping matrix `f` and solution vector *S*.

    Parameters
    -----------
    transformed_mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """
    mapped_reconstructed_visibilities = (0.0 + 0.0j) * np.zeros(
        transformed_mapping_matrix.shape[0]
    )

    transformed_mapping_matrix_real = transformed_mapping_matrix.real
    transformed_mapping_matrix_imag = transformed_mapping_matrix.imag

    for i in range(transformed_mapping_matrix.shape[0]):
        for j in range(reconstruction.shape[0]):
            mapped_reconstructed_visibilities[i] += (
                reconstruction[j] * transformed_mapping_matrix_real[i, j]
            ) + 1.0j * (reconstruction[j] * transformed_mapping_matrix_imag[i, j])

    return mapped_reconstructed_visibilities


def inversion_residual_map_from(
    *,
    pixelization_values: np.ndarray,
    data: np.ndarray,
    slim_index_for_sub_slim_index: np.ndarray,
    all_sub_slim_indexes_for_pixelization_index: [list],
):
    """
    Returns the residual-map of the `reconstruction` of an `Inversion` on its pixel-grid.

    For this residual-map, each pixel on the `reconstruction`'s pixel-grid corresponds to the sum of absolute residual
    values in the `residual_map` of the reconstructed `data` divided by the number of data-points that it maps too,
    (to normalize its value).

    This provides information on where in the `Inversion`'s `reconstruction` it is least able to accurately fit the
    `data`.

    Parameters
    ----------
    pixelization_values
        The values computed by the `Inversion` for the `reconstruction`, which are used in this function to compute
        the `residual_map` values.
    data
        The array of `data` that the `Inversion` fits.
    slim_index_for_sub_slim_index
        The mappings between the observed grid's sub-pixels and observed grid's pixels.
    all_sub_slim_indexes_for_pixelization_index
        The mapping of every pixel on the `Inversion`'s `reconstruction`'s pixel-grid to the `data` pixels.

    Returns
    -------
    np.ndarray
        The residuals of the `Inversion`'s `reconstruction` on its pixel-grid, computed by mapping the `residual_map`
        from the fit to the data.
    """
    residual_map = np.zeros(shape=len(all_sub_slim_indexes_for_pixelization_index))

    for pix_index, sub_slim_indexes in enumerate(
        all_sub_slim_indexes_for_pixelization_index
    ):

        sub_mask_total = 0
        for sub_mask_1d_index in sub_slim_indexes:
            sub_mask_total += 1
            mask_1d_index = slim_index_for_sub_slim_index[sub_mask_1d_index]
            residual = data[mask_1d_index] - pixelization_values[pix_index]
            residual_map[pix_index] += np.abs(residual)

        if sub_mask_total > 0:
            residual_map[pix_index] /= sub_mask_total

    return residual_map.copy()


def inversion_normalized_residual_map_from(
    *,
    pixelization_values,
    data,
    noise_map_1d,
    slim_index_for_sub_slim_index,
    all_sub_slim_indexes_for_pixelization_index,
):
    """
    Returns the normalized residual-map of the `reconstruction` of an `Inversion` on its pixel-grid.

    For this normalized residual-map, each pixel on the `reconstruction`'s pixel-grid corresponds to the sum of
    absolute normalized residual values in the `normalized residual_map` of the reconstructed `data` divided by the
    number of data-points that it maps too (to normalize its value).

    This provides information on where in the `Inversion`'s `reconstruction` it is least able to accurately fit the
    `data`.

    Parameters
    ----------
    pixelization_values
        The values computed by the `Inversion` for the `reconstruction`, which are used in this function to compute
        the `normalized residual_map` values.
    data
        The array of `data` that the `Inversion` fits.
    slim_index_for_sub_slim_index
        The mappings between the observed grid's sub-pixels and observed grid's pixels.
    all_sub_slim_indexes_for_pixelization_index
        The mapping of every pixel on the `Inversion`'s `reconstruction`'s pixel-grid to the `data` pixels.

    Returns
    -------
    np.ndarray
        The normalized residuals of the `Inversion`'s `reconstruction` on its pixel-grid, computed by mapping the
        `normalized_residual_map` from the fit to the data.
    """
    normalized_residual_map = np.zeros(
        shape=len(all_sub_slim_indexes_for_pixelization_index)
    )

    for pix_index, sub_slim_indexes in enumerate(
        all_sub_slim_indexes_for_pixelization_index
    ):
        sub_mask_total = 0
        for sub_mask_1d_index in sub_slim_indexes:
            sub_mask_total += 1
            mask_1d_index = slim_index_for_sub_slim_index[sub_mask_1d_index]
            residual = data[mask_1d_index] - pixelization_values[pix_index]
            normalized_residual_map[pix_index] += np.abs(
                (residual / noise_map_1d[mask_1d_index])
            )

        if sub_mask_total > 0:
            normalized_residual_map[pix_index] /= sub_mask_total

    return normalized_residual_map.copy()


def inversion_chi_squared_map_from(
    *,
    pixelization_values,
    data,
    noise_map_1d,
    slim_index_for_sub_slim_index,
    all_sub_slim_indexes_for_pixelization_index,
):
    """
    Returns the chi-squared-map of the `reconstruction` of an `Inversion` on its pixel-grid.

    For this chi-squared-map, each pixel on the `reconstruction`'s pixel-grid corresponds to the sum of chi-squared
    values in the `chi_squared_map` of the reconstructed `data` divided by the number of data-points that it maps too,
    (to normalize its value).

    This provides information on where in the `Inversion`'s `reconstruction` it is least able to accurately fit the
    `data`.

    Parameters
    ----------
    pixelization_values
        The values computed by the `Inversion` for the `reconstruction`, which are used in this function to compute
        the `chi_squared_map` values.
    data
        The array of `data` that the `Inversion` fits.
    slim_index_for_sub_slim_index
        The mappings between the observed grid's sub-pixels and observed grid's pixels.
    all_sub_slim_indexes_for_pixelization_index
        The mapping of every pixel on the `Inversion`'s `reconstruction`'s pixel-grid to the `data` pixels.

    Returns
    -------
    np.ndarray
        The chi-squareds of the `Inversion`'s `reconstruction` on its pixel-grid, computed by mapping the `chi-squared_map`
        from the fit to the data.
    """
    chi_squared_map = np.zeros(shape=len(all_sub_slim_indexes_for_pixelization_index))

    for pix_index, sub_slim_indexes in enumerate(
        all_sub_slim_indexes_for_pixelization_index
    ):
        sub_mask_total = 0
        for sub_mask_1d_index in sub_slim_indexes:
            sub_mask_total += 1
            mask_1d_index = slim_index_for_sub_slim_index[sub_mask_1d_index]
            residual = data[mask_1d_index] - pixelization_values[pix_index]
            chi_squared_map[pix_index] += (
                residual / noise_map_1d[mask_1d_index]
            ) ** 2.0

        if sub_mask_total > 0:
            chi_squared_map[pix_index] /= sub_mask_total

    return chi_squared_map.copy()


def preconditioner_matrix_via_mapping_matrix_from(
    mapping_matrix: np.ndarray,
    regularization_matrix: np.ndarray,
    preconditioner_noise_normalization: float,
) -> np.ndarray:
    """
    Returns the preconditioner matrix `{` from a mapping matrix `f` and the sum of the inverse of the 1D noise-map
    values squared (see Powell et al. 2020).

    Parameters
    -----------
    mapping_matrix
        The matrix representing the mappings between sub-grid pixels and pixelization pixels.
    regularization_matrix
        The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
    preconditioner_noise_normalization
        The sum of (1.0 / noise-map**2.0) every value in the noise-map.
    """

    curvature_matrix = curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=mapping_matrix,
        noise_map=np.ones(shape=(mapping_matrix.shape[0])),
    )

    return (
        preconditioner_noise_normalization * curvature_matrix
    ) + regularization_matrix
