from autoarray import decorator_util
import numpy as np

from typing import Tuple


@decorator_util.jit()
def curvature_matrix_via_w_tilde_from(
    w_tilde: np.ndarray, mapping_matrix: np.ndarray
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) from `w_tilde`.

    The dimensions of `w_tilde` are [image_pixels, image_pixels], meaning that for datasets with many image pixels
    this matrix can take up 10's of GB of memory. The calculation of the `curvature_matrix` via this function will
    therefore be very slow, and the method `curvature_matrix_via_w_tilde_imaging_sparse_from` should be used
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


@decorator_util.jit()
def w_tilde_imaging_from(
    noise_map_native: np.ndarray, kernel_native: np.ndarray, native_index_for_slim_index
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_imaging_sparse_from` describes a compressed representation that overcomes this hurdles. It is
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

    w_tilde = np.zeros((image_pixels, image_pixels))

    for ip0 in range(w_tilde.shape[0]):

        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        for ip1 in range(ip0, w_tilde.shape[1]):

            ip1_y, ip1_x = native_index_for_slim_index[ip1]

            w_tilde[ip0, ip1] += w_tilde_value_from(
                noise_map_native=noise_map_native,
                kernel_2d=kernel_native,
                ip0_y=ip0_y,
                ip0_x=ip0_x,
                ip1_y=ip1_y,
                ip1_x=ip1_x,
            )

    for ip0 in range(w_tilde.shape[0]):
        for ip1 in range(ip0, w_tilde.shape[1]):
            w_tilde[ip1, ip0] = w_tilde[ip0, ip1]

    return w_tilde


@decorator_util.jit()
def w_tilde_imaging_sparse_from(
    noise_map_native: np.ndarray, kernel_native: np.ndarray, native_index_for_slim_index
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. This methods creates
    a sparse matrix that can compute the matrix w_tilde.

    For imaging data, w_tilde is a sparse meatrix, whereby non-zero entries are only contained for pairs of image pixels
    where the two pixels overlap due to the kernel size. For example, if the kernel size is (11, 11) and two image
    pixels are separated by more than 20 pixels, the kernel will never convolve flux between the two pixels. Two image
    pixels will only share a convolution if they are within `kernel_overlap_size = 2 * kernel_shape - 1` pixels within
    one another.

    Thus, a `w_tilde_preload` matrix of dimensions [image_pixels, kernel_overlap_size ** 2] can be computed which
    significantly reduces the memory consumption by removing the sparsity. Because the dimensions of the the second
    axes is no longer image_pixels, a second matrix `w_tilde_indexes` must also be computed containing the slim image
    pixel indexes of every entry of `w_tilde_preload`.

    In order for the preload to store half the number of values, owing to the symmetry of the w_tilde matrix, the
    image pixel pairs corresponding to the same image pixel are divided by two. This ensures that when the curvature
    matrix is computed these pixels are not double-counted.

    This matrix can then be used to compute the curvature_matrix in a memory efficient way that exploits the sparsity
    of the linear algebra.

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
        the curvature matrix, where the dimensions are reduced to save memory..
    """

    image_pixels = len(native_index_for_slim_index)

    kernel_overlap_size = (2 * kernel_native.shape[0] - 1) * (
        2 * kernel_native.shape[1] - 1
    )

    w_tilde_preload = np.zeros((image_pixels, kernel_overlap_size))
    w_tilde_indexes = np.zeros((image_pixels, kernel_overlap_size))
    w_tilde_lengths = np.zeros(image_pixels)

    for ip0 in range(w_tilde_preload.shape[0]):

        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        kernel_index = 0

        for ip1 in range(ip0, w_tilde_preload.shape[0]):

            ip1_y, ip1_x = native_index_for_slim_index[ip1]

            value = w_tilde_value_from(
                noise_map_native=noise_map_native,
                kernel_2d=kernel_native,
                ip0_y=ip0_y,
                ip0_x=ip0_x,
                ip1_y=ip1_y,
                ip1_x=ip1_x,
            )

            if ip0 == ip1:
                value /= 2.0

            w_tilde_preload[ip0, kernel_index] = value

            if value > 0.0:

                w_tilde_indexes[ip0, kernel_index] = ip1
                kernel_index += 1

        w_tilde_lengths[ip0] = kernel_index

    return w_tilde_preload, w_tilde_indexes, w_tilde_lengths


@decorator_util.jit()
def w_tilde_value_from(
    noise_map_native: np.ndarray, kernel_2d: np.ndarray, ip0_y, ip0_x, ip1_y, ip1_x
) -> float:
    """
    Compute the value of an entry of the `w_tilde` matrix, where this entry encodes the PSF convolution of the
    noise-map between two image pixels.

    The calculation is performed by over-laying the PSF kernel over two noise-map image pixels in 2D. For all image
    pixels where the two overlaid PSF kernels overlap, the following calculation is performed for every noise map
    value:

    `value = kernel_value_0 * kernel_value_1 * (1.0 / noise_value) ** 2.0`

    This calculation infers the fraction of flux that every PSF convolution will move between each pair of image pixels
    and can therefore be used to efficiently calculate the curvature_matrix that is used in the linear algebra
    calculation of an inversion.

    The sum of all values where kernel pixels overlap is returned to give the `w_tilde` value.

    Parameters
    ----------
    noise_map_native
        The two dimensional masked noise-map of values which w_tilde is computed from.
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
    value = 0.0

    kernel_shift_y = -(kernel_2d.shape[1] // 2)
    kernel_shift_x = -(kernel_2d.shape[0] // 2)

    ip_y_offset = ip0_y - ip1_y
    ip_x_offset = ip0_x - ip1_x

    if (
        ip_y_offset < 2 * kernel_shift_y
        or ip_y_offset > -2 * kernel_shift_y
        or ip_x_offset < 2 * kernel_shift_x
        or ip_x_offset > -2 * kernel_shift_x
    ):
        return value

    for k0_y in range(kernel_2d.shape[0]):
        for k0_x in range(kernel_2d.shape[1]):

            noise_value = noise_map_native[
                ip0_y + k0_y + kernel_shift_y, ip0_x + k0_x + kernel_shift_x
            ]

            if noise_value > 0.0:

                k1_y = k0_y + ip_y_offset
                k1_x = k0_x + ip_x_offset

                if (
                    k1_y >= 0
                    and k1_x >= 0
                    and k1_y < kernel_2d.shape[0]
                    and k1_x < kernel_2d.shape[1]
                ):

                    kernel_value_0 = kernel_2d[k0_y, k0_x]
                    kernel_value_1 = kernel_2d[k1_y, k1_x]

                    value += (
                        kernel_value_0 * kernel_value_1 * (1.0 / noise_value) ** 2.0
                    )

    return value


@decorator_util.jit()
def curvature_matrix_via_w_tilde_imaging_sparse_from(
    w_tilde_preload: np.ndarray,
    w_tilde_indexes: np.ndarray,
    w_tilde_lengths: np.ndarray,
    pixelization_index_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    pixelization_pixels: int,
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
    w_tilde_preload
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    pixelization_index_for_sub_slim_index
        The mappings between the pixelization grid's pixels and the data's slimmed pixels.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pixelization_pixels
        The total number of pixels in the pixelization that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    image_pixels = len(native_index_for_slim_index)

    curvature_matrix = np.zeros((pixelization_pixels, pixelization_pixels))

    for ip0 in range(image_pixels):

        sp0 = pixelization_index_for_sub_slim_index[ip0]

        for ip1_index in range(w_tilde_lengths[ip0]):

            ip1 = w_tilde_indexes[ip0, ip1_index]

            sp1 = pixelization_index_for_sub_slim_index[ip1]

            curvature_matrix[sp0, sp1] += w_tilde_preload[ip0, ip1_index]

    for i in range(pixelization_pixels):
        for j in range(i, pixelization_pixels):
            curvature_matrix[i, j] += curvature_matrix[j, i]

    for i in range(pixelization_pixels):
        for j in range(i, pixelization_pixels):
            curvature_matrix[j, i] = curvature_matrix[i, j]

    return curvature_matrix


@decorator_util.jit()
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

    mapping_shape = blurred_mapping_matrix.shape

    data_vector = np.zeros(mapping_shape[1])

    for mask_1d_index in range(mapping_shape[0]):
        for pix_index in range(mapping_shape[1]):
            data_vector[pix_index] += (
                image[mask_1d_index]
                * blurred_mapping_matrix[mask_1d_index, pix_index]
                / (noise_map[mask_1d_index] ** 2.0)
            )

    return data_vector


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


@decorator_util.jit()
def curvature_matrix_sparse_preload_via_mapping_matrix_from(
    mapping_matrix: np.ndarray,
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
            if mapping_matrix[mask_1d_index, pix_index] > 0.0:
                curvature_matrix_preload_counts[mask_1d_index] += 1

    preload_max = np.max(curvature_matrix_preload_counts)

    curvature_matrix_sparse_preload = np.zeros(
        (mapping_matrix.shape[0], int(preload_max))
    )

    for mask_1d_index in range(mapping_matrix.shape[0]):
        index = 0
        for pix_index in range(mapping_matrix.shape[1]):
            if mapping_matrix[mask_1d_index, pix_index] > 0.0:
                curvature_matrix_sparse_preload[mask_1d_index, index] = pix_index
                index += 1

    return curvature_matrix_sparse_preload, curvature_matrix_preload_counts


@decorator_util.jit()
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


@decorator_util.jit()
def mapped_reconstructed_data_from(
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


@decorator_util.jit()
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


@decorator_util.jit()
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
