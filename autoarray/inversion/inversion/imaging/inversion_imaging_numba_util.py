import numpy as np
from typing import Tuple

from autoarray import numba_util

import numpy as np

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
