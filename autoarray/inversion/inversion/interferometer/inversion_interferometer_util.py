import logging
import numpy as np
import time
import multiprocessing as mp
import os
from typing import Tuple

from autoarray import numba_util

logger = logging.getLogger(__name__)


@numba_util.jit()
def w_tilde_data_interferometer_from(
    visibilities_real: np.ndarray,
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
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

    image_pixels = len(native_index_for_slim_index)

    w_tilde_data = np.zeros(image_pixels)

    weight_map_real = visibilities_real / noise_map_real**2.0

    for ip0 in range(image_pixels):
        value = 0.0

        y = grid_radians_slim[ip0, 1]
        x = grid_radians_slim[ip0, 0]

        for vis_1d_index in range(uv_wavelengths.shape[0]):
            value += weight_map_real[vis_1d_index] ** -2.0 * np.cos(
                2.0
                * np.pi
                * (
                    y * uv_wavelengths[vis_1d_index, 0]
                    + x * uv_wavelengths[vis_1d_index, 1]
                )
            )

        w_tilde_data[ip0] = value

    return w_tilde_data


@numba_util.jit()
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data.
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            y_offset = grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            x_offset = grid_radians_slim[i, 0] - grid_radians_slim[j, 0]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        y_offset * uv_wavelengths[vis_1d_index, 0]
                        + x_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
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

    The `w_tilde_preload` method instead only computes each value once. To do this, it stores the preload values in a
    matrix of dimensions [shape_masked_pixels_y, shape_masked_pixels_x, 2], where `shape_masked_pixels` is the (y,x)
    size of the vertical and horizontal extent of unmasked pixels, e.g. the spatial extent over which the real space
    grid extends.

    Each entry in the matrix `w_tilde_preload[:,:,0]` provides the the precomputed NUFFT value mapping an image pixel
    to a pixel offset by that much in the y and x directions, for example:

    - w_tilde_preload[0,0,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
    in the x direction by 0 - the values of pixels paired with themselves.
    - w_tilde_preload[1,0,0] gives the precomputed values of image pixels that are offset in the y direction by 1 and
    in the x direction by 0 - the values of pixel pairs [0,3], [1,4], [2,5], [3,6], [4,7] and [5,8]
    - w_tilde_preload[0,1,0] gives the precomputed values of image pixels that are offset in the y direction by 0 and
    in the x direction by 1 - the values of pixel pairs [0,1], [1,2], [3,4], [4,5], [6,7] and [7,9].

    Flipped pairs:

    The above preloaded values pair all image pixel NUFFT values when a pixel is to the right and / or down of the
    first image pixel. However, one must also precompute pairs where the paired pixel is to the left of the host
    pixels. These pairings are stored in `w_tilde_preload[:,:,1]`, and the ordering of these pairings is flipped in the
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

    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    for i in range(y_shape):
        for j in range(x_shape):
            y_offset = grid_radians_2d[0, 0, 0] - grid_radians_2d[i, j, 0]
            x_offset = grid_radians_2d[0, 0, 1] - grid_radians_2d[i, j, 1]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                curvature_preload[i, j] += noise_map_real[
                    vis_1d_index
                ] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        x_offset * uv_wavelengths[vis_1d_index, 0]
                        + y_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(y_shape):
        for j in range(x_shape):
            if j > 0:
                y_offset = (
                    grid_radians_2d[0, -1, 0]
                    - grid_radians_2d[i, grid_x_shape - j - 1, 0]
                )
                x_offset = (
                    grid_radians_2d[0, -1, 1]
                    - grid_radians_2d[i, grid_x_shape - j - 1, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[i, -j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0:
                y_offset = (
                    grid_radians_2d[-1, 0, 0]
                    - grid_radians_2d[grid_y_shape - i - 1, j, 0]
                )
                x_offset = (
                    grid_radians_2d[-1, 0, 1]
                    - grid_radians_2d[grid_y_shape - i - 1, j, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[-i, j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0 and j > 0:
                y_offset = (
                    grid_radians_2d[-1, -1, 0]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 0]
                )
                x_offset = (
                    grid_radians_2d[-1, -1, 1]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload[-i, -j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    return curvature_preload


@numba_util.jit()
def w_tilde_via_preload_from(w_tilde_preload, native_index_for_slim_index):
    """
    Use the preloaded w_tilde matrix (see `w_tilde_preload_interferometer_from`) to compute
    w_tilde (see `w_tilde_interferometer_from`) efficiently.

    Parameters
    ----------
    w_tilde_preload
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

            w_tilde_via_preload[i, j] = w_tilde_preload[y_diff, x_diff]

    for i in range(slim_size):
        for j in range(i, slim_size):
            w_tilde_via_preload[j, i] = w_tilde_via_preload[i, j]

    return w_tilde_via_preload


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


@numba_util.jit()
def curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
    curvature_preload: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    pix_pixels: int,
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) by computing it using `w_tilde_preload`
    (see `w_tilde_preload_interferometer_from`) for an interferometer inversion.

    To compute the curvature matrix via w_tilde the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * w_tilde * mapping matrix

    This function speeds this calculation up in two ways:

    1) Instead of using `w_tilde` (dimensions [image_pixels, image_pixels] it uses `w_tilde_preload` (dimensions
    [image_pixels, 2]). The massive reduction in the size of this matrix in memory allows for much fast computation.

    2) It omits the `mapping_matrix` and instead uses directly the 1D vector that maps every image pixel to a source
    pixel `native_index_for_slim_index`. This exploits the sparsity in the `mapping_matrix` to directly
    compute the `curvature_matrix` (e.g. it condenses the triple matrix multiplication into a double for loop!).

    Parameters
    ----------
    curvature_preload
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelizaiton pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelizaiton pixel.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    image_pixels = len(native_index_for_slim_index)

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    for ip0 in range(image_pixels):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            ip0_weight = pix_weights_for_sub_slim_index[ip0, ip0_pix]

            sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]

            for ip1 in range(image_pixels):
                ip1_y, ip1_x = native_index_for_slim_index[ip1]

                for ip1_pix in range(pix_size_for_sub_slim_index[ip1]):
                    ip1_weight = pix_weights_for_sub_slim_index[ip1, ip1_pix]

                    sp1 = pix_indexes_for_sub_slim_index[ip1, ip1_pix]

                    y_diff = ip1_y - ip0_y
                    x_diff = ip1_x - ip0_x

                    curvature_matrix[sp0, sp1] += (
                        curvature_preload[y_diff, x_diff] * ip0_weight * ip1_weight
                    )

    return curvature_matrix


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


"""
Welcome to the quagmire!
"""


@numba_util.jit()
def curvature_matrix_via_w_tilde_curvature_preload_interferometer_from_2(
    curvature_preload: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    pix_pixels: int,
    sub_slim_indexes_for_pix_index,
    sub_slim_sizes_for_pix_index,
    sub_slim_weights_for_pix_index,
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) by computing it using `w_tilde_preload`
    (see `w_tilde_preload_interferometer_from`) for an interferometer inversion.

    To compute the curvature matrix via w_tilde the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * w_tilde * mapping matrix

    This function speeds this calculation up in two ways:

    1) Instead of using `w_tilde` (dimensions [image_pixels, image_pixels] it uses `w_tilde_preload` (dimensions
       [image_pixels, 2]). The massive reduction in the size of this matrix in memory allows for much fast computation.

    2) It omits the `mapping_matrix` and instead uses directly the 1D vector that maps every image pixel to a source
       pixel `native_index_for_slim_index`. This exploits the sparsity in the `mapping_matrix` to directly
       compute the `curvature_matrix` (e.g. it condenses the triple matrix multiplication into a double for loop!).

    Parameters
    ----------
    curvature_preload
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization's mesh pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelization pixel.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pix_pixels
        The total number of pixels in the pixelization's mesh that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    for sp0 in range(pix_pixels):
        ip_size_0 = sub_slim_sizes_for_pix_index[sp0]

        for sp1 in range(sp0, pix_pixels):
            val = 0.0
            ip_size_1 = sub_slim_sizes_for_pix_index[sp1]

            for ip0_tmp in range(ip_size_0):
                ip0 = sub_slim_indexes_for_pix_index[sp0, ip0_tmp]
                ip0_weight = sub_slim_weights_for_pix_index[sp0, ip0_tmp]

                ip0_y, ip0_x = native_index_for_slim_index[ip0]

                for ip1_tmp in range(ip_size_1):
                    ip1 = sub_slim_indexes_for_pix_index[sp1, ip1_tmp]
                    ip1_weight = sub_slim_weights_for_pix_index[sp1, ip1_tmp]

                    ip1_y, ip1_x = native_index_for_slim_index[ip1]

                    y_diff = ip1_y - ip0_y
                    x_diff = ip1_x - ip0_x

                    val += curvature_preload[y_diff, x_diff] * ip0_weight * ip1_weight

            curvature_matrix[sp0, sp1] += val

    for i in range(pix_pixels):
        for j in range(i, pix_pixels):
            curvature_matrix[j, i] = curvature_matrix[i, j]

    return curvature_matrix


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_stage_1_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
) -> np.ndarray:
    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload_stage_1 = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    for i in range(y_shape):
        for j in range(x_shape):
            y_offset = grid_radians_2d[0, 0, 0] - grid_radians_2d[i, j, 0]
            x_offset = grid_radians_2d[0, 0, 1] - grid_radians_2d[i, j, 1]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                curvature_preload_stage_1[i, j] += noise_map_real[
                    vis_1d_index
                ] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        x_offset * uv_wavelengths[vis_1d_index, 0]
                        + y_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    return curvature_preload_stage_1


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_stage_2_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
) -> np.ndarray:
    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload_stage_2 = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    for i in range(y_shape):
        for j in range(x_shape):
            if j > 0:
                y_offset = (
                    grid_radians_2d[0, -1, 0]
                    - grid_radians_2d[i, grid_x_shape - j - 1, 0]
                )
                x_offset = (
                    grid_radians_2d[0, -1, 1]
                    - grid_radians_2d[i, grid_x_shape - j - 1, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload_stage_2[i, -j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    return curvature_preload_stage_2


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_stage_3_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
) -> np.ndarray:
    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload_stage_3 = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0:
                y_offset = (
                    grid_radians_2d[-1, 0, 0]
                    - grid_radians_2d[grid_y_shape - i - 1, j, 0]
                )
                x_offset = (
                    grid_radians_2d[-1, 0, 1]
                    - grid_radians_2d[grid_y_shape - i - 1, j, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload_stage_3[-i, j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    return curvature_preload_stage_3


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_stage_4_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
) -> np.ndarray:
    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload_stage_4 = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    for i in range(y_shape):
        for j in range(x_shape):
            if i > 0 and j > 0:
                y_offset = (
                    grid_radians_2d[-1, -1, 0]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 0]
                )
                x_offset = (
                    grid_radians_2d[-1, -1, 1]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload_stage_4[-i, -j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    return curvature_preload_stage_4


def w_tilde_curvature_preload_interferometer_in_stages_with_chunks_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
    stage="1",
    chunk: int = 100,
    check=True,
    directory=None,
) -> np.ndarray:

    from astropy.io import fits

    if directory is None:
        raise NotImplementedError()

    y_shape = shape_masked_pixels_2d[0]
    if chunk > y_shape:
        raise NotImplementedError()

    size = 0
    while size < y_shape:
        check_condition = True

        if size + chunk < y_shape:
            limits = [size, size + chunk]
        else:
            limits = [size, y_shape]
        print("limits =", limits)

        filename = "{}/curvature_preload_stage_{}_limits_{}_{}.fits".format(
            directory,
            stage,
            limits[0],
            limits[1],
        )
        print("filename =", filename)

        filename_check = "{}/stage_{}_limits_{}_{}_in_progress".format(
            directory,
            stage,
            limits[0],
            limits[1],
        )

        if check:
            if os.path.isfile(filename_check):
                check_condition = False
            else:
                os.system("touch {}".format(filename_check))

        if check_condition:
            print("computing ...")
            if stage == "1":
                data = w_tilde_curvature_preload_interferometer_stage_1_with_limits_placeholder_from(
                    noise_map_real=noise_map_real,
                    uv_wavelengths=uv_wavelengths,
                    shape_masked_pixels_2d=shape_masked_pixels_2d,
                    grid_radians_2d=grid_radians_2d,
                    limits=limits,
                )
            if stage == "2":
                data = w_tilde_curvature_preload_interferometer_stage_2_with_limits_placeholder_from(
                    noise_map_real=noise_map_real,
                    uv_wavelengths=uv_wavelengths,
                    shape_masked_pixels_2d=shape_masked_pixels_2d,
                    grid_radians_2d=grid_radians_2d,
                    limits=limits,
                )
            if stage == "3":
                data = w_tilde_curvature_preload_interferometer_stage_3_with_limits_placeholder_from(
                    noise_map_real=noise_map_real,
                    uv_wavelengths=uv_wavelengths,
                    shape_masked_pixels_2d=shape_masked_pixels_2d,
                    grid_radians_2d=grid_radians_2d,
                    limits=limits,
                )
            if stage == "4":
                data = w_tilde_curvature_preload_interferometer_stage_4_with_limits_placeholder_from(
                    noise_map_real=noise_map_real,
                    uv_wavelengths=uv_wavelengths,
                    shape_masked_pixels_2d=shape_masked_pixels_2d,
                    grid_radians_2d=grid_radians_2d,
                    limits=limits,
                )

            fits.writeto(filename, data=data)

        size = size + chunk


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_stage_1_with_limits_placeholder_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
    limits: list = [],
) -> np.ndarray:
    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload_stage_1 = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    i_lower, i_upper = limits
    for i in range(i_lower, i_upper):
        for j in range(x_shape):
            y_offset = grid_radians_2d[0, 0, 0] - grid_radians_2d[i, j, 0]
            x_offset = grid_radians_2d[0, 0, 1] - grid_radians_2d[i, j, 1]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                curvature_preload_stage_1[i, j] += noise_map_real[
                    vis_1d_index
                ] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        x_offset * uv_wavelengths[vis_1d_index, 0]
                        + y_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    return curvature_preload_stage_1


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_stage_2_with_limits_placeholder_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
    limits: list = [],
) -> np.ndarray:
    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload_stage_2 = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    i_lower, i_upper = limits
    for i in range(i_lower, i_upper):
        for j in range(x_shape):
            if j > 0:
                y_offset = (
                    grid_radians_2d[0, -1, 0]
                    - grid_radians_2d[i, grid_x_shape - j - 1, 0]
                )
                x_offset = (
                    grid_radians_2d[0, -1, 1]
                    - grid_radians_2d[i, grid_x_shape - j - 1, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload_stage_2[i, -j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    return curvature_preload_stage_2


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_stage_3_with_limits_placeholder_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
    limits: list = [],
) -> np.ndarray:
    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload_stage_3 = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    i_lower, i_upper = limits
    for i in range(i_lower, i_upper):
        for j in range(x_shape):
            if i > 0:
                y_offset = (
                    grid_radians_2d[-1, 0, 0]
                    - grid_radians_2d[grid_y_shape - i - 1, j, 0]
                )
                x_offset = (
                    grid_radians_2d[-1, 0, 1]
                    - grid_radians_2d[grid_y_shape - i - 1, j, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload_stage_3[-i, j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    return curvature_preload_stage_3


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_stage_4_with_limits_placeholder_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
    limits: list = [],
) -> np.ndarray:
    y_shape = shape_masked_pixels_2d[0]
    x_shape = shape_masked_pixels_2d[1]

    curvature_preload_stage_4 = np.zeros((y_shape * 2, x_shape * 2))

    #  For the second preload to index backwards correctly we have to extracted the 2D grid to its shape.
    grid_radians_2d = grid_radians_2d[0:y_shape, 0:x_shape]

    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]

    i_lower, i_upper = limits
    for i in range(i_lower, i_upper):
        for j in range(x_shape):
            if i > 0 and j > 0:
                y_offset = (
                    grid_radians_2d[-1, -1, 0]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 0]
                )
                x_offset = (
                    grid_radians_2d[-1, -1, 1]
                    - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 1]
                )

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    curvature_preload_stage_4[-i, -j] += noise_map_real[
                        vis_1d_index
                    ] ** -2.0 * np.cos(
                        2.0
                        * np.pi
                        * (
                            x_offset * uv_wavelengths[vis_1d_index, 0]
                            + y_offset * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

    return curvature_preload_stage_4


def make_2d(arr: mp.Array, y_shape: int, x_shape: int) -> np.ndarray:
    """
    Converts shared multiprocessing array into a non-square Numpy array of a given shape. Multiprocessing arrays must have only a single dimension.

    Parameters
    ----------
    arr
        Shared multiprocessing array to convert.
    y_shape
        Size of y-dimension of output array.
    x_shape
        Size of x-dimension of output array.

    Returns
    -------
    para_result
        Reshaped array in Numpy array format.
    """
    para_result_np = np.frombuffer(arr.get_obj(), dtype="float64")
    para_result = para_result_np.reshape((y_shape, x_shape))
    return para_result


def parallel_preload(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_2d: np.ndarray,
    curvature_preload: np.ndarray,
    x_shape: int,
    i0: int,
    i1: int,
    loop_number: int,
):
    """
    Runs the each loop in the curvature preload calculation by calling the associated JIT accelerated function.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_2d
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.
    curvature_preload
        Output array to construct, shared across half of the parallel threads.
    x_shape
        The x shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask. From shape_masked_pixels_2d.
    i0
        The lowest index of curvature_preload this particular parallel process operates over.
    i1
        The largest index of curvature_preload this particular parallel process operates over.
    loop_number
        Determines which JIT-accelerated function to run i.e. which stage of the calculation.

    Returns
    -------
    none
        Updates shared object
    """
    if loop_number == 1:
        for i in range(i0, i1):
            jit_loop_preload_1(
                noise_map_real,
                uv_wavelengths,
                grid_radians_2d,
                curvature_preload,
                x_shape,
                i,
            )
    elif loop_number == 2:
        for i in range(i0, i1):
            jit_loop_preload_2(
                noise_map_real,
                uv_wavelengths,
                grid_radians_2d,
                curvature_preload,
                x_shape,
                i,
            )
    elif loop_number == 3:
        for i in range(i0, i1):
            jit_loop_preload_3(
                noise_map_real,
                uv_wavelengths,
                grid_radians_2d,
                curvature_preload,
                x_shape,
                i,
            )
    elif loop_number == 4:
        for i in range(i0, i1):
            jit_loop_preload_4(
                noise_map_real,
                uv_wavelengths,
                grid_radians_2d,
                curvature_preload,
                x_shape,
                i,
            )


@numba_util.jit(cache=True)
def jit_loop_preload_1(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_2d: np.ndarray,
    curvature_preload: np.ndarray,
    x_shape: int,
    i: int,
):
    """
    JIT-accelerated function for the first loop of the curvature preload calculation.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_2d
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.
    curvature_preload
        Output array to construct, shared across half of the parallel threads.
    x_shape
        The x shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask. From shape_masked_pixels_2d.
    i
        the y-index of curvature preload this function operates over.

    Returns
    -------
    none
        Updates shared object
    """
    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]
    for j in range(x_shape):
        y_offset = grid_radians_2d[0, 0, 0] - grid_radians_2d[i, j, 0]
        x_offset = grid_radians_2d[0, 0, 1] - grid_radians_2d[i, j, 1]

        for vis_1d_index in range(uv_wavelengths.shape[0]):
            curvature_preload[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                2.0
                * np.pi
                * (
                    x_offset * uv_wavelengths[vis_1d_index, 0]
                    + y_offset * uv_wavelengths[vis_1d_index, 1]
                )
            )


@numba_util.jit(cache=True)
def jit_loop_preload_2(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_2d: np.ndarray,
    curvature_preload: np.ndarray,
    x_shape: int,
    i: int,
):
    """
    JIT-accelerated function for the second loop of the curvature preload calculation.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_2d
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.
    curvature_preload
        Output array to construct, shared across half of the parallel threads.
    x_shape
        The x shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask. From shape_masked_pixels_2d.
    i
        the y-index of curvature preload this function operates over.

    Returns
    -------
    none
        Updates shared object
    """
    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]
    for j in range(x_shape):
        if j > 0:
            y_offset = (
                grid_radians_2d[0, -1, 0] - grid_radians_2d[i, grid_x_shape - j - 1, 0]
            )
            x_offset = (
                grid_radians_2d[0, -1, 1] - grid_radians_2d[i, grid_x_shape - j - 1, 1]
            )

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                curvature_preload[i, -j] += noise_map_real[
                    vis_1d_index
                ] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        x_offset * uv_wavelengths[vis_1d_index, 0]
                        + y_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )


@numba_util.jit(cache=True)
def jit_loop_preload_3(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_2d: np.ndarray,
    curvature_preload: np.ndarray,
    x_shape: int,
    i: int,
):
    """
    JIT-accelerated function for the third loop of the curvature preload calculation.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_2d
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.
    curvature_preload
        Output array to construct, shared across half of the parallel threads.
    x_shape
        The x shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask. From shape_masked_pixels_2d.
    i
        the y-index of curvature preload this function operates over.

    Returns
    -------
    none
        Updates shared object
    """
    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]
    for j in range(x_shape):
        if i > 0:
            y_offset = (
                grid_radians_2d[-1, 0, 0] - grid_radians_2d[grid_y_shape - i - 1, j, 0]
            )
            x_offset = (
                grid_radians_2d[-1, 0, 1] - grid_radians_2d[grid_y_shape - i - 1, j, 1]
            )

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                curvature_preload[-i, j] += noise_map_real[
                    vis_1d_index
                ] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        x_offset * uv_wavelengths[vis_1d_index, 0]
                        + y_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )


@numba_util.jit(cache=True)
def jit_loop_preload_4(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_2d: np.ndarray,
    curvature_preload: np.ndarray,
    x_shape: int,
    i: int,
):
    """
    JIT-accelerated function for the forth loop of the curvature preload calculation.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_2d
        The 2D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.
    curvature_preload
        Output array to construct, shared across half of the parallel threads.
    x_shape
        The x shape corresponding to the extent of unmasked pixels that go vertically and horizontally across the
        mask. From shape_masked_pixels_2d.
    i
        the y-index of curvature preload this function operates over.

    Returns
    -------
    none
        Updates shared object
    """
    grid_y_shape = grid_radians_2d.shape[0]
    grid_x_shape = grid_radians_2d.shape[1]
    for j in range(x_shape):
        if i > 0 and j > 0:
            y_offset = (
                grid_radians_2d[-1, -1, 0]
                - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 0]
            )
            x_offset = (
                grid_radians_2d[-1, -1, 1]
                - grid_radians_2d[grid_y_shape - i - 1, grid_x_shape - j - 1, 1]
            )

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                curvature_preload[-i, -j] += noise_map_real[
                    vis_1d_index
                ] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        x_offset * uv_wavelengths[vis_1d_index, 0]
                        + y_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )


try:
    import numba
    from numba import prange

    @numba.jit("void(f8[:,:], i8)", nopython=True, parallel=True, cache=True)
    def jit_loop2(curvature_matrix: np.ndarray, pix_pixels: int):
        """
        Performs second stage of curvature matrix calculation using Numba parallelisation and JIT.

        Parameters
        ----------
        curvature_matrix
            Curvature matrix this function operates on. Still requires third stage of calculation.
        pix_pixels
            Size of one dimension of the curvature matrix.

        Returns
        -------
        none
            Updates shared object.
        """

        curvature_matrix_temp = curvature_matrix.copy()
        for i in prange(pix_pixels):
            for j in range(pix_pixels):
                curvature_matrix[i, j] = (
                    curvature_matrix_temp[i, j] + curvature_matrix_temp[j, i]
                )

except ModuleNotFoundError:
    pass


@numba_util.jit(cache=True)
def jit_loop3(
    curvature_matrix: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    preload: np.float64,
    image_pixels: int,
) -> np.ndarray:
    """
    Third stage of curvature matrix calculation.

    Parameters
    ----------
    curvature_matrix
        Curvature matrix this function operates on. This function completes the calculation and returns the final curvature matrix F.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelization pixel.
    preload
        Zeroth element of the curvature preload matrix.
    image_pixels
        Length of native_index_for_slim_index.

    Returns
    -------
    ndarray
        Fully computed curvature preload matrix F.
    """
    for ip0 in range(image_pixels):
        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            for ip1_pix in range(pix_size_for_sub_slim_index[ip0]):
                sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]
                sp1 = pix_indexes_for_sub_slim_index[ip0, ip1_pix]

                ip0_weight = pix_weights_for_sub_slim_index[ip0, ip0_pix]
                ip1_weight = pix_weights_for_sub_slim_index[ip0, ip1_pix]

                if sp0 > sp1:
                    curvature_matrix[sp0, sp1] += preload * ip0_weight * ip1_weight
                    curvature_matrix[sp1, sp0] += preload * ip0_weight * ip1_weight
                elif sp0 == sp1:
                    curvature_matrix[sp0, sp1] += preload * ip0_weight * ip1_weight
    return curvature_matrix


def parallel_loop1(
    curvature_preload: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    curvature_matrix: np.ndarray,
    i0: int,
    i1: int,
    lock: mp.Lock,
):
    """
    This function prepares the first part of the curvature matrix calculation and is called by a multiprocessing process.

    Parameters
    ----------
    curvature_preload
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelization pixel.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    curvature_matrix
        Output of first stage of the calculation, shared across multiple threads.
    i0
        First index of native_index_for_slim_index that a particular thread operates over.
    i1
        Last index of native_index_for_slim_index that a particular thread operates over.
    lock
        Mutex lock shared across all processes to prevent a race condition.

    Returns
    ------
    none
        Updates shared object, doesn not return anything.
    """
    print(f"calling parallel_loop1 for process {mp.current_process().pid}.")
    image_pixels = len(native_index_for_slim_index)
    for ip0 in range(i0, i1):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]
        # print(f"Processing ip0={ip0}, ip0_y={ip0_y}, ip0_x={ip0_x}")
        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]
            result_vector = jit_calc_loop1(
                image_pixels,
                native_index_for_slim_index,
                pix_indexes_for_sub_slim_index,
                pix_size_for_sub_slim_index,
                pix_weights_for_sub_slim_index,
                curvature_preload,
                curvature_matrix[sp0, :].shape,
                ip0,
                ip0_pix,
                i1,
                ip0_y,
                ip0_x,
            )
            with lock:
                curvature_matrix[sp0, :] += result_vector
    print(f"finished parallel_loop1 for process {mp.current_process().pid}.")


# ---------------------------------------------------------------------------- #
"""
def parallel_loop1_ChatGPT( # NOTE: THIS DID NOT FIX THE ISSUE ON COSMA ...
    curvature_preload: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    curvature_matrix: np.ndarray,
    i0: int,
    i1: int
):


    image_pixels = len(native_index_for_slim_index)
    local_results = np.zeros(curvature_matrix.shape)  # Local accumulation

    for ip0 in range(i0, i1):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]
        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]
            result_vector = jit_calc_loop1(image_pixels,
                                            native_index_for_slim_index,
                                            pix_indexes_for_sub_slim_index,
                                            pix_size_for_sub_slim_index,
                                            pix_weights_for_sub_slim_index,
                                            curvature_preload,
                                            curvature_matrix[sp0, :].shape,
                                            ip0, ip0_pix, i1, ip0_y, ip0_x)
            local_results[sp0, :] += result_vector  # Accumulate locally

    # Merge local results into the shared curvature_matrix
    np.add.at(curvature_matrix, np.nonzero(local_results), local_results[np.nonzero(local_results)])
"""


def parallel_loop1_ChatGPT(
    curvature_preload: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    curvature_matrix: np.ndarray,
    i0: int,
    i1: int,
    lock: mp.Lock,
):
    print(f"calling parallel_loop1 for process {mp.current_process().pid}.")

    image_pixels = len(native_index_for_slim_index)

    # Create a local copy of the result to reduce lock contention
    local_curvature_matrix = np.zeros_like(curvature_matrix)

    for ip0 in range(i0, i1):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]
        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]
            result_vector = jit_calc_loop1(
                image_pixels,
                native_index_for_slim_index,
                pix_indexes_for_sub_slim_index,
                pix_size_for_sub_slim_index,
                pix_weights_for_sub_slim_index,
                curvature_preload,
                local_curvature_matrix[sp0, :].shape,
                ip0,
                ip0_pix,
                i1,
                ip0_y,
                ip0_x,
            )
            local_curvature_matrix[sp0, :] += result_vector

    # Write the local results to the shared memory with a single lock acquisition
    with lock:
        print(f"{mp.current_process().pid} has lock.")
        curvature_matrix += local_curvature_matrix

    print(f"finished parallel_loop1 for process {mp.current_process().pid}.")


# ---------------------------------------------------------------------------- #


@numba_util.jit(cache=True)
def jit_calc_loop1(
    image_pixels: int,
    native_index_for_slim_index: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    curvature_preload: np.ndarray,
    result_vector_shape: tuple,
    ip0: int,
    ip0_pix: int,
    i1: int,
    ip0_y: int,
    ip0_x: int,
) -> np.ndarray:
    """
    Performs first stage of curvature matrix calculation in parallel using JIT. Returns a single column of the curvature matrix per function call.

    Parameters
    ----------
    image_pixels
        Length of native_index_for_slim_index, precomputed outside of the loop to reduce overhead.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelization pixel.
    curvature_preload
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    result_vector_shape
        The shape of the output of this function, a vector of one column of the curvature_matrix.
    ip0, ip0_pix
        Indices for ip0_weight for this iteration.
    i1
        Last index of native_index_for_slim_index that a particular thread operates over.
    ip0_y
        Index used to calculate y_diff values for this loop iteration.
    ip0_x
        Index used to calculate x_diff values for this loop iteration.

    Returns
    -------
    result_vector
        The column of the curvature matrix calculated in this loop iteration for this subprocess.
    """

    result_vector = np.zeros(result_vector_shape)
    ip0_weight = pix_weights_for_sub_slim_index[ip0, ip0_pix]

    for ip1 in range(ip0 + 1, image_pixels):
        ip1_y, ip1_x = native_index_for_slim_index[ip1]

        for ip1_pix in range(pix_size_for_sub_slim_index[ip1]):
            sp1 = pix_indexes_for_sub_slim_index[ip1, ip1_pix]
            ip1_weight = pix_weights_for_sub_slim_index[ip1, ip1_pix]

            y_diff = ip1_y - ip0_y
            x_diff = ip1_x - ip0_x

            result = curvature_preload[y_diff, x_diff] * ip0_weight * ip1_weight
            result_vector[sp1] += result
    return result_vector


def curvature_matrix_via_w_tilde_curvature_preload_interferometer_para_from(
    curvature_preload: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    pix_size_for_sub_slim_index: np.ndarray,
    pix_weights_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    pix_pixels: int,
    n_processes: int = mp.cpu_count(),
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) by computing it using `w_tilde_preload`
    (see `w_tilde_preload_interferometer_from`) for an interferometer inversion.

    To compute the curvature matrix via w_tilde the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * w_tilde * mapping matrix

    This function speeds this calculation up in two ways:

    1) Instead of using `w_tilde` (dimensions [image_pixels, image_pixels] it uses `w_tilde_preload` (dimensions
    [2*y_image_pixels, 2*x_image_pixels]). The massive reduction in the size of this matrix in memory allows for much
    fast computation.

    2) It omits the `mapping_matrix` and instead uses directly the 1D vector that maps every image pixel to a source
    pixel `native_index_for_slim_index`. This exploits the sparsity in the `mapping_matrix` to directly
    compute the `curvature_matrix` (e.g. it condenses the triple matrix multiplication into a double for loop!).

    This version of the function uses Python Multiprocessing to parallelise the calculation over multiple CPUs in three stages.

    Parameters
    ----------
    curvature_preload
        A matrix that precomputes the values for fast computation of w_tilde, which in this function is used to bypass
        the creation of w_tilde altogether and go directly to the `curvature_matrix`.
    pix_indexes_for_sub_slim_index
        The mappings from a data sub-pixel index to a pixelization pixel index.
    pix_size_for_sub_slim_index
        The number of mappings between each data sub pixel and pixelization pixel.
    pix_weights_for_sub_slim_index
        The weights of the mappings of every data sub pixel and pixelization pixel.
    native_index_for_slim_index
        An array of shape [total_unmasked_pixels*sub_size] that maps every unmasked sub-pixel to its corresponding
        native 2D pixel using its (y,x) pixel indexes.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.
    n_processes
        The number of cores to parallelise over, defaults to the maximum number available

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).

    """
    print(
        "calling 'curvature_matrix_via_w_tilde_curvature_preload_interferometer_para_from'."
    )
    preload = curvature_preload[0, 0]
    image_pixels = len(native_index_for_slim_index)

    # Make sure there isn't more cores assigned than there is indices to loop over
    if n_processes > image_pixels:
        n_processes = image_pixels

    # Set up parallel code
    idx_diff = int(image_pixels / n_processes)
    idxs = []
    for n in range(n_processes):
        idxs.append(idx_diff * n)
    idxs.append(len(native_index_for_slim_index))

    idx_access_list = []
    for i in range(len(idxs) - 1):
        id0 = idxs[i]
        id1 = idxs[i + 1]
        idx_access_list.append([id0, id1])

    lock = mp.Lock()
    para_result_jit_arr = mp.Array("d", pix_pixels * pix_pixels)

    # Run first loop in parallel
    print("starting 1st loop.")

    processes = [
        mp.Process(
            target=parallel_loop1,
            args=(
                curvature_preload,
                pix_indexes_for_sub_slim_index,
                pix_size_for_sub_slim_index,
                pix_weights_for_sub_slim_index,
                native_index_for_slim_index,
                make_2d(para_result_jit_arr, pix_pixels, pix_pixels),
                i0,
                i1,
                lock,
            ),
        )
        for i0, i1 in idx_access_list
    ]

    """
    processes = [
    mp.Process(target = parallel_loop1_ChatGPT,
    args = (curvature_preload,
            pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index,
            pix_weights_for_sub_slim_index,
            native_index_for_slim_index,
            make_2d(para_result_jit_arr, pix_pixels, pix_pixels),
            i0, i1)) for i0, i1 in idx_access_list]
    """
    for i, p in enumerate(processes):
        p.start()
        time.sleep(0.01)
        # logging.info(f"Started process {p.pid}.")
        print("process {} started (id = {}).".format(i, p.pid))
    for j, p in enumerate(processes):
        p.join()
        # logging.info(f"Process {p.pid} finished.")
        print("process {} finished (id = {}).".format(j, p.pid))
    print("finished 1st loop.")

    # Run second loop
    print("starting 2nd loop.")
    curvature_matrix = make_2d(para_result_jit_arr, pix_pixels, pix_pixels)
    jit_loop2(curvature_matrix, pix_pixels)
    print("finished 2nd loop.")

    # Run final loop
    print("starting 3rd loop.")
    curvature_matrix = jit_loop3(
        curvature_matrix,
        pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        preload,
        image_pixels,
    )
    print("finished 3rd loop.")
    return curvature_matrix
