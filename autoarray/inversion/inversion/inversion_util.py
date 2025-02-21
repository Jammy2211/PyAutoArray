import numpy as np

from typing import List, Optional, Tuple

from autoconf import conf

from autoarray.inversion.inversion.settings import SettingsInversion

from autoarray import numba_util
from autoarray import exc
from autoarray.util.fnnls import fnnls_cholesky


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
    Each entry in the matrix `w_tilde_preload[:,:,0]` provides the precomputed NUFFT value mapping an image pixel
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

    Each entry in the matrix `w_tilde_preload[:,:,0]` provides the precomputed NUFFT value mapping an image pixel
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
def curvature_matrix_with_added_to_diag_from(
    curvature_matrix: np.ndarray,
    value: float,
    no_regularization_index_list: Optional[List] = None,
) -> np.ndarray:
    """
    It is common for the `curvature_matrix` computed to not be positive-definite, leading for the inversion
    via `np.linalg.solve` to fail and raise a `LinAlgError`.

    In many circumstances, adding a small numerical value of `1.0e-8` to the diagonal of the `curvature_matrix`
    makes it positive definite, such that the inversion is performed without raising an error.

    This function adds this numerical value to the diagonal of the curvature matrix.

    Parameters
    ----------
    curvature_matrix
        The curvature matrix which is being constructed in order to solve a linear system of equations.
    """

    for i in no_regularization_index_list:
        curvature_matrix[i, i] += value

    return curvature_matrix


@numba_util.jit()
def curvature_matrix_mirrored_from(
    curvature_matrix: np.ndarray,
) -> np.ndarray:
    curvature_matrix_mirrored = np.zeros(
        (curvature_matrix.shape[0], curvature_matrix.shape[1])
    )

    for i in range(curvature_matrix.shape[0]):
        for j in range(curvature_matrix.shape[1]):
            if curvature_matrix[i, j] != 0:
                curvature_matrix_mirrored[i, j] = curvature_matrix[i, j]
                curvature_matrix_mirrored[j, i] = curvature_matrix[i, j]
            if curvature_matrix[j, i] != 0:
                curvature_matrix_mirrored[i, j] = curvature_matrix[j, i]
                curvature_matrix_mirrored[j, i] = curvature_matrix[j, i]

    return curvature_matrix_mirrored


def curvature_matrix_via_mapping_matrix_from(
    mapping_matrix: np.ndarray,
    noise_map: np.ndarray,
    add_to_curvature_diag: bool = False,
    no_regularization_index_list: Optional[List] = None,
    settings: SettingsInversion = SettingsInversion(),
) -> np.ndarray:
    """
    Returns the curvature matrix `F` from a blurred mapping matrix `f` and the 1D noise-map $\sigma$
     (see Warren & Dye 2003).

    Parameters
    ----------
    mapping_matrix
        The matrix representing the mappings (these could be blurred or transfomed) between sub-grid pixels and
        pixelization pixels.
    noise_map
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """
    array = mapping_matrix / noise_map[:, None]
    curvature_matrix = np.dot(array.T, array)

    if add_to_curvature_diag and len(no_regularization_index_list) > 0:
        curvature_matrix = curvature_matrix_with_added_to_diag_from(
            curvature_matrix=curvature_matrix,
            value=settings.no_regularization_add_to_curvature_diag_value,
            no_regularization_index_list=no_regularization_index_list,
        )

    return curvature_matrix


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

    preload = curvature_preload[0, 0]

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    image_pixels = len(native_index_for_slim_index)

    for ip0 in range(image_pixels):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        for ip0_pix in range(pix_size_for_sub_slim_index[ip0]):
            sp0 = pix_indexes_for_sub_slim_index[ip0, ip0_pix]

            ip0_weight = pix_weights_for_sub_slim_index[ip0, ip0_pix]

            for ip1 in range(ip0 + 1, image_pixels):
                ip1_y, ip1_x = native_index_for_slim_index[ip1]

                for ip1_pix in range(pix_size_for_sub_slim_index[ip1]):
                    sp1 = pix_indexes_for_sub_slim_index[ip1, ip1_pix]

                    ip1_weight = pix_weights_for_sub_slim_index[ip1, ip1_pix]

                    y_diff = ip1_y - ip0_y
                    x_diff = ip1_x - ip0_x

                    curvature_matrix[sp0, sp1] += (
                        curvature_preload[y_diff, x_diff] * ip0_weight * ip1_weight
                    )

    curvature_matrix_new = np.zeros((pix_pixels, pix_pixels))

    for i in range(pix_pixels):
        for j in range(pix_pixels):
            curvature_matrix_new[i, j] = curvature_matrix[i, j] + curvature_matrix[j, i]

    curvature_matrix = curvature_matrix_new

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
def mapped_reconstructed_data_via_image_to_pix_unique_from(
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    reconstruction: np.ndarray,
) -> np.ndarray:
    """
    Returns the reconstructed data vector from the blurred mapping matrix `f` and solution vector *S*.

    Parameters
    ----------
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
    ----------
    mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """
    mapped_reconstructed_data = np.zeros(mapping_matrix.shape[0])
    for i in range(mapping_matrix.shape[0]):
        for j in range(reconstruction.shape[0]):
            mapped_reconstructed_data[i] += reconstruction[j] * mapping_matrix[i, j]

    return mapped_reconstructed_data


def reconstruction_positive_negative_from(
    data_vector: np.ndarray,
    curvature_reg_matrix: np.ndarray,
    mapper_param_range_list,
    force_check_reconstruction: bool = False,
):
    """
    Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
    of https://arxiv.org/pdf/astro-ph/0302587.pdf

    S is the vector of reconstructed inversion values.

    This reconstruction uses a linear algebra solver that allows for negative and positives values in the solution.
    By allowing negative values, the solver is efficient, but there are many inference problems where negative values
    are nonphysical or undesirable.

    This function checks that the solution does not give a linear algebra error (e.g. because the input matrix is
    not positive-definitive).

    It also explicitly checks solutions where all reconstructed values go to the same value, and raises an exception if
    this occurs. This solution occurs in many scenarios when it is clear not a valid solution, and therefore is checked
    for and removed.

    Parameters
    ----------
    data_vector
        The `data_vector` D which is solved for.
    curvature_reg_matrix
        The sum of the curvature and regularization matrices.
    mapper_param_range_list
        A list of lists, where each list contains the range of values in the solution vector (reconstruction) that
        correspond to values that are part of a mapper's mesh.
    force_check_reconstruction
        If `True`, the reconstruction is forced to check for solutions where all reconstructed values go to the same
        value irrespective of the configuration file value.

    Returns
    -------
    curvature_reg_matrix
        The curvature_matrix plus regularization matrix, overwriting the curvature_matrix in memory.
    """
    try:
        reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)
    except np.linalg.LinAlgError as e:
        raise exc.InversionException() from e

    if (
        conf.instance["general"]["inversion"]["check_reconstruction"]
        or force_check_reconstruction
    ):
        for mapper_param_range in mapper_param_range_list:
            if np.allclose(
                a=reconstruction[mapper_param_range[0] : mapper_param_range[1]],
                b=reconstruction[mapper_param_range[0]],
            ):
                raise exc.InversionException()

    return reconstruction


def reconstruction_positive_only_from(
    data_vector: np.ndarray,
    curvature_reg_matrix: np.ndarray,
    settings: SettingsInversion = SettingsInversion(),
):
    """
    Solve the linear system Eq.(2) (in terms of minimizing the quadratic value) of
    https://arxiv.org/pdf/astro-ph/0302587.pdf. Not finding the exact solution of Eq.(3) or Eq.(4).

    This reconstruction uses a linear algebra optimizer that allows only positives values in the solution.
    By not allowing negative values, the solver is slower than methods which allow negative values, but there are
    many inference problems where negative values are nonphysical or undesirable and removing them improves the solution.

    The non-negative optimizer we use is a modified version of fnnls (https://github.com/jvendrow/fnnls). The algorithm
    is published by:

    Bro & Jong (1997) ("A fast non‐negativity‐constrained least squares algorithm."
                Journal of Chemometrics: A Journal of the Chemometrics Society 11, no. 5 (1997): 393-401.)

    The modification we made here is that we create a function called fnnls_Cholesky which directly takes ZTZ and ZTx
    as inputs. The reason is that we realize for this specific algorithm (Bro & Jong (1997)), ZTZ and ZTx happen to
    be the curvature_reg_matrix and data_vector, respectively, already defined in PyAutoArray (verified). Besides,
    we build a Cholesky scheme that solves the lstsq problem in each iteration within the fnnls algorithm by updating
    the Cholesky factorisation.

    Please note that we are trying to find non-negative solution S that minimizes |Z * S - x|^2. We are not trying to
    find a solution that minimizes |ZTZ * S - ZTx|^2! ZTZ and ZTx are just some variables help to
    minimize |Z * S - x|^2. It is just a coincidence (or fundamentally not) that ZTZ and ZTx are the
    curvature_reg_matrix and data_vector, respectively.

    If we no longer uses fnnls (the algorithm of Bro & Jong (1997)), we need to check if the algorithm takes Z or
    ZTZ (x or ZTx) as an input. If not, we need to build Z and x in PyAutoArray.

    Parameters
    ----------
    data_vector
        The `data_vector` D happens to be the ZTx.
    curvature_reg_matrix
        The sum of the curvature and regularization matrices. Taken as ZTZ in our problem.
    settings
        Controls the settings of the inversion, for this function where the solution is checked to not be all
        the same values.\

    Returns
    -------
    Non-negative S that minimizes the Eq.(2) of https://arxiv.org/pdf/astro-ph/0302587.pdf.
    """

    if len(data_vector):
        try:
            if settings.positive_only_uses_p_initial:
                P_initial = np.linalg.solve(curvature_reg_matrix, data_vector) > 0
            else:
                P_initial = np.zeros(0, dtype=int)

            reconstruction = fnnls_cholesky(
                curvature_reg_matrix,
                (data_vector).T,
                P_initial=P_initial,
            )

        except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
            raise exc.InversionException() from e

    else:
        raise exc.InversionException()

    return reconstruction


def preconditioner_matrix_via_mapping_matrix_from(
    mapping_matrix: np.ndarray,
    regularization_matrix: np.ndarray,
    preconditioner_noise_normalization: float,
) -> np.ndarray:
    """
    Returns the preconditioner matrix `{` from a mapping matrix `f` and the sum of the inverse of the 1D noise-map
    values squared (see Powell et al. 2020).

    Parameters
    ----------
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
