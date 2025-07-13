import numpy as np
from typing import Tuple

from autoarray import exc
from autoarray import numba_util


@numba_util.jit()
def zeroth_regularization_matrix_from(coefficient: float, pixels: int) -> np.ndarray:
    """
    Apply zeroth order regularization which penalizes every pixel's deviation from zero by addiing non-zero terms
    to the regularization matrix.

    A complete description of regularization and the `regularization_matrix` can be found in the `Regularization`
    class in the module `autoarray.inversion.regularization`.

    Parameters
    ----------
    pixels
        The number of pixels in the linear object which is to be regularized, being used to in the inversion.
    coefficient
        The regularization coefficients which controls the degree of smoothing of the inversion reconstruction.

    Returns
    -------
    np.ndarray
        The regularization matrix computed using Regularization where the effective regularization
        coefficient of every source pixel is the same.
    """

    regularization_matrix = np.zeros(shape=(pixels, pixels))

    regularization_coefficient = coefficient**2.0

    for i in range(pixels):
        regularization_matrix[i, i] += regularization_coefficient

    return regularization_matrix


@numba_util.jit()
def constant_regularization_matrix_from(
    coefficient: float, neighbors: np.ndarray, neighbors_sizes: np.ndarray
) -> np.ndarray:
    """
    From the pixel-neighbors array, setup the regularization matrix using the instance regularization scheme.

    A complete description of regularizatin and the `regularization_matrix` can be found in the `Regularization`
    class in the module `autoarray.inversion.regularization`.

    Parameters
    ----------
    coefficient
        The regularization coefficients which controls the degree of smoothing of the inversion reconstruction.
    neighbors
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.

    Returns
    -------
    np.ndarray
        The regularization matrix computed using Regularization where the effective regularization
        coefficient of every source pixel is the same.
    """

    parameters = len(neighbors)

    regularization_matrix = np.zeros(shape=(parameters, parameters))

    regularization_coefficient = coefficient**2.0

    for i in range(parameters):
        regularization_matrix[i, i] += 1e-8
        for j in range(neighbors_sizes[i]):
            neighbor_index = neighbors[i, j]
            regularization_matrix[i, i] += regularization_coefficient
            regularization_matrix[i, neighbor_index] -= regularization_coefficient

    return regularization_matrix


@numba_util.jit()
def constant_zeroth_regularization_matrix_from(
    coefficient: float,
    coefficient_zeroth: float,
    neighbors: np.ndarray,
    neighbors_sizes: np.ndarray,
) -> np.ndarray:
    """
    From the pixel-neighbors array, setup the regularization matrix using the instance regularization scheme.

    A complete description of regularizatin and the ``regularization_matrix`` can be found in the ``Regularization``
    class in the module ``autoarray.inversion.regularization``.

    Parameters
    ----------
    coefficients
        The regularization coefficients which controls the degree of smoothing of the inversion reconstruction.
    neighbors
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.

    Returns
    -------
    np.ndarray
        The regularization matrix computed using Regularization where the effective regularization
        coefficient of every source pixel is the same.
    """

    pixels = len(neighbors)

    regularization_matrix = np.zeros(shape=(pixels, pixels))

    regularization_coefficient = coefficient**2.0
    regularization_coefficient_zeroth = coefficient_zeroth**2.0

    for i in range(pixels):
        regularization_matrix[i, i] += 1e-8
        regularization_matrix[i, i] += regularization_coefficient_zeroth
        for j in range(neighbors_sizes[i]):
            neighbor_index = neighbors[i, j]
            regularization_matrix[i, i] += regularization_coefficient
            regularization_matrix[i, neighbor_index] -= regularization_coefficient

    return regularization_matrix


def adaptive_regularization_weights_from(
    inner_coefficient: float, outer_coefficient: float, pixel_signals: np.ndarray
) -> np.ndarray:
    """
    Returns the regularization weights for the adaptive regularization scheme (e.g. ``AdaptiveBrightness``).

    The weights define the effective regularization coefficient of every mesh parameter (typically pixels
    of a ``Mapper``).

    They are computed using an estimate of the expected signal in each pixel.

    Two regularization coefficients are used, corresponding to the:

    1) pixel_signals: pixels with a high pixel-signal (i.e. where the signal is located in the pixelization).
    2) 1.0 - pixel_signals: pixels with a low pixel-signal (i.e. where the signal is not located in the pixelization).

    Parameters
    ----------
    inner_coefficient
        The inner regularization coefficients which controls the degree of smoothing of the inversion reconstruction
        in the inner regions of a mesh's reconstruction.
    outer_coefficient
        The outer regularization coefficients which controls the degree of smoothing of the inversion reconstruction
        in the outer regions of a mesh's reconstruction.
    pixel_signals
        The estimated signal in every pixelization pixel, used to change the regularization weighting of high signal
        and low signal pixelizations.

    Returns
    -------
    np.ndarray
        The adaptive regularization weights which act as the effective regularization coefficients of
        every source pixel.
    """
    return (
        inner_coefficient * pixel_signals + outer_coefficient * (1.0 - pixel_signals)
    ) ** 2.0


def brightness_zeroth_regularization_weights_from(
    coefficient: float, pixel_signals: np.ndarray
) -> np.ndarray:
    """
    Returns the regularization weights for the brightness zeroth regularization scheme (e.g. ``BrightnessZeroth``).

    The weights define the level of zeroth order regularization applied to every mesh parameter (typically pixels
    of a ``Mapper``).

    They are computed using an estimate of the expected signal in each pixel.

    The zeroth order regularization coefficients is applied in combination with  1.0 - pixel_signals, which are
    the pixels with a low pixel-signal (i.e. where the signal is not located near the source being reconstructed in
    the pixelization).

    Parameters
    ----------
    coefficient
        The level of zeroth order regularization applied to every mesh parameter (typically pixels of a ``Mapper``),
        with the degree applied varying based on the ``pixel_signals``.
    pixel_signals
        The estimated signal in every pixelization pixel, used to change the regularization weighting of high signal
        and low signal pixelizations.

    Returns
    -------
    np.ndarray
        The zeroth order regularization weights which act as the effective level of zeroth order regularization
        applied to every mesh parameter.
    """
    return coefficient * (1.0 - pixel_signals)


# @numba_util.jit()
def weighted_regularization_matrix_from(
    regularization_weights: np.ndarray,
    neighbors: np.ndarray,
    neighbors_sizes: np.ndarray,
) -> np.ndarray:
    """
    Returns the regularization matrix of the adaptive regularization scheme (e.g. ``AdaptiveBrightness``).

    This matrix is computed using the regularization weights of every mesh pixel, which are computed using the
    function ``adaptive_regularization_weights_from``. These act as the effective regularization coefficients of
    every mesh pixel.

    The regularization matrix is computed using the pixel-neighbors array, which is setup using the appropriate
    neighbor calculation of the corresponding ``Mapper`` class.

    Parameters
    ----------
    regularization_weights
        The regularization weight of each pixel, adaptively governing the degree of gradient regularization
        applied to each inversion parameter (e.g. mesh pixels of a ``Mapper``).
    neighbors
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in
        the mesh grid (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.

    Returns
    -------
    np.ndarray
        The regularization matrix computed using an adaptive regularization scheme where the effective regularization
        coefficient of every source pixel is different.
    """
    parameters = len(regularization_weights)
    regularization_matrix = np.zeros((parameters, parameters))
    regularization_weight = regularization_weights**2.0

    # Add small diagonal offset
    np.fill_diagonal(regularization_matrix, 1e-8)

    for i in range(parameters):
        for j in range(neighbors_sizes[i]):
            neighbor_index = neighbors[i, j]
            w = regularization_weight[neighbor_index]

            regularization_matrix[i, i] += w
            regularization_matrix[neighbor_index, neighbor_index] += w
            regularization_matrix[i, neighbor_index] -= w
            regularization_matrix[neighbor_index, i] -= w

    return regularization_matrix


# def weighted_regularization_matrix_from(
#     regularization_weights: np.ndarray,
#     neighbors: np.ndarray,
#     neighbors_sizes: np.ndarray,
# ) -> np.ndarray:
#     """
#     Returns the regularization matrix of the adaptive regularization scheme (e.g. ``AdaptiveBrightness``).
#
#     This matrix is computed using the regularization weights of every mesh pixel, which are computed using the
#     function ``adaptive_regularization_weights_from``. These act as the effective regularization coefficients of
#     every mesh pixel.
#
#     The regularization matrix is computed using the pixel-neighbors array, which is setup using the appropriate
#     neighbor calculation of the corresponding ``Mapper`` class.
#
#     Parameters
#     ----------
#     regularization_weights
#         The regularization weight of each pixel, adaptively governing the degree of gradient regularization
#         applied to each inversion parameter (e.g. mesh pixels of a ``Mapper``).
#     neighbors
#         An array of length (total_pixels) which provides the index of all neighbors of every pixel in
#         the mesh grid (entries of -1 correspond to no neighbor).
#     neighbors_sizes
#         An array of length (total_pixels) which gives the number of neighbors of every pixel in the
#         Voronoi grid.
#
#     Returns
#     -------
#     np.ndarray
#         The regularization matrix computed using an adaptive regularization scheme where the effective regularization
#         coefficient of every source pixel is different.
#     """
#     parameters = len(regularization_weights)
#     regularization_matrix = np.zeros((parameters, parameters))
#     regularization_weight = regularization_weights**2.0
#
#     # Add small diagonal offset
#     np.fill_diagonal(regularization_matrix, 1e-8)
#
#     for i in range(parameters):
#         for j in range(neighbors_sizes[i]):
#             neighbor_index = neighbors[i, j]
#             w = regularization_weight[neighbor_index]
#
#             regularization_matrix[i, i] += w
#             regularization_matrix[neighbor_index, neighbor_index] += w
#             regularization_matrix[i, neighbor_index] -= w
#             regularization_matrix[neighbor_index, i] -= w
#
#     return regularization_matrix

def brightness_zeroth_regularization_matrix_from(
    regularization_weights: np.ndarray,
) -> np.ndarray:
    """
    Returns the regularization matrix for the zeroth-order brightness regularization scheme.

    Parameters
    ----------
    regularization_weights
        The regularization weights for each pixel, governing the strength of zeroth-order
        regularization applied per inversion parameter.

    Returns
    -------
    A diagonal regularization matrix where each diagonal element is the squared regularization weight
    for that pixel.
    """
    regularization_weight_squared = regularization_weights**2.0
    return np.diag(regularization_weight_squared)



def reg_split_from(
    splitted_mappings: np.ndarray,
    splitted_sizes: np.ndarray,
    splitted_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    When creating the regularization matrix of a source pixelization, this function assumes each source pixel has been
    split into a cross of four points (the size of which is based on the area of the source pixel). This cross of
    points represents points which together can evaluate the gradient of the pixelization's reconstructed values.

    This function takes each cross of points and determines the regularization weights of every point on the cross,
    to construct a regulariaztion matrix based on the gradient of each pixel.

    The size of each cross depends on the Voronoi pixel area, thus this regularization scheme and its weights depend
    on the pixel area (there are larger weights for bigger pixels). This ensures that bigger pixels are regularized
    more.

    The number of pixel neighbors over which regularization is 4 * the total number of source pixels. This contrasts
    other regularization schemes, where the number of neighbors changes depending on, for example, the Voronoi mesh
    geometry. By having a fixed number of neighbors this removes stochasticty in the regularization that is applied
    to a solution.

    There are cases where a grid has over 100 neighbors, corresponding to very coordinate transformations. In such
    extreme cases, we raise a `exc.FitException`.

    Parameters
    ----------
    splitted_mappings
    splitted_sizes
    splitted_weights

    Returns
    -------

    """

    max_j = np.shape(splitted_weights)[1] - 1

    splitted_weights *= -1.0

    for i in range(len(splitted_mappings)):
        pixel_index = i // 4

        flag = 0

        for j in range(splitted_sizes[i]):
            if splitted_mappings[i][j] == pixel_index:
                splitted_weights[i][j] += 1.0
                flag = 1

            if j >= max_j:
                raise exc.MeshException(
                    f"The number of Voronoi natural neighbours exceeds {max_j}."
                )

        if flag == 0:
            splitted_mappings[i][j + 1] = pixel_index
            splitted_sizes[i] += 1
            splitted_weights[i][j + 1] = 1.0

    return splitted_mappings, splitted_sizes, splitted_weights


def pixel_splitted_regularization_matrix_from(
    regularization_weights: np.ndarray,
    splitted_mappings: np.ndarray,
    splitted_sizes: np.ndarray,
    splitted_weights: np.ndarray,
) -> np.ndarray:
    """
    Returns the regularization matrix for the adaptive split-pixel regularization scheme.

    This scheme splits each source pixel into a cross of four regularization points and interpolates
    to those points to smooth the inversion solution. It is designed to mitigate stochasticity in
    the regularization that can arise when the number of neighboring pixels varies across a
    mesh (e.g., in a Voronoi tessellation).

    A visual description and further details are provided in the appendix of He et al. (2024):
    https://arxiv.org/abs/2403.16253

    Parameters
    ----------
    regularization_weights
        The regularization weight per pixel, adaptively controlling the strength of regularization
        applied to each inversion parameter.
    splitted_mappings
        The image pixel index mappings for each of the four regularization points into which each source pixel is split.
    splitted_sizes
        The number of neighbors or interpolation terms associated with each regularization point.
    splitted_weights
        The interpolation weights corresponding to each mapping entry, used to apply regularization
        between split points.

    Returns
    -------
    The regularization matrix of shape [source_pixels, source_pixels].
    """

    parameters = splitted_mappings.shape[0] // 4
    regularization_matrix = np.zeros((parameters, parameters))
    regularization_weight = regularization_weights**2.0

    # Add small constant to diagonal
    np.fill_diagonal(regularization_matrix, 2e-8)

    # Compute regularization contributions
    for i in range(parameters):
        reg_w = regularization_weight[i]
        for j in range(4):
            k = i * 4 + j
            size = splitted_sizes[k]
            mapping = splitted_mappings[k][:size]
            weight = splitted_weights[k][:size]

            # Outer product of weights and symmetric updates
            outer = np.outer(weight, weight) * reg_w
            rows, cols = np.meshgrid(mapping, mapping, indexing='ij')
            regularization_matrix[rows, cols] += outer

    # Correct diagonal entries
    np.fill_diagonal(regularization_matrix, np.diag(regularization_matrix) / 2.0)

    return regularization_matrix

