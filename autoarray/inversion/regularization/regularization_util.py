import jax.numpy as jnp
import numpy as np
from typing import Tuple

from autoarray import exc
from autoarray import numba_util


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
    reg_coeff = coefficient ** 2.0
    # Identity matrix scaled by reg_coeff does exactly ∑_i reg_coeff * e_i e_i^T
    return jnp.eye(pixels) * reg_coeff


def constant_regularization_matrix_from(
    coefficient: float,
    neighbors: np.ndarray[[int, int], np.int64],
    neighbors_sizes: np.ndarray[[int], np.int64],
) -> np.ndarray[[int, int], np.float64]:
    """
    From the pixel-neighbors array, setup the regularization matrix using the instance regularization scheme.

    A complete description of regularizatin and the `regularization_matrix` can be found in the `Regularization`
    class in the module `autoarray.inversion.regularization`.

    Memory requirement: 2SP + S^2
    FLOPS: 1 + 2S + 2SP

    Parameters
    ----------
    coefficient
        The regularization coefficients which controls the degree of smoothing of the inversion reconstruction.
    neighbors : ndarray, shape (S, P), dtype=int64
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    neighbors_sizes : ndarray, shape (S,), dtype=int64
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.

    Returns
    -------
    regularization_matrix : ndarray, shape (S, S), dtype=float64
        The regularization matrix computed using Regularization where the effective regularization
        coefficient of every source pixel is the same.
    """
    S, P = neighbors.shape
    # as the regularization matrix is S by S, S would be out of bound (any out of bound index would do)
    OUT_OF_BOUND_IDX = S
    regularization_coefficient = coefficient * coefficient

    # flatten it for feeding into the matrix as j indices
    neighbors = neighbors.flatten()
    # now create the corresponding i indices
    I_IDX = jnp.repeat(jnp.arange(S), P)
    # Entries of `-1` in `neighbors` (indicating no neighbor) are replaced with an out-of-bounds index.
    # This ensures that JAX can efficiently drop these entries during matrix updates.
    neighbors = jnp.where(neighbors == -1, OUT_OF_BOUND_IDX, neighbors)
    return (
        jnp.diag(1e-8 + regularization_coefficient * neighbors_sizes).at[I_IDX, neighbors]
        # unique indices should be guranteed by neighbors-spec
        .add(-regularization_coefficient, mode="drop", unique_indices=True)
    )


def constant_zeroth_regularization_matrix_from(
    coefficient: float,
    coefficient_zeroth: float,
    neighbors: np.ndarray,
    neighbors_sizes: np.ndarray[[int], np.int64],
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
    S, P = neighbors.shape
    # as the regularization matrix is S by S, S would be out of bound (any out of bound index would do)
    OUT_OF_BOUND_IDX = S
    regularization_coefficient = coefficient * coefficient

    # flatten it for feeding into the matrix as j indices
    neighbors = neighbors.flatten()
    # now create the corresponding i indices
    I_IDX = jnp.repeat(jnp.arange(S), P)
    # Entries of `-1` in `neighbors` (indicating no neighbor) are replaced with an out-of-bounds index.
    # This ensures that JAX can efficiently drop these entries during matrix updates.
    neighbors = jnp.where(neighbors == -1, OUT_OF_BOUND_IDX, neighbors)
    const = (
        jnp.diag(1e-8 + regularization_coefficient * neighbors_sizes).at[I_IDX, neighbors]
        # unique indices should be guranteed by neighbors-spec
        .add(-regularization_coefficient, mode="drop", unique_indices=True)
    )

    reg_coeff = coefficient_zeroth ** 2.0
    # Identity matrix scaled by reg_coeff does exactly ∑_i reg_coeff * e_i e_i^T
    zeroth = jnp.eye(P) * reg_coeff

    return const + zeroth

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


def weighted_regularization_matrix_from(
    regularization_weights: np.ndarray,
    neighbors: np.ndarray,
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
    S, P = neighbors.shape
    reg_w = regularization_weights ** 2

    # 1) Flatten the (i→j) neighbor pairs
    I = jnp.repeat(jnp.arange(S), P)            # (S*P,)
    J = neighbors.reshape(-1)                   # (S*P,)

    # 2) Remap “no neighbor” entries to an extra slot S, whose weight=0
    OUT = S
    J = jnp.where(J < 0, OUT, J)

    # 3) Build an extended weight vector with a zero at index S
    reg_w_ext = jnp.concatenate([reg_w, jnp.zeros((1,))], axis=0)
    w_ij = reg_w_ext[J]                         # (S*P,)

    # 4) Start with zeros on an (S+1)x(S+1) canvas so we can scatter into row S safely
    mat = jnp.zeros((S + 1, S + 1), dtype=regularization_weights.dtype)

    # 5) Scatter into the diagonal:
    #    - the tiny 1e-8 floor on each i < S
    #    - sum_j reg_w[j] into diag[i]
    #    - sum contributions reg_w[j] into diag[j]
    #    (diagonal at OUT=S picks up zeros only)
    diag_updates_i = jnp.concatenate([
        jnp.full((S,), 1e-8),
        jnp.zeros((1,))  # out‐of‐bounds slot stays zero
    ], axis=0)
    mat = mat.at[jnp.diag_indices(S + 1)].add(diag_updates_i)
    mat = mat.at[I, I].add(w_ij)
    mat = mat.at[J, J].add(w_ij)

    # 6) Scatter the off‐diagonal subtractions:
    mat = mat.at[I, J].add(-w_ij)
    mat = mat.at[J, I].add(-w_ij)

    # 7) Drop the extra row/column S and return the S×S result
    return mat[:S, :S]

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
            rows, cols = np.meshgrid(mapping, mapping, indexing="ij")
            regularization_matrix[rows, cols] += outer

    return regularization_matrix
