import numpy as np
from typing import Tuple

from autoarray import exc

from autoarray.inversion.regularization.adaptive_brightness import (
    adaptive_regularization_weights_from,
)
from autoarray.inversion.regularization.adaptive_brightness import (
    weighted_regularization_matrix_from,
)
from autoarray.inversion.regularization.brightness_zeroth import (
    brightness_zeroth_regularization_matrix_from,
)
from autoarray.inversion.regularization.brightness_zeroth import (
    brightness_zeroth_regularization_weights_from,
)
from autoarray.inversion.regularization.constant import (
    constant_regularization_matrix_from,
)
from autoarray.inversion.regularization.constant_zeroth import (
    constant_zeroth_regularization_matrix_from,
)
from autoarray.inversion.regularization.exponential_kernel import exp_cov_matrix_from
from autoarray.inversion.regularization.gaussian_kernel import gauss_cov_matrix_from
from autoarray.inversion.regularization.matern_kernel import matern_kernel
from autoarray.inversion.regularization.zeroth import zeroth_regularization_matrix_from


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
