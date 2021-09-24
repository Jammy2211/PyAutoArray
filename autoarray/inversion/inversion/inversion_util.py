import numpy as np
from scipy.linalg import cho_solve

from autoarray.inversion.inversion.settings import SettingsInversion

from autoarray import numba_util
from autoarray import exc

from autoarray.inversion.linear_eqn import linear_eqn_util


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

    curvature_matrix = linear_eqn_util.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=mapping_matrix,
        noise_map=np.ones(shape=(mapping_matrix.shape[0])),
    )

    return (
        preconditioner_noise_normalization * curvature_matrix
    ) + regularization_matrix
