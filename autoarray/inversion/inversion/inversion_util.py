import numpy as np
import scipy

# from scipy.optimize import nnls
from typing import List, Optional

from autoconf import conf

from autoarray.inversion.inversion.settings import SettingsInversion

from autoarray import numba_util
from autoarray import exc
from autoarray.util.fnnls import fnnls_Cholesky


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
    curvature_matrix: np.ndarray, no_regularization_index_list: Optional[List] = None
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
        curvature_matrix[i, i] += 1e-8

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
            no_regularization_index_list=no_regularization_index_list,
        )

    return curvature_matrix


@numba_util.jit()
def curvature_matrix_preload_from(
    mapping_matrix: np.ndarray, mapping_matrix_threshold=1.0e-8
) -> np.ndarray:
    """
    Returns a matrix that expresses the non-zero entries of the blurred mapping matrix for an efficient construction of
    the curvature matrix `F` (see Warren & Dye 2003).

    This is used for models where the blurred mapping matrix does not change, but the noise-map of the values that
    are used to construct that blurred mapping matrix do change.

    Parameters
    ----------
    mapping_matrix
        The matrix representing the mappings (these could be blurred or transfomed) between sub-grid pixels and
        pixelization pixels.
    """

    curvature_matrix_counts = np.zeros(mapping_matrix.shape[0])

    for mask_1d_index in range(mapping_matrix.shape[0]):
        for pix_index in range(mapping_matrix.shape[1]):
            if mapping_matrix[mask_1d_index, pix_index] > mapping_matrix_threshold:
                curvature_matrix_counts[mask_1d_index] += 1

    preload_max = np.max(curvature_matrix_counts)

    curvature_matrix_preload = np.zeros((mapping_matrix.shape[0], int(preload_max)))

    for mask_1d_index in range(mapping_matrix.shape[0]):
        index = 0
        for pix_index in range(mapping_matrix.shape[1]):
            if mapping_matrix[mask_1d_index, pix_index] > mapping_matrix_threshold:
                curvature_matrix_preload[mask_1d_index, index] = pix_index
                index += 1

    return curvature_matrix_preload, curvature_matrix_counts


@numba_util.jit()
def curvature_matrix_via_sparse_preload_from(
    mapping_matrix: np.ndarray,
    noise_map: np.ndarray,
    curvature_matrix_preload,
    curvature_matrix_counts,
) -> np.ndarray:
    """
    Returns the curvature matrix `F` from a blurred mapping matrix `f` and the 1D noise-map $\sigma$
     (see Warren & Dye 2003) via a sparse preload matrix, which has already mapped out where all non-zero entries
     and multiplications take place.

    This is used for models where the blurred mapping matrix does not change, but the noise-map of the values that
    are used to construct that blurred mapping matrix do change.

    Parameters
    ----------
    mapping_matrix
        The matrix representing the mappings (these could be blurred or transfomed) between sub-grid pixels and
        pixelization pixels.
    noise_map
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """
    curvature_matrix = np.zeros((mapping_matrix.shape[1], mapping_matrix.shape[1]))

    for mask_1d_index in range(mapping_matrix.shape[0]):

        total_pix = curvature_matrix_counts[mask_1d_index]

        for preload_index_0 in range(total_pix):
            for preload_index_1 in range(
                preload_index_0, curvature_matrix_counts[mask_1d_index]
            ):

                pix_index_0 = curvature_matrix_preload[mask_1d_index, preload_index_0]
                pix_index_1 = curvature_matrix_preload[mask_1d_index, preload_index_1]

                curvature_matrix[pix_index_0, pix_index_1] += (
                    mapping_matrix[mask_1d_index, pix_index_0]
                    * mapping_matrix[mask_1d_index, pix_index_1]
                ) / noise_map[mask_1d_index] ** 2

    for i in range(mapping_matrix.shape[1]):
        for j in range(mapping_matrix.shape[1]):
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

    The non-negative optimizer we use is a modified version of fnnls (https://github.com/jvendrow/fnnls). The algorithm is published by
    Bro & Jong (1997) ("A fast non‐negativity‐constrained least squares algorithm."
                Journal of Chemometrics: A Journal of the Chemometrics Society 11, no. 5 (1997): 393-401.)

    The modification we made here is that we create a function called fnnls_Cholesky which directly takes ZTZ and ZTx as inputs. The reason
    is that we realize for this specific algorithm (Bro & Jong (1997)), ZTZ and ZTx happen to be the curvature_reg_matrix and
    data_vector, respectively, already defined in PyAutoArray (verified). Besides, we build a Cholesky scheme that solves the lstsq problem
    in each iteration within the fnnls algorithm by updating the Cholesky factorisation.

    Please note that we are trying to find non-negative solution S that minimizes |Z * S - x|^2. We are not trying to find a solution that
    minimizes |ZTZ * S - ZTx|^2! ZTZ and ZTx are just some variables help to minimize |Z * S - x|^2. It is just a coincidence (or fundamentally not)
    that ZTZ and ZTx are the curvature_reg_matrix and data_vector, respectively.

    If we no longer uses fnnls (the algorithm of Bro & Jong (1997)), we need to check if the algorithm takes Z or ZTZ (x or ZTx) as an input. If not,
    we need to build Z and x in PyAutoArray.

    Parameters
    ----------
    data_vector
        The `data_vector` D happens to be the ZTx.
    curvature_reg_matrix
        The sum of the curvature and regularization matrices. Taken as ZTZ in our problem.
    settings
        Controls the settings of the inversion, for this function where the solution is checked to not be all
        the same values.

    Returns
    -------
    Non-negative S that minimizes the Eq.(2) of https://arxiv.org/pdf/astro-ph/0302587.pdf.
    """

    try:

        #np.save(file="ZTZ", arr=curvature_reg_matrix)
        #np.save(file="ZTx", arr=data_vector)

        #print('Here!')

        reconstruction = fnnls_Cholesky(
            curvature_reg_matrix,
            (data_vector).T,
            P_initial=scipy.linalg.solve(curvature_reg_matrix, data_vector.T, assume_a='pos') > 0,
            lstsq=lambda A, x: scipy.linalg.solve(
                A,
                x,
                assume_a="pos",
                overwrite_a=True,
                overwrite_b=True,
                check_finite=False,
            ),
        )

        #print('Finish Cholesky!')

    except (RuntimeError, np.linalg.LinAlgError) as e:
        raise exc.InversionException() from e

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


def inversion_residual_map_from(
    *,
    reconstruction: np.ndarray,
    data: np.ndarray,
    slim_index_for_sub_slim_index: np.ndarray,
    sub_slim_indexes_for_pix_index: [list],
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
    reconstruction
        The values computed by the `Inversion` for the `reconstruction`, which are used in this function to compute
        the `residual_map` values.
    data
        The array of `data` that the `Inversion` fits.
    slim_index_for_sub_slim_index
        The mappings between the observed grid's sub-pixels and observed grid's pixels.
    sub_slim_indexes_for_pix_index
        The mapping of every pixel on the `LinearEqn`'s `reconstruction`'s pixel-grid to the `data` pixels.

    Returns
    -------
    np.ndarray
        The residuals of the `Inversion`'s `reconstruction` on its pixel-grid, computed by mapping the `residual_map`
        from the fit to the data.
    """
    residual_map = np.zeros(shape=len(sub_slim_indexes_for_pix_index))

    for pix_index, sub_slim_indexes in enumerate(sub_slim_indexes_for_pix_index):

        sub_mask_total = 0
        for sub_mask_1d_index in sub_slim_indexes:
            sub_mask_total += 1
            mask_1d_index = slim_index_for_sub_slim_index[sub_mask_1d_index]
            residual = data[mask_1d_index] - reconstruction[pix_index]
            residual_map[pix_index] += np.abs(residual)

        if sub_mask_total > 0:
            residual_map[pix_index] /= sub_mask_total

    return residual_map.copy()


def inversion_normalized_residual_map_from(
    *,
    reconstruction,
    data,
    noise_map_1d,
    slim_index_for_sub_slim_index,
    sub_slim_indexes_for_pix_index,
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
    reconstruction
        The values computed by the `Inversion` for the `reconstruction`, which are used in this function to compute
        the `normalized residual_map` values.
    data
        The array of `data` that the `Inversion` fits.
    slim_index_for_sub_slim_index
        The mappings between the observed grid's sub-pixels and observed grid's pixels.
    sub_slim_indexes_for_pix_index
        The mapping of every pixel on the `LinearEqn`'s `reconstruction`'s pixel-grid to the `data` pixels.

    Returns
    -------
    np.ndarray
        The normalized residuals of the `Inversion`'s `reconstruction` on its pixel-grid, computed by mapping the
        `normalized_residual_map` from the fit to the data.
    """
    normalized_residual_map = np.zeros(shape=len(sub_slim_indexes_for_pix_index))

    for pix_index, sub_slim_indexes in enumerate(sub_slim_indexes_for_pix_index):
        sub_mask_total = 0
        for sub_mask_1d_index in sub_slim_indexes:
            sub_mask_total += 1
            mask_1d_index = slim_index_for_sub_slim_index[sub_mask_1d_index]
            residual = data[mask_1d_index] - reconstruction[pix_index]
            normalized_residual_map[pix_index] += np.abs(
                (residual / noise_map_1d[mask_1d_index])
            )

        if sub_mask_total > 0:
            normalized_residual_map[pix_index] /= sub_mask_total

    return normalized_residual_map.copy()


def inversion_chi_squared_map_from(
    *,
    reconstruction,
    data,
    noise_map_1d,
    slim_index_for_sub_slim_index,
    sub_slim_indexes_for_pix_index,
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
    reconstruction
        The values computed by the `Inversion` for the `reconstruction`, which are used in this function to compute
        the `chi_squared_map` values.
    data
        The array of `data` that the `Inversion` fits.
    slim_index_for_sub_slim_index
        The mappings between the observed grid's sub-pixels and observed grid's pixels.
    sub_slim_indexes_for_pix_index
        The mapping of every pixel on the `LinearEqn`'s `reconstruction`'s pixel-grid to the `data` pixels.

    Returns
    -------
    np.ndarray
        The chi-squareds of the `Inversion`'s `reconstruction` on its pixel-grid, computed by mapping the `chi-squared_map`
        from the fit to the data.
    """
    chi_squared_map = np.zeros(shape=len(sub_slim_indexes_for_pix_index))

    for pix_index, sub_slim_indexes in enumerate(sub_slim_indexes_for_pix_index):
        sub_mask_total = 0
        for sub_mask_1d_index in sub_slim_indexes:
            sub_mask_total += 1
            mask_1d_index = slim_index_for_sub_slim_index[sub_mask_1d_index]
            residual = data[mask_1d_index] - reconstruction[pix_index]
            chi_squared_map[pix_index] += (
                residual / noise_map_1d[mask_1d_index]
            ) ** 2.0

        if sub_mask_total > 0:
            chi_squared_map[pix_index] /= sub_mask_total

    return chi_squared_map.copy()
