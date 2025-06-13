import jax.numpy as jnp
import jaxnnls
import numpy as np

from typing import List, Optional, Tuple

from autoconf import conf

from autoarray.inversion.inversion.settings import SettingsInversion

from autoarray import numba_util
from autoarray import exc
from autoarray.util.fnnls import fnnls_cholesky


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


# def curvature_matrix_with_added_to_diag_from(
#     curvature_matrix: np.ndarray,
#     value: float,
#     no_regularization_index_list: Optional[List] = None,
# ) -> np.ndarray:
#     """
#     It is common for the `curvature_matrix` computed to not be positive-definite, leading for the inversion
#     via `np.linalg.solve` to fail and raise a `LinAlgError`.
#
#     In many circumstances, adding a small numerical value of `1.0e-8` to the diagonal of the `curvature_matrix`
#     makes it positive definite, such that the inversion is performed without raising an error.
#
#     This function adds this numerical value to the diagonal of the curvature matrix.
#
#     Parameters
#     ----------
#     curvature_matrix
#         The curvature matrix which is being constructed in order to solve a linear system of equations.
#     """
#     return curvature_matrix.at[
#         no_regularization_index_list, no_regularization_index_list
#     ].add(value)


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
    return np.dot(mapping_matrix, reconstruction)


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
        reconstruction = jnp.linalg.solve(curvature_reg_matrix, data_vector)
    except np.linalg.LinAlgError as e:
        raise exc.InversionException() from e

    if jnp.isnan(reconstruction).any():
        raise exc.InversionException

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

    try:
        return jaxnnls.solve_nnls_primal(curvature_reg_matrix, data_vector)
    except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
        raise exc.InversionException() from e

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
