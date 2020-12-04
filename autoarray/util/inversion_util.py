from autoarray import decorator_util
import numpy as np


@decorator_util.jit()
def data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix: np.ndarray, image: np.ndarray, noise_map: np.ndarray
) -> np.ndarray:
    """
    Returns the data vector `D` from a blurred mapping matrix `f` and the 1D image `d` and 1D noise-map $\sigma$`
    (see Warren & Dye 2003).

    Parameters
    -----------
    blurred_mapping_matrix : np.ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    image : np.ndarray
        Flattened 1D array of the observed image the inversion is fitting.
    noise_map : np.ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """

    mapping_shape = blurred_mapping_matrix.shape

    data_vector = np.zeros(mapping_shape[1])

    for mask_1d_index in range(mapping_shape[0]):
        for pix_1_index in range(mapping_shape[1]):
            data_vector[pix_1_index] += (
                image[mask_1d_index]
                * blurred_mapping_matrix[mask_1d_index, pix_1_index]
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
    mapping_matrix : np.ndarray
        The matrix representing the mappings (these could be blurred or transfomed) between sub-grid pixels and
        pixelization pixels.
    noise_map : np.ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """
    array = mapping_matrix / noise_map[:, None]
    curvature_matrix = np.dot(array.T, np.matrix.transpose(array.T))
    return curvature_matrix


@decorator_util.jit()
def mapped_reconstructed_data_from(
    mapping_matrix: np.ndarray, reconstruction: np.ndarray
) -> np.ndarray:
    """
    Returns the reconstructed data vector from the blurrred mapping matrix `f` and solution vector *S*.

    Parameters
    -----------
    mapping_matrix : np.ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """
    mapped_reconstructred_data = np.zeros(mapping_matrix.shape[0])
    for i in range(mapping_matrix.shape[0]):
        for j in range(reconstruction.shape[0]):
            mapped_reconstructred_data[i] += reconstruction[j] * mapping_matrix[i, j]

    return mapped_reconstructred_data


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
    transformed_mapping_matrix : np.ndarray
        The matrix representing the transformed mappings between sub-grid pixels and pixelization pixels.
    image : np.ndarray
        Flattened 1D array of the observed image the inversion is fitting.
    noise_map : np.ndarray
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


# @decorator_util.jit()
def mapped_reconstructed_visibilities_from(
    transformed_mapping_matrix: np.ndarray, reconstruction: np.ndarray
) -> np.ndarray:
    """
    Returns the reconstructed data vector from the blurrred mapping matrix `f` and solution vector *S*.

    Parameters
    -----------
    transformed_mapping_matrix : np.ndarray
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
    mask_1d_index_for_sub_mask_1d_index: np.ndarray,
    all_sub_mask_1d_indexes_for_pixelization_1d_index: [list],
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
    pixelization_values : np.ndarray
        The values computed by the `Inversion` for the `reconstruction`, which are used in this function to compute
        the `residual_map` values.
    data : np.ndarray
        The array of `data` that the `Inversion` fits.
    mask_1d_index_for_sub_mask_1d_index : np.ndarray
        The mappings between the observed grid's sub-pixels and observed grid's pixels.
    all_sub_mask_1d_indexes_for_pixelization_1d_index : np.ndarray
        The mapping of every pixel on the `Inversion`'s `reconstruction`'s pixel-grid to the `data` pixels.

    Returns
    -------
    np.ndarray
        The residuals of the `Inversion`'s `reconstruction` on its pixel-grid, computed by mapping the `residual_map`
        from the fit to the data.
    """
    residual_map = np.zeros(
        shape=len(all_sub_mask_1d_indexes_for_pixelization_1d_index)
    )

    for pix_1_index, sub_mask_1d_indexes in enumerate(
        all_sub_mask_1d_indexes_for_pixelization_1d_index
    ):

        sub_mask_total = 0
        for sub_mask_1d_index in sub_mask_1d_indexes:
            sub_mask_total += 1
            mask_1d_index = mask_1d_index_for_sub_mask_1d_index[sub_mask_1d_index]
            residual = data[mask_1d_index] - pixelization_values[pix_1_index]
            residual_map[pix_1_index] += np.abs(residual)

        if sub_mask_total > 0:
            residual_map[pix_1_index] /= sub_mask_total

    return residual_map.copy()


def inversion_normalized_residual_map_from(
    *,
    pixelization_values,
    data,
    noise_map_1d,
    mask_1d_index_for_sub_mask_1d_index,
    all_sub_mask_1d_indexes_for_pixelization_1d_index,
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
    pixelization_values : np.ndarray
        The values computed by the `Inversion` for the `reconstruction`, which are used in this function to compute
        the `normalized residual_map` values.
    data : np.ndarray
        The array of `data` that the `Inversion` fits.
    mask_1d_index_for_sub_mask_1d_index : np.ndarray
        The mappings between the observed grid's sub-pixels and observed grid's pixels.
    all_sub_mask_1d_indexes_for_pixelization_1d_index : np.ndarray
        The mapping of every pixel on the `Inversion`'s `reconstruction`'s pixel-grid to the `data` pixels.

    Returns
    -------
    np.ndarray
        The normalized residuals of the `Inversion`'s `reconstruction` on its pixel-grid, computed by mapping the
        `normalized_residual_map` from the fit to the data.
    """
    normalized_residual_map = np.zeros(
        shape=len(all_sub_mask_1d_indexes_for_pixelization_1d_index)
    )

    for pix_1_index, sub_mask_1d_indexes in enumerate(
        all_sub_mask_1d_indexes_for_pixelization_1d_index
    ):
        sub_mask_total = 0
        for sub_mask_1d_index in sub_mask_1d_indexes:
            sub_mask_total += 1
            mask_1d_index = mask_1d_index_for_sub_mask_1d_index[sub_mask_1d_index]
            residual = data[mask_1d_index] - pixelization_values[pix_1_index]
            normalized_residual_map[pix_1_index] += np.abs(
                (residual / noise_map_1d[mask_1d_index])
            )

        if sub_mask_total > 0:
            normalized_residual_map[pix_1_index] /= sub_mask_total

    return normalized_residual_map.copy()


def inversion_chi_squared_map_from(
    *,
    pixelization_values,
    data,
    noise_map_1d,
    mask_1d_index_for_sub_mask_1d_index,
    all_sub_mask_1d_indexes_for_pixelization_1d_index,
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
    pixelization_values : np.ndarray
        The values computed by the `Inversion` for the `reconstruction`, which are used in this function to compute
        the `chi_squared_map` values.
    data : np.ndarray
        The array of `data` that the `Inversion` fits.
    mask_1d_index_for_sub_mask_1d_index : np.ndarray
        The mappings between the observed grid's sub-pixels and observed grid's pixels.
    all_sub_mask_1d_indexes_for_pixelization_1d_index : np.ndarray
        The mapping of every pixel on the `Inversion`'s `reconstruction`'s pixel-grid to the `data` pixels.

    Returns
    -------
    np.ndarray
        The chi-squareds of the `Inversion`'s `reconstruction` on its pixel-grid, computed by mapping the `chi-squared_map`
        from the fit to the data.
    """
    chi_squared_map = np.zeros(
        shape=len(all_sub_mask_1d_indexes_for_pixelization_1d_index)
    )

    for pix_1_index, sub_mask_1d_indexes in enumerate(
        all_sub_mask_1d_indexes_for_pixelization_1d_index
    ):
        sub_mask_total = 0
        for sub_mask_1d_index in sub_mask_1d_indexes:
            sub_mask_total += 1
            mask_1d_index = mask_1d_index_for_sub_mask_1d_index[sub_mask_1d_index]
            residual = data[mask_1d_index] - pixelization_values[pix_1_index]
            chi_squared_map[pix_1_index] += (
                residual / noise_map_1d[mask_1d_index]
            ) ** 2.0

        if sub_mask_total > 0:
            chi_squared_map[pix_1_index] /= sub_mask_total

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
    mapping_matrix : np.ndarray
        The matrix representing the mappings between sub-grid pixels and pixelization pixels.
    regularization_matrix : np.ndarray
        The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
    preconditioner_noise_normalization : np.ndarray
        The sum of (1.0 / noise-map**2.0) every value in the noise-map.
    """

    curvature_matrix = curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=mapping_matrix,
        noise_map=np.ones(shape=(mapping_matrix.shape[0])),
    )

    return (
        preconditioner_noise_normalization * curvature_matrix
    ) + regularization_matrix
