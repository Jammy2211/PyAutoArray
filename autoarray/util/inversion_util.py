from autoarray import decorator_util
import numpy as np


@decorator_util.jit()
def data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix, image, noise_map
):
    """Compute the hyper_galaxies vector *D* from a blurred util matrix *f* and the 1D image *d* and 1D noise-map *\sigma* \
    (see Warren & Dye 2003).
    
    Parameters
    -----------
    blurred_mapping_matrix : ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    image : ndarray
        Flattened 1D array of the observed image the inversion is fitting.
    noise_map : ndarray
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


def curvature_matrix_via_blurred_mapping_matrix_from(blurred_mapping_matrix, noise_map):
    """Compute the curvature matrix *F* from a blurred util matrix *f* and the 1D noise-map *\sigma* \
     (see Warren & Dye 2003).

    Parameters
    -----------
    blurred_mapping_matrix : ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    noise_map : ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """

    flist = np.zeros(blurred_mapping_matrix.shape[1])
    iflist = np.zeros(blurred_mapping_matrix.shape[1], dtype="int")
    return curvature_matrix_via_blurred_mapping_matrix_jit(
        blurred_mapping_matrix, noise_map, flist, iflist
    )


@decorator_util.jit()
def curvature_matrix_via_blurred_mapping_matrix_jit(
    blurred_mapping_matrix, noise_map, flist, iflist
):
    """Compute the curvature matrix *F* from a blurred util matrix *f* and the 1D noise-map *\sigma* \
    (see Warren & Dye 2003).

    Parameters
    -----------
    blurred_mapping_matrix : ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    noise_map : ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    flist : ndarray
        NumPy array of floats used to store mappings for efficienctly calculation.
    iflist : ndarray
        NumPy array of integers used to store mappings for efficienctly calculation.
    """
    curvature_matrix = np.zeros(
        (blurred_mapping_matrix.shape[1], blurred_mapping_matrix.shape[1])
    )

    for mask_1d_index in range(blurred_mapping_matrix.shape[0]):
        index = 0
        for pix_1_index in range(blurred_mapping_matrix.shape[1]):
            if blurred_mapping_matrix[mask_1d_index, pix_1_index] > 0.0:
                flist[index] = (
                    blurred_mapping_matrix[mask_1d_index, pix_1_index]
                    / noise_map[mask_1d_index]
                )
                iflist[index] = pix_1_index
                index += 1

        if index > 0:
            for i1 in range(index):
                for j1 in range(index):
                    ix = iflist[i1]
                    iy = iflist[j1]
                    curvature_matrix[ix, iy] += flist[i1] * flist[j1]

    for i in range(blurred_mapping_matrix.shape[1]):
        for j in range(blurred_mapping_matrix.shape[1]):
            curvature_matrix[i, j] = curvature_matrix[j, i]

    return curvature_matrix


@decorator_util.jit()
def mapped_reconstructed_data_from(mapping_matrix, reconstruction):
    """ Compute the reconstructed hyper_galaxies vector from the blurrred util matrix *f* and solution vector *S*.

    Parameters
    -----------
    mapping_matrix : ndarray
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """
    mapped_reconstructred_data = np.zeros(mapping_matrix.shape[0])
    for i in range(mapping_matrix.shape[0]):
        for j in range(reconstruction.shape[0]):
            mapped_reconstructred_data[i] += reconstruction[j] * mapping_matrix[i, j]

    return mapped_reconstructred_data


@decorator_util.jit()
def data_vector_via_transformed_mapping_matrix_from(
    transformed_mapping_matrix, visibilities, noise_map
):
    """Compute the hyper_galaxies vector *D* from a transformed util matrix *f* and the 1D image *d* and 1D noise-map *\sigma* \
    (see Warren & Dye 2003).

    Parameters
    -----------
    transformed_mapping_matrix : ndarray
        The matrix representing the transformed mappings between sub-grid pixels and pixelization pixels.
    image : ndarray
        Flattened 1D array of the observed image the inversion is fitting.
    noise_map : ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """

    data_vector = np.zeros(transformed_mapping_matrix.shape[1])

    for vis_1d_index in range(transformed_mapping_matrix.shape[0]):
        for pix_1d_index in range(transformed_mapping_matrix.shape[1]):
            data_vector[pix_1d_index] += (
                visibilities[vis_1d_index]
                * transformed_mapping_matrix[vis_1d_index, pix_1d_index]
                / (noise_map[vis_1d_index] ** 2.0)
            )

    return data_vector


def curvature_matrix_via_transformed_mapping_matrix_from(
    transformed_mapping_matrix, noise_map
):
    """Compute the curvature matrix *F* from a transformed util matrix *f* and the 1D noise-map *\sigma* \
    (see Warren & Dye 2003).

    Parameters
    -----------
    transformed_mapping_matrix : ndarray
        The matrix representing the transformed mappings between sub-grid pixels and pixelization pixels.
    noise_map : ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    flist : ndarray
        NumPy array of floats used to store mappings for efficienctly calculation.
    iflist : ndarray
        NumPy array of integers used to store mappings for efficienctly calculation.
    """

    array = transformed_mapping_matrix / noise_map[:, None]
    curvature_matrix = np.dot(array.T, np.matrix.transpose(array.T))

    return curvature_matrix


def inversion_residual_map_from(
    *,
    pixelization_values,
    data,
    mask_1d_index_for_sub_mask_1d_index,
    all_sub_mask_1d_indexes_for_pixelization_1d_index,
):

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
