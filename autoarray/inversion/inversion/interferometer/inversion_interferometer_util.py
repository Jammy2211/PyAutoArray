import logging
import numpy as np

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
    pass


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
    pass


@numba_util.jit()
def w_tilde_curvature_preload_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    shape_masked_pixels_2d: Tuple[int, int],
    grid_radians_2d: np.ndarray,
) -> np.ndarray:
    pass


@numba_util.jit()
def w_tilde_via_preload_from(w_tilde_preload, native_index_for_slim_index):
    pass


@numba_util.jit()
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


@numba_util.jit()
def curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
    curvature_preload: np.ndarray,
    pix_indexes_for_sub_slim_index: np.ndarray,
    native_index_for_slim_index: np.ndarray,
    pixelization_pixels: int,
) -> np.ndarray:
    pass


@numba_util.jit()
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
