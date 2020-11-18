from autoarray import decorator_util

import numpy as np


@decorator_util.jit()
def preload_real_transforms(
    grid_radians: np.ndarray, uv_wavelengths: np.ndarray
) -> np.ndarray:
    """
    Sets up the real preloaded values used by the direct fourier transform (`TransformerDFT`) to speed up
    the Fourier transform calculations.

    The preloaded values are the cosine terms of every (y,x) radian coordinate on the real-space grid multiplied by
    everu `uv_wavelength` value.

    For large numbers of visibilities (> 100000) this array requires large amounts of memory ( > 1 GB) and it is
    recommended this preloading is not used.

    Parameters
    ----------
    grid_radians : np.ndarray
        The grid in radians corresponding to real-space mask within which the image that is Fourier transformed is
        computed.
    uv_wavelengths : np.ndarray
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.

    Returns
    -------
    np.ndarray
        The preloaded values of the cosine terms in the calculation of real entries of the direct Fourier transform.

    """

    preloaded_real_transforms = np.zeros(
        shape=(grid_radians.shape[0], uv_wavelengths.shape[0])
    )

    for image_1d_index in range(grid_radians.shape[0]):
        for vis_1d_index in range(uv_wavelengths.shape[0]):
            preloaded_real_transforms[image_1d_index, vis_1d_index] += np.cos(
                -2.0
                * np.pi
                * (
                    grid_radians[image_1d_index, 1] * uv_wavelengths[vis_1d_index, 0]
                    + grid_radians[image_1d_index, 0] * uv_wavelengths[vis_1d_index, 1]
                )
            )

    return preloaded_real_transforms


@decorator_util.jit()
def preload_imag_transforms(grid_radians, uv_wavelengths):

    preloaded_imag_transforms = np.zeros(
        shape=(grid_radians.shape[0], uv_wavelengths.shape[0])
    )

    for image_1d_index in range(grid_radians.shape[0]):
        for vis_1d_index in range(uv_wavelengths.shape[0]):
            preloaded_imag_transforms[image_1d_index, vis_1d_index] += np.sin(
                -2.0
                * np.pi
                * (
                    grid_radians[image_1d_index, 1] * uv_wavelengths[vis_1d_index, 0]
                    + grid_radians[image_1d_index, 0] * uv_wavelengths[vis_1d_index, 1]
                )
            )

    return preloaded_imag_transforms


@decorator_util.jit()
def real_visibilities_via_preload_jit_from(image_1d, preloaded_reals):

    real_visibilities = np.zeros(shape=(preloaded_reals.shape[1]))

    for image_1d_index in range(image_1d.shape[0]):
        for vis_1d_index in range(preloaded_reals.shape[1]):
            real_visibilities[vis_1d_index] += (
                image_1d[image_1d_index] * preloaded_reals[image_1d_index, vis_1d_index]
            )

    return real_visibilities


@decorator_util.jit()
def real_visibilities_jit(image_1d, grid_radians, uv_wavelengths):

    real_visibilities = np.zeros(shape=(uv_wavelengths.shape[0]))

    for image_1d_index in range(image_1d.shape[0]):
        for vis_1d_index in range(uv_wavelengths.shape[0]):
            real_visibilities[vis_1d_index] += image_1d[image_1d_index] * np.cos(
                -2.0
                * np.pi
                * (
                    grid_radians[image_1d_index, 1] * uv_wavelengths[vis_1d_index, 0]
                    + grid_radians[image_1d_index, 0] * uv_wavelengths[vis_1d_index, 1]
                )
            )

    return real_visibilities


@decorator_util.jit()
def imag_visibilities_from_via_preload_jit_from(image_1d, preloaded_imags):

    imag_visibilities = np.zeros(shape=(preloaded_imags.shape[1]))

    for image_1d_index in range(image_1d.shape[0]):
        for vis_1d_index in range(preloaded_imags.shape[1]):
            imag_visibilities[vis_1d_index] += (
                image_1d[image_1d_index] * preloaded_imags[image_1d_index, vis_1d_index]
            )

    return imag_visibilities


@decorator_util.jit()
def imag_visibilities_jit(image_1d, grid_radians, uv_wavelengths):

    imag_visibilities = np.zeros(shape=(uv_wavelengths.shape[0]))

    for image_1d_index in range(image_1d.shape[0]):
        for vis_1d_index in range(uv_wavelengths.shape[0]):
            imag_visibilities[vis_1d_index] += image_1d[image_1d_index] * np.sin(
                -2.0
                * np.pi
                * (
                    grid_radians[image_1d_index, 1] * uv_wavelengths[vis_1d_index, 0]
                    + grid_radians[image_1d_index, 0] * uv_wavelengths[vis_1d_index, 1]
                )
            )

    return imag_visibilities


@decorator_util.jit()
def real_transformed_mapping_matrix_via_preload_jit_from(
    mapping_matrix, preloaded_reals
):

    transfomed_mapping_matrix = np.zeros(
        (preloaded_reals.shape[1], mapping_matrix.shape[1])
    )

    for pixel_1d_index in range(mapping_matrix.shape[1]):
        for image_1d_index in range(mapping_matrix.shape[0]):

            value = mapping_matrix[image_1d_index, pixel_1d_index]

            if value > 0:

                for vis_1d_index in range(preloaded_reals.shape[1]):
                    transfomed_mapping_matrix[vis_1d_index, pixel_1d_index] += (
                        value * preloaded_reals[image_1d_index, vis_1d_index]
                    )

    return transfomed_mapping_matrix


@decorator_util.jit()
def real_transformed_mapping_matrix_jit(mapping_matrix, grid_radians, uv_wavelengths):

    transfomed_mapping_matrix = np.zeros(
        (uv_wavelengths.shape[0], mapping_matrix.shape[1])
    )

    for pixel_1d_index in range(mapping_matrix.shape[1]):
        for image_1d_index in range(mapping_matrix.shape[0]):

            value = mapping_matrix[image_1d_index, pixel_1d_index]

            if value > 0:

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    transfomed_mapping_matrix[vis_1d_index, pixel_1d_index] += (
                        value
                        * np.cos(
                            -2.0
                            * np.pi
                            * (
                                grid_radians[image_1d_index, 1]
                                * uv_wavelengths[vis_1d_index, 0]
                                + grid_radians[image_1d_index, 0]
                                * uv_wavelengths[vis_1d_index, 1]
                            )
                        )
                    )

    return transfomed_mapping_matrix


@decorator_util.jit()
def imag_transformed_mapping_matrix_via_preload_jit_from(
    mapping_matrix, preloaded_imags
):

    transfomed_mapping_matrix = np.zeros(
        (preloaded_imags.shape[1], mapping_matrix.shape[1])
    )

    for pixel_1d_index in range(mapping_matrix.shape[1]):
        for image_1d_index in range(mapping_matrix.shape[0]):

            value = mapping_matrix[image_1d_index, pixel_1d_index]

            if value > 0:

                for vis_1d_index in range(preloaded_imags.shape[1]):
                    transfomed_mapping_matrix[vis_1d_index, pixel_1d_index] += (
                        value * preloaded_imags[image_1d_index, vis_1d_index]
                    )

    return transfomed_mapping_matrix


@decorator_util.jit()
def imag_transformed_mapping_matrix_jit(mapping_matrix, grid_radians, uv_wavelengths):

    transfomed_mapping_matrix = np.zeros(
        (uv_wavelengths.shape[0], mapping_matrix.shape[1])
    )

    for pixel_1d_index in range(mapping_matrix.shape[1]):
        for image_1d_index in range(mapping_matrix.shape[0]):

            value = mapping_matrix[image_1d_index, pixel_1d_index]

            if value > 0:

                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    transfomed_mapping_matrix[vis_1d_index, pixel_1d_index] += (
                        value
                        * np.sin(
                            -2.0
                            * np.pi
                            * (
                                grid_radians[image_1d_index, 1]
                                * uv_wavelengths[vis_1d_index, 0]
                                + grid_radians[image_1d_index, 0]
                                * uv_wavelengths[vis_1d_index, 1]
                            )
                        )
                    )

    return transfomed_mapping_matrix
