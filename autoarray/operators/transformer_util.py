import numpy as np

from autoarray import numba_util


def preload_real_transforms_from(
    grid_radians: np.ndarray, uv_wavelengths: np.ndarray
) -> np.ndarray:
    """
    Sets up the real preloaded values used by the direct Fourier transform (`TransformerDFT`) to speed up
    the Fourier transform calculations.

    The preloaded values are the cosine terms of every (y,x) radian coordinate on the real-space grid multiplied by
    every `uv_wavelength` value.

    For large numbers of visibilities (> 100000) this array requires large amounts of memory (> 1 GB) and it is
    recommended this preloading is not used.

    Parameters
    ----------
    grid_radians
        The grid in radians corresponding to real-space mask within which the image that is Fourier transformed is
        computed.
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.

    Returns
    -------
    The preloaded values of the cosine terms in the calculation of real entries of the direct Fourier transform.
    """
    # Compute the phase matrix: shape (n_pixels, n_visibilities)
    phase = (
        -2.0
        * np.pi
        * (
            np.outer(grid_radians[:, 1], uv_wavelengths[:, 0])  # y * u
            + np.outer(grid_radians[:, 0], uv_wavelengths[:, 1])  # x * v
        )
    )

    # Compute cosine of the phase matrix
    preloaded_real_transforms = np.cos(phase)

    return preloaded_real_transforms


def preload_imag_transforms_from(
    grid_radians: np.ndarray, uv_wavelengths: np.ndarray
) -> np.ndarray:
    """
    Sets up the imaginary preloaded values used by the direct Fourier transform (`TransformerDFT`) to speed up
    the Fourier transform calculations in interferometric imaging.

    The preloaded values are the sine terms of every (y,x) radian coordinate on the real-space grid multiplied by
    every `uv_wavelength` value. These are used to compute the imaginary components of visibilities.

    For large numbers of visibilities (> 100000), this array can require significant memory (> 1 GB), so preloading
    should be used with care.

    Parameters
    ----------
    grid_radians
        The grid in radians corresponding to the (y,x) coordinates in real space.
    uv_wavelengths
        The (u,v) coordinates in the Fourier plane (in units of wavelengths).

    Returns
    -------
    The sine term preloads used in imaginary-part DFT calculations.
    """
    # Compute the phase matrix: shape (n_pixels, n_visibilities)
    phase = (
        -2.0
        * np.pi
        * (
            np.outer(grid_radians[:, 1], uv_wavelengths[:, 0])  # y * u
            + np.outer(grid_radians[:, 0], uv_wavelengths[:, 1])  # x * v
        )
    )

    # Compute sine of the phase matrix
    preloaded_imag_transforms = np.sin(phase)

    return preloaded_imag_transforms


def visibilities_via_preload_from(
    image_1d: np.ndarray, preloaded_reals: np.ndarray, preloaded_imags: np.ndarray
) -> np.ndarray:
    """
    Computes interferometric visibilities using preloaded real and imaginary DFT transform components.

    This function performs a direct Fourier transform (DFT) using precomputed cosine (real) and sine (imaginary)
    terms. It is used in radio astronomy to compute visibilities from an image for a given interferometric
    observation setup.

    Parameters
    ----------
    image_1d : ndarray of shape (n_pixels,)
        The 1D image vector (real-space brightness values).
    preloaded_reals : ndarray of shape (n_pixels, n_visibilities)
        The preloaded cosine terms (real part of DFT matrix).
    preloaded_imags : ndarray of shape (n_pixels, n_visibilities)
        The preloaded sine terms (imaginary part of DFT matrix).

    Returns
    -------
    visibilities : ndarray of shape (n_visibilities,)
        The complex visibilities computed by summing over all pixels.
    """
    # Perform the dot product between the image and preloaded transform matrices
    vis_real = np.dot(image_1d, preloaded_reals)  # shape (n_visibilities,)
    vis_imag = np.dot(image_1d, preloaded_imags)  # shape (n_visibilities,)

    visibilities = vis_real + 1j * vis_imag

    return visibilities


def visibilities_from(
    image_1d: np.ndarray, grid_radians: np.ndarray, uv_wavelengths: np.ndarray
) -> np.ndarray:
    """
    Compute complex visibilities from an input sky image using the Fourier transform,
    simulating the response of an astronomical radio interferometer.

    This function converts an image defined on a sky coordinate grid into its
    visibility-space representation, given a set of (u,v) spatial frequency
    coordinates (in wavelengths), as sampled by a radio interferometer.

    Parameters
    ----------
    image_1d
        The 1D flattened sky brightness values corresponding to each pixel in the grid.
    grid_radians
        The angular (y, x) positions of each image pixel in radians, matching image_1d.
    uv_wavelengths
        The (u, v) spatial frequencies in units of wavelengths, for each baseline
        of the interferometer.

    Returns
    -------
    visibilities
        The complex visibilities (Fourier components) corresponding to each
        (u, v) coordinate, representing the interferometer’s measurement.
    """

    # Compute the dot product for each pixel-uv pair
    phase = (
        -2.0
        * np.pi
        * (
            np.outer(grid_radians[:, 1], uv_wavelengths[:, 0])
            + np.outer(grid_radians[:, 0], uv_wavelengths[:, 1])
        )
    )  # shape (n_pixels, n_vis)

    # Multiply image values with phase terms
    vis_real = image_1d[:, None] * np.cos(phase)
    vis_imag = image_1d[:, None] * np.sin(phase)

    # Sum over all pixels for each visibility
    visibilities = np.sum(vis_real + 1j * vis_imag, axis=0)

    return visibilities


def image_direct_from(
    visibilities: np.ndarray, grid_radians: np.ndarray, uv_wavelengths: np.ndarray
) -> np.ndarray:
    """
    Reconstruct a real-valued sky image from complex interferometric visibilities
    using an inverse Fourier transform approximation.

    This function simulates the synthesis imaging equation of a radio interferometer
    by summing sinusoidal components across all (u, v) spatial frequencies.

    Parameters
    ----------
    visibilities
        The real and imaginary parts of the complex visibilities for each (u, v) point.

    grid_radians
        The angular (y, x) coordinates of each pixel in radians.

    uv_wavelengths
        The (u, v) spatial frequencies in units of wavelengths for each baseline.

    Returns
    -------
    image_1d
        The reconstructed real-valued image in sky coordinates.
    """
    # Compute the phase term for each (pixel, visibility) pair
    phase = (
        2.0
        * np.pi
        * (
            np.outer(grid_radians[:, 1], uv_wavelengths[:, 0])
            + np.outer(grid_radians[:, 0], uv_wavelengths[:, 1])
        )
    )

    real_part = np.dot(np.cos(phase), visibilities[:, 0])
    imag_part = np.dot(np.sin(phase), visibilities[:, 1])

    image_1d = real_part - imag_part

    return image_1d


@numba_util.jit()
def transformed_mapping_matrix_via_preload_jit_from(
    mapping_matrix, preloaded_reals, preloaded_imags
):
    transfomed_mapping_matrix = 0 + 0j * np.zeros(
        (preloaded_reals.shape[1], mapping_matrix.shape[1])
    )

    for pixel_1d_index in range(mapping_matrix.shape[1]):
        for image_1d_index in range(mapping_matrix.shape[0]):
            value = mapping_matrix[image_1d_index, pixel_1d_index]

            if value > 0:
                for vis_1d_index in range(preloaded_reals.shape[1]):
                    vis_real = value * preloaded_reals[image_1d_index, vis_1d_index]
                    vis_imag = value * preloaded_imags[image_1d_index, vis_1d_index]
                    transfomed_mapping_matrix[vis_1d_index, pixel_1d_index] += (
                        vis_real + 1j * vis_imag
                    )

    return transfomed_mapping_matrix


@numba_util.jit()
def transformed_mapping_matrix_jit(mapping_matrix, grid_radians, uv_wavelengths):
    transfomed_mapping_matrix = 0 + 0j * np.zeros(
        (uv_wavelengths.shape[0], mapping_matrix.shape[1])
    )

    for pixel_1d_index in range(mapping_matrix.shape[1]):
        for image_1d_index in range(mapping_matrix.shape[0]):
            value = mapping_matrix[image_1d_index, pixel_1d_index]

            if value > 0:
                for vis_1d_index in range(uv_wavelengths.shape[0]):
                    vis_real = value * np.cos(
                        -2.0
                        * np.pi
                        * (
                            grid_radians[image_1d_index, 1]
                            * uv_wavelengths[vis_1d_index, 0]
                            + grid_radians[image_1d_index, 0]
                            * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

                    vis_imag = value * np.sin(
                        -2.0
                        * np.pi
                        * (
                            grid_radians[image_1d_index, 1]
                            * uv_wavelengths[vis_1d_index, 0]
                            + grid_radians[image_1d_index, 0]
                            * uv_wavelengths[vis_1d_index, 1]
                        )
                    )

                    transfomed_mapping_matrix[vis_1d_index, pixel_1d_index] += (
                        vis_real + 1j * vis_imag
                    )

    return transfomed_mapping_matrix
