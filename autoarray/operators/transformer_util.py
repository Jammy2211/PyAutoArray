import numpy as np


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
    image_1d: np.ndarray,
    preloaded_reals: np.ndarray,
    preloaded_imags: np.ndarray,
    xp=np,
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
    vis_real = xp.dot(image_1d, preloaded_reals)  # shape (n_visibilities,)
    vis_imag = xp.dot(image_1d, preloaded_imags)  # shape (n_visibilities,)

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


def transformed_mapping_matrix_via_preload_from(
    mapping_matrix: np.ndarray, preloaded_reals: np.ndarray, preloaded_imags: np.ndarray
) -> np.ndarray:
    """
    Computes the Fourier-transformed mapping matrix using preloaded sine and cosine terms for efficiency.

    This function transforms each source pixel's mapping to visibilities by using precomputed
    real (cosine) and imaginary (sine) terms from the direct Fourier transform.
    It is used in radio interferometric imaging where source-to-image mappings are projected
    into the visibility space.

    Parameters
    ----------
    mapping_matrix
        The mapping matrix from image-plane pixels to source-plane pixels.
    preloaded_reals
        Precomputed cosine terms for each pixel-vis pair: cos(-2π(yu + xv)).
    preloaded_imags
        Precomputed sine terms for each pixel-vis pair: sin(-2π(yu + xv)).

    Returns
    -------
    Complex-valued matrix mapping source pixels to visibilities.
    """

    # Broadcasted multiplication and matrix multiplication over non-zero entries

    vis_real = preloaded_reals.T @ mapping_matrix  # (n_visibilities, n_source_pixels)
    vis_imag = preloaded_imags.T @ mapping_matrix

    transformed_matrix = vis_real + 1j * vis_imag

    return transformed_matrix


def transformed_mapping_matrix_from(
    mapping_matrix: np.ndarray, grid_radians: np.ndarray, uv_wavelengths: np.ndarray
) -> np.ndarray:
    """
    Computes the Fourier-transformed mapping matrix used in radio interferometric imaging.

    This function applies a direct Fourier transform to each pixel column of the mapping matrix using the
    uv-wavelength coordinates. The result is a matrix that maps source pixel intensities to complex visibilities,
    which represent how a model image would appear to an interferometer.

    Parameters
    ----------
    mapping_matrix : ndarray of shape (n_image_pixels, n_source_pixels)
        The mapping matrix from image-plane pixels to source-plane pixels.
    grid_radians : ndarray of shape (n_image_pixels, 2)
        The (y,x) positions of each image pixel in radians.
    uv_wavelengths : ndarray of shape (n_visibilities, 2)
        The (u,v) coordinates of the sampled Fourier modes in units of wavelength.

    Returns
    -------
    transformed_matrix : ndarray of shape (n_visibilities, n_source_pixels)
        The transformed mapping matrix in the visibility domain (complex-valued).
    """
    # Compute phase term: (n_image_pixels, n_visibilities)
    phase = (
        -2.0
        * np.pi
        * (
            np.outer(grid_radians[:, 1], uv_wavelengths[:, 0])  # y * u
            + np.outer(grid_radians[:, 0], uv_wavelengths[:, 1])  # x * v
        )
    )

    # Compute real and imaginary Fourier matrices
    fourier_real = np.cos(phase)
    fourier_imag = np.sin(phase)

    # Only compute contributions from non-zero mapping entries
    # This matrix multiplication is: (n_visibilities x n_image_pixels) dot (n_image_pixels x n_source_pixels)
    vis_real = fourier_real.T @ mapping_matrix  # (n_vis, n_src)
    vis_imag = fourier_imag.T @ mapping_matrix  # (n_vis, n_src)

    transformed_matrix = vis_real + 1j * vis_imag

    return transformed_matrix
