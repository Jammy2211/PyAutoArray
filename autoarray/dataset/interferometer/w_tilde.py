import numpy as np

from autoarray.dataset.abstract.w_tilde import AbstractWTilde
from autoarray.mask.mask_2d import Mask2D


class WTildeInterferometer(AbstractWTilde):
    def __init__(
        self,
        w_matrix: np.ndarray,
        curvature_preload: np.ndarray,
        dirty_image: np.ndarray,
        real_space_mask: Mask2D,
        noise_map_value: float,
    ):
        """
        Packages together all derived data quantities necessary to fit `Interferometer` data using an ` Inversion` via
        the w_tilde formalism.

        The w_tilde formalism performs linear algebra formalism in a way that speeds up the construction of  the
        simultaneous linear equations by bypassing the construction of a `mapping_matrix` and precomputing the
        Fourier transform operations performed using the interferometer's `uv_wavelengths`.

        Parameters
        ----------
        w_matrix
            The w_tilde matrix used by the w-tilde formalism to construct the data vector and
            curvature matrix during an inversion efficiently..
        curvature_preload
            A matrix which uses the interferometer `uv_wavelengths` to preload as much of the computation of the
            curvature matrix as possible.
        dirty_image
            The real-space image of the visibilities computed via the transform, which is used to construct the
            curvature matrix.
        real_space_mask
            The 2D mask in real-space defining the area where the interferometer data's visibilities are observing
            a signal.
        noise_map_value
            The first value of the noise-map used to construct the curvature preload, which is used as a sanity
            check when performing the inversion to ensure the preload corresponds to the data being fitted.
        """
        super().__init__(
            curvature_preload=curvature_preload, noise_map_value=noise_map_value
        )

        self.dirty_image = dirty_image
        self.real_space_mask = real_space_mask

        self.w_matrix = w_matrix
