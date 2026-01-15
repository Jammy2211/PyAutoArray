import numpy as np

from autoarray.dataset.abstract.w_tilde import AbstractWTilde
from autoarray.mask.mask_2d import Mask2D


class WTildeInterferometer(AbstractWTilde):
    def __init__(
        self,
        curvature_preload: np.ndarray,
        dirty_image: np.ndarray,
        fft_mask: Mask2D,
        batch_size: int = 128,
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
        fft_mask
            The 2D mask in real-space defining the area where the interferometer data's visibilities are observing
            a signal.
        batch_size
            The size of batches used to compute the w-tilde curvature matrix via FFT-based convolution,
            which can be reduced to produce lower memory usage at the cost of speed.
        """
        super().__init__(
            curvature_preload=curvature_preload, fft_mask=fft_mask
        )

        self.dirty_image = dirty_image

        from autoarray.inversion.inversion.interferometer import (
            inversion_interferometer_util,
        )

        self.fft_state = inversion_interferometer_util.w_tilde_fft_state_from(
            curvature_preload=self.curvature_preload, batch_size=batch_size
        )


