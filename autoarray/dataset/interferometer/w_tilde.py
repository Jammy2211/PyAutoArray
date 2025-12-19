import numpy as np

from autoarray.dataset.abstract.w_tilde import AbstractWTilde
from autoarray.mask.mask_2d import Mask2D


class WTildeInterferometer(AbstractWTilde):
    def __init__(
        self,
        curvature_preload: np.ndarray,
        dirty_image: np.ndarray,
        real_space_mask: Mask2D,
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
        real_space_mask
            The 2D mask in real-space defining the area where the interferometer data's visibilities are observing
            a signal.
        batch_size
            The size of batches used to compute the w-tilde curvature matrix via FFT-based convolution,
            which can be reduced to produce lower memory usage at the cost of speed.
        """
        super().__init__(
            curvature_preload=curvature_preload,
        )

        self.dirty_image = dirty_image
        self.real_space_mask = real_space_mask

        from autoarray.inversion.inversion.interferometer import (
            inversion_interferometer_util,
        )

        self.fft_state = inversion_interferometer_util.w_tilde_fft_state_from(
            curvature_preload=self.curvature_preload, batch_size=batch_size
        )

    @property
    def mask_rectangular_w_tilde(self) -> np.ndarray:
        """
        Returns a rectangular boolean mask that tightly bounds the unmasked region
        of the interferometer mask.

        This rectangular mask is used for computing the W-tilde curvature matrix
        via FFT-based convolution, which requires a full rectangular grid.

        Pixels outside the bounding box of the original mask are set to True
        (masked), and pixels inside are False (unmasked).

        Returns
        -------
        np.ndarray
            Boolean mask of shape (Ny, Nx), where False denotes unmasked pixels.
        """
        mask = self.real_space_mask

        ys, xs = np.where(~mask)

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        rect_mask = np.ones(mask.shape, dtype=bool)
        rect_mask[y_min : y_max + 1, x_min : x_max + 1] = False

        return rect_mask

    @property
    def rect_index_for_mask_index(self) -> np.ndarray:
        """
        Mapping from masked-grid pixel indices to rectangular-grid pixel indices.

        This array enables extraction of a curvature matrix computed on a full
        rectangular grid back to the original masked grid.

        If:
            - C_rect is the curvature matrix computed on the rectangular grid
            - idx = rect_index_for_mask_index

        then the masked curvature matrix is:
            C_mask = C_rect[idx[:, None], idx[None, :]]

        Returns
        -------
        np.ndarray
            Array of shape (N_masked_pixels,), where each entry gives the
            corresponding index in the rectangular grid (row-major order).
        """
        mask = self.real_space_mask
        rect_mask = self.mask_rectangular_w_tilde

        # Bounding box of the rectangular region
        ys, xs = np.where(~rect_mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        rect_width = x_max - x_min + 1

        # Coordinates of unmasked pixels in the original mask (slim order)
        mask_ys, mask_xs = np.where(~mask)

        # Convert (y, x) → rectangular flat index
        rect_indices = ((mask_ys - y_min) * rect_width + (mask_xs - x_min)).astype(
            np.int32
        )

        return rect_indices
