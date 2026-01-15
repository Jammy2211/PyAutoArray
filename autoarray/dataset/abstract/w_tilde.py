import numpy as np

from autoarray import exc


class AbstractWTilde:
    def __init__(self, curvature_preload : np.ndarray, fft_mask: np.ndarray):
        """
        Packages together all derived data quantities necessary to fit `data (e.g. `Imaging`, Interferometer`) using
        an ` Inversion` via the w_tilde formalism.

        The w_tilde formalism performs linear algebra formalism in a way that speeds up the construction of the
        simultaneous linear equations by bypassing the construction of a `mapping_matrix` and precomputing
        operations like blurring or a Fourier transform.

        Parameters
        ----------
        curvature_preload
            A matrix which uses the imaging's noise-map and PSF to preload as much of the computation of the
            curvature matrix as possible.
        """
        self.curvature_preload = curvature_preload
        self.fft_mask = fft_mask

    @property
    def fft_index_for_masked_pixel(self) -> np.ndarray:
        """
        Return a mapping from masked-pixel (slim) indices to flat indices
        on the rectangular FFT grid.

        This array is used to translate between:

            - "masked pixel space" (a compact 1D indexing over unmasked pixels)
            - the 2D rectangular grid on which FFT-based convolutions are performed

        The FFT grid is assumed to be rectangular and already suitable for FFTs
        (e.g. padded and centered appropriately). Masked pixels are present on
        this grid but are ignored in computations via zero-weighting.

        Returns
        -------
        np.ndarray
            A 1D array of shape (N_unmasked,), where element `i` gives the flat
            (row-major) index into the FFT grid corresponding to the `i`-th
            unmasked pixel in slim ordering.

        Notes
        -----
        - The slim ordering is defined as the order returned by `np.where(~mask)`.
        - The flat FFT index is computed assuming row-major (C-style) ordering:
              flat_index = y * width + x
        - This method is intentionally backend-agnostic and can be used by both
          imaging and interferometer curvature pipelines.
        """

        # Boolean mask defined on the rectangular FFT grid
        # True  = masked pixel
        # False = unmasked pixel
        mask_fft = self.fft_mask

        # Coordinates of unmasked pixels in the FFT grid
        ys, xs = np.where(~mask_fft)

        # Width of the FFT grid (number of columns)
        width = mask_fft.shape[1]

        # Convert (y, x) coordinates to flat row-major indices
        return (ys * width + xs).astype(np.int32)