import logging
import numpy as np

from autoarray.dataset.abstract.w_tilde import AbstractWTilde

from autoarray.inversion.inversion.imaging import inversion_imaging_util
from autoarray.inversion.inversion.imaging import inversion_imaging_numba_util

logger = logging.getLogger(__name__)


class WTildeImaging(AbstractWTilde):
    def __init__(
        self,
        data: np.ndarray,
        noise_map: np.ndarray,
        psf: np.ndarray,
        fft_mask: np.ndarray,
    ):
        """
        Packages together all derived data quantities necessary to fit `Imaging` data using an ` Inversion` via the
        w_tilde formalism.

        The w_tilde formalism performs linear algebra formalism in a way that speeds up the construction of the
        simultaneous linear equations by bypassing the construction of a `mapping_matrix` and precomputing the
        blurring operations performed using the imaging's PSF.

        Parameters
        ----------
        curvature_preload
            A matrix which uses the imaging's noise-map and PSF to preload as much of the computation of the
            curvature matrix as possible.
        indexes
            The image-pixel indexes of the curvature preload matrix, which are used to compute the curvature matrix
            efficiently when performing an inversion.
        lengths
            The lengths of how many indexes each curvature preload contains, again used to compute the curvature
            matrix efficienctly.
        """
        super().__init__(
            curvature_preload=None,
            fft_mask=fft_mask
        )

        self.data = data
        self.noise_map = noise_map
        self.psf = psf

        self.data_native = data.native
        self.noise_map_native = noise_map.native

    @property
    def psf_operator_matrix_dense(self):

        return inversion_imaging_util.psf_operator_matrix_dense_from(
            kernel_native=self.psf.native.array,
            native_index_for_slim_index=np.array(
                self.mask.derive_indexes.native_for_slim
            ).astype("int"),
            native_shape=self.noise_map.shape_native,
            correlate=False,
        )
