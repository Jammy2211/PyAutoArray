import copy
import logging
import numpy as np

from autoarray.dataset.abstract.w_tilde import AbstractWTilde

logger = logging.getLogger(__name__)


class WTildeImaging(AbstractWTilde):
    def __init__(
        self,
        curvature_preload: np.ndarray,
        indexes: np.ndim,
        lengths: np.ndarray,
        noise_map_value: float,
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
        noise_map_value
            The first value of the noise-map used to construct the curvature preload, which is used as a sanity
            check when performing the inversion to ensure the preload corresponds to the data being fitted.
        """
        super().__init__(
            curvature_preload=curvature_preload, noise_map_value=noise_map_value
        )

        self.indexes = indexes
        self.lengths = lengths
