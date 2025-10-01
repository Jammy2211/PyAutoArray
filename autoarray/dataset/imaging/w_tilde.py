import logging
import numpy as np

from autoconf import cached_property

from autoarray.dataset.abstract.w_tilde import AbstractWTilde

from autoarray.inversion.inversion.imaging import inversion_imaging_util
from autoarray.inversion.inversion.imaging import inversion_imaging_numba_util

logger = logging.getLogger(__name__)


class WTildeImaging(AbstractWTilde):
    def __init__(
        self,
        curvature_preload: np.ndarray,
        indexes: np.ndim,
        lengths: np.ndarray,
        noise_map: np.ndarray,
        psf: np.ndarray,
        mask: np.ndarray,
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
        self.noise_map = noise_map
        self.psf = psf
        self.mask = mask

    @cached_property
    def w_matrix(self):
        """
        The matrix `w_tilde_curvature` is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF
        convolution of every pair of image pixels given the noise map. This can be used to efficiently compute the
        curvature matrix via the mappings between image and source pixels, in a way that omits having to perform the
        PSF convolution on every individual source pixel. This provides a significant speed up for inversions of imaging
        datasets.

        The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
        making it impossible to store in memory and its use in linear algebra calculations extremely. The method
        `w_tilde_curvature_preload_imaging_from` describes a compressed representation that overcomes this hurdles. It is
        advised `w_tilde` and this method are only used for testing.

        Parameters
        ----------
        noise_map_native
            The two dimensional masked noise-map of values which w_tilde is computed from.
        kernel_native
            The two dimensional PSF kernel that w_tilde encodes the convolution of.
        native_index_for_slim_index
            An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

        Returns
        -------
        ndarray
            A matrix that encodes the PSF convolution values between the noise map that enables efficient calculation of
            the curvature matrix.
        """

        return inversion_imaging_numba_util.w_tilde_curvature_imaging_from(
            noise_map_native=np.array(self.noise_map.native.array).astype("float64"),
            kernel_native=np.array(self.psf.native.array).astype("float64"),
            native_index_for_slim_index=np.array(
                self.mask.derive_indexes.native_for_slim
            ).astype("int"),
        )

    @cached_property
    def psf_operator_matrix_dense(self):

        return inversion_imaging_util.psf_operator_matrix_dense_from(
            kernel_native=np.array(self.psf.native.array).astype("float64"),
            native_index_for_slim_index=np.array(
                self.mask.derive_indexes.native_for_slim
            ).astype("int"),
            native_shape=self.noise_map.shape_native,
            correlate=False,
        )
