import logging
import numpy as np

from autoarray.dataset.abstract.w_tilde import AbstractWTilde
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray.inversion.inversion.imaging import inversion_imaging_util

logger = logging.getLogger(__name__)


class WTildeImaging(AbstractWTilde):
    def __init__(
        self,
        data: np.ndarray,
        noise_map: np.ndarray,
        psf: np.ndarray,
        fft_mask: np.ndarray,
        batch_size: int = 128,
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
        super().__init__(curvature_preload=None, fft_mask=fft_mask)

        self.data = data
        self.noise_map = noise_map
        self.psf = psf

        self.data_native = data.native
        self.noise_map_native = noise_map.native

        inverse_noise_variances = 1.0 / noise_map ** 2
        inverse_noise_variances = Array2D(values=inverse_noise_variances, mask=data.mask)
        self.inverse_noise_variances_native = inverse_noise_variances.native

        import jax.numpy as jnp

#         self.inv_noise_var = jnp.asarray(self.inv_noise_var, dtype=jnp.float64)

        self.curvature_matrix_diag_func = (
            inversion_imaging_util.curvature_matrix_diag_via_w_tilde_from_func(
                psf=self.psf.native.array,
                y_shape=data.shape_native[0],
                x_shape=data.shape_native[1],
            )
        )

        self.curvature_matrix_off_diag_func = inversion_imaging_util.build_curvature_matrix_off_diag_via_w_tilde_from_func(
            psf=self.psf.native.array,
            y_shape=data.shape_native[0],
            x_shape=data.shape_native[1],
        )

        self.curvature_matrix_off_diag_light_profiles_func = inversion_imaging_util.build_curvature_matrix_off_diag_with_light_profiles_via_w_tilde_from_func(
            psf=self.psf.native.array,
            y_shape=data.shape_native[0],
            x_shape=data.shape_native[1],
        )

        self.batch_size = batch_size
