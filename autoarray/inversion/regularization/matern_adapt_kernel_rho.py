from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from autoarray.inversion.regularization.matern_kernel import MaternKernel

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.matern_kernel import matern_kernel
from autoarray.inversion.regularization.matern_kernel import matern_cov_matrix_from
from autoarray.inversion.regularization.matern_kernel import inv_via_cholesky


class MaternAdaptKernelRho(MaternKernel):
    def __init__(
        self,
        coefficient: float = 1.0,
        scale: float = 1.0,
        nu: float = 0.5,
        rho: float = 1.0,
    ):
        """
        Regularization which uses a Matern smoothing kernel to regularize the solution with regularization weights
        that adapt to the brightness of the source being reconstructed.

        For this regularization scheme, every pixel is regularized with every other pixel. This contrasts many other
        schemes, where regularization is based on neighboring (e.g. do the pixels share a Delaunay edge?) or computing
        derivatives around the center of the pixel (where nearby pixels are regularization locally in similar ways).

        This makes the regularization matrix fully dense and therefore may change the run times of the solution.
        It also leads to more overall smoothing which can lead to more stable linear inversions.

        For the weighted regularization scheme, each pixel is given an 'effective regularization weight', which is
        applied when each set of pixel neighbors are regularized with one another. The motivation of this is that
        different regions of a pixelization's mesh require different levels of regularization (e.g., high smoothing where the
        no signal is present and less smoothing where it is, see (Nightingale, Dye and Massey 2018)).

        This scheme is not used by Vernardos et al. (2022): https://arxiv.org/abs/2202.09378, but it follows
        a similar approach.

        A full description of regularization and this matrix can be found in the parent `AbstractRegularization` class.

        Parameters
        ----------
        coefficient
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        scale
            The typical scale (correlation length) of the Matérn regularization kernel.
        nu
            Controls the smoothness (differentiability) of the Matérn kernel; ``nu=0.5`` corresponds to an
            exponential (Ornstein–Uhlenbeck) kernel, while a Gaussian covariance is obtained in the limit
            as ``nu`` approaches infinity.
        rho
            Controls how strongly the kernel weights adapt to pixel brightness. Larger values make bright pixels
            receive significantly higher weights (and faint pixels lower weights), while smaller values produce a
            more uniform weighting. Typical values are of order unity (e.g. 0.5–2.0).
        """
        super().__init__(coefficient=coefficient, scale=scale, nu=nu)
        self.rho = rho

    def regularization_weights_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.
        """
        # Assumes linear_obj.pixel_signals_from is xp-aware elsewhere in the codebase.
        pixel_signals = linear_obj.pixel_signals_from(signal_scale=1.0, xp=xp)

        max_signal = xp.max(pixel_signals)
        max_signal = xp.maximum(max_signal, 1e-8)  # avoid divide-by-zero (JAX-safe)

        weights = xp.exp(-self.rho * (1.0 - pixel_signals / max_signal))

        return 1.0 / weights

    def regularization_matrix_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        kernel_weights = 1.0 / self.regularization_weights_from(linear_obj=linear_obj, xp=xp)

        # Follow the xp pattern used in the Matérn kernel module (often `.array` for grids).
        pixel_points = linear_obj.source_plane_mesh_grid.array

        covariance_matrix = matern_cov_matrix_from(
            scale=self.scale,
            pixel_points=pixel_points,
            nu=self.nu,
            weights=kernel_weights,
            xp=xp,
        )

        return self.coefficient * inv_via_cholesky(covariance_matrix, xp=xp)
