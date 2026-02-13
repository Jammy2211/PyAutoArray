from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from autoarray.inversion.regularization.matern_kernel import MaternKernel

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.matern_kernel import matern_kernel


def matern_cov_matrix_from(
    scale: float,
    nu: float,
    pixel_points,
    weights=None,
    xp=np,
):
    """
    Construct the regularization covariance matrix (N x N) using a Matérn kernel,
    optionally modulated by per-pixel weights.

    If `weights` is provided (shape [N]), the covariance is:
        C_ij = K(d_ij; scale, nu) * w_i * w_j
    with a small diagonal jitter added for numerical stability.

    Parameters
    ----------
    scale
        Typical correlation length of the Matérn kernel.
    nu
        Smoothness parameter of the Matérn kernel.
    pixel_points
        Array-like of shape [N, 2] with (y, x) coordinates (or any 2D coords; only distances matter).
    weights
        Optional array-like of shape [N]. If None, treated as all ones.
    xp
        Backend (numpy or jax.numpy).

    Returns
    -------
    covariance_matrix
        Array of shape [N, N].
    """

    # --------------------------------
    # Pairwise distances (broadcasted)
    # --------------------------------
    diff = pixel_points[:, None, :] - pixel_points[None, :, :]  # (N, N, 2)
    d_ij = xp.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)  # (N, N)

    # --------------------------------
    # Base Matérn covariance
    # --------------------------------
    covariance_matrix = matern_kernel(d_ij, l=scale, v=nu, xp=xp)  # (N, N)

    # --------------------------------
    # Apply weights: C_ij *= w_i * w_j
    # (broadcasted outer product, JAX-safe)
    # --------------------------------
    if weights is not None:
        w = xp.asarray(weights)
        # Ensure shape (N,) -> outer product (N,1)*(1,N) -> (N,N)
        covariance_matrix = covariance_matrix * (w[:, None] * w[None, :])

    # --------------------------------
    # Add diagonal jitter (JAX-safe)
    # --------------------------------
    pixels = pixel_points.shape[0]
    covariance_matrix = covariance_matrix + 1e-8 * xp.eye(pixels)

    return covariance_matrix


class MaternAdaptiveBrightnessKernel(MaternKernel):
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
        derivates around the center of the pixel (where nearby pixels are regularization locally in similar ways).

        This makes the regularization matrix fully dense and therefore maybe change the run times of the solution.
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
            The typical scale of the exponential regularization pattern.
        nu
            Controls the derivative of the regularization pattern (`nu=0.5` is a Gaussian).
        rho
            Controls how strongly the kernel weights adapt to pixel brightness. Larger values make bright pixels
            receive significantly higher weights (and faint pixels lower weights), while smaller values produce a
            more uniform weighting. Typical values are of order unity (e.g. 0.5–2.0).
        """
        super().__init__(coefficient=coefficient, scale=scale, nu=nu)
        self.rho = rho

    def covariance_kernel_weights_from(
        self, linear_obj: LinearObj, xp=np
    ) -> np.ndarray:
        """
        Returns per-pixel kernel weights that adapt to the reconstructed pixel brightness.
        """
        # Assumes linear_obj.pixel_signals_from is xp-aware elsewhere in the codebase.
        pixel_signals = linear_obj.pixel_signals_from(signal_scale=1.0, xp=xp)

        max_signal = xp.max(pixel_signals)
        max_signal = xp.maximum(max_signal, 1e-8)  # avoid divide-by-zero (JAX-safe)

        return xp.exp(-self.rho * (1.0 - pixel_signals / max_signal))

    def regularization_matrix_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        kernel_weights = self.covariance_kernel_weights_from(
            linear_obj=linear_obj, xp=xp
        )

        # Follow the xp pattern used in the Matérn kernel module (often `.array` for grids).
        pixel_points = linear_obj.source_plane_mesh_grid.array

        covariance_matrix = matern_cov_matrix_from(
            scale=self.scale,
            pixel_points=pixel_points,
            nu=self.nu,
            weights=kernel_weights,
            xp=xp,
        )

        return self.coefficient * xp.linalg.inv(covariance_matrix)

    def regularization_weights_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.
        """
        return 1.0 / self.covariance_kernel_weights_from(linear_obj=linear_obj, xp=xp)
