from __future__ import annotations
import jax.numpy as jnp
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization

def gauss_cov_matrix_from(
    scale: float,
    pixel_points: jnp.ndarray,  # shape (N, 2)
) -> jnp.ndarray:
    """
    Construct the source‐pixel Gaussian covariance matrix for regularization.

    For N source‐pixels at coordinates (y_i, x_i), we define

      C_ij = exp( -||p_i - p_j||^2 / (2 scale^2) )

    plus a tiny diagonal “jitter” (1e-8) to ensure numerical stability.

    Parameters
    ----------
    scale
        The characteristic length scale of the Gaussian kernel.
    pixel_points
        Array of shape (N, 2), giving the (y, x) coordinates of each source pixel.

    Returns
    -------
    cov : jnp.ndarray, shape (N, N)
        The Gaussian covariance matrix.
    """
    # Ensure array:
    pts = jnp.asarray(pixel_points)               # (N, 2)
    # Compute squared distances: ||p_i - p_j||^2
    diffs = pts[:, None, :] - pts[None, :, :]     # (N, N, 2)
    d2   = jnp.sum(diffs**2, axis=-1)             # (N, N)

    # Gaussian kernel
    cov  = jnp.exp(-d2 / (2.0 * scale**2))        # (N, N)

    # Add tiny jitter on the diagonal
    N    = pts.shape[0]
    cov  = cov + jnp.eye(N, dtype=cov.dtype) * 1e-8

    return cov


class GaussianKernel(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0, scale: float = 1.0):
        """
        Regularization which uses a Gaussian smoothing kernel to regularize the solution.

        For this regularization scheme, every pixel is regularized with every other pixel. This contrasts many other
        schemes, where regularization is based on neighboring (e.g. do the pixels share a Voronoi edge?) or computing
        derivates around the center of the pixel (where nearby pixels are regularization locally in similar ways).

        This makes the regularization matrix fully dense and therefore maybe change the run times of the solution.
        It also leads to more overall smoothing which can lead to more stable linear inversions.

        This scheme is introduced by Vernardos et al. (2022): https://arxiv.org/abs/2202.09378

        A full description of regularization and this matrix can be found in the parent `AbstractRegularization` class.

        Parameters
        ----------
        coefficient
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        scale
            The typical scale of the exponential regularization pattern.
        """
        self.coefficient = coefficient
        self.scale = scale
        super().__init__()

    def regularization_weights_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.

        The regularization weights define the level of regularization applied to each parameter in the linear object
        (e.g. the ``pixels`` in a ``Mapper``).

        For standard regularization (e.g. ``Constant``) are weights are equal, however for adaptive schemes
        (e.g. ``AdaptiveBrightness``) they vary to adapt to the data being reconstructed.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses these weights when performing regularization.

        Returns
        -------
        The regularization weights.
        """
        return self.coefficient * np.ones(linear_obj.params)

    def regularization_matrix_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        Returns the regularization matrix with shape [pixels, pixels].

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses this matrix to perform regularization.

        Returns
        -------
        The regularization matrix.
        """
        covariance_matrix = gauss_cov_matrix_from(
            scale=self.scale,
            pixel_points=linear_obj.source_plane_mesh_grid.array
        )

        return self.coefficient * jnp.linalg.inv(covariance_matrix)
