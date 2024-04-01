from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray import numba_util


#@numba_util.jit()
def exp_cov_matrix_from(
    scale: float,
    pixel_points: np.ndarray,
) -> np.ndarray:
    """
    Consutruct the source brightness covariance matrix, which is used to determined the regularization
    pattern (i.e, how the different  source pixels are smoothed).

    The covariance matrix includes one non-linear parameters, the scale coefficient, which is used to determine
    the typical scale of the regularization pattern.

    Parameters
    ----------
    scale
        The typical scale of the regularization pattern .
    pixel_points
        An 2d array with shape [N_source_pixels, 2], which save the source pixelization coordinates (on source plane).
        Something like [[y1,x1], [y2,x2], ...]

    Returns
    -------
    np.ndarray
        The source covariance matrix (2d array), shape [N_source_pixels, N_source_pixels].
    """

    pixels = len(pixel_points)
    covariance_matrix = np.zeros(shape=(pixels, pixels))

    for i in range(pixels):
        covariance_matrix[i, i] += 1e-8
        for j in range(pixels):
            xi = pixel_points[i, 1]
            yi = pixel_points[i, 0]
            xj = pixel_points[j, 1]
            yj = pixel_points[j, 0]
            d_ij = np.sqrt(
                (xi - xj) ** 2 + (yi - yj) ** 2
            )  # distance between the pixel i and j

            covariance_matrix[i, j] += np.exp(-1.0 * d_ij / scale)

    return covariance_matrix


class ExponentialKernel(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0, scale: float = 1.0):
        """
        Regularization which uses an Exponential smoothing kernel to regularize the solution.

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
        covariance_matrix = exp_cov_matrix_from(
            scale=self.scale,
            pixel_points=np.array(linear_obj.source_plane_mesh_grid),
        )

        return self.coefficient * np.linalg.inv(covariance_matrix)
