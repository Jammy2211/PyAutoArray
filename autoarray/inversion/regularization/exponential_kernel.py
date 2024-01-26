from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray import numba_util


@numba_util.jit()
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
        self.coefficient = coefficient
        self.scale = scale

        super().__init__()

    def regularization_matrix_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        points: the position of mesh that is regularized
        """

        covariance_matrix = exp_cov_matrix_from(
            scale=self.scale, pixel_points=linear_obj.source_plane_mesh_grid
        )

        return self.coefficient * np.linalg.inv(covariance_matrix)
