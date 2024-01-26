from __future__ import annotations
import numpy as np

import math
import scipy.special as sc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray import numba_util


@numba_util.jit(cache=False)
def matern_kernel(r: float, l: float = 1.0, v: float = 0.5):
    """
    need to `pip install numba-scipy `
    see https://gaussianprocess.org/gpml/chapters/RW4.pdf for more info

    the distance r need to be scalar
    l is the scale
    v is the order, better < 30, otherwise may have numerical NaN issue.

    v control the smoothness level. the larger the v, the stronger smoothing condition (i.e., the solution is
    v-th differentiable) imposed by the kernel.
    """
    r = abs(r)
    if r == 0:
        r = 0.00000001
    part1 = 2 ** (1 - v) / math.gamma(v)
    part2 = (math.sqrt(2 * v) * r / l) ** v
    part3 = sc.kv(v, math.sqrt(2 * v) * r / l)
    return part1 * part2 * part3


@numba_util.jit(cache=False)
def matern_cov_matrix_from(
    scale: float,
    nu: float,
    pixel_points: np.ndarray,
) -> np.ndarray:
    """
    Consutruct the regularization covariance matrix, which is used to determined the regularization pattern (i.e,
    how the different pixels are correlated).

    the covariance matrix includes two non-linear parameters, the scale coefficient, which is used to determine
    the typical scale of the regularization pattern. The smoothness order parameters mu, whose value determie
    the inversion solution is mu-th differentiable.

    Parameters
    ----------
    scale
        the typical scale of the regularization pattern .
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

            covariance_matrix[i, j] += matern_kernel(d_ij, l=scale, v=nu)

    return covariance_matrix


class NumbaScipyPlaceholder:
    pass


try:
    import numba_scipy
    numba_scipy = object
except ModuleNotFoundError:
    numba_scipy = NumbaScipyPlaceholder()


def numba_scipy_exception():
    raise ModuleNotFoundError(
        "\n--------------------\n"
        "You are attempting to use the MaternKernel for Regularization.\n\n"
        "However, the optional library numba_scipy (https://pypi.org/project/numba-scipy/) is not installed.\n\n"
        "Install it via the command `pip install numba-scipy==0.3.1`.\n\n"
        "----------------------"
    )


class MaternKernel(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0, scale: float = 1.0, nu: float = 0.5):

        if isinstance(numba_scipy, NumbaScipyPlaceholder):
            numba_scipy_exception()

        self.coefficient = coefficient
        self.scale = float(scale)
        self.nu = float(nu)
        super().__init__()

    def regularization_matrix_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        points: the position of mesh that is regularized
        """
        covariance_matrix = matern_cov_matrix_from(
            scale=self.scale,
            pixel_points=linear_obj.source_plane_mesh_grid,
            nu=self.nu,
        )

        return self.coefficient * np.linalg.inv(covariance_matrix)
