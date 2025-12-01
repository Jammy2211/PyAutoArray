from functools import partial
import numpy as np
from typing import Tuple


def forward_interp_np(xp, yp, x):
    """
    xp: (N, M)
    yp: (N, M)
    x : (M,)  ← one x per column
    """

    if yp.ndim == 1 and xp.ndim == 2:
        yp = np.broadcast_to(yp[:, None], xp.shape)

    K, M = x.shape

    out = np.empty((K, 2), dtype=xp.dtype)

    for j in range(2):
        out[:, j] = np.interp(x[:, j], xp[:, j], yp[:, j], left=0, right=1)

    return out


def reverse_interp_np(xp, yp, x):
    """
    xp : (N,) or (N, M)
    yp : (N, M)
    x  : (K, M)   query points per column
    """

    # Ensure xp is 2D: (N, M)
    if xp.ndim == 1 and yp.ndim == 2:  # (N, 1)
        xp = np.broadcast_to(xp[:, None], yp.shape)

    # Shapes
    K, M = x.shape

    # Output
    out = np.empty((K, 2), dtype=yp.dtype)

    # Column-wise interpolation (cannot avoid this loop in pure NumPy)
    for j in range(2):
        out[:, j] = np.interp(x[:, j], xp[:, j], yp[:, j])

    return out


def create_transforms(traced_points, mesh_weight_map=None):

    N = traced_points.shape[0]  # // 2

    if mesh_weight_map is None:
        t = np.arange(1, N + 1) / (N + 1)
        t = np.stack([t, t], axis=1)
        sort_points = np.sort(traced_points, axis=0)  # [::2]
    else:
        sdx = np.argsort(traced_points, axis=0)
        sort_points = np.take_along_axis(traced_points, sdx, axis=0)
        t = np.stack([mesh_weight_map, mesh_weight_map], axis=1)
        t = np.take_along_axis(t, sdx, axis=0)
        t = np.cumsum(t, axis=0)

    transform = partial(forward_interp_np, sort_points, t)
    inv_transform = partial(reverse_interp_np, t, sort_points)
    return transform, inv_transform


