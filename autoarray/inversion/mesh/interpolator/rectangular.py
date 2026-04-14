import numpy as np
from functools import partial

from autoconf import cached_property

from autoarray.inversion.mesh.interpolator.abstract import AbstractInterpolator


def forward_interp(xp, yp, x):

    import jax
    import jax.numpy as jnp

    return jax.vmap(jnp.interp, in_axes=(1, 1, 1, None, None), out_axes=(1))(
        x, xp, yp, 0, 1
    )


def reverse_interp(xp, yp, x):
    import jax
    import jax.numpy as jnp

    return jax.vmap(jnp.interp, in_axes=(1, 1, 1), out_axes=(1))(x, xp, yp)


def forward_interp_safe(xp, yp, x, eps=1e-30):
    """
    NOT CURRENTLY USED — kept as a reference implementation for a follow-up
    switch away from the ``JITTER`` offset in ``create_transforms``. See
    ``admin_jammy/prompt/autoarray/rectangular_interp_custom_vjp.md``.

    Drop-in replacement for ``forward_interp`` that is robust to duplicate
    and near-duplicate knots in ``xp`` without perturbing the forward
    interpolation value.

    Motivation
    ----------
    ``jnp.interp``'s autodiff rule divides by ``xp[i+1] - xp[i]`` in both
    the knot-gradient and query-gradient branches. Ray-traced source-plane
    grids regularly contain large runs of exact-duplicate coordinates
    (e.g. ~50% for an Isothermal lens over a circular mask), which makes
    that division a literal 0/0 and emits O(1e24) cotangents.

    The current shipping fix (see below in ``create_transforms``) adds a
    ``JITTER ~ 1e-7`` monotonic offset to ``sort_points`` before feeding
    them to ``jnp.interp``. That fixes the gradient but perturbs the
    forward interpolation value by up to ``N * JITTER ~ 1.5e-3`` in
    scaled source-plane units, which is enough to drift integration-test
    reference likelihoods by ~1e-4 relative.

    This function keeps the forward value **exact** (no jitter) and uses
    a ``jnp.where`` (double-where) guard on the slope computation so the
    backward pass floors the denominator at ``eps``. The gradient
    magnitude is then bounded by ``|dy| / eps``, which is finite, and
    the forward path is identical to ``jnp.interp`` at well-separated
    knots.

    At duplicate knots, this implementation returns ``yp[i]`` (the
    left-duplicate's value) rather than ``jnp.interp``'s implementation-
    defined behaviour, but the mapping matrix consumes only
    ``floor``/``ceil`` of the result, so the choice at a zero-measure
    set of query points does not affect bilinear weights in practice.

    Notes / TODO for the expert reviewer
    ------------------------------------
    * Does ``jnp.searchsorted(..., side='right') - 1`` reproduce
      ``jnp.interp``'s bin convention in the edge cases ``x == xp[0]``
      and ``x == xp[-1]``?  The extrapolation clamping below uses
      ``left=0, right=1`` to mirror the current ``forward_interp``
      signature, but the bin picked at the boundary may differ from
      what ``jnp.interp`` picks internally.
    * Should this use ``jax.custom_vjp`` instead, so the *forward*
      stays bit-identical to ``jnp.interp`` and only the backward is
      customised? That may be preferable if any downstream test is
      sensitive to the duplicate-bin return value.
    * The default ``eps=1e-30`` floors the slope at ``|dy|/1e-30``,
      which is effectively "no floor" — relying on the double-``where``
      to block the 0/0 path entirely. A realistic EPS like ``1e-12``
      would bound the slope at ``1/N * 1e12 ~ 1e8`` for ``N ~ 1e4``
      — still finite, still harmless. Choice of ``eps`` interacts
      with downstream numerical stability.
    """
    import jax
    import jax.numpy as jnp

    def _safe_interp_1d(xp_col, yp_col, x_col):
        idx = jnp.clip(
            jnp.searchsorted(xp_col, x_col, side="right") - 1,
            0,
            xp_col.shape[0] - 2,
        )
        x0 = xp_col[idx]
        x1 = xp_col[idx + 1]
        y0 = yp_col[idx]
        y1 = yp_col[idx + 1]

        gap = x1 - x0
        safe_gap = jnp.where(gap > eps, gap, jnp.ones_like(gap))
        t = jnp.where(
            gap > eps,
            (x_col - x0) / safe_gap,
            jnp.zeros_like(x_col),
        )
        result = y0 + t * (y1 - y0)

        # Match jnp.interp(..., left=0, right=1) clamping used by forward_interp.
        result = jnp.where(x_col < xp_col[0], jnp.zeros_like(result), result)
        result = jnp.where(x_col > xp_col[-1], jnp.ones_like(result), result)
        return result

    return jax.vmap(_safe_interp_1d, in_axes=(1, 1, 1), out_axes=1)(xp, yp, x)


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


def create_transforms(traced_points, mesh_weight_map=None, xp=np):

    N = traced_points.shape[0]  # // 2

    if mesh_weight_map is None:
        t = xp.arange(1, N + 1) / (N + 1)
        t = xp.stack([t, t], axis=1)
        sort_points = xp.sort(traced_points, axis=0)  # [::2]
    else:
        sdx = xp.argsort(traced_points, axis=0)
        sort_points = xp.take_along_axis(traced_points, sdx, axis=0)
        t = xp.stack([mesh_weight_map, mesh_weight_map], axis=1)
        t = xp.take_along_axis(t, sdx, axis=0)
        t = xp.cumsum(t, axis=0)

    if xp.__name__.startswith("jax"):
        # --------------------------------------------------------------
        # Gradient stabilisation for `jnp.interp`.
        #
        # Ray-traced source grids commonly contain near-duplicate or
        # exactly-duplicate coordinates — e.g. an Isothermal lens over
        # a circular mask produces ~50% gaps that are exactly zero
        # after sorting. This breaks `jnp.interp` for autodiff in two
        # distinct ways, both of which have to be patched here:
        #
        # (1) Knot-gradient term. The vjp of `jnp.interp` w.r.t. its
        #     knot array `xp` divides by `xp[i+1] - xp[i]`, which is
        #     0/0 at duplicate knots and emits O(1e24) cotangents.
        #     We freeze this path with `stop_gradient`. This is
        #     semantically correct: the only downstream consumer of
        #     the transformed grid is `adaptive_rectangular_mappings_
        #     weights_..._from`, which uses `floor`/`ceil` to select
        #     the 4 corner pixels. That bin assignment already has
        #     zero gradient, so the knot-gradient term has no
        #     downstream consumer anyway.
        #
        # (2) Query-gradient term. The vjp w.r.t. the query `x` is
        #     the local slope `(yp[i+1] - yp[i]) / (xp[i+1] - xp[i])`.
        #     Even with frozen knots, this blows up when the knot gap
        #     is near zero. We prevent that by adding a strictly-
        #     monotonic offset `arange(N) * JITTER` to `sort_points`.
        #     For the default `mesh_weight_map=None` path, `t` moves
        #     in steps of `1/(N+1)`; with `JITTER = 1e-7
        #     `N ~ 1.5e4`, the worst-case slope is bounded by
        #     `(1/(N+1)) / JITTER ~ 650`, which is harmless, and the
        #     forward interpolation value is perturbed by at most
        #     `N * JITTER ~ 1.5e-3` in the source-plane scaled units
        #     — well below the `(source_grid_size - 3)` downstream
        #     multiplier's sensitivity to sub-pixel placement.
        #
        # Together these two patches make the rectangular interpolator
        # differentiable end-to-end and bring the mapping-matrix
        # gradient into agreement with finite differences.
        import jax

        JITTER = 1e-7
        jitter = xp.arange(sort_points.shape[0], dtype=sort_points.dtype) * JITTER
        jitter = xp.stack([jitter, jitter], axis=1)
        sort_points = jax.lax.stop_gradient(sort_points + jitter)

        transform = partial(forward_interp, sort_points, t)
        inv_transform = partial(reverse_interp, t, sort_points)
        return transform, inv_transform

    transform = partial(forward_interp_np, sort_points, t)
    inv_transform = partial(reverse_interp_np, t, sort_points)
    return transform, inv_transform


def adaptive_rectangular_transformed_grid_from(
    data_grid, grid, mesh_weight_map=None, xp=np
):

    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    transform, inv_transform = create_transforms(
        source_grid_scaled, mesh_weight_map=mesh_weight_map, xp=xp
    )

    def inv_full(U):
        return inv_transform(U) * scale + mu

    return inv_full(grid)


def adaptive_rectangular_mappings_weights_via_interpolation_from(
    source_grid_size: int,
    data_grid,
    data_grid_over_sampled,
    mesh_weight_map=None,
    xp=np,
):
    """
    Compute bilinear interpolation indices and weights for mapping an oversampled
    source-plane grid onto a regular rectangular pixelization.

    This function takes a set of irregularly-sampled source-plane coordinates and
    builds an adaptive mapping onto a `source_grid_size x source_grid_size` rectangular
    pixelization using bilinear interpolation. The interpolation is expressed as:

        f(x, y) ≈ w_bl * f(ix_down, iy_down) +
                  w_br * f(ix_up,   iy_down) +
                  w_tl * f(ix_down, iy_up) +
                  w_tr * f(ix_up,   iy_up)

    where `(ix_down, ix_up, iy_down, iy_up)` are the integer grid coordinates
    surrounding the continuous position `(x, y)`.

    Steps performed:
      1. Normalize the source-plane grid by subtracting its mean and dividing by
         the minimum axis standard deviation (to balance scaling).
      2. Construct forward/inverse transforms which map the grid into the unit square [0,1]^2.
      3. Transform the oversampled source-plane grid into [0,1]^2, then scale it
         to index space `[0, source_grid_size)`.
      4. Compute floor/ceil along x and y axes to find the enclosing rectangular cell.
      5. Build the four corner indices: bottom-left (bl), bottom-right (br),
         top-left (tl), and top-right (tr).
      6. Flatten the 2D indices into 1D indices suitable for scatter operations,
         with a flipped row-major convention: row = source_grid_size - i, col = j.
      7. Compute bilinear interpolation weights (`w_bl, w_br, w_tl, w_tr`).
      8. Return arrays of flattened indices and weights of shape `(N, 4)`, where
         `N` is the number of oversampled coordinates.

    Parameters
    ----------
    source_grid_size : int
        The number of pixels along one dimension of the rectangular pixelization.
        The grid is square: (source_grid_size x source_grid_size).
    data_grid : (M, 2) ndarray
        The base source-plane coordinates, used to define normalization and transforms.
    data_grid_over_sampled : (N, 2) ndarray
        Oversampled source-plane coordinates to be interpolated onto the rectangular grid.
    mesh_weight_map
        The weight map used to weight the creation of the rectangular mesh grid, which is used for the
        `RectangularBrightness` mesh which adapts the size of its pixels to where the source is reconstructed.

    Returns
    -------
    flat_indices : (N, 4) int ndarray
        The flattened indices of the four neighboring pixel corners for each oversampled point.
        Order: [bl, br, tl, tr].
    weights : (N, 4) float ndarray
        The bilinear interpolation weights for each of the four neighboring pixels.
        Order: [w_bl, w_br, w_tl, w_tr].
    """

    # --- Step 1. Normalize grid ---
    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    # --- Step 2. Build transforms ---
    transform, inv_transform = create_transforms(
        source_grid_scaled, mesh_weight_map=mesh_weight_map, xp=xp
    )

    # --- Step 3. Transform oversampled grid into index space ---
    grid_over_sampled_scaled = (data_grid_over_sampled - mu) / scale
    grid_over_sampled_transformed = transform(grid_over_sampled_scaled)
    grid_over_index = (source_grid_size - 3) * grid_over_sampled_transformed + 1

    # --- Step 4. Floor/ceil indices ---
    ix_down = xp.floor(grid_over_index[:, 0])
    ix_up = xp.ceil(grid_over_index[:, 0])
    iy_down = xp.floor(grid_over_index[:, 1])
    iy_up = xp.ceil(grid_over_index[:, 1])

    # --- Step 5. Four corners ---
    idx_tl = xp.stack([ix_up, iy_down], axis=1)
    idx_tr = xp.stack([ix_up, iy_up], axis=1)
    idx_br = xp.stack([ix_down, iy_up], axis=1)
    idx_bl = xp.stack([ix_down, iy_down], axis=1)

    # --- Step 6. Flatten indices ---
    def flatten(idx, n):
        row = n - idx[:, 0]
        col = idx[:, 1]
        return row * n + col

    flat_tl = flatten(idx_tl, source_grid_size)
    flat_tr = flatten(idx_tr, source_grid_size)
    flat_bl = flatten(idx_bl, source_grid_size)
    flat_br = flatten(idx_br, source_grid_size)

    flat_indices = xp.stack([flat_tl, flat_tr, flat_bl, flat_br], axis=1).astype(
        "int64"
    )

    # --- Step 7. Bilinear interpolation weights ---
    t_row = (grid_over_index[:, 0] - ix_down) / (ix_up - ix_down + 1e-12)
    t_col = (grid_over_index[:, 1] - iy_down) / (iy_up - iy_down + 1e-12)

    # Weights
    w_tl = (1 - t_row) * (1 - t_col)
    w_tr = (1 - t_row) * t_col
    w_bl = t_row * (1 - t_col)
    w_br = t_row * t_col
    weights = xp.stack([w_tl, w_tr, w_bl, w_br], axis=1)

    return flat_indices, weights


class InterpolatorRectangular(AbstractInterpolator):

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        mesh_weight_map,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
        """
        A grid of (y,x) coordinates which represent a uniform rectangular pixelization.

        A `InterpolatorRectangular` is ordered such pixels begin from the top-row and go rightwards and then downwards.
        It is an ndarray of shape [total_pixels, 2], where the first dimension of the ndarray corresponds to the
        pixelization's pixel index and second element whether it is a y or x arc-second coordinate.

        For example:

        - grid[3,0] = the y-coordinate of the 4th pixel in the rectangular pixelization.
        - grid[6,1] = the x-coordinate of the 7th pixel in the rectangular pixelization.

        This class is used in conjuction with the `inversion/pixelizations` package to create rectangular pixelizations
        and mappers that perform an `Inversion`.

        Parameters
        ----------
        values
            The grid of (y,x) coordinates corresponding to the centres of each pixel in the rectangular pixelization.
        shape_native
            The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) origin of the pixelization.
        """
        super().__init__(
            mesh=mesh,
            mesh_grid=mesh_grid,
            data_grid=data_grid,
            adapt_data=adapt_data,
            xp=xp,
        )
        self.mesh_weight_map = mesh_weight_map

    @cached_property
    def mesh_geometry(self):

        from autoarray.inversion.mesh.mesh_geometry.rectangular import (
            MeshGeometryRectangular,
        )

        return MeshGeometryRectangular(
            mesh=self.mesh,
            mesh_grid=self.mesh_grid,
            data_grid=self.data_grid,
            mesh_weight_map=self.mesh_weight_map,
            xp=self._xp,
        )

    @cached_property
    def _mappings_sizes_weights(self):

        mappings, weights = (
            adaptive_rectangular_mappings_weights_via_interpolation_from(
                source_grid_size=self.mesh.shape[0],
                data_grid=self.data_grid.array,
                data_grid_over_sampled=self.data_grid.over_sampled.array,
                mesh_weight_map=self.mesh_weight_map,
                xp=self._xp,
            )
        )

        sizes = 4 * self._xp.ones(len(mappings), dtype="int")

        return mappings, sizes, weights

    @cached_property
    def _mappings_sizes_weights_split(self):
        # Rectangular pixelizations use bilinear interpolation which already factors
        # in the 4-corner neighbourhood, so no separate split-cross calculation is
        # needed — split regularization reuses the same mappings.
        return self._mappings_sizes_weights
