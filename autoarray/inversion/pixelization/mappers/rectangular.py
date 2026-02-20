from typing import Optional, Tuple
import numpy as np
from functools import partial

from autoconf import cached_property

from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.settings import Settings
from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.inversion.pixelization.mappers.abstract import PixSubWeights
from autoarray.inversion.pixelization.interpolator.rectangular import InterpolatorRectangular
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.pixelization.border_relocator import BorderRelocator


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
        transform = partial(forward_interp, sort_points, t)
        inv_transform = partial(reverse_interp, t, sort_points)
        return transform, inv_transform

    transform = partial(forward_interp_np, sort_points, t)
    inv_transform = partial(reverse_interp_np, t, sort_points)
    return transform, inv_transform

def adaptive_rectangular_transformed_grid_from(
    source_plane_data_grid, grid, mesh_weight_map=None, xp=np
):

    mu = source_plane_data_grid.mean(axis=0)
    scale = source_plane_data_grid.std(axis=0).min()
    source_grid_scaled = (source_plane_data_grid - mu) / scale

    transform, inv_transform = create_transforms(
        source_grid_scaled, mesh_weight_map=mesh_weight_map, xp=xp
    )

    def inv_full(U):
        return inv_transform(U) * scale + mu

    return inv_full(grid)


def adaptive_rectangular_areas_from(
    source_grid_shape, source_plane_data_grid, mesh_weight_map=None, xp=np
):

    edges_y = xp.linspace(1, 0, source_grid_shape[0] + 1)
    edges_x = xp.linspace(0, 1, source_grid_shape[1] + 1)

    mu = source_plane_data_grid.mean(axis=0)
    scale = source_plane_data_grid.std(axis=0).min()
    source_grid_scaled = (source_plane_data_grid - mu) / scale

    transform, inv_transform = create_transforms(
        source_grid_scaled, mesh_weight_map=mesh_weight_map, xp=xp
    )

    def inv_full(U):
        return inv_transform(U) * scale + mu

    pixel_edges = inv_full(xp.stack([edges_y, edges_x]).T)
    pixel_lengths = xp.diff(pixel_edges, axis=0).squeeze()  # shape (N_source, 2)

    dy = pixel_lengths[:, 0]
    dx = pixel_lengths[:, 1]

    return xp.abs(xp.outer(dy, dx).flatten())


def adaptive_rectangular_mappings_weights_via_interpolation_from(
    source_grid_size: int,
    source_plane_data_grid,
    source_plane_data_grid_over_sampled,
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
    source_plane_data_grid : (M, 2) ndarray
        The base source-plane coordinates, used to define normalization and transforms.
    source_plane_data_grid_over_sampled : (N, 2) ndarray
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
    mu = source_plane_data_grid.mean(axis=0)
    scale = source_plane_data_grid.std(axis=0).min()
    source_grid_scaled = (source_plane_data_grid - mu) / scale

    # --- Step 2. Build transforms ---
    transform, inv_transform = create_transforms(
        source_grid_scaled, mesh_weight_map=mesh_weight_map, xp=xp
    )

    # --- Step 3. Transform oversampled grid into index space ---
    grid_over_sampled_scaled = (source_plane_data_grid_over_sampled - mu) / scale
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


class MapperRectangular(AbstractMapper):

    def __init__(
        self,
        mask,
        mesh,
        source_plane_data_grid: Grid2DIrregular,
        source_plane_mesh_grid: Grid2DIrregular,
        regularization: Optional[AbstractRegularization],
        border_relocator: BorderRelocator,
        adapt_data: Optional[np.ndarray] = None,
        settings: Settings = None,
        preloads=None,
        mesh_weight_map: Optional[np.ndarray] = None,
        xp=np,
    ):
        """
        To understand a `Mapper` one must be familiar `Mesh` objects and the `mesh` and `pixelization` packages, where
        the four grids are explained (`image_plane_data_grid`, `source_plane_data_grid`,
        `image_plane_mesh_grid`,`source_plane_mesh_grid`)

        If you are unfamliar withe above objects, read through the docstrings of the `pixelization`, `mesh` and
        `image_mesh` packages.

        A `Mapper` determines the mappings between the masked data grid's pixels (`image_plane_data_grid` and
        `source_plane_data_grid`) and the mesh's pixels (`image_plane_mesh_grid` and `source_plane_mesh_grid`).

        The 1D Indexing of each grid is identical in the `data` and `source` frames (e.g. the transformation does not
        change the indexing, such that `source_plane_data_grid[0]` corresponds to the transformed value
        of `image_plane_data_grid[0]` and so on).

        A mapper therefore only needs to determine the index mappings between the `grid_slim` and `mesh_grid`,
        noting that associations are made by pairing `source_plane_mesh_grid` with `source_plane_data_grid`.

        Mappings are represented in the 2D ndarray `pix_indexes_for_sub_slim_index`, whereby the index of
        a pixel on the `mesh_grid` maps to the index of a pixel on the `grid_slim` as follows:

        - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel maps to the mesh's 1st pixel.
        - pix_indexes_for_sub_slim_index[1, 0] = 3: the data's 2nd sub-pixel maps to the mesh's 4th pixel.
        - pix_indexes_for_sub_slim_index[2, 0] = 1: the data's 3rd sub-pixel maps to the mesh's 2nd pixel.

        The second dimension of this array (where all three examples above are 0) is used for cases where a
        single pixel on the `grid_slim` maps to multiple pixels on the `mesh_grid`. For example, a
        `Delaunay` triangulation, where every `grid_slim` pixel maps to three Delaunay pixels (the corners of the
        triangles) with varying interpolation weights .

        For a `RectangularAdaptDensity` mesh every pixel in the masked data maps to only one pixel, thus the second
        dimension of `pix_indexes_for_sub_slim_index` is always of size 1.

        The mapper allows us to create a mapping matrix, which is a matrix representing the mapping between every
        unmasked data pixel annd the pixels of a mesh. This matrix is the basis of performing an `Inversion`,
        which reconstructs the data using the `source_plane_mesh_grid`.

        Parameters
        ----------
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            The 2D grid of (y,x) centres of every pixelization pixel in the `source` frame.
        image_plane_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame. This has a
            transformation applied to it to create the `source_plane_mesh_grid`.
        adapt_data
            An image which is used to determine the `image_plane_mesh_grid` and therefore adapt the distribution of
            pixels of the Delaunay grid to the data it discretizes.
        mesh_weight_map
            The weight map used to weight the creation of the rectangular mesh grid, which is used for the
            `RectangularBrightness` mesh which adapts the size of its pixels to where the source is reconstructed.
        regularization
            The regularization scheme which may be applied to this linear object in order to smooth its solution,
            which for a mapper smooths neighboring pixels on the mesh.
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        settings
            Settings controlling the pixelization for example if a border is used to relocate its exterior coordinates.
        preloads
            The JAX preloads, storing shape information so that JAX knows in advance the shapes of arrays used
            in the mapping matrix and indexes of certain array entries, for example to zero source pixels in the
            linear inversion.
        mesh_weight_map
            The weight map used to weight the creation of the rectangular mesh grid, which adapts the size of
            the rectangular pixels to be smaller in the brighter regions of the source.
         xp
            The array module (e.g. `numpy` or `jax.numpy`) used to perform calculations and store arrays in the mapper.
        """
        super().__init__(
            mask=mask,
            mesh=mesh,
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=source_plane_mesh_grid,
            regularization=regularization,
            border_relocator=border_relocator,
            adapt_data=adapt_data,
            settings=settings,
            preloads=preloads,
            xp=xp,
        )
        self.mesh_weight_map = mesh_weight_map

    @property
    def interpolator(self):
        """
        Return the rectangular `source_plane_mesh_grid` as a `InterpolatorRectangular` object, which provides additional
        functionality for perform operatons that exploit the geometry of a rectangular pixelization.

        Parameters
        ----------
        source_plane_data_grid
            The (y,x) grid of coordinates over which the rectangular pixelization is overlaid, where this grid may have
            had exterior pixels relocated to its edge via the border.
        source_plane_mesh_grid
            Not used for a rectangular pixelization, because the pixelization grid in the `source` frame is computed
            by overlaying the `source_plane_data_grid` with the rectangular pixelization.
        """
        return InterpolatorRectangular(
            mesh=self.mesh,
            mesh_grid=self.source_plane_mesh_grid,
            data_grid_over_sampled=self.source_plane_data_grid.over_sampled,
            preloads=self.preloads,
            _xp=self._xp,
        )

    @property
    def shape_native(self) -> Tuple[int, ...]:
        return self.mesh.shape

    @cached_property
    def pix_sub_weights(self) -> PixSubWeights:
        """
        Computes the following three quantities describing the mappings between of every sub-pixel in the masked data
        and pixel in the `RectangularAdaptDensity` mesh.

        - `pix_indexes_for_sub_slim_index`: the mapping of every data pixel (given its `sub_slim_index`)
        to mesh pixels (given their `pix_indexes`).

        - `pix_sizes_for_sub_slim_index`: the number of mappings of every data pixel to mesh pixels.

        - `pix_weights_for_sub_slim_index`: the interpolation weights of every data pixel's mesh
        pixel mapping

        These are packaged into the class `PixSubWeights` with attributes `mappings`, `sizes` and `weights`.

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the mesh pixel indexes,
        for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the RectangularAdaptDensity
        mesh's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[2, 0] = 4`: The data's third (index 2) sub-pixel maps to the RectangularAdaptDensity
        mesh's fifth (index 4) pixel.

        The second dimension of the array `pix_indexes_for_sub_slim_index`, which is 0 in both examples above, is used
        for cases where a data pixel maps to more than one mesh pixel (for example a `Delaunay` triangulation
        where each data pixel maps to 3 Delaunay triangles with interpolation weights). The weights of multiple mappings
        are stored in the array `pix_weights_for_sub_slim_index`.

        For a RectangularAdaptDensity pixelization each data sub-pixel maps to a single mesh pixel, thus the second
        dimension of the array `pix_indexes_for_sub_slim_index` 1 and all entries in `pix_weights_for_sub_slim_index`
        are equal to 1.0.
        """
        mappings, weights = (
            adaptive_rectangular_mappings_weights_via_interpolation_from(
                source_grid_size=self.shape_native[0],
                source_plane_data_grid=self.source_plane_data_grid.array,
                source_plane_data_grid_over_sampled=self.source_plane_data_grid.over_sampled,
                mesh_weight_map=self.mesh_weight_map,
                xp=self._xp,
            )
        )

        return PixSubWeights(
            mappings=mappings,
            sizes=4 * self._xp.ones(len(mappings), dtype="int"),
            weights=weights,
        )

    @property
    def areas_transformed(self):
        """
        A class packing the ndarrays describing the neighbors of every pixel in the rectangular pixelization (see
        `Neighbors` for a complete description of the neighboring scheme).

        The neighbors of a rectangular pixelization are computed by exploiting the uniform and symmetric nature of the
        rectangular grid, as described in the method `rectangular_neighbors_from`.
        """
        return adaptive_rectangular_areas_from(
            source_grid_shape=self.shape_native,
            source_plane_data_grid=self.source_plane_data_grid.array,
            mesh_weight_map=self.mesh_weight_map,
            xp=self._xp,
        )

    @property
    def areas_for_magnification(self):
        """
        The area of every pixel in the rectangular pixelization.

        Returns
        -------
        ndarray
            The area of every pixel in the rectangular pixelization.
        """
        return self.areas_transformed

    @property
    def edges_transformed(self):
        """
        A class packing the ndarrays describing the neighbors of every pixel in the rectangular pixelization (see
        `Neighbors` for a complete description of the neighboring scheme).

        The neighbors of a rectangular pixelization are computed by exploiting the uniform and symmetric nature of the
        rectangular grid, as described in the method `rectangular_neighbors_from`.
        """

        # edges defined in 0 -> 1 space, there is one more edge than pixel centers on each side
        edges_y = self._xp.linspace(1, 0, self.shape_native[0] + 1)
        edges_x = self._xp.linspace(0, 1, self.shape_native[1] + 1)

        edges_reshaped = self._xp.stack([edges_y, edges_x]).T

        return adaptive_rectangular_transformed_grid_from(
            source_plane_data_grid=self.source_plane_data_grid.array,
            grid=edges_reshaped,
            mesh_weight_map=self.mesh_weight_map,
            xp=self._xp,
        )
