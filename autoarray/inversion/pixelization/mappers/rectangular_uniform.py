import numpy as np
from typing import Tuple

from autoarray.inversion.pixelization.mappers.rectangular import MapperRectangular
from autoarray.inversion.pixelization.mappers.abstract import PixSubWeights


def rectangular_mappings_weights_via_interpolation_from(
    shape_native: Tuple[int, int],
    source_plane_data_grid: np.ndarray,
    source_plane_mesh_grid: np.ndarray,
    xp=np,
):
    """
    Compute bilinear interpolation weights and corresponding rectangular mesh indices for an irregular grid.

    Given a flattened regular rectangular mesh grid and an irregular grid of data points, this function
    determines for each irregular point:
    - the indices of the 4 nearest rectangular mesh pixels (top-left, top-right, bottom-left, bottom-right), and
    - the bilinear interpolation weights with respect to those pixels.

    The function supports JAX and is compatible with JIT compilation.

    Parameters
    ----------
    shape_native
        The shape (Ny, Nx) of the original rectangular mesh grid before flattening.
    source_plane_data_grid
        The irregular grid of (y, x) points to interpolate.
    source_plane_mesh_grid
        The flattened regular rectangular mesh grid of (y, x) coordinates.

    Returns
    -------
    mappings : np.ndarray of shape (N, 4)
        Indices of the four nearest rectangular mesh pixels in the flattened mesh grid.
        Order is: top-left, top-right, bottom-left, bottom-right.
    weights : np.ndarray of shape (N, 4)
        Bilinear interpolation weights corresponding to the four nearest mesh pixels.

    Notes
    -----
    - Assumes the mesh grid is uniformly spaced.
    - The weights sum to 1 for each irregular point.
    - Uses bilinear interpolation in the (y, x) coordinate system.
    """
    source_plane_mesh_grid = source_plane_mesh_grid.reshape(*shape_native, 2)

    # Assume mesh is shaped (Ny, Nx, 2)
    Ny, Nx = source_plane_mesh_grid.shape[:2]

    # Get mesh spacings and lower corner
    y_coords = source_plane_mesh_grid[:, 0, 0]  # shape (Ny,)
    x_coords = source_plane_mesh_grid[0, :, 1]  # shape (Nx,)

    dy = y_coords[1] - y_coords[0]
    dx = x_coords[1] - x_coords[0]

    y_min = y_coords[0]
    x_min = x_coords[0]

    # shape (N_irregular, 2)
    irregular = source_plane_data_grid

    # Compute normalized mesh coordinates (floating indices)
    fy = (irregular[:, 0] - y_min) / dy
    fx = (irregular[:, 1] - x_min) / dx

    # Integer indices of top-left corners
    ix = xp.floor(fx).astype(xp.int32)
    iy = xp.floor(fy).astype(xp.int32)

    # Clip to stay within bounds
    ix = xp.clip(ix, 0, Nx - 2)
    iy = xp.clip(iy, 0, Ny - 2)

    # Local coordinates inside the cell (0 <= tx, ty <= 1)
    tx = fx - ix
    ty = fy - iy

    # Bilinear weights
    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty

    weights = xp.stack([w00, w10, w01, w11], axis=1)  # shape (N_irregular, 4)

    # Compute indices of 4 surrounding pixels in the flattened mesh
    i00 = iy * Nx + ix
    i10 = iy * Nx + (ix + 1)
    i01 = (iy + 1) * Nx + ix
    i11 = (iy + 1) * Nx + (ix + 1)

    mappings = xp.stack([i00, i10, i01, i11], axis=1)  # shape (N_irregular, 4)

    return mappings, weights


class MapperRectangularUniform(MapperRectangular):
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
    regularization
        The regularization scheme which may be applied to this linear object in order to smooth its solution,
        which for a mapper smooths neighboring pixels on the mesh.
    """

    @property
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
            rectangular_mappings_weights_via_interpolation_from(
                shape_native=self.shape_native,
                source_plane_mesh_grid=self.source_plane_mesh_grid.array,
                source_plane_data_grid=self.source_plane_data_grid.over_sampled,
                xp=self._xp,
            )
        )

        return PixSubWeights(
            mappings=mappings,
            sizes=4 * self._xp.ones(len(mappings), dtype="int"),
            weights=weights,
        )
