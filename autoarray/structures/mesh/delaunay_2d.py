import numpy as np
import scipy.spatial
from scipy.spatial import cKDTree, Delaunay, Voronoi
from typing import List, Union, Optional, Tuple

from autoconf import cached_property

from autoarray.geometry.geometry_2d_irregular import Geometry2DIrregular
from autoarray.structures.mesh.abstract_2d import Abstract2DMesh
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.inversion.linear_obj.neighbors import Neighbors

from autoarray import exc
from autoarray.inversion.pixelization.mesh import mesh_numba_util


def scipy_delaunay(points_np, query_points_np, use_voronoi_areas, areas_factor):
    """Compute Delaunay simplices (simplices_padded) and Voronoi areas in one call."""

    max_simplices = 2 * points_np.shape[0]

    # --- Delaunay mesh using source plane data grid ---
    tri = Delaunay(points_np)

    points = tri.points.astype(points_np.dtype)
    simplices = tri.simplices.astype(np.int32)

    # Pad simplices to max_simplices
    simplices_padded = -np.ones((max_simplices, 3), dtype=np.int32)
    simplices_padded[: simplices.shape[0]] = simplices

    # ---------- find_simplex for source plane data grid ----------
    simplex_idx = tri.find_simplex(query_points_np).astype(np.int32)  # (Q,)

    mappings = pix_indexes_for_sub_slim_index_delaunay_from(
        source_plane_data_grid=query_points_np,
        simplex_index_for_sub_slim_index=simplex_idx,
        pix_indexes_for_simplex_index=simplices,
        delaunay_points=points_np,
    )

    # ---------- Voronoi or Barycentric Areas used to weight split points ----------

    if use_voronoi_areas:

        areas = voronoi_areas_numpy(
            points,
        )

        max_area = np.percentile(areas, 90.0)

        areas[areas == -1] = max_area
        areas[areas > max_area] = max_area

    else:

        areas = barycentric_dual_area_from(
            points,
            simplices,
            xp=np,
        )

    split_point_areas = areas_factor * np.sqrt(areas)

    # ---------- Compute split cross points for Split regularization ----------
    split_points = split_points_from(
        points=points_np,
        area_weights=split_point_areas,
    )

    # ---------- find_simplex for split cross points ----------
    split_points_idx = tri.find_simplex(split_points)

    splitted_mappings = pix_indexes_for_sub_slim_index_delaunay_from(
        source_plane_data_grid=split_points,
        simplex_index_for_sub_slim_index=split_points_idx,
        pix_indexes_for_simplex_index=simplices,
        delaunay_points=points_np,
    )

    return points, simplices_padded, mappings, split_points, splitted_mappings


def jax_delaunay(points, query_points, use_voronoi_areas, areas_factor=0.5):
    import jax
    import jax.numpy as jnp

    N = points.shape[0]
    Q = query_points.shape[0]
    max_simplices = 2 * N

    points_shape = jax.ShapeDtypeStruct((N, 2), points.dtype)
    simplices_padded_shape = jax.ShapeDtypeStruct((max_simplices, 3), jnp.int32)
    mappings_shape = jax.ShapeDtypeStruct((Q, 3), jnp.int32)
    split_points_shape = jax.ShapeDtypeStruct((N * 4, 2), points.dtype)
    splitted_mappings_shape = jax.ShapeDtypeStruct((N * 4, 3), jnp.int32)

    return jax.pure_callback(
        lambda points, qpts: scipy_delaunay(
            np.asarray(points), np.asarray(qpts), use_voronoi_areas, areas_factor
        ),
        (
            points_shape,
            simplices_padded_shape,
            mappings_shape,
            split_points_shape,
            splitted_mappings_shape,
        ),
        points,
        query_points,
    )


def barycentric_dual_area_from(
    mesh_grid,  # (N_pix, 2) vertex positions
    simplices,  # (N_tri, 3) triangle vertex indices
    xp=np,  # xp = np or jnp
):
    """
    Compute barycentric dual area for each vertex in a Delaunay triangulation.

    Dual area A_i = sum over triangles containing vertex i of (triangle_area / 3).

    Parameters
    ----------
    mesh_grid : (N_pix, 2)
        Coordinates of all mesh vertices.
    simplices : (N_tri, 3)
        Vertex indices for each triangle.
    xp : module
        numpy or jax.numpy

    Returns
    -------
    dual_area : (N_pix,)
        Barycentric dual area for each vertex.
    """

    # -------------------------------
    # gather triangle vertices
    # -------------------------------
    p0 = mesh_grid[simplices[:, 0]]  # (N_tri, 2)
    p1 = mesh_grid[simplices[:, 1]]
    p2 = mesh_grid[simplices[:, 2]]

    # -------------------------------
    # triangle areas
    # -------------------------------
    # parallelogram area = |(p1 - p0) × (p2 - p0)|
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
        p2[:, 0] - p0[:, 0]
    )

    tri_area = 0.5 * xp.abs(cross)  # (N_tri,)

    # each triangle contributes area/3 to 3 vertices
    contrib = tri_area / 3.0

    # -------------------------------
    # scatter-add into dual area array
    # -------------------------------
    N_pix = mesh_grid.shape[0]
    dual_area = xp.zeros(N_pix)

    # xp.add.at works for np and jnp
    for k in range(3):
        xp.add.at(dual_area, simplices[:, k], contrib)

    return dual_area


def voronoi_areas_numpy(points, qhull_options="Qbb Qc Qx Qm"):
    """
    Compute Voronoi cell areas with a fully optimized pure-NumPy pipeline.
    Exact match to the per-cell SciPy Voronoi loop but much faster.
    """
    vor = Voronoi(points, qhull_options=qhull_options)

    vertices = vor.vertices
    point_region = vor.point_region
    regions = vor.regions
    N = len(point_region)

    # ------------------------------------------------------------
    # 1) Collect all region lists in one go (list comprehension is fast)
    # ------------------------------------------------------------
    region_lists = [regions[r] for r in point_region]

    # Precompute which regions are unbounded (vectorized test)
    unbounded = np.array([(-1 in r) for r in region_lists], dtype=bool)

    # Filter only bounded region vertex indices
    clean_regions = [
        np.asarray([v for v in r if v != -1], dtype=int) for r in region_lists
    ]

    # Compute lengths once
    lengths = np.array([len(r) for r in clean_regions], dtype=int)
    max_len = lengths.max()

    # ------------------------------------------------------------
    # 2) Build padded idx + mask in a vectorized-like way
    #
    # Instead of doing Python work inside the loop, we pre-pack
    # the flattened data and then reshape.
    # ------------------------------------------------------------
    idx = np.full((N, max_len), -1, dtype=int)
    mask = np.zeros((N, max_len), dtype=bool)

    # Single loop remaining: extremely cheap
    for i, (r, L) in enumerate(zip(clean_regions, lengths)):
        if L:
            idx[i, :L] = r
            mask[i, :L] = True

    # ------------------------------------------------------------
    # 3) Gather polygon vertices (vectorized)
    # ------------------------------------------------------------
    safe_idx = idx.clip(min=0)
    verts = vertices[safe_idx]  # (N, max_len, 2)

    # Extract x, y with masked invalid entries zeroed
    x = np.where(mask, verts[..., 1], 0.0)
    y = np.where(mask, verts[..., 0], 0.0)

    # ------------------------------------------------------------
    # 4) Vectorized "previous index" per polygon
    # ------------------------------------------------------------
    safe_lengths = np.where(lengths == 0, 1, lengths)
    j = np.arange(max_len)
    prev = (j[None, :] - 1) % safe_lengths[:, None]

    # Efficient take-along-axis
    x_prev = np.take_along_axis(x, prev, axis=1)
    y_prev = np.take_along_axis(y, prev, axis=1)

    # ------------------------------------------------------------
    # 5) Shoelace vectorized
    # ------------------------------------------------------------
    cross = x * y_prev - y * x_prev
    areas = 0.5 * np.abs(cross.sum(axis=1))

    # ------------------------------------------------------------
    # 6) Mark unbounded regions
    # ------------------------------------------------------------
    areas[unbounded] = -1.0

    return areas


def split_points_from(points, area_weights, xp=np):
    """
    points : (N, 2)
    areas  : (N,)
    xp     : np or jnp

    Returns (4*N, 2)
    """

    N = points.shape[0]
    offsets = area_weights

    x = points[:, 0]
    y = points[:, 1]

    # Allocate output (N, 4, 2)
    out = xp.zeros((N, 4, 2), dtype=points.dtype)

    if xp.__name__.startswith("jax"):
        # ----------------------------
        # JAX → use .at[] updates
        # ----------------------------
        out = out.at[:, 0, 0].set(x + offsets)
        out = out.at[:, 0, 1].set(y)

        out = out.at[:, 1, 0].set(x - offsets)
        out = out.at[:, 1, 1].set(y)

        out = out.at[:, 2, 0].set(x)
        out = out.at[:, 2, 1].set(y + offsets)

        out = out.at[:, 3, 0].set(x)
        out = out.at[:, 3, 1].set(y - offsets)

    else:

        # ----------------------------
        # NumPy → direct assignment OK
        # ----------------------------
        out[:, 0, 0] = x + offsets
        out[:, 0, 1] = y

        out[:, 1, 0] = x - offsets
        out[:, 1, 1] = y

        out[:, 2, 0] = x
        out[:, 2, 1] = y + offsets

        out[:, 3, 0] = x
        out[:, 3, 1] = y - offsets

    return out.reshape((N * 4, 2))


def pix_indexes_for_sub_slim_index_delaunay_from(
    source_plane_data_grid,  # (N_sub, 2)
    simplex_index_for_sub_slim_index,  # (N_sub,)
    pix_indexes_for_simplex_index,  # (M, 3)
    delaunay_points,  # (N_pix, 2)
):

    N_sub = source_plane_data_grid.shape[0]

    inside_mask = simplex_index_for_sub_slim_index >= 0
    outside_mask = ~inside_mask

    # ---------------------------
    # Preallocate output
    # ---------------------------
    out = np.full((N_sub, 3), -1, dtype=np.int32)

    # ---------------------------
    # Case 1: Inside simplex (fast gather)
    # ---------------------------
    if inside_mask.any():
        out[inside_mask] = pix_indexes_for_simplex_index[
            simplex_index_for_sub_slim_index[inside_mask]
        ]

    # ---------------------------
    # Case 2: Outside → KDTree NN
    # ---------------------------
    if outside_mask.any():
        tree = cKDTree(delaunay_points)
        _, idx = tree.query(source_plane_data_grid[outside_mask], k=1)
        out[outside_mask, 0] = idx.astype(np.int32)

    out = out.astype(np.int32)

    return out


class DelaunayInterface:

    def __init__(
        self, points, simplices, mappings, split_points, splitted_mappings, xp=np
    ):

        self.points = points
        self.simplices = simplices
        self.mappings = mappings
        self.split_points = split_points
        self.splitted_mappings = splitted_mappings

        self.xp = xp

    @cached_property
    def sizes(self):
        return self.xp.sum(self.mappings >= 0, axis=1).astype(np.int32)

    @cached_property
    def splitted_sizes(self):
        return self.xp.sum(self.splitted_mappings >= 0, axis=1).astype(np.int32)


class Mesh2DDelaunay(Abstract2DMesh):
    def __init__(
        self,
        values: Union[np.ndarray, List],
        source_plane_data_grid_over_sampled=None,
        preloads=None,
        _xp=np,
    ):
        """
        An irregular 2D grid of (y,x) coordinates which represents both a Delaunay triangulation and Voronoi mesh.

        The input irregular `2D` grid represents both of the following quantities:

        - The corners of the Delaunay triangulles used to construct a Delaunay triangulation.
        - The centers of a Voronoi pixels used to constract a Voronoi mesh.

        These reflect the closely related geometric properties of the Delaunay and Voronoi grids, whereby the corner
        points of Delaunay triangles by definition represent the centres of the corresponding Voronoi mesh.

        Different pixelizations, mappers and regularization schemes combine the Delaunay and Voronoi
        geometries in different ways to perform an Inversion. Thus, having all geometric methods contained in the
        single class here is necessary.

        The input `grid` of source pixel centres is ordered arbitrarily, given that there is no regular pattern
        for a Delaunay triangulation and Voronoi mesh's indexing to follow.

        This class is used in conjuction with the `inversion/pixelizations` package to create Voronoi meshs
        and mappers that perform an `Inversion`.

        Parameters
        ----------
        values
            The grid of (y,x) coordinates corresponding to the Delaunay triangle corners and Voronoi pixel centres.
        """

        if type(values) is list:
            values = np.asarray(values)

        super().__init__(values, xp=_xp)

        self._source_plane_data_grid_over_sampled = source_plane_data_grid_over_sampled
        self.preloads = preloads

    @property
    def geometry(self):
        shape_native_scaled = (
            np.amax(self[:, 0]).astype("float") - np.amin(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float") - np.amin(self[:, 1]).astype("float"),
        )

        scaled_maxima = (
            np.amax(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float"),
        )

        scaled_minima = (
            np.amin(self[:, 0]).astype("float"),
            np.amin(self[:, 1]).astype("float"),
        )

        return Geometry2DIrregular(
            shape_native_scaled=shape_native_scaled,
            scaled_maxima=scaled_maxima,
            scaled_minima=scaled_minima,
        )

    @cached_property
    def mesh_grid_xy(self):
        """
        The default convention in `scipy.spatial` is to represent 2D coordinates as (x,y) pairs, whereas PyAutoArray
        represents 2D coordinates as (y,x) pairs.

        Therefore, this property simply converts the (y,x) grid of irregular coordinates into an (x,y) grid.
        """
        return self._xp.stack([self.array[:, 0], self.array[:, 1]]).T

    @cached_property
    def delaunay(self) -> "scipy.spatial.Delaunay":
        """
        Returns a `scipy.spatial.Delaunay` object from the 2D (y,x) grid of irregular coordinates, which correspond to
        the corner of every triangle of a Delaunay triangulation.

        This object contains numerous attributes describing a Delaunay triangulation. PyAutoArray uses the `ridge_points`
        attribute to determine the neighbors of every Voronoi pixel and the `vertices`, `regions` and `point_region`
        properties to determine the Voronoi pixel areas.

        There are numerous exceptions that `scipy.spatial.Voronoi` may raise when the input grid of coordinates used
        to compute the Voronoi mesh are ill posed. These exceptions are caught and combined into a single
        `MeshException`, which helps exception handling in the `inversion` package.
        """

        if self._source_plane_data_grid_over_sampled is None:

            raise ValueError(
                """
                You must input the `source_plane_data_grid_over_sampled` parameter of the `Mesh2DDelaunay` object
                in order to compute the Delaunay triangulation.
                """
            )

        if self.preloads is not None:

            use_voronoi_areas = self.preloads.use_voronoi_areas
            areas_factor = self.preloads.areas_factor

        else:

            use_voronoi_areas = True
            areas_factor = 0.5

        if self._xp.__name__.startswith("jax"):

            import jax.numpy as jnp

            points, simplices, mappings, split_points, splitted_mappings = jax_delaunay(
                points=self.mesh_grid_xy,
                query_points=self._source_plane_data_grid_over_sampled,
                use_voronoi_areas=use_voronoi_areas,
                areas_factor=areas_factor,
            )

        else:

            points, simplices, mappings, split_points, splitted_mappings = (
                scipy_delaunay(
                    points_np=self.mesh_grid_xy,
                    query_points_np=self._source_plane_data_grid_over_sampled,
                    use_voronoi_areas=use_voronoi_areas,
                    areas_factor=areas_factor,
                )
            )

        return DelaunayInterface(
            points=points,
            simplices=simplices,
            mappings=mappings,
            split_points=split_points,
            splitted_mappings=splitted_mappings,
            xp=self._xp,
        )

    @property
    def split_points(self) -> np.ndarray:
        """
        For every 2d (y,x) coordinate corresponding to a Voronoi pixel centre, this property splits them into a cross
        of four coordinates in the vertical and horizontal directions. The function therefore returns a irregular
        2D grid with four times the number of (y,x) coordinates.

        The distance between each centre and the 4 cross points is given by half the square root of its Voronoi
        pixel area.

        The reason for creating this grid is that the cross points allow one to estimate the gradient of the value of
        the Voronoi mesh, once the Voronoi pixels have values associated with them (e.g. after using the Voronoi
        mesh to fit data and perform an `Inversion`).

        The grid returned by this function is used by certain regularization schemes in the `Inversion` module to apply
        gradient regularization to an `Inversion` using a Delaunay triangulation or Voronoi mesh.
        """
        return self.delaunay.split_points

    @cached_property
    def neighbors(self) -> Neighbors:
        """
        Returns a ndarray describing the neighbors of every pixel in a Delaunay triangulation, where a neighbor is
        defined as two Delaunay triangles which are directly connected to one another in the triangulation.

        see `Neighbors` for a complete description of the neighboring scheme.

        The neighbors of a Voronoi mesh are computed using the `ridge_points` attribute of the scipy `Voronoi`
        object, as described in the method `mesh_util.voronoi_neighbors_from`.
        """

        delaunay = scipy.spatial.Delaunay(self.mesh_grid_xy)

        indptr, indices = delaunay.vertex_neighbor_vertices

        sizes = indptr[1:] - indptr[:-1]

        neighbors = -1 * np.ones(
            shape=(self.mesh_grid_xy.shape[0], int(np.max(sizes))), dtype="int"
        )

        for k in range(self.mesh_grid_xy.shape[0]):
            neighbors[k][0 : sizes[k]] = indices[indptr[k] : indptr[k + 1]]

        return Neighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))

    def interpolated_array_from(
        self,
        values: np.ndarray,
        shape_native: Tuple[int, int] = (401, 401),
        extent: Optional[Tuple[float, float, float, float]] = None,
    ) -> Array2D:
        """
        The reconstruction of data on a `Delaunay` triangulation (e.g. the `reconstruction` output from an `Inversion`)
        is on  irregular pixelization.

        Analysing the reconstruction can therefore be difficult and require specific functionality tailored to the
        `Delaunay` triangulation.

        This function therefore interpolates the irregular reconstruction on to a regular grid of square pixels.
        The routine uses the Delaunay triangulation interpolation weights based on the area of each triangle to
        perform this interpolation.

        The output interpolated reconstruction cis by default returned on a grid of 401 x 401 square pixels. This
        can be customized by changing the `shape_native` input, and a rectangular grid with rectangular pixels can
        be returned by instead inputting the optional `shape_scaled` tuple.

        Parameters
        ----------
        values
            The value corresponding to the reconstructed value of Delaunay triangle vertex.
        shape_native
            The 2D shape in pixels of the interpolated reconstruction, which is always returned using square pixels.
        extent
            The (x0, x1, y0, y1) extent of the grid in scaled coordinates over which the grid is created if it
            is input.
        """
        # Uses find simplex so recomputes delaunay internally
        delaunay = Delaunay(self.mesh_grid_xy)

        interpolation_grid = self.interpolation_grid_from(
            shape_native=shape_native, extent=extent
        )

        interpolated_array = mesh_numba_util.delaunay_interpolated_array_from(
            shape_native=shape_native,
            interpolation_grid_slim=np.array(interpolation_grid.slim.array),
            delaunay=delaunay,
            pixel_values=values,
        )

        return Array2D.no_mask(
            values=interpolated_array, pixel_scales=interpolation_grid.pixel_scales
        )

    @cached_property
    def voronoi(self) -> "scipy.spatial.Voronoi":
        """
        Returns a `scipy.spatial.Voronoi` object from the 2D (y,x) grid of irregular coordinates, which correspond to
        the centre of every Voronoi pixel.

        This object contains numerous attributes describing a Voronoi mesh. PyAutoArray uses
        the `vertex_neighbor_vertices` attribute to determine the neighbors of every Delaunay triangle.

        There are numerous exceptions that `scipy.spatial.Delaunay` may raise when the input grid of coordinates used
        to compute the Delaunay triangulation are ill posed. These exceptions are caught and combined into a single
        `MeshException`, which helps exception handling in the `inversion` package.
        """
        import scipy.spatial
        from scipy.spatial import QhullError

        try:
            return scipy.spatial.Voronoi(
                self.mesh_grid_xy,
                qhull_options="Qbb Qc Qx Qm",
            )
        except (ValueError, OverflowError, QhullError) as e:
            raise exc.MeshException() from e

    @property
    def voronoi_areas(self):
        return voronoi_areas_numpy(points=self.mesh_grid_xy)

    @property
    def areas_for_magnification(self) -> np.ndarray:
        """
        Returns the area of every Voronoi pixel in the Voronoi mesh.

        Pixels at boundaries can sometimes have large unrealistic areas, which can impact the magnification
        calculation. This method therefore sets their areas to zero so they do not impact the magnification
        calculation.
        """
        areas = self.voronoi_areas

        areas[areas == -1] = 0.0

        return areas

    @property
    def origin(self) -> Tuple[float, float]:
        """
        The (y,x) origin of the Voronoi grid, which is fixed to (0.0, 0.0) for simplicity.
        """
        return 0.0, 0.0

    @property
    def pixels(self) -> int:
        """
        The total number of pixels in the Voronoi mesh.
        """
        return self.shape[0]
