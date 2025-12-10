import numpy as np
import scipy.spatial
from scipy.spatial import cKDTree
from typing import List, Union, Optional, Tuple

from autoconf import cached_property

from autoarray.geometry.geometry_2d_irregular import Geometry2DIrregular
from autoarray.structures.mesh.abstract_2d import Abstract2DMesh
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.inversion.linear_obj.neighbors import Neighbors

from autoarray import exc
from autoarray.inversion.pixelization.mesh import mesh_numba_util

def voronoi_areas_from(points_np):

    from scipy.spatial import Voronoi

    N = points_np.shape[0]

    # --- Voronoi ---
    vor = Voronoi(points_np, qhull_options="Qbb Qc Qx Qm")

    voronoi_vertices = vor.vertices
    voronoi_regions = vor.regions
    voronoi_point_region = vor.point_region
    voronoi_areas = np.zeros(N)

    for i in range(N):
        region_vertices_indexes = voronoi_regions[voronoi_point_region[i]]
        if -1 in region_vertices_indexes:
            voronoi_areas[i] = -1
        else:

            points_of_region = voronoi_vertices[region_vertices_indexes]

            x = points_of_region[:, 1]
            y = points_of_region[:, 0]

            voronoi_areas[i] = 0.5 * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )

    voronoi_pixel_areas_for_split = voronoi_areas.copy()

    # 90th percentile
    max_area = np.percentile(voronoi_pixel_areas_for_split, 90.0)

    voronoi_pixel_areas_for_split[voronoi_pixel_areas_for_split == -1] = max_area
    voronoi_pixel_areas_for_split[voronoi_pixel_areas_for_split > max_area] = max_area

    half_region_area_sqrt_lengths = 0.5 * np.sqrt(voronoi_pixel_areas_for_split)

    split_points = split_points_from(
        points=points_np,
        area_weights=half_region_area_sqrt_lengths,
    )

    return voronoi_areas, split_points


def scipy_delaunay_voronoi(points_np, query_points_np, max_simplices):
    """Compute Delaunay simplices (padded) and Voronoi areas in one call."""
    from scipy.spatial import Delaunay
    import numpy as np

    # --- Delaunay ---
    tri = Delaunay(points_np)

    pts = tri.points.astype(points_np.dtype)
    simplices = tri.simplices.astype(np.int32)

    # Pad simplices to max_simplices
    padded = -np.ones((max_simplices, 3), dtype=np.int32)
    padded[: simplices.shape[0]] = simplices

    # ---------- Voronoi cell areas ----------
    areas, split_points = voronoi_areas_from(points_np)

    # ---------- find_simplex ----------
    simplex_idx = tri.find_simplex(query_points_np).astype(np.int32)  # (Q,)

    mappings, sizes = pix_indexes_for_sub_slim_index_delaunay_from(
        source_plane_data_grid=query_points_np,
        simplex_index_for_sub_slim_index=simplex_idx,
        pix_indexes_for_simplex_index=simplices,
        delaunay_points=points_np,
    )

    mappings = mappings.astype(np.int32)

    # ---------- find_simplex for split cross points ----------
    split_cross_idx = tri.find_simplex(split_points)

    (
        splitted_mappings,
        splitted_sizes,
    ) = pix_indexes_for_sub_slim_index_delaunay_from(
        source_plane_data_grid=split_points,
        simplex_index_for_sub_slim_index=split_cross_idx,
        pix_indexes_for_simplex_index=simplices,
        delaunay_points=points_np,
    )

    splitted_mappings = splitted_mappings.astype(np.int32)

    return pts, padded, areas, split_points, mappings, splitted_mappings


def jax_delaunay_voronoi(points, query_points):
    import jax
    import jax.numpy as jnp

    N = points.shape[0]
    Q = query_points.shape[0]

    # Conservative pad (you can pass exact M if you want)
    max_simplices = 2 * N  # same logic as before

    pts_shape = jax.ShapeDtypeStruct((N, 2), points.dtype)
    simp_shape = jax.ShapeDtypeStruct((max_simplices, 3), jnp.int32)
    area_shape = jax.ShapeDtypeStruct((N,), points.dtype)
    split_points_shape = jax.ShapeDtypeStruct((N*4, 2), points.dtype)
    mappings_shape    = jax.ShapeDtypeStruct((Q, 3), jnp.int32)
    splitted_mappings_shape = jax.ShapeDtypeStruct((N*4, 3), jnp.int32)

    return jax.pure_callback(
        lambda pts, qpts: scipy_delaunay_voronoi(
            np.asarray(pts), np.asarray(qpts), max_simplices
        ),
        (pts_shape, simp_shape, area_shape, split_points_shape, mappings_shape, splitted_mappings_shape),
        points, query_points,
    )


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
    source_plane_data_grid,          # (N_sub, 2)
    simplex_index_for_sub_slim_index,# (N_sub,)
    pix_indexes_for_simplex_index,   # (M, 3)
    delaunay_points,                 # (N_pix, 2)
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

    # ---------------------------
    # Sizes
    # ---------------------------
    sizes = np.sum(out >= 0, axis=1).astype(np.int32)

    return out, sizes


# def pix_indexes_for_sub_slim_index_delaunay_from(
#     source_plane_data_grid,  # (N_sub, 2)
#     simplex_index_for_sub_slim_index,  # (N_sub,)
#     pix_indexes_for_simplex_index,  # (M, 3)
#     delaunay_points,  # (N_points, 2)
#     xp=np,  # <--- choose backend: np or jnp
# ):
#     """
#     XP-compatible version of pix_indexes_for_sub_slim_index_delaunay_from.
#
#     If xp=np: runs with NumPy (no JAX needed)
#     If xp=jnp: runs inside JAX (jit, vmap, GPU)
#
#     Returns
#     -------
#     pix_indexes_for_sub_slim_index : (N_sub, 3)
#     pix_indexes_for_sub_slim_index_sizes : (N_sub,)
#     """
#
#     # Helper: xp.ones_like that supports setting dtype
#     def ones_like(x, dtype):
#         return xp.ones(x.shape, dtype=dtype)
#
#     N_sub = source_plane_data_grid.shape[0]
#
#     # Boolean mask for points that fall inside simplices
#     inside_mask = simplex_index_for_sub_slim_index >= 0  # shape (N_sub,)
#     outside_mask = ~inside_mask
#
#     # ----------------------------
#     # Case 1: inside a simplex
#     # ----------------------------
#     # (N_sub, 3)
#     pix_inside = xp.where(
#         inside_mask[:, None],
#         pix_indexes_for_simplex_index[simplex_index_for_sub_slim_index],
#         -ones_like((inside_mask[:, None] + 0), dtype=np.int32),  # -1 filler
#     )
#
#     # ----------------------------
#     # Case 2: outside any simplex → nearest delaunay point
#     # ----------------------------
#     # Squared distances: (N_sub, N_points)
#     d2 = xp.sum(
#         (source_plane_data_grid[:, None, :] - delaunay_points[None, :, :]) ** 2.0,
#         axis=-1,
#     )
#
#     nearest = xp.argmin(d2, axis=1).astype(np.int32)  # (N_sub,)
#
#     # (N_sub, 3) → [nearest, -1, -1]
#     nn_triplets = xp.stack(
#         [
#             nearest,
#             -ones_like(nearest, dtype=np.int32),
#             -ones_like(nearest, dtype=np.int32),
#         ],
#         axis=1,
#     )
#
#     pix_outside = xp.where(
#         outside_mask[:, None],
#         nn_triplets,
#         -ones_like((outside_mask[:, None] + 0), dtype=np.int32),
#     )
#
#     # ----------------------------
#     # Combine inside + outside
#     # ----------------------------
#     pix_indexes_for_sub_slim_index = xp.where(
#         inside_mask[:, None],
#         pix_inside,
#         pix_outside,
#     )
#
#     # ----------------------------
#     # Count valid entries
#     # ----------------------------
#     pix_sizes = xp.sum(pix_indexes_for_sub_slim_index >= 0, axis=1)
#
#     return pix_indexes_for_sub_slim_index, pix_sizes


class DelaunayInterface:

    def __init__(self, points, simplices, voronoi_areas, split_cross, vertex_neighbor_vertices, mappings, sizes, splitted_mappings, splitted_sizes):

        self.points = points
        self.simplices = simplices
        self.voronoi_areas = voronoi_areas
        self.split_cross = split_cross
        self.vertex_neighbor_vertices = vertex_neighbor_vertices
        self.mappings = mappings
        self.sizes = sizes
        self.splitted_mappings = splitted_mappings
        self.splitted_sizes = splitted_sizes


class Mesh2DDelaunay(Abstract2DMesh):
    def __init__(self, values: Union[np.ndarray, List], source_plane_data_grid_over_sampled=None, _xp=np):
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

        mesh_grid = self._xp.stack([self.array[:, 0], self.array[:, 1]]).T

        if self._xp.__name__.startswith("jax"):

            import jax.numpy as jnp

            points, simplices, voronoi_areas, split_cross, mappings, splitted_mappings = jax_delaunay_voronoi(mesh_grid, self._source_plane_data_grid_over_sampled)
            vertex_neighbor_vertices = None

        else:

            delaunay = scipy.spatial.Delaunay(mesh_grid)

            points = delaunay.points
            simplices = delaunay.simplices.astype(np.int32)
            vertex_neighbor_vertices = delaunay.vertex_neighbor_vertices

            voronoi_areas = voronoi_areas_from(mesh_grid)

            simplex_index_for_sub_slim_index = delaunay.find_simplex(self.source_plane_data_grid_over_sampled)
            splitted_simplex_index_for_sub_slim_index = delaunay.find_simplex(
                self.split_cross
            )

            mappings = pix_indexes_for_sub_slim_index_delaunay_from(
                source_plane_data_grid=self.source_plane_data_grid.over_sampled,
                simplex_index_for_sub_slim_index=simplex_index_for_sub_slim_index,
                pix_indexes_for_simplex_index=delaunay.simplices,
                delaunay_points=delaunay.points,
            )

            (
                splitted_mappings
            ) = pix_indexes_for_sub_slim_index_delaunay_from(
                source_plane_data_grid=self.source_plane_mesh_grid.split_cross,
                simplex_index_for_sub_slim_index=splitted_simplex_index_for_sub_slim_index,
                pix_indexes_for_simplex_index=delaunay.simplices,
                delaunay_points=delaunay.points,
            )

        sizes = self._xp.sum(mappings >= 0, axis=1).astype(self._xp.int32)
        splitted_sizes = self._xp.sum(splitted_mappings >= 0, axis=1).astype(self._xp.int32)

        return DelaunayInterface(
            points, simplices, voronoi_areas, split_cross, vertex_neighbor_vertices, mappings, sizes, splitted_mappings, splitted_sizes
        )

    @property
    def edge_pixel_list(self) -> List:
        """
        Returns a list of the Voronoi pixel indexes that are on the edge of the mesh.
        """
        return mesh_numba_util.voronoi_edge_pixels_from(
            regions=self.voronoi.regions, point_region=self.voronoi.point_region
        )

    @property
    def split_cross(self) -> np.ndarray:
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

        half_region_area_sqrt_lengths = 0.5 * self._xp.sqrt(
            self.voronoi_pixel_areas_for_split
        )

        return split_points_from(
            points=self.array,
            area_weights=half_region_area_sqrt_lengths,
            xp=self._xp,
        )

    @property
    def voronoi_pixel_areas_for_split(self) -> np.ndarray:
        """
        Returns the area of every Voronoi pixel in the Voronoi mesh.

        These areas are used when performing gradient regularization in order to determine the size of the cross of
        points where the derivative is evaluated and therefore where regularization is evaluated (see `split_cross`).

        Pixels at boundaries can sometimes have large unrealistic areas, in which case we set the maximum area to be
        90.0% the maximum area of the Voronoi mesh. This large area values ensures that the pixels are regularized
        with large regularization coefficients, which is preferred at the edge of the mesh where the reconstruction
        goes to zero.
        """
        areas = self._xp.asarray(self.delaunay.voronoi_areas)

        # 90th percentile
        max_area = self._xp.percentile(areas, 90.0)

        if self._xp is np:
            # NumPy allows in-place mutation
            areas[areas == -1] = max_area
            areas[areas > max_area] = max_area
            return areas

        # JAX arrays are immutable → use .at[]
        areas = self._xp.where(areas == -1, max_area, areas)
        areas = self._xp.where(areas > max_area, max_area, areas)
        return areas

    @property
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
                np.asarray([self.array[:, 1], self.array[:, 0]]).T,
                qhull_options="Qbb Qc Qx Qm",
            )
        except (ValueError, OverflowError, QhullError) as e:
            raise exc.MeshException() from e

    @property
    def neighbors(self) -> Neighbors:
        """
        Returns a ndarray describing the neighbors of every pixel in a Delaunay triangulation, where a neighbor is
        defined as two Delaunay triangles which are directly connected to one another in the triangulation.

        see `Neighbors` for a complete description of the neighboring scheme.

        The neighbors of a Voronoi mesh are computed using the `ridge_points` attribute of the scipy `Voronoi`
        object, as described in the method `mesh_util.voronoi_neighbors_from`.
        """
        indptr, indices = self.delaunay.vertex_neighbor_vertices

        sizes = indptr[1:] - indptr[:-1]

        neighbors = -1 * np.ones(
            shape=(self.parameters, int(np.max(sizes))), dtype="int"
        )

        for k in range(self.parameters):
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
        interpolation_grid = self.interpolation_grid_from(
            shape_native=shape_native, extent=extent
        )

        interpolated_array = mesh_numba_util.delaunay_interpolated_array_from(
            shape_native=shape_native,
            interpolation_grid_slim=np.array(interpolation_grid.slim.array),
            delaunay=self.delaunay,
            pixel_values=values,
        )

        return Array2D.no_mask(
            values=interpolated_array, pixel_scales=interpolation_grid.pixel_scales
        )

    @property
    def areas_for_magnification(self) -> np.ndarray:
        """
        Returns the area of every Voronoi pixel in the Voronoi mesh.

        Pixels at boundaries can sometimes have large unrealistic areas, which can impact the magnification
        calculation. This method therefore sets their areas to zero so they do not impact the magnification
        calculation.
        """
        areas = self.delaunay.voronoi_areas

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
