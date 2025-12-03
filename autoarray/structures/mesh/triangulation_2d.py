import numpy as np
import scipy.spatial
from typing import List, Union, Tuple

from autoconf import cached_property

from autoarray.geometry.geometry_2d_irregular import Geometry2DIrregular
from autoarray.structures.mesh.abstract_2d import Abstract2DMesh

from autoarray import exc
from autoarray.inversion.pixelization.mesh import mesh_numba_util
from autoarray.structures.grids import grid_2d_util





def scipy_delaunay_padded(points_np, max_simplices):
    tri = scipy.spatial.Delaunay(points_np)

    pts = tri.points  # same dtype as input
    simplices = tri.simplices.astype(np.int32)

    # Pad simplices to fixed size (max_simplices, 3)
    padded = -np.ones((max_simplices, 3), dtype=np.int32)
    padded[: simplices.shape[0]] = simplices

    return pts, padded


def jax_delaunay(points):

    import jax
    import jax.numpy as jnp

    N = points.shape[0]
    max_simplices = 2 * N

    pts_shape = jax.ShapeDtypeStruct((N, 2), points.dtype)
    simp_shape = jax.ShapeDtypeStruct((max_simplices, 3), jnp.int32)

    return jax.pure_callback(
        lambda pts: scipy_delaunay_padded(pts, max_simplices),
        (pts_shape, simp_shape),
        points,
    )


def find_simplex_from(query_points, points, simplices):
    """
    Return simplex index for each query point.
    Returns -1 where no simplex contains the point.
    """

    import jax
    import jax.numpy as jnp

    # Mask padded simplices (marked with -1)
    valid = simplices[:, 0] >= 0  # (M,)
    simplices_clipped = simplices.clip(min=0)

    # Triangle vertices: (M, 3, 2)
    tri = points[simplices_clipped]

    p0 = tri[:, 0]  # (M, 2)
    p1 = tri[:, 1]
    p2 = tri[:, 2]

    # Edges
    v0 = p1 - p0  # (M, 2)
    v1 = p2 - p0

    # Precomputed dot products
    d00 = jnp.sum(v0 * v0, axis=1)  # (M,)
    d01 = jnp.sum(v0 * v1, axis=1)
    d11 = jnp.sum(v1 * v1, axis=1)
    denom = d00 * d11 - d01 * d01  # (M,)

    # Barycentric computation for each query point vs each triangle
    diff = query_points[:, None, :] - p0[None, :, :]  # (Q, M, 2)

    a = jnp.sum(diff * v0[None, :, :], axis=-1)  # (Q, M)
    b = jnp.sum(diff * v1[None, :, :], axis=-1)

    b0 = (a * d11 - b * d01) / denom  # (Q, M)
    b1 = (b * d00 - a * d01) / denom

    # Inside test
    inside = (b0 >= 0.0) & (b1 >= 0.0) & (b0 + b1 <= 1.0)  # (Q, M)

    # Remove padded simplices
    inside = inside & valid[None, :]

    # First valid simplex per point
    simplex_idx = jnp.argmax(inside, axis=1)  # (Q,)

    # Detect points with no simplex match
    has_match = jnp.any(inside, axis=1)  # (Q,)

    # Replace unmatched with -1
    simplex_idx = jnp.where(has_match, simplex_idx, -1)

    return simplex_idx


def vertex_areas_from_delaunay(points, simplices, xp=np):
    """
    Compute per-vertex areas using:
        vertex_area[v] = sum(area(triangles incident to v)) / 3

    Parameters
    ----------
    points : (N_pts, 2) array
    simplices : (N_tris, 3) array of triangle vertex indices
    xp : np or jnp
        Backend to use.

    Returns
    -------
    vertex_area : (N_pts,) array
        Estimated area associated with each vertex.
    """

    # Triangle vertices
    p0 = points[simplices[:, 0]]  # (N_tris, 2)
    p1 = points[simplices[:, 1]]
    p2 = points[simplices[:, 2]]

    # Compute triangle areas (vectorized)
    tri_area = 0.5 * xp.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )  # (N_tris,)

    # Area contribution to each vertex: (N_tris, 3)
    contrib = (tri_area / 3.0)[:, None] * xp.ones((1, 3))

    # Flatten for scatter:
    # Each triangle contributes 3 entries, one per vertex.
    scatter_idx = simplices.reshape(-1)  # (3*N_tris,)
    scatter_vals = contrib.reshape(-1)  # (3*N_tris,)

    # Allocate output
    n_pts = points.shape[0]
    vertex_area = xp.zeros(n_pts)

    # Scatter-add: NumPy and JAX both support this API!
    if xp.__name__.startswith("jax"):
        vertex_area = vertex_area.at[scatter_idx].add(scatter_vals)
    else:
        np.add.at(vertex_area, scatter_idx, scatter_vals)

    return vertex_area


def split_points_from(points, area_weights, xp=np):
    """
    points : (N, 2)
    areas  : (N,)
    xp     : np or jnp

    Returns (4*N, 2)
    """

    N = points.shape[0]
    offsets = 0.5 * area_weights

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


class DelaunayInterface:

    def __init__(self, ppoints, simplices, areas, vertex_neighbor_vertices):

        self.points = ppoints
        self.simplices = simplices
        self.areas = areas
        self.vertex_neighbor_vertices = vertex_neighbor_vertices

    def find_simplex(self, query_points):
        return find_simplex_from(query_points, self.points, self.simplices)


class Abstract2DMeshTriangulation(Abstract2DMesh):
    def __init__(self, values: Union[np.ndarray, List], _xp=np):
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

            points, simplices = jax_delaunay(mesh_grid)
            vertex_neighbor_vertices = None

        else:

            delaunay = scipy.spatial.Delaunay(mesh_grid)

            points = delaunay.points
            simplices = delaunay.simplices.astype(np.int32)
            vertex_neighbor_vertices = delaunay.vertex_neighbor_vertices

        areas = vertex_areas_from_delaunay(
            points=points, simplices=simplices, xp=self._xp
        )

        return DelaunayInterface(points, simplices, areas, vertex_neighbor_vertices)

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

        return split_points_from(
            points=self.array,
            area_weights=0.5 * self.delaunay.areas,
            xp=self._xp,
        )

    @property
    def voronoi_pixel_areas(self) -> np.ndarray:
        """
        Returns the area of every Voronoi pixel in the Voronoi mesh.

        Pixels at boundaries can sometimes have large unrealistic areas, in which case we set the maximum area to be
        an input value of N% the maximum area of the Voronoi mesh, which this value is suitable for different
        calculations.
        """
        voronoi_vertices = self.voronoi.vertices
        voronoi_regions = self.voronoi.regions
        voronoi_point_region = self.voronoi.point_region
        region_areas = np.zeros(self.pixels)

        for i in range(self.pixels):
            region_vertices_indexes = voronoi_regions[voronoi_point_region[i]]
            if -1 in region_vertices_indexes:
                region_areas[i] = -1
            else:
                region_areas[i] = grid_2d_util.compute_polygon_area(
                    voronoi_vertices[region_vertices_indexes]
                )

        return region_areas

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
        areas = self.voronoi_pixel_areas

        max_area = np.percentile(areas, 90.0)

        areas[areas == -1] = max_area
        areas[areas > max_area] = max_area

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
