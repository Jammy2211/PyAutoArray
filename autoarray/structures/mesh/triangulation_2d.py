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


def circumcenters_from(points, simplices, xp=np):
    """
    Compute triangle circumcenters for a Delaunay triangulation, using either NumPy or JAX.

    Parameters
    ----------
    points : (N, 2)
    simplices : (M, 3)
    xp : np or jnp

    Returns
    -------
    circumcenters : (M, 2)
    """
    pts = xp.asarray(points)
    tris = xp.asarray(simplices, dtype=int)

    tri_pts = pts[tris]  # (M, 3, 2)

    x0 = tri_pts[:, 0, 0]
    y0 = tri_pts[:, 0, 1]
    x1 = tri_pts[:, 1, 0]
    y1 = tri_pts[:, 1, 1]
    x2 = tri_pts[:, 2, 0]
    y2 = tri_pts[:, 2, 1]

    a = 2.0 * (x1 - x0)
    b = 2.0 * (y1 - y0)
    c = 2.0 * (x2 - x0)
    d = 2.0 * (y2 - y0)

    v1 = (x1**2 + y1**2) - (x0**2 + y0**2)
    v2 = (x2**2 + y2**2) - (x0**2 + y0**2)

    det = a * d - b * c
    detx = v1 * d - v2 * b
    dety = a * v2 - c * v1

    x = detx / det
    y = dety / det

    centers = xp.stack([x, y], axis=1)  # (M, 2)
    return centers


MAX_DEG_JAX = 64

def voronoi_areas_via_delaunay_from(points, simplices, xp=np):
    """
    Compute 'Voronoi-ish' cell areas for each vertex in a 2D Delaunay triangulation.

    For each vertex v:
      - find all incident triangles
      - get their circumcenters
      - sort these circumcenters by angle around v
      - compute polygon area by the shoelace formula

    Parameters
    ----------
    points : (N, 2)
        Delaunay vertices.
    simplices : (M, 3)
        Delaunay triangles (indices into `points`).
    xp : np or jnp
        Backend.

    Returns
    -------
    areas : (N,)
        Voronoi-ish area associated with each vertex.
    """
    import jax
    import jax.numpy as jnp

    pts = xp.asarray(points)
    tris = xp.asarray(simplices, dtype=int)
    N = pts.shape[0]
    M = tris.shape[0]

    # 1) Circumcenters for all triangles
    centers = circumcenters_from(pts, tris, xp=xp)  # (M, 2)

    # 2) Build a flattened vertex-triangle incidence
    # Each triangle contributes its circumcenter to 3 vertices.
    vert_ids = tris.reshape(-1)           # (3M,)
    centers_rep = xp.repeat(centers, 3, axis=0)  # (3M, 2)

    # 3) Sort by vertex id so all entries for a given vertex are contiguous
    order = xp.argsort(vert_ids)
    vert_sorted = vert_ids[order]        # (3M,)
    centers_sorted = centers_rep[order]  # (3M, 2)

    # 4) Compute how many triangles are incident to each vertex
    if xp is np:
        counts = xp.bincount(vert_sorted, minlength=N)   # (N,)
        max_deg = int(counts.max())
    else:
        counts = xp.bincount(vert_sorted, length=N)     # (N,)
        max_deg = MAX_DEG_JAX                            # static upper bound

    # 5) Compute start index for each vertex's block in vert_sorted
    #    start[v] = cumulative sum of counts up to v
    start = xp.concatenate([xp.array([0], dtype=int), xp.cumsum(counts[:-1])])

    # Global indices 0..3M-1
    arange_all = xp.arange(3 * M, dtype=int)

    # Position within each vertex block: pos = i - start[vertex]
    start_per_entry = start[vert_sorted]       # (3M,)
    pos = arange_all - start_per_entry         # (3M,)

    # 6) Scatter into a padded (N, max_deg, 2) array of circumcenters
    circum_padded = xp.zeros((N, max_deg, 2), dtype=pts.dtype)
    if xp is np:
        circum_padded[vert_sorted, pos, :] = centers_sorted
    else:
        circum_padded = circum_padded.at[vert_sorted, pos, :].set(centers_sorted)

    # 7) For each vertex, sort its circumcenters by angle around the vertex
    # Compute angles: (N, max_deg)
    dx = circum_padded[..., 0] - pts[:, None, 0]
    dy = circum_padded[..., 1] - pts[:, None, 1]
    angles = xp.arctan2(dy, dx)

    # Mark which slots are valid (j < count[v])
    j_idx = xp.arange(max_deg)[None, :]
    valid_mask = j_idx < counts[:, None]        # (N, max_deg)

    # For invalid entries, set angle to a big constant so they go to the end
    big_angle = xp.array(1e9, dtype=angles.dtype)
    angles_masked = xp.where(valid_mask, angles, big_angle)

    # Sort indices by angle for each vertex
    order_angles = xp.argsort(angles_masked, axis=1)   # (N, max_deg)

    # Reorder circumcenters accordingly
    centers_sorted2 = xp.take_along_axis(
        circum_padded,
        order_angles[..., None].repeat(2, axis=2),
        axis=1,
    )  # (N, max_deg, 2)

    # 8) Compute polygon area with shoelace formula per vertex
    x = centers_sorted2[..., 0]  # (N, max_deg)
    y = centers_sorted2[..., 1]  # (N, max_deg)

    # roll by -1 so j+1 wraps around
    x_next = xp.roll(x, shift=-1, axis=1)
    y_next = xp.roll(y, shift=-1, axis=1)

    # A contribution is valid if both current and next vertices are valid
    valid_pair = valid_mask & xp.roll(valid_mask, shift=-1, axis=1)

    cross = x * y_next - x_next * y
    cross = xp.where(valid_pair, cross, 0.0)

    area = 0.5 * xp.abs(xp.sum(cross, axis=1))  # (N,)

    return area


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

    def __init__(self, ppoints, simplices, vertex_neighbor_vertices):

        self.points = ppoints
        self.simplices = simplices
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

        return DelaunayInterface(points, simplices, vertex_neighbor_vertices)

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

        half_region_area_sqrt_lengths = 0.5 * self._xp.sqrt(
            self.voronoi_pixel_areas_for_split
        )

        return split_points_from(
            points=self.array,
            area_weights=half_region_area_sqrt_lengths,
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
        return voronoi_areas_via_delaunay_from(
            points=self.delaunay.points,
            simplices=self.delaunay.simplices,
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
        areas = self._xp.asarray(self.voronoi_pixel_areas)

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
