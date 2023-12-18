import numpy as np
import scipy.spatial
from typing import List, Union, Tuple

from autoarray.structures.abstract_structure import Structure
from autoconf import cached_property

from autoarray.geometry.geometry_2d_irregular import Geometry2DIrregular
from autoarray.structures.mesh.abstract_2d import Abstract2DMesh

from autoarray import exc
from autoarray.inversion.pixelization.mesh import mesh_util
from autoarray.structures.grids import grid_2d_util


class Abstract2DMeshTriangulation(Abstract2DMesh):
    @property
    def slim(self) -> "Structure":
        raise NotImplementedError()

    def structure_2d_list_from(self, result_list: list) -> List["Structure"]:
        raise NotImplementedError()

    def structure_2d_from(self, result: np.ndarray) -> "Structure":
        raise NotImplementedError()

    def trimmed_after_convolution_from(self, kernel_shape) -> "Structure":
        raise NotImplementedError()

    @property
    def native(self) -> Structure:
        raise NotImplementedError()

    def __init__(
        self,
        values: Union[np.ndarray, List],
        uses_interpolation: bool = False,
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

        self.uses_interpolation = uses_interpolation

        super().__init__(values)

    def __array_finalize__(self, obj: object):
        """
        Ensures that the attribute `uses_interpolation` are retained when numpy array calculations are performed.
        """

        if hasattr(obj, "uses_interpolation"):
            self.uses_interpolation = obj.uses_interpolation

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
    def delaunay(self) -> scipy.spatial.Delaunay:
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
        try:
            return scipy.spatial.Delaunay(np.asarray([self[:, 0], self[:, 1]]).T)
        except (ValueError, OverflowError, scipy.spatial.qhull.QhullError) as e:
            raise exc.MeshException() from e

    @cached_property
    def voronoi(self) -> scipy.spatial.Voronoi:
        """
        Returns a `scipy.spatial.Voronoi` object from the 2D (y,x) grid of irregular coordinates, which correspond to
        the centre of every Voronoi pixel.

        This object contains numerous attributes describing a Voronoi mesh. PyAutoArray uses
        the `vertex_neighbor_vertices` attribute to determine the neighbors of every Delaunay triangle.

        There are numerous exceptions that `scipy.spatial.Delaunay` may raise when the input grid of coordinates used
        to compute the Delaunay triangulation are ill posed. These exceptions are caught and combined into a single
        `MeshException`, which helps exception handling in the `inversion` package.
        """
        try:
            return scipy.spatial.Voronoi(
                np.asarray([self[:, 1], self[:, 0]]).T, qhull_options="Qbb Qc Qx Qm"
            )
        except (ValueError, OverflowError, scipy.spatial.qhull.QhullError) as e:
            raise exc.MeshException() from e

    @cached_property
    def edge_pixel_list(self) -> List:
        """
        Returns a list of the Voronoi pixel indexes that are on the edge of the mesh.
        """

        return mesh_util.voronoi_edge_pixels_from(
            regions=self.voronoi.regions, point_region=self.voronoi.point_region
        )

    @cached_property
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
        half_region_area_sqrt_lengths = 0.5 * np.sqrt(self.voronoi_pixel_areas)

        splitted_array = np.zeros((self.pixels, 4, 2))

        splitted_array[:, 0][:, 0] = self[:, 0] + half_region_area_sqrt_lengths
        splitted_array[:, 0][:, 1] = self[:, 1]

        splitted_array[:, 1][:, 0] = self[:, 0] - half_region_area_sqrt_lengths
        splitted_array[:, 1][:, 1] = self[:, 1]

        splitted_array[:, 2][:, 0] = self[:, 0]
        splitted_array[:, 2][:, 1] = self[:, 1] + half_region_area_sqrt_lengths

        splitted_array[:, 3][:, 0] = self[:, 0]
        splitted_array[:, 3][:, 1] = self[:, 1] - half_region_area_sqrt_lengths

        return splitted_array.reshape((self.pixels * 4, 2))

    @cached_property
    def voronoi_pixel_areas(self) -> np.ndarray:
        """
        Returns the area of every Voronoi pixel in the Voronoi mesh.

        These areas are used when performing gradient regularization in order to determine the size of the cross of
        points where the derivative is evaluated and therefore where regularization is evaluated (see `split_cross`).

        Pixels at boundaries can sometimes have large unrealistic areas, in which case we set the maximum area to be
        90.0% the maximum area of the Voronoi mesh.
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

        max_area = np.percentile(region_areas, 90.0)

        region_areas[region_areas == -1] = max_area
        region_areas[region_areas > max_area] = max_area

        return region_areas

    @property
    def sub_border_grid(self) -> np.ndarray:
        """
        The (y,x) grid of all sub-pixels which are at the border of the mask.

        This is NOT all sub-pixels which are in mask pixels at the mask's border, but specifically the sub-pixels
        within these border pixels which are at the extreme edge of the border.
        """
        return self[self.mask.derive_indexes.sub_border_slim]

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
