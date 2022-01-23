import numpy as np
import scipy.spatial
import scipy.spatial.qhull as qhull
from typing import Optional, List, Union, Tuple

from autoconf import cached_property

from autoarray.structures.abstract_structure import AbstractStructure2D
from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.structures.grids.two_d import grid_2d_util
from autoarray.inversion.pixelizations import pixelization_util

from autoarray import type as ty


class PixelNeighbors(np.ndarray):
    def __new__(cls, arr: np.ndarray, sizes: np.ndarray):
        """
        Class packaging ndarrays describing the neighbors of every pixel in a pixelization (e.g. `Rectangular`,
        `Voronoi`).

        The array `arr` contains the pixel indexes of the neighbors of every pixel. Its has shape [total_pixels,
        max_neighbors_in_single_pixel].

        The array `sizes` contains the number of neighbors of every pixel in the pixelixzation.

        For example, for a 3x3 `Rectangular` grid:

        - `total_pixels=9` and `max_neighbors_in_single_pixel=4` (because the central pixel has 4 neighbors whereas
        edge / corner pixels have 3 and 2).

        - The shape of `arr` is therefore [9, 4], with entries where there is no neighbor (e.g. arr[0, 3]) containing
        values of -1.

        - Pixel 0 is at the top-left of the rectangular pixelization and has two neighbors, the pixel to its right
        (with index 1) and the pixel below it (with index 3). Therefore, `arr[0,:] = [1, 3, -1, -1]` and `sizes[0] = 2`.

        - Pixel 1 is at the top-middle and has three neighbors, to its left (index 0, right (index 2) and below it
        (index 4). Therefore, pixel_neighbors[1,:] = [0, 2, 4, 1] and pixel_neighbors_sizes[1] = 3.

        - For pixel 4, the central pixel, pixel_neighbors[4,:] = [1, 3, 5, 7] and pixel_neighbors_sizes[4] = 4.

        The same arrays can be generalized for other pixelizations, for example a `Voronoi` grid.

        Parameters
        ----------
        arr
            An array which maps every pixelization pixel to the indexes of its neighbors.
        sizes
            An array containing the number of neighbors of every pixelization pixel.
        """
        obj = arr.view(cls)
        obj.sizes = sizes

        return obj


class Grid2DRectangular(AbstractStructure2D):
    def __new__(
        cls,
        grid: np.ndarray,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        *args,
        **kwargs
    ):
        """
        A grid of (y,x) coordinates which represent a uniform rectangular pixelization.

        A `Grid2DRectangular` is ordered such pixels begin from the top-row and go rightwards and then downwards.
        It is an ndarray of shape [total_pixels, 2], where the first dimension of the ndarray corresponds to the
        pixelization's pixel index and second element whether it is a y or x arc-second coordinate.

        For example:

        - grid[3,0] = the y-coordinate of the 4th pixel in the rectangular pixelization.
        - grid[6,1] = the x-coordinate of the 7th pixel in the rectangular pixelization.

        This class is used in conjuction with the `inversion/pixelizations` package to create rectangular pixelizations
        and mappers that perform an `Inversion`.

        Parameters
        -----------
        grid
            The grid of (y,x) coordinates corresponding to the centres of each pixel in the rectangular pixelization.
        shape_native
            The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
        pixel_scales
            The (y,x) scaled units to pixel units conversion factors of every pixel. If this is input as a `float`,
            it is converted to a (float, float) structure.
        origin
            The (y,x) origin of the pixelization.
        nearest_pixelization_index_for_slim_index
            A 1D array that maps every grid pixel to its nearest pixelization-grid pixel.
        """

        mask = Mask2D.unmasked(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
        )

        obj = grid.view(cls)
        obj.mask = mask

        return obj

    @classmethod
    def overlay_grid(
        cls, shape_native: Tuple[int, int], grid: np.ndarray, buffer: float = 1e-8
    ) -> "Grid2DRectangular":
        """
        Creates a `Grid2DRecntagular` by overlaying the rectangular pixelization over an input grid of (y,x)
        coordinates.

        This is performed by first computing the minimum and maximum y and x coordinates of the input grid. A
        rectangular pixelization with dimensions `shape_native` is then laid over the grid using these coordinates,
        such that the extreme edges of this rectangular pixelization overlap these maximum and minimum (y,x) coordinates.

        A a `buffer` can be included which increases the size of the rectangular pixelization, placing additional
        spacing beyond these maximum and minimum coordinates.

        Parameters
        -----------
        shape_native
            The 2D dimensions of the rectangular pixelization with shape (y_pixels, x_pixel).
        grid
            A grid of (y,x) coordinates which the rectangular pixelization is laid-over.
        buffer
            The size of the extra spacing placed between the edges of the rectangular pixelization and input grid.
        """

        y_min = np.min(grid[:, 0]) - buffer
        y_max = np.max(grid[:, 0]) + buffer
        x_min = np.min(grid[:, 1]) - buffer
        x_max = np.max(grid[:, 1]) + buffer

        pixel_scales = (
            float((y_max - y_min) / shape_native[0]),
            float((x_max - x_min) / shape_native[1]),
        )

        origin = ((y_max + y_min) / 2.0, (x_max + x_min) / 2.0)

        grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sub_size=1,
            origin=origin,
        )

        return Grid2DRectangular(
            grid=grid_slim,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )

    @cached_property
    def pixel_neighbors(self) -> PixelNeighbors:
        """
        A class packing the ndarrays describing the neighbors of every pixel in the rectangular pixelization (see
        `PixelNeighbors` for a complete description of the neighboring scheme).

        The neighbors of a rectangular pixelization are computed by exploiting the uniform and symmetric nature of the
        rectangular grid, as described in the method `pixelization_util.rectangular_neighbors_from`.
        """
        neighbors, sizes = pixelization_util.rectangular_neighbors_from(
            shape_native=self.shape_native
        )

        return PixelNeighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))

    @property
    def pixels(self) -> int:
        """
        The total number of pixels in the rectangular pixelization.
        """
        return self.shape_native[0] * self.shape_native[1]

    @property
    def shape_native_scaled(self) -> Tuple[float, float]:
        """
        The (y,x) 2D shape of the rectangular pixelization in scaled units, computed from the 2D `shape_native` (units
        pixels) and the `pixel_scales` (units scaled/pixels) conversion factor.
        """
        return (
            (self.shape_native[0] * self.pixel_scales[0]),
            (self.shape_native[1] * self.pixel_scales[1]),
        )

    @property
    def scaled_maxima(self) -> Tuple[float, float]:
        """
        The maximum (y,x) values of the rectangular pixelization in scaled coordinates returned as a
        tuple (y_max, x_max).
        """
        return (
            self.origin[0] + (self.shape_native_scaled[0] / 2.0),
            self.origin[1] + (self.shape_native_scaled[1] / 2.0),
        )

    @property
    def scaled_minima(self) -> Tuple[float, float]:
        """
        The minimum (y,x) values of the rectangular pixelization in scaled coordinates returned as a
        tuple (y_min, x_min).
        """
        return (
            (self.origin[0] - (self.shape_native_scaled[0] / 2.0)),
            (self.origin[1] - (self.shape_native_scaled[1] / 2.0)),
        )

    @property
    def extent(self) -> np.ndarray:
        """
        The extent of the grid in scaled units returned as an ndarray of the form [x_min, x_max, y_min, y_max].

        This follows the format of the extent input parameter in the matplotlib method imshow (and other methods) and
        is used for visualization in the plot module, which is why the x and y coordinates are swapped compared to
        the normal PyAutoArray convention.
        """
        return np.asarray(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )


class AbstractGrid2DMeshTriangulation(AbstractStructure2D):
    def __new__(
        cls,
        grid: Union[np.ndarray, List],
        nearest_pixelization_index_for_slim_index: Optional[np.ndarray] = None,
        uses_interpolation: bool = False,
        *args,
        **kwargs
    ):
        """
        An irregular 2D grid of (y,x) coordinates which represents both a Delaunay triangulation and Voronoi mesh.

        The input irregular `2D` grid represents both of the following quantities:

        - The corners of the Delaunay triangulles used to construct a Delaunay triangulation.
        - The centers of a Voronoi pixels used to constract a Voronoi mesh.

        These reflect the closely related geometric properties of the Delaunay and Voronoi grids, whereby the corner
        points of Delaunay triangles by definition represent the centres of the corresponding Voronoi mesh.

        Different pixelizations, mappers and regularization schemes combine the the Delaunay and Voronoi
        geometries in different ways to perform an Inversion. Thus, having all geometric methods contained in the
        single class here is necessary.

        The input `grid` of source pixel centres is ordered arbitrarily, given that there is no regular pattern
        for a Delaunay triangulation and Voronoi mesh's indexing to follow.

        This class is used in conjuction with the `inversion/pixelizations` package to create Voronoi pixelizations
        and mappers that perform an `Inversion`.

        Parameters
        -----------
        grid
            The grid of (y,x) coordinates corresponding to the Delaunay triangle corners and Voronoi pixel centres.
        nearest_pixelization_index_for_slim_index
            When a Voronoi grid is used to create a mapper and inversion, there are mappings between the `data` pixels
            and Voronoi pixelization. This array contains these mappings and it is used to speed up the creation of the
            mapper.
        """

        if type(grid) is list:
            grid = np.asarray(grid)

        obj = grid.view(cls)
        obj.nearest_pixelization_index_for_slim_index = (
            nearest_pixelization_index_for_slim_index
        )
        obj.uses_interpolation = uses_interpolation

        return obj

    def __array_finalize__(self, obj: object):
        """
        Ensures that the attributes `nearest_pixelization_index_for_slim_index` and `uses_interpolation` are retained
        when numpy array calculations are performed.
        """
        if hasattr(obj, "nearest_pixelization_index_for_slim_index"):
            self.nearest_pixelization_index_for_slim_index = (
                obj.nearest_pixelization_index_for_slim_index
            )

        if hasattr(obj, "uses_interpolation"):
            self.uses_interpolation = obj.uses_interpolation

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
        `PixelizationException`, which helps exception handling in the `inversion` package.
        """
        try:
            return scipy.spatial.Delaunay(np.asarray([self[:, 0], self[:, 1]]).T)
        except (ValueError, OverflowError, scipy.spatial.qhull.QhullError) as e:
            raise exc.PixelizationException() from e

    @cached_property
    def voronoi(self) -> scipy.spatial.Voronoi:
        """
        Returns a `scipy.spatial.Voronoi` object from the 2D (y,x) grid of irregular coordinates, which correspond to
        the centre of every Voronoi pixel.

        This object contains numerous attributes describing a Voronoi mesh. PyAutoArray uses
        the `vertex_neighbor_vertices` attribute to determine the neighbors of every Delaunay triangle.

        There are numerous exceptions that `scipy.spatial.Delaunay` may raise when the input grid of coordinates used
        to compute the Delaunay triangulation are ill posed. These exceptions are caught and combined into a single
        `PixelizationException`, which helps exception handling in the `inversion` package.
        """
        try:
            return scipy.spatial.Voronoi(
                np.asarray([self[:, 1], self[:, 0]]).T, qhull_options="Qbb Qc Qx Qm"
            )
        except (ValueError, OverflowError, scipy.spatial.qhull.QhullError) as e:
            raise exc.PixelizationException() from e

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
        return self[self.mask.sub_border_flat_indexes]

    @property
    def origin(self) -> Tuple[float, float]:
        """
        The (y,x) origin of the Voronoi grid, which is fixed to (0.0, 0.0) for simplicity.
        """
        return 0.0, 0.0

    @property
    def pixels(self) -> int:
        """
        The total number of pixels in the Voronoi pixelization.
        """
        return self.shape[0]

    @property
    def shape_native_scaled(self) -> Tuple[float, float]:
        """
        The (y,x) 2D shape of the Voronoi pixelization in scaled units, computed from the minimum and maximum y and x v
        alues of the pixelization.
        """
        return (
            np.amax(self[:, 0]).astype("float") - np.amin(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float") - np.amin(self[:, 1]).astype("float"),
        )

    @property
    def scaled_maxima(self) -> Tuple[float, float]:
        """
        The maximum (y,x) values of the Voronoi pixelization in scaled coordinates returned as a tuple (y_max, x_max).
        """
        return (
            np.amax(self[:, 0]).astype("float"),
            np.amax(self[:, 1]).astype("float"),
        )

    @property
    def scaled_minima(self) -> Tuple[float, float]:
        """
        The minimum (y,x) values of the Voronoi pixelization in scaled coordinates returned as a tuple (y_min, x_min).
        """
        return (
            np.amin(self[:, 0]).astype("float"),
            np.amin(self[:, 1]).astype("float"),
        )

    @property
    def extent(self) -> np.ndarray:
        return np.array(
            [
                self.scaled_minima[1],
                self.scaled_maxima[1],
                self.scaled_minima[0],
                self.scaled_maxima[0],
            ]
        )


class Grid2DVoronoi(AbstractGrid2DMeshTriangulation):
    @cached_property
    def pixel_neighbors(self) -> PixelNeighbors:
        """
        Returns a ndarray describing the neighbors of every pixel in a Voronoi mesh, where a neighbor is defined as
        two Voronoi cells which share an adjacent vertex.

        see `PixelNeighbors` for a complete description of the neighboring scheme.

        The neighbors of a Voronoi pixelization are computed using the `ridge_points` attribute of the scipy `Voronoi`
        object, as described in the method `pixelization_util.voronoi_neighbors_from`.
        """
        neighbors, sizes = pixelization_util.voronoi_neighbors_from(
            pixels=self.pixels, ridge_points=np.asarray(self.voronoi.ridge_points)
        )

        return PixelNeighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))

    @classmethod
    def manual_slim(cls, grid) -> "Grid2DVoronoi":
        """
        Convenience method which mimicks the API of other `Grid2D` objects in PyAutoArray.
        """
        return Grid2DVoronoi(grid=grid)


class Grid2DDelaunay(AbstractGrid2DMeshTriangulation):
    @cached_property
    def pixel_neighbors(self) -> PixelNeighbors:
        """
        Returns a ndarray describing the neighbors of every pixel in a Delaunay triangulation, where a neighbor is
        defined as two Delaunay triangles which are directly connected to one another in the triangulation.

        see `PixelNeighbors` for a complete description of the neighboring scheme.

        The neighbors of a Voronoi pixelization are computed using the `ridge_points` attribute of the scipy `Voronoi`
        object, as described in the method `pixelization_util.voronoi_neighbors_from`.
        """
        indptr, indices = self.delaunay.vertex_neighbor_vertices

        sizes = indptr[1:] - indptr[:-1]

        neighbors = -1 * np.ones(shape=(self.pixels, int(np.max(sizes))), dtype="int")

        for k in range(self.pixels):
            neighbors[k][0 : sizes[k]] = indices[indptr[k] : indptr[k + 1]]

        return PixelNeighbors(arr=neighbors.astype("int"), sizes=sizes.astype("int"))

    @classmethod
    def manual_slim(cls, grid) -> "Grid2DDelaunay":
        return Grid2DDelaunay(grid=grid)
