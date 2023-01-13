import numpy as np

from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.sparse_2d import Grid2DSparse
from autoarray.structures.mesh.delaunay_2d import Mesh2DDelaunay
from autoarray.inversion.pixelization.mesh.triangulation import Triangulation
from autoarray.inversion.pixelization.settings import SettingsPixelization

from autoarray.numba_util import profile_func


class Delaunay(Triangulation):
    def __init__(self):
        """
        An irregular mesh of Delaunay triangle pixels, which using linear barycentric interpolation are paired with
        a 2D grid of (y,x) coordinates.

        For a full description of how a mesh is paired with another grid,
        see the :meth:`Pixelization API documentation <autoarray.inversion.pixelization.pixelization.Pixelization>`.

        The Delaunay mesh represents pixels as an irregular 2D grid of Delaunay triangles.

        - ``image_plane_data_grid``: The observed data grid in the image-plane (which is paired with the mesh in
          the source-plane).
        - ``image_plane_mesh_grid``: The (y,x) mesh coordinates in the image-plane (which are the corners of Delaunay
          triangles in the source-plane).
        - ``source_plane_data_grid``: The observed data grid mapped to the source-plane after gravitational lensing.
        - ``source_plane_mesh_grid``: The corner of each Delaunay triangle in the source-plane
          (the ``image_plane_mesh_grid`` maps to this after gravitational lensing).

        Each (y,x) coordinate in the ``source_plane_data_grid`` is paired with the three nearest Delaunay triangle
        corners, using a weighted interpolation scheme.

        Coordinates on the ``source_plane_data_grid`` are therefore given higher weights when paired with Delaunay
        triangle corners they are a closer distance to.
        """
        super().__init__()

    @property
    def uses_interpolation(self):
        return False

    @profile_func
    def mesh_grid_from(
        self,
        source_plane_data_grid=None,
        source_plane_mesh_grid=None,
        sparse_index_for_slim_index=None,
    ):
        """
        Return the Delaunay ``source_plane_mesh_grid`` as a ``Mesh2DDelaunay`` object, which provides additional
        functionality for performing operations that exploit the geometry of a Delaunay mesh.

        Parameters
        ----------
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            ``source`` reference frame.
        source_plane_mesh_grid
            The centres of every Delaunay pixel in the ``source`` frame, which are initially derived by computing a sparse
            set of (y,x) coordinates computed from the unmasked data in the image-plane and applying a transformation
            to this.
        settings
            Settings controlling the pixelization for example if a border is used to relocate its exterior coordinates.
        """

        return Mesh2DDelaunay(
            values=source_plane_mesh_grid, uses_interpolation=self.uses_interpolation
        )


class DelaunayMagnification(Delaunay):
    def __init__(self, shape=(3, 3)):
        """
        An irregular mesh of Delaunay triangle pixels, which using linear barycentric interpolation are paired with
        a 2D grid of (y,x) coordinates. The Delaunay corners are derived in the image-plane by overlaying a uniform
        grid over the image.

        For a full description of how a mesh is paired with another grid,
        see the :meth:`Pixelization API documentation <autoarray.inversion.pixelization.pixelization.Pixelization>`.

        The Delaunay mesh represents pixels as an irregular 2D grid of Delaunay triangles.

        - ``image_plane_data_grid``: The observed data grid in the image-plane (which is paired with the mesh in
          the source-plane).
        - ``image_plane_mesh_grid``: The (y,x) mesh coordinates in the image-plane (which are the corners of Delaunay
          triangles in the source-plane).
        - ``source_plane_data_grid``: The observed data grid mapped to the source-plane after gravitational lensing.
        - ``source_plane_mesh_grid``: The corner of each Delaunay triangle in the source-plane
          (the ``image_plane_mesh_grid`` maps to this after gravitational lensing).

        Each (y,x) coordinate in the ``source_plane_data_grid`` is paired with the three nearest Delaunay triangle
        corners, using a weighted interpolation scheme. Coordinates on the ``source_plane_data_grid`` are therefore
        given higher weights when paired with Delaunay triangle corners they are a closer distance to.

        The corners of the Delaunay pixels are derived in the image-plane by overlaying a uniform grid with the
        input ``shape`` over the masked image data's grid. All coordinates in this uniform grid which are contained
        within the mask are kept and mapped to the source-plane via gravitational lensing, where they form
        the Delaunay pixel corners.

        Parameters
        ----------
        shape
            The shape of the unmasked uniform grid in the image-plane which is laid over the masked image, in
            order to derive the image-plane (y,x) coordinates which act as the corners of the Delaunay pixels after
            being mapped to the source-plane via gravitational lensing.
        """
        super().__init__()
        self.shape = (int(shape[0]), int(shape[1]))
        self.pixels = self.shape[0] * self.shape[1]

    def image_plane_mesh_grid_from(
        self,
        image_plane_data_grid: Grid2D,
        hyper_data: np.ndarray = None,
        settings=SettingsPixelization(),
    ):
        """
        Computes the ``mesh_grid`` in the image-plane, by overlaying a uniform grid of coordinates over the
        masked 2D data (see ``Grid2DSparse.from_grid_and_unmasked_2d_grid_shape()``).

        For a ``DelaunayMagnification`` this grid is computed by overlaying a 2D grid with dimensions ``shape`` over the
        masked 2D data in the image-plane, whereby all (y,x) coordinates in this grid which are not masked are
        retained.

        Parameters
        ----------
        image_plane_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the image-plane. This has a
            transformation applied to it to create the ``source_plane_mesh_grid``.
        hyper_data
            An image which is used to determine the ``image_plane_mesh_grid`` and therefore adapt the distribution of
            pixels of the Delaunay grid to the data it discretizes.
        settings
            Settings controlling the pixelization for example if a border is used to relocate its exterior coordinates.
        """
        return Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
            grid=image_plane_data_grid, unmasked_sparse_shape=self.shape
        )


class DelaunayBrightnessImage(Delaunay):
    def __init__(self, pixels=10, weight_floor=0.0, weight_power=0.0):
        """
        An irregular mesh of Delaunay triangle pixels, which using linear barycentric interpolation are paired with
        a 2D grid of (y,x) coordinates. The Delaunay corners are derived in the image-plane by applying a KMeans
        clustering algorithm to the image's weight map.

        For a full description of how a mesh is paired with another grid,
        see the :meth:`Pixelization API documentation <autoarray.inversion.pixelization.pixelization.Pixelization>`.

        The Delaunay mesh represents pixels as an irregular 2D grid of Delaunay triangles.

        - ``image_plane_data_grid``: The observed data grid in the image-plane (which is paired with the mesh in
          the source-plane).
        - ``image_plane_mesh_grid``: The (y,x) mesh coordinates in the image-plane (which are the corners of Delaunay
          triangles in the source-plane).
        - ``source_plane_data_grid``: The observed data grid mapped to the source-plane after gravitational lensing.
        - ``source_plane_mesh_grid``: The corner of each Delaunay triangle in the source-plane
          (the ``image_plane_mesh_grid`` maps to this after gravitational lensing).

        Each (y,x) coordinate in the ``source_plane_data_grid`` is paired with the three nearest Delaunay triangle
        corners, using a weighted interpolation scheme. Coordinates on the ``source_plane_data_grid`` are therefore
        given higher weights when paired with Delaunay triangle corners they are a closer distance to.

        The corners of the Delaunay pixels are derived in the image plane, by applying a KMeans clustering algorithm
        to the masked image data's weight-map. The ``weight_floor`` and ``weight_power`` allow the KMeans algorithm to
        adapt the image-plane coordinates to the image's brightest or faintest values. The computed valies are
        mapped to the source-plane  via gravitational lensing, where they form the Delaunay pixel corners.

        Parameters
        ----------
        pixels
            The total number of pixels in the mesh, which is therefore also the number of (y,x) coordinates computed
            via the KMeans clustering algorithm in image-plane.
        weight_floor
            A parameter which reweights the data values the KMeans algorithm is applied too; as the floor increases
            more weight is applied to values with lower values thus allowing mesh pixels to be placed in these
            regions of the data.
        weight_power
            A parameter which reweights the data values the KMeans algorithm is applied too; as the power increases
            more weight is applied to values with higher values thus allowing mesh pixels to be placed in these
            regions of the data.
        """
        super().__init__()

        self.pixels = int(pixels)
        self.weight_floor = weight_floor
        self.weight_power = weight_power

    def weight_map_from(self, hyper_data: np.ndarray):
        """
        Computes a ``weight_map`` from an input ``hyper_data``, where this image represents components in the masked 2d
        data in the image-plane. This applies the ``weight_floor`` and ``weight_power`` attributes of the class, which
        scale the weights to make different components upweighted relative to one another.

        Parameters
        ----------
        hyper_data
            A image which represents one or more components in the masked 2D data in the image-plane.

        Returns
        -------
        The weight map which is used to adapt the Delaunay pixels in the image-plane to components in the data.
        """
        weight_map = (hyper_data - np.min(hyper_data)) / (
            np.max(hyper_data) - np.min(hyper_data)
        ) + self.weight_floor * np.max(hyper_data)

        return np.power(weight_map, self.weight_power)

    def image_plane_mesh_grid_from(
        self,
        image_plane_data_grid: Grid2D,
        hyper_data: np.ndarray,
        settings=SettingsPixelization(),
    ):
        """
        Computes the ``mesh_grid`` in the image-plane, by overlaying a uniform grid of coordinates over the
        masked 2D data (see ``Grid2DSparse.from_grid_and_unmasked_2d_grid_shape()``).

        The ``data_pixelization_grid`` is transformed to the ``source_plane_mesh_grid``, and it is these (y,x) values
        which then act the centres of the Delaunay pixelization's pixels.

        For a ``DelaunayBrightnessImage`` this grid is computed by applying a KMeans clustering algorithm to the masked
        data's values, where these values are reweighted by the ``hyper_data`` so that the algorithm can adapt to
        specific parts of the data.

        Parameters
        ----------
        image_plane_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the image-plane. This has a
            transformation applied to it to create the ``source_plane_mesh_grid``.
        hyper_data
            An image which is used to determine the ``image_plane_mesh_grid`` and therefore adapt the distribution of
            pixels of the Delaunay grid to the data it discretizes.
        settings
            Settings controlling the pixelization for example if a border is used to relocate its exterior coordinates.
        """
        weight_map = self.weight_map_from(hyper_data=hyper_data)

        return Grid2DSparse.from_total_pixels_grid_and_weight_map(
            total_pixels=self.pixels,
            grid=image_plane_data_grid,
            weight_map=weight_map,
            seed=settings.kmeans_seed,
            stochastic=settings.is_stochastic,
        )

    @property
    def is_stochastic(self) -> bool:
        return True
