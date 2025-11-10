import numpy as np

from autoarray.structures.mesh.delaunay_2d import Mesh2DDelaunay
from autoarray.inversion.pixelization.mesh.triangulation import Triangulation


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

    def mesh_grid_from(
        self,
        source_plane_data_grid=None,
        source_plane_mesh_grid=None,
        xp=np,
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
            values=source_plane_mesh_grid,
        )
