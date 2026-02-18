import numpy as np

from autoarray.inversion.pixelization.mesh.delaunay_2d import Mesh2DDelaunay


class Mesh2DDelaunayKNN(Mesh2DDelaunay):

    def mesh_grid_from(
        self,
        source_plane_data_grid=None,
        source_plane_mesh_grid=None,
        preloads=None,
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

        return Mesh2DDelaunayKNN(
            values=source_plane_mesh_grid,
            source_plane_data_grid_over_sampled=source_plane_data_grid,
            preloads=preloads,
            _xp=xp,
        )
