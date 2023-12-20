from autoarray.structures.mesh.voronoi_2d import Mesh2DVoronoi
from autoarray.inversion.pixelization.mesh.triangulation import Triangulation

from autoarray.numba_util import profile_func


class Voronoi(Triangulation):
    def __init__(self):
        """
        An irregular mesh of Voronoi pixels, which using no interpolation are paired with a 2D grid of (y,x)
        coordinates.

        For a full description of how a mesh is paired with another grid,
        see the :meth:`Pixelization API documentation <autoarray.inversion.pixelization.pixelization.Pixelization>`.

        The Voronoi mesh represents pixels as an irregular 2D grid of Voronoi cells.

        A ``Pixelization`` using a ``Voronoi`` mesh has four grids associated with it:

        - ``image_plane_data_grid``: The observed data grid in the image-plane (which is paired with the mesh in
          the source-plane).

        - ``image_plane_mesh_grid``: The (y,x) mesh coordinates in the image-plane (which are the centres of Voronoi
          cells in the source-plane).

        - ``source_plane_data_grid``: The observed data grid mapped to the source-plane (e.g. after gravitational lensing).

        - ``source_plane_mesh_grid``: The centre of each Voronoi cell in the source-plane
          (the ``image_plane_mesh_grid`` maps to this after gravitational lensing).

        Each (y,x) coordinate in the ``source_plane_data_grid`` is paired with all Voronoi cells it falls within,
        without using an interpolation scheme.
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
    ) -> Mesh2DVoronoi:
        """
        Return the Voronoi `source_plane_mesh_grid` as a `Mesh2DVoronoi` object, which provides additional
        functionality for performing operations that exploit the geometry of a Voronoi mesh.

        Parameters
        ----------
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            The centres of every Voronoi pixel in the `source` frame, which are initially derived by computing a sparse
            set of (y,x) coordinates computed from the unmasked data in the `data` frame and applying a transformation
            to this.
        settings
            Settings controlling the pixelization for example if a border is used to relocate its exterior coordinates.
        """

        return Mesh2DVoronoi(
            values=source_plane_mesh_grid,
            uses_interpolation=self.uses_interpolation,
        )
