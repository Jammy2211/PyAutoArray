from autoconf import cached_property

from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay
from autoarray.inversion.pixelization.interpolator.knn import (
    InterpolatorKNearestNeighbor,
)


class MapperKNNInterpolator(MapperDelaunay):

    @cached_property
    def interpolator(self):
        """
        Return the Delaunay ``source_plane_mesh_grid`` as a ``InterpolatorDelaunay`` object, which provides additional
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
        return InterpolatorKNearestNeighbor(
            mesh=self.mesh,
            mesh_grid=self.source_plane_mesh_grid,
            data_grid=self.source_plane_data_grid,
            preloads=self.preloads,
            _xp=self._xp,
        )
