from autoconf import cached_property

from autoarray.inversion.pixelization.mappers.abstract import (
    PixSubWeights,
)
from autoarray.inversion.pixelization.mappers.delaunay import MapperDelaunay
from autoarray.inversion.pixelization.interpolator.knn import (
    InterpolatorKNearestNeighbor,
)


class MapperKNNInterpolator(MapperDelaunay):

    @property
    def mapper_cls(self):

        from autoarray.inversion.pixelization.mappers.knn import MapperKNNInterpolator

        return MapperKNNInterpolator

    @property
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

    @cached_property
    def pix_sub_weights(self) -> PixSubWeights:
        """
        kNN mappings + kernel weights for every oversampled source-plane data-grid point.
        """
        weights = self.interpolator.weights
        mappings = self.interpolator.mappings

        sizes = self._xp.full(
            (mappings.shape[0],),
            mappings.shape[1],
        )

        return PixSubWeights(
            mappings=mappings,
            sizes=sizes,
            weights=weights,
        )

    @property
    def pix_sub_weights_split_points(self) -> PixSubWeights:
        """
        kNN mappings + kernel weights computed at split points (for split regularization schemes),
        with split-point step sizes derived from kNN local spacing (no Delaunay / simplices).
        """
        from autoarray.inversion.pixelization.interpolator.delaunay import (
            split_points_from,
        )

        neighbor_index = int(self.mesh.k_neighbors) // self.mesh.split_neighbor_division
        # e.g. k=10, division=2 -> neighbor_index=5

        distance_to_self = self.interpolator.distance_to_self  # (N, k_neighbors), col 0 is self

        others = distance_to_self[:, 1:]  # (N, k_neighbors-1)

        # Clamp to valid range (0-based indexing into `others`)
        idx = int(neighbor_index) - 1
        idx = max(0, min(idx, others.shape[1] - 1))

        r_k = others[:, idx]  # (N,)

        # Split cross step size (length): sqrt(area) ~ r_k
        split_step = self.mesh.areas_factor * r_k  # (N,)

        # Split points (xp-native)
        split_points = split_points_from(
            points=self.source_plane_mesh_grid.array,
            area_weights=split_step,
            xp=self._xp,
        )

        interpolator = InterpolatorKNearestNeighbor(
            mesh=self.mesh,
            mesh_grid=self.source_plane_mesh_grid,
            data_grid=split_points,
            preloads=self.preloads,
            _xp=self._xp,
        )

        mappings = interpolator.mappings
        weights = interpolator.weights

        sizes = self._xp.full(
            (mappings.shape[0],),
            mappings.shape[1],
        )

        return PixSubWeights(
            mappings=mappings,
            sizes=sizes,
            weights=weights,
        )
