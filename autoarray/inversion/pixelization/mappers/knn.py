import numpy as np
from autoconf import cached_property

from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper, PixSubWeights

from autoarray.inversion.pixelization.mesh.knn import get_interpolation_weights


class MapperKNNInterpolator(AbstractMapper):
    """
    Mapper using kNN + compact Wendland kernel interpolation (partition of unity).
    """

    def _pix_sub_weights_from_query_points(self, query_points) -> PixSubWeights:
        """
        Compute PixSubWeights for arbitrary query points using the kNN kernel module.
        Arrays are created in self._xp (numpy or jax.numpy) from the start.
        """

        k_neighbors = 10
        kernel = 'wendland_c4'
        radius_scale = 1.5

        xp = self._xp  # numpy or jax.numpy

        # ------------------------------------------------------------------
        # Source nodes (pixelization "pixels") on the source-plane mesh grid
        # Shape: (N, 2)
        # ------------------------------------------------------------------
        points = xp.asarray(self.source_plane_mesh_grid.array, dtype=xp.float64)

        # ------------------------------------------------------------------
        # Query points (oversampled source-plane data grid or split points)
        # Shape: (M, 2)
        # ------------------------------------------------------------------
        query_points = xp.asarray(query_points, dtype=xp.float64)

        # ------------------------------------------------------------------
        # kNN kernel weights (runs in JAX, but accepts NumPy or JAX inputs)
        # Always returns JAX arrays
        # ------------------------------------------------------------------
        weights_jax, indices_jax, _ = get_interpolation_weights(
            points=points,
            query_points=query_points,
            k_neighbors=int(k_neighbors),
            kernel=kernel,
            radius_scale=float(radius_scale),
        )

        # ------------------------------------------------------------------
        # Convert outputs to xp backend *only if needed*
        # ------------------------------------------------------------------
        if xp is jnp:
            weights = weights_jax
            mappings = indices_jax
        else:
            # xp is numpy
            weights = np.asarray(weights_jax)
            mappings = np.asarray(indices_jax)

        # ------------------------------------------------------------------
        # Sizes: always k for kNN
        # Shape: (M,)
        # ------------------------------------------------------------------
        sizes = xp.full(
            (mappings.shape[0],),
            mappings.shape[1],
            dtype=xp.int32,
        )

        # Ensure correct dtypes
        mappings = mappings.astype(xp.int32)
        weights = weights.astype(xp.float64)

        return PixSubWeights(
            mappings=mappings,
            sizes=sizes,
            weights=weights,
        )

    @cached_property
    def pix_sub_weights(self) -> PixSubWeights:
        """
        kNN mappings + kernel weights for every oversampled source-plane data-grid point.
        """
        return self._pix_sub_weights_from_query_points(
            query_points=self.source_plane_data_grid.over_sampled
        )

    @property
    def pix_sub_weights_split_points(self) -> PixSubWeights:
        """
        kNN mappings + kernel weights computed at split points (for split regularization schemes).
        """
        # Your Delaunay mesh exposes split points via self.delaunay.split_points.
        # For KNN mesh, you should expose the same property. If not, route appropriately:
        #   split_points = self.mesh.split_points
        split_points = self.delaunay.split_points  # keep consistent with existing API

        return self._pix_sub_weights_from_query_points(query_points=split_points)
