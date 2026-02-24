from typing import Optional

from autoarray.inversion.mesh.mesh.delaunay import Delaunay


class KNearestNeighbor(Delaunay):

    def __init__(
        self,
        pixels: int,
        k_neighbors=10,
        radius_scale=1.5,
        areas_factor=0.5,
        split_neighbor_division=2,
        zeroed_pixels: Optional[int] = 0,
    ):

        self.k_neighbors = k_neighbors
        self.radius_scale = radius_scale
        self.areas_factor = areas_factor
        self.split_neighbor_division = split_neighbor_division

        super().__init__(pixels=pixels, zeroed_pixels=zeroed_pixels)

    @property
    def skip_areas(self):
        """
        Whether to skip barycentric  area calculations and split point computations during Delaunay triangulation.
        When True, the Delaunay interface returns only the minimal set of outputs (points, simplices, mappings)
        without computing split_points or splitted_mappings. This optimization is useful for regularization
        schemes like Matérn kernels that don't require area-based calculations. Default is False.
        """
        return False

    @property
    def interpolator_cls(self):

        from autoarray.inversion.mesh.interpolator.knn import (
            InterpolatorKNearestNeighbor,
        )

        return InterpolatorKNearestNeighbor
