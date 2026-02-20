from autoarray.inversion.pixelization.mesh.delaunay import Delaunay


class KNearestNeighbor(Delaunay):

    def __init__(
        self,
        k_neighbors=10,
        radius_scale=1.5,
        areas_factor=0.5,
        split_neighbor_division=2,
    ):

        self.k_neighbors = k_neighbors
        self.radius_scale = radius_scale
        self.areas_factor = areas_factor
        self.split_neighbor_division = split_neighbor_division

        super().__init__()

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
    def mapper_cls(self):

        from autoarray.inversion.pixelization.mappers.knn import MapperKNNInterpolator

        return MapperKNNInterpolator
