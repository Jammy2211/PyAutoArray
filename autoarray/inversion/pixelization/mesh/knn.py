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
    def mapper_cls(self):

        from autoarray.inversion.pixelization.mappers.knn import MapperKNNInterpolator

        return MapperKNNInterpolator
