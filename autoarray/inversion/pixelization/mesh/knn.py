from autoarray.inversion.pixelization.mesh.delaunay import Delaunay


class KNearestNeighbor(Delaunay):

    def __init__(self, k_neighbors=10, radius_scale=1.5):

        self.k_neighbors = k_neighbors
        self.radius_scale = radius_scale

        super().__init__()

    @property
    def mapper_cls(self):

        from autoarray.inversion.pixelization.mappers.knn import MapperKNNInterpolator

        return MapperKNNInterpolator
