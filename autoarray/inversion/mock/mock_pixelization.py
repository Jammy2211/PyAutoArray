from autoarray.inversion.mesh.abstract import AbstractMesh


class MockPixelization(AbstractMesh):
    def __init__(self, mapper=None, data_mesh_grid=None):

        super().__init__()

        self.mapper = mapper
        self.data_mesh_grid = data_mesh_grid

    # noinspection PyUnusedLocal,PyShadowingNames
    def mapper_from(
        self,
        source_grid_slim,
        source_mesh_grid,
        data_mesh_grid=None,
        hyper_image=None,
        settings=None,
        preloads=None,
        profiling_dict=None,
    ):
        return self.mapper

    def data_pixelization_grid_from(self, data_grid_slim, hyper_image, settings=None):

        if hyper_image is not None and self.data_mesh_grid is not None:
            return hyper_image * self.data_mesh_grid

        return self.data_mesh_grid
