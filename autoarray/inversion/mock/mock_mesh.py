from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh


class MockMesh(AbstractMesh):
    def __init__(self, data_mesh_grid=None):

        super().__init__()

        self.data_mesh_grid = data_mesh_grid

    def data_pixelization_grid_from(self, data_grid_slim, hyper_image, settings=None):

        if hyper_image is not None and self.data_mesh_grid is not None:
            return hyper_image * self.data_mesh_grid

        return self.data_mesh_grid
