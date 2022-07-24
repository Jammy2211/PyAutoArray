from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh


class MockMesh(AbstractMesh):
    def __init__(self, data_mesh_grid=None):

        super().__init__()

        self.data_mesh_grid = data_mesh_grid

    def data_mesh_grid_from(self, data_grid_slim, hyper_data, settings=None):

        if hyper_data is not None and self.data_mesh_grid is not None:
            return hyper_data * self.data_mesh_grid

        return self.data_mesh_grid
