from autoarray.inversion.pixelization.pixelization import Pixelization


class MockPixelization(Pixelization):
    def __init__(
        self, mesh=None, regularization=None, mapper=None, data_mesh_grid=None
    ):

        super().__init__(mesh=mesh, regularization=regularization)

        self.mapper = mapper
        self.data_mesh_grid = data_mesh_grid

    # noinspection PyUnusedLocal,PyShadowingNames
    def mapper_grids_from(
        self,
        source_grid_slim,
        source_mesh_grid,
        data_mesh_grid=None,
        hyper_data=None,
        settings=None,
        preloads=None,
        profiling_dict=None,
    ):
        return self.mapper

    def data_mesh_grid_from(self, data_grid_slim, hyper_data, settings=None):

        if hyper_data is not None and self.data_mesh_grid is not None:
            return hyper_data * self.data_mesh_grid

        return self.data_mesh_grid
