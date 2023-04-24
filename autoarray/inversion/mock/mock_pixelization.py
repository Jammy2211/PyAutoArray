from autoarray.inversion.pixelization.pixelization import Pixelization


class MockPixelization(Pixelization):
    def __init__(
        self, mesh=None, regularization=None, mapper=None, image_plane_mesh_grid=None
    ):
        super().__init__(mesh=mesh, regularization=regularization)

        self.mapper = mapper
        self.image_plane_mesh_grid = image_plane_mesh_grid

    # noinspection PyUnusedLocal,PyShadowingNames
    def mapper_grids_from(
        self,
        source_plane_data_grid,
        source_plane_mesh_grid,
        image_plane_mesh_grid=None,
        hyper_data=None,
        settings=None,
        preloads=None,
        profiling_dict=None,
    ):
        return self.mapper

    def image_plane_mesh_grid_from(
        self, image_plane_data_grid, hyper_data, settings=None
    ):
        if hyper_data is not None and self.image_plane_mesh_grid is not None:
            return hyper_data * self.image_plane_mesh_grid

        return self.image_plane_mesh_grid
