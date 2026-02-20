import numpy as np


class AbstractMeshGeometry:

    def __init__(self, mesh, mesh_grid, data_grid, mesh_weight_map=None, xp=np):

        self.mesh = mesh
        self.mesh_grid = mesh_grid
        self.data_grid = data_grid
        self.mesh_weight_map = mesh_weight_map
        self._xp = xp
