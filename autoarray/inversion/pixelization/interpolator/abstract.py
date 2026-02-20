import numpy as np


class AbstractInterpolator:

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        preloads=None,
        _xp=np,
    ):
        self.mesh = mesh
        self.mesh_grid = mesh_grid
        self.data_grid = data_grid
        self.preloads = preloads
        self._xp = _xp

    @property
    def _interpolation_and_weights(self):
        raise NotImplementedError(
            "Subclasses of AbstractInterpolator must implement the _interpolation_and_weights property."
        )

    @property
    def weights(self):
        _, weights = self._interpolation_and_weights
        return weights

    @property
    def mappings(self):
        mappings, _ = self._interpolation_and_weights
        return mappings
