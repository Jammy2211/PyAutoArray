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
    def _mappings_sizes_weights(self):
        raise NotImplementedError(
            "Subclasses of AbstractInterpolator must implement the _mappings_sizes_weights property."
        )

    @property
    def mappings(self):
        mappings, _, _ = self._mappings_sizes_weights
        return mappings

    @property
    def sizes(self):
        _, sizes, _ = self._mappings_sizes_weights
        return sizes

    @property
    def weights(self):
        _, _, weights = self._mappings_sizes_weights
        return weights
