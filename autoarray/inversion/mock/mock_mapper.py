import numpy as np
from typing import Optional, Tuple

from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper


class MockMapper(AbstractMapper):
    def __init__(
        self,
        mask=None,
        mesh=None,
        mesh_geometry=None,
        interpolator=None,
        source_plane_data_grid=None,
        source_plane_mesh_grid=None,
        over_sampler=None,
        border_relocator=None,
        adapt_data=None,
        regularization=None,
        mapping_matrix=None,
        pixel_signals=None,
        parameters=None,
    ):

        super().__init__(
            mask=mask,
            mesh=mesh,
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=source_plane_mesh_grid,
            adapt_data=adapt_data,
            border_relocator=border_relocator,
            regularization=regularization,
        )

        self._interpolator = interpolator
        self._mesh_geometry = mesh_geometry
        self._over_sampler = over_sampler
        self._mapping_matrix = mapping_matrix
        self._parameters = parameters
        self._pixel_signals = pixel_signals

    def pixel_signals_from(self, signal_scale, xp=np):
        if self._pixel_signals is None:
            return super().pixel_signals_from(signal_scale=signal_scale)
        return self._pixel_signals

    @property
    def interpolator(self):
        if self._interpolator is None:
            return super().interpolator
        return self._interpolator

    @property
    def mesh_geometry(self):
        if self._mesh_geometry is None:
            return super().mesh_geometry
        return self._mesh_geometry

    @property
    def params(self):
        if self._parameters is None:
            return super().params
        return self._parameters

    @property
    def over_sampler(self):
        if self._over_sampler is None:
            return super().over_sampler
        return self._over_sampler

    @property
    def mapping_matrix(self):
        return self._mapping_matrix
