import numpy as np
from typing import Optional, Tuple

from autoarray.inversion.mappers.abstract import AbstractMapper


class MockMapper(AbstractMapper):
    def __init__(
        self,
        source_grid_slim=None,
        source_mesh_grid=None,
        hyper_image=None,
        pix_sub_weights=None,
        mapping_matrix=None,
        pixel_signals=None,
        pixels=None,
        interpolated_array=None,
    ):

        super().__init__(
            source_grid_slim=source_grid_slim,
            source_mesh_grid=source_mesh_grid,
            hyper_image=hyper_image,
        )

        self._pix_sub_weights = pix_sub_weights

        self._mapping_matrix = mapping_matrix

        self._pixels = pixels

        self._pixel_signals = pixel_signals

        self._interpolated_array = interpolated_array

    def pixel_signals_from(self, signal_scale):
        if self._pixel_signals is None:
            return super().pixel_signals_from(signal_scale=signal_scale)
        return self._pixel_signals

    @property
    def pixels(self):
        if self._pixels is None:
            return super().pixels
        return self._pixels

    @property
    def pix_sub_weights(self):
        return self._pix_sub_weights

    @property
    def mapping_matrix(self):
        return self._mapping_matrix

    def interpolated_array_from(
        self,
        values: np.ndarray,
        shape_native: Tuple[int, int] = (401, 401),
        extent: Optional[Tuple[float, float, float, float]] = None,
    ):
        return self._interpolated_array
