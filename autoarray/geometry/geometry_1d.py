from __future__ import annotations
import logging
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from autoarray.structures.grids.uniform_1d import Grid1D
    from autoarray.mask.mask_2d import Mask2D

from autoarray.mask.abstract_mask import Mask

from autoarray import exc
from autoarray.structures.arrays import array_1d_util
from autoarray.structures.grids import grid_1d_util
from autoarray import type as ty

logging.basicConfig()
logger = logging.getLogger(__name__)


class Geometry1D:
    def __init__(
        self,
        shape_native: Tuple[int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float] = (0.0,),
    ):

        self.shape_native = shape_native
        self.pixel_scales = pixel_scales
        self.origin = origin

    @property
    def shape_slim_scaled(self) -> Tuple[float]:
        return (float(self.pixel_scales[0] * self.shape_native[0]),)

    @property
    def scaled_maxima(self) -> Tuple[float]:
        return (float(self.shape_slim_scaled[0] / 2.0 + self.origin[0]),)

    @property
    def scaled_minima(self) -> Tuple[float]:
        return (-float(self.shape_slim_scaled[0] / 2.0) + self.origin[0],)

    @property
    def extent(self):
        return np.array([self.scaled_minima[0], self.scaled_maxima[0]])
